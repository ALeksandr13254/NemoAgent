"""Microphone capture with Silero VAD end-pointing, push-to-talk and barge-in detection."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .config import settings
from .vad import SileroVAD

log = logging.getLogger("mic")

SR = 16000
FRAME = 512  # 32 ms


class Microphone:
    def __init__(self, on_utterance: Callable[[np.ndarray, float], None], on_speech_start: Callable[[], None],
                 is_agent_speaking: Callable[[], bool], on_level: Optional[Callable[[float, bool], None]] = None,
                 device=None):
        self.on_utterance = on_utterance
        self.on_speech_start = on_speech_start
        self.is_agent_speaking = is_agent_speaking
        self.on_level = on_level or (lambda level, speech: None)
        self.vad = SileroVAD()
        self.enabled = settings.AUTO_LISTEN
        self.ptt = False          # push-to-talk held
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._device = device
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mic-vad")
        self._running = True
        self._thread.start()
        self._open(device)

    def _open(self, device) -> None:
        import sounddevice as sd
        dev = device
        if isinstance(dev, str) and dev.strip():
            dev = dev.strip()
            if dev.isdigit():
                dev = int(dev)
        elif not dev:
            dev = None
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME, device=dev,
                                      callback=self._callback)
        self._stream.start()
        log.info("microphone open: %s", sd.query_devices(self._stream.device)["name"] if self._stream.device is not None else "default")

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        self._q.put(indata[:, 0].copy())

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._reset_state()

    def set_ptt(self, held: bool) -> None:
        self.ptt = held

    def _reset_state(self) -> None:
        self.vad.reset()

    # ------------------------------------------------------------------ VAD loop
    def _loop(self) -> None:
        pre_roll_frames = max(1, settings.VAD_PRE_ROLL_MS // 32)
        min_speech = max(1, settings.VAD_MIN_SPEECH_MS // 32)
        silence_frames = max(1, settings.VAD_SILENCE_MS // 32)
        barge_frames = max(1, settings.BARGE_IN_MIN_SPEECH_MS // 32)
        max_frames = settings.VAD_MAX_UTTERANCE_S * SR // FRAME

        pre: list[np.ndarray] = []
        collecting: list[np.ndarray] = []
        speech_run = 0
        silence_run = 0
        in_utterance = False
        started_at = 0.0
        ptt_prev = False
        barge_notified = False

        while self._running:
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame.shape[0] != FRAME:
                continue
            level = float(np.sqrt(np.mean(frame ** 2)))

            # ---------------- push-to-talk: record while held, no VAD
            if self.ptt or ptt_prev:
                if self.ptt and not ptt_prev:
                    collecting = list(pre)
                    started_at = time.time()
                    self.on_level(level, True)
                if self.ptt:
                    collecting.append(frame)
                    self.on_level(level, True)
                else:  # released
                    if collecting and len(collecting) > min_speech:
                        self.on_utterance(np.concatenate(collecting), time.time() - started_at)
                    collecting = []
                ptt_prev = self.ptt
                in_utterance = False
                speech_run = silence_run = 0
                continue

            if not self.enabled:
                pre.append(frame)
                if len(pre) > pre_roll_frames:
                    pre.pop(0)
                self.on_level(level, False)
                continue

            prob = self.vad(frame)
            agent_talking = self.is_agent_speaking()
            # while the agent talks through the speakers, be stricter to ignore echo
            threshold = settings.VAD_THRESHOLD + (0.25 if agent_talking else 0.0)
            is_speech = prob >= min(0.95, threshold)
            self.on_level(level, is_speech)

            if not in_utterance:
                pre.append(frame)
                if len(pre) > pre_roll_frames:
                    pre.pop(0)
                if is_speech:
                    speech_run += 1
                    if agent_talking and settings.BARGE_IN and speech_run >= barge_frames and not barge_notified:
                        barge_notified = True
                        self.on_speech_start()
                    if speech_run >= min_speech and (not agent_talking or settings.BARGE_IN):
                        in_utterance = True
                        collecting = list(pre)
                        started_at = time.time()
                        silence_run = 0
                else:
                    speech_run = max(0, speech_run - 1)
                    if speech_run == 0:
                        barge_notified = False
                continue

            collecting.append(frame)
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1
            if silence_run >= silence_frames or len(collecting) >= max_frames:
                audio = np.concatenate(collecting)
                duration = time.time() - started_at
                in_utterance = False
                collecting = []
                speech_run = silence_run = 0
                barge_notified = False
                self.vad.reset()
                # trim trailing silence a bit (keep 150 ms)
                keep = max(0, len(audio) - (silence_frames - 5) * FRAME)
                audio = audio[:max(keep, FRAME * min_speech)]
                self.on_utterance(audio, duration)

    def close(self) -> None:
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
