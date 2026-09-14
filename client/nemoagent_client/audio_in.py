"""Microphone capture with Silero VAD end-pointing, push-to-talk and barge-in.

While the agent talks the VAD threshold is raised (speaker bleed) and a longer run of speech is
needed; once it is reached `on_speech_start` fires (the core stops the playback) and the utterance
is collected and transcribed like any other.
"""
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
    def __init__(self, on_utterance: Callable[[np.ndarray, float, bool], None], on_speech_start: Callable[[], None],
                 is_agent_speaking: Callable[[], bool], on_level: Optional[Callable[[float, bool], None]] = None,
                 device=None):
        self.on_utterance = on_utterance          # (audio16k, duration_s, during_agent_speech)
        self.on_speech_start = on_speech_start    # UI hint only ("possible barge-in")
        self.is_agent_speaking = is_agent_speaking
        self.on_level = on_level or (lambda level, speech: None)
        self.vad = SileroVAD()
        self.enabled = settings.AUTO_LISTEN
        self.ptt = False          # push-to-talk held
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._device = device
        self._rate = SR
        self.device_name = ""
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
            else:  # partial device name
                wanted = dev.lower()
                dev = next((i for i, d in enumerate(sd.query_devices())
                            if d["max_input_channels"] > 0 and wanted in d["name"].lower()), dev)
        elif not dev:
            dev = None
        # 16 kHz capture: WASAPI shared mode may refuse it -> auto-convert, then native rate + decimation
        attempts = [dict(samplerate=SR)]
        try:
            info = sd.query_devices(dev if dev is not None else sd.default.device[0])
            if "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]:
                attempts.append(dict(samplerate=SR, extra_settings=sd.WasapiSettings(auto_convert=True)))
            native = int(info["default_samplerate"])
            if native != SR:
                attempts.append(dict(samplerate=native))
        except Exception:
            pass
        last = None
        for kw in attempts:
            try:
                rate = int(kw["samplerate"])
                block = FRAME * rate // SR
                self._stream = sd.InputStream(channels=1, dtype="float32", blocksize=block, device=dev,
                                              callback=self._callback, **kw)
                self._rate = rate
                self._stream.start()
                break
            except Exception as e:  # noqa: BLE001
                last = e
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"cannot open microphone {device!r}: {last}")
        self._device = device
        try:
            self.device_name = sd.query_devices(self._stream.device)["name"]
        except Exception:
            self.device_name = str(device or "default")
        log.info("microphone open: %s @ %d Hz", self.device_name, self._rate)

    def reopen(self, device) -> None:
        """Switch to another input device (settings dropdown)."""
        old = self._stream
        self._stream = None
        try:
            if old is not None:
                old.stop()
                old.close()
        except Exception:
            pass
        self._open(device)
        self.vad.reset()

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        mono = indata[:, 0]
        if self._rate != SR:   # device opened at its native rate: resample to 16 kHz for the VAD/STT
            n_out = int(round(mono.size * SR / self._rate))
            mono = np.interp(np.linspace(0.0, 1.0, num=n_out, endpoint=False),
                             np.linspace(0.0, 1.0, num=mono.size, endpoint=False), mono).astype(np.float32)
        self._q.put(mono.copy())

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
        during_speech = False

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
                        self.on_utterance(np.concatenate(collecting), time.time() - started_at, False)
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
                    # while the agent talks we need a longer run before trusting it (speaker bleed)
                    need = barge_frames if agent_talking else min_speech
                    if speech_run >= need and (not agent_talking or settings.BARGE_IN):
                        in_utterance = True
                        during_speech = agent_talking
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
                self.on_utterance(audio, duration, during_speech)
                during_speech = False

    def close(self) -> None:
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
