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
                 device=None, on_health: Optional[Callable[[str], None]] = None):
        self.on_utterance = on_utterance          # (audio16k, duration_s, during_agent_speech)
        self.on_speech_start = on_speech_start    # UI hint only ("possible barge-in")
        self.is_agent_speaking = is_agent_speaking
        self.on_level = on_level or (lambda level, speech: None)
        self.on_health = on_health or (lambda text: None)   # "ready" / "no audio from the device" for the UI
        self.vad = SileroVAD()
        self.enabled = settings.AUTO_LISTEN
        self.ptt = False          # push-to-talk held
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._device = device
        self._rate = SR
        self.device_name = ""
        self.health = "ready"
        self._last_frame_at = time.time()
        self._last_reopen_at = 0.0
        self._frames_seen = 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mic-vad")
        self._running = True
        self._thread.start()
        self._open(device)

    def _candidates(self, device) -> list[dict]:
        """Ways to open the microphone, best first. Each is {device, samplerate, extra_settings?}.

        The default device is the MME entry, and an MME stream at 16 kHz on this webcam opens fine but
        delivers no audio after a restart (and blocks the endpoint for every other program while it is
        open), so the WASAPI entry of the same microphone at its native rate goes first; we resample."""
        import sounddevice as sd
        devs = sd.query_devices()
        apis = sd.query_hostapis()

        def api_of(i: int) -> str:
            return apis[devs[i]["hostapi"]]["name"]

        dev = device
        if isinstance(dev, str) and dev.strip():
            dev = dev.strip()
            if dev.isdigit():
                dev = int(dev)
            else:  # partial device name
                wanted = dev.lower()
                dev = next((i for i, d in enumerate(devs) if d["max_input_channels"] > 0 and wanted in d["name"].lower()), None)
        elif not dev:
            dev = None
        base = dev if dev is not None else sd.default.device[0]
        if base is None or base < 0 or base >= len(devs):
            return [dict(device=None, samplerate=SR)]
        name = devs[base]["name"]
        # the same physical microphone under every host API (MME truncates names, so compare prefixes)
        key = name.rstrip(")").lower()[:28]
        same = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0 and d["name"].rstrip(")").lower()[:28] == key]
        out: list[dict] = []
        for i in sorted(same, key=lambda i: (0 if "WASAPI" in api_of(i) else 1 if i == base else 2 if "DirectSound" in api_of(i) else 3)):
            native = int(devs[i]["default_samplerate"] or SR)
            if "WASAPI" in api_of(i):
                out.append(dict(device=i, samplerate=native))
                out.append(dict(device=i, samplerate=SR, extra_settings=sd.WasapiSettings(auto_convert=True)))
            elif "WDM-KS" in api_of(i):
                continue   # exclusive: would lock the microphone for other programs
            else:
                out.append(dict(device=i, samplerate=SR))
                if native != SR:
                    out.append(dict(device=i, samplerate=native))
        return out or [dict(device=base, samplerate=SR)]

    def _open(self, device) -> None:
        import sounddevice as sd
        last = None
        for kw in self._candidates(device):
            rate = int(kw["samplerate"])
            block = FRAME * rate // SR
            self._frames_seen = 0
            try:
                stream = sd.InputStream(channels=1, dtype="float32", blocksize=block, callback=self._callback, **kw)
                stream.start()
            except Exception as e:  # noqa: BLE001
                last = e
                continue
            # an "open" stream is not proof of anything: wait for real frames before trusting it
            deadline = time.time() + 1.5
            while time.time() < deadline and self._frames_seen < 3:
                time.sleep(0.05)
            if self._frames_seen >= 3:
                self._stream = stream
                self._rate = rate
                break
            last = RuntimeError("stream opened but delivers no audio")
            log.warning("microphone %r @ %d Hz delivers nothing — trying the next way", kw.get("device"), rate)
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if self._stream is None:
            raise RuntimeError(f"cannot open microphone {device!r}: {last}")
        self._device = device
        self._last_frame_at = time.time()
        try:
            self.device_name = sd.query_devices(self._stream.device)["name"]
        except Exception:
            self.device_name = str(device or "default")
        log.info("microphone open: %s @ %d Hz", self.device_name, self._rate)

    def _watchdog(self) -> None:
        """No frames from the device for a while although the stream is 'open': Windows re-created the
        endpoint or PortAudio silently stopped the stream. Re-open it (at most once per 5 s) and tell the UI."""
        now = time.time()
        if now - self._last_frame_at < 3.0:
            return
        if self.health == "ready":
            self.health = "нет звука с микрофона"
            self.on_health(self.health)
            log.warning("microphone %s delivers no audio — reopening", self.device_name)
        if now - self._last_reopen_at < 5.0:
            return
        self._last_reopen_at = now
        try:
            self.reopen(self._device)
        except Exception as e:  # noqa: BLE001
            log.warning("microphone reopen failed: %s", e)

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
        self._frames_seen += 1
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
                self._watchdog()
                continue
            self._last_frame_at = time.time()
            if self.health != "ready":
                self.health = "ready"
                self.on_health(self.health)
                log.info("microphone delivers audio again")
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
