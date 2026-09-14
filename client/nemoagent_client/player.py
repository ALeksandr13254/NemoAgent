"""Gapless audio player: one persistent OutputStream fed from a queue (from the OmniVoice pipeline)."""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("player")


class StreamPlayer:
    def __init__(self, samplerate: int, device=None, blocksize: int = 1024):
        import sounddevice as sd
        self._sd = sd
        self.samplerate = samplerate
        self.q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._cur = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._last_audio_at = 0.0
        self._generation = 0
        self._stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32", blocksize=blocksize,
                                       device=device, callback=self._callback)
        self._stream.start()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def feed(self, audio: np.ndarray, generation: int | None = None) -> None:
        if generation is not None and generation != self._generation:
            return  # stale audio from a cancelled utterance
        self.q.put(np.asarray(audio, dtype=np.float32).reshape(-1))
        self._playing.set()

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        out = outdata[:, 0]
        i = 0
        while i < frames:
            if self._cur.size == 0:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    out[i:] = 0.0
                    if self._playing.is_set() and time.time() - self._last_audio_at > 0.25:
                        self._playing.clear()
                    return
                if item is None:
                    continue
                self._cur = item
            n = min(frames - i, self._cur.size)
            out[i:i + n] = self._cur[:n]
            self._cur = self._cur[n:]
            i += n
        self._last_audio_at = time.time()

    def flush(self) -> int:
        """Drop everything queued and playing; returns a new generation number."""
        with self._lock:
            self._generation += 1
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            self._cur = np.zeros(0, dtype=np.float32)
            self._playing.clear()
            return self._generation

    @property
    def generation(self) -> int:
        return self._generation

    def wait_idle(self, timeout: float = 30.0) -> None:
        t0 = time.time()
        while (self._playing.is_set() or not self.q.empty()) and time.time() - t0 < timeout:
            time.sleep(0.02)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
