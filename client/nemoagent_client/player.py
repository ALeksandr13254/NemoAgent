"""Gapless audio player: one persistent OutputStream fed from a queue (from the OmniVoice pipeline)."""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("player")


IDLE_CLOSE_S = 20.0   # close the output stream after this much silence; it is re-opened on the next feed


class StreamPlayer:
    """Gapless player. The PortAudio stream is opened lazily and closed after ~20 s of silence:
    a stream that stays open for hours keeps pointing at the endpoint it was opened on, and when
    Windows/FxSound re-creates the default device (headphones plugged, enhancer restarted) that old
    endpoint plays into nothing. Re-opening per utterance costs ~20 ms and always hits the current
    default device.
    """

    def __init__(self, samplerate: int, device=None, blocksize: int = 1024):
        import sounddevice as sd
        self._sd = sd
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device_spec = device
        self.q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._cur = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._last_audio_at = 0.0
        self._generation = 0
        self.samples_played = 0
        self.samples_fed = 0
        self.device_rate = samplerate       # rate of the opened stream (may differ -> we resample on feed)
        self._stream = None
        self.device_name = ""
        self._open_stream()                 # fail early if the device is wrong
        self._closed = False
        threading.Thread(target=self._idle_watch, daemon=True, name="player-idle").start()

    def _open_stream(self) -> None:
        sd = self._sd
        dev = resolve_device(self.device_spec)
        self._stream = self._open(dev, self.samplerate, self.blocksize)
        self._stream.start()
        self._last_audio_at = time.time()
        try:
            self.device_name = sd.query_devices(self._stream.device)["name"]
        except Exception:
            self.device_name = str(self.device_spec or "default")
        log.info("speaker output: %s @ %d Hz", self.device_name, self.device_rate)

    def _close_stream(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def _idle_watch(self) -> None:
        while not getattr(self, "_closed", False):
            time.sleep(2.0)
            with self._lock:
                idle = time.time() - self._last_audio_at
                if self._stream is not None and not self._playing.is_set() and self.q.empty() and idle > IDLE_CLOSE_S:
                    self._close_stream()
                    log.debug("output stream closed after %.0fs idle", idle)

    def _open(self, dev, samplerate: int, blocksize: int):
        """Open the output stream; WASAPI shared mode needs the device's own rate, so fall back to
        auto-conversion and finally to opening at the device rate and resampling ourselves."""
        sd = self._sd
        attempts = [dict(samplerate=samplerate)]
        try:
            api_name = sd.query_hostapis(sd.query_devices(dev if dev is not None else sd.default.device[1])["hostapi"])["name"]
        except Exception:
            api_name = ""
        if "WASAPI" in api_name:
            attempts.append(dict(samplerate=samplerate, extra_settings=sd.WasapiSettings(auto_convert=True)))
        try:
            native = int(sd.query_devices(dev if dev is not None else sd.default.device[1])["default_samplerate"])
            if native != samplerate:
                attempts.append(dict(samplerate=native))
        except Exception:
            pass
        last = None
        for kw in attempts:
            try:
                stream = sd.OutputStream(channels=1, dtype="float32", blocksize=blocksize, device=dev,
                                         callback=self._callback, **kw)
                self.device_rate = int(kw["samplerate"])
                return stream
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"cannot open output device {dev!r}: {last}")

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        if self.device_rate == self.samplerate or audio.size == 0:
            return audio
        n_out = int(round(audio.size * self.device_rate / self.samplerate))
        x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def feed(self, audio: np.ndarray, generation: int | None = None) -> None:
        if generation is not None and generation != self._generation:
            return  # stale audio from a cancelled utterance
        with self._lock:
            if self._stream is None:
                try:
                    self._open_stream()
                except Exception as e:  # noqa: BLE001
                    log.warning("cannot open output device: %s", e)
                    return
        arr = self._resample(np.asarray(audio, dtype=np.float32).reshape(-1))
        self.samples_fed += arr.size
        self._last_audio_at = time.time()
        self.q.put(arr)
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
            self.samples_played += n
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
        self._closed = True
        with self._lock:
            self._close_stream()


def resolve_device(spec) -> int | str | None:
    """'' / None -> default; digits -> index; otherwise a (partial, case-insensitive) device name."""
    import sounddevice as sd
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "default":
        return None
    s = str(spec).strip()
    if s.isdigit():
        return int(s)
    wanted = s.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and wanted in d["name"].lower():
            return i
    return s  # let PortAudio complain


_API_RANK = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2, "Windows WDM-KS": 3}
_HIDDEN = ("переназначение звуковых", "sound mapper", "первичный звуковой", "primary sound")


def _list_devices(kind: str) -> list[dict]:
    """Devices worth showing: one entry per physical device (MME truncates names to 31 chars, so the
    key is the first 30), preferring WASAPI (lowest latency on Windows)."""
    import sounddevice as sd
    apis = sd.query_hostapis()
    field = "max_output_channels" if kind == "output" else "max_input_channels"
    seen: dict[str, dict] = {}
    for i, d in enumerate(sd.query_devices()):
        if d[field] <= 0:
            continue
        name = d["name"].strip()
        if any(h in name.lower() for h in _HIDDEN):
            continue
        api = apis[d["hostapi"]]["name"]
        key = name.lower()[:30]
        entry = {"index": i, "name": name, "api": api}
        if key not in seen or _API_RANK.get(api, 9) < _API_RANK.get(seen[key]["api"], 9):
            seen[key] = entry
    try:
        default = sd.query_devices(kind=kind)["name"].strip()
    except Exception:
        default = ""
    return [{"index": None, "name": f"по умолчанию ({default})" if default else "по умолчанию", "api": ""}] + sorted(seen.values(), key=lambda e: e["name"])


def list_output_devices() -> list[dict]:
    return _list_devices("output")


def list_input_devices() -> list[dict]:
    return _list_devices("input")
