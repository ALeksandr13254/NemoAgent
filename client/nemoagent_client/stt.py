"""Speech-to-text with faster-whisper (CTranslate2). GPU float16 when available."""
from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

from .config import settings

log = logging.getLogger("stt")


def _collapse_repeats(text: str) -> str:
    """Whisper loops on noise ('Привет. Привет. Привет…'); keep at most two repeats."""
    if len(text) < 12:
        return text
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out: list[str] = []
    for p in parts:
        if len(out) >= 2 and p == out[-1] == out[-2]:
            continue
        out.append(p)
    words = " ".join(out).split()
    out2: list[str] = []
    for w in words:
        if len(out2) >= 2 and w == out2[-1] == out2[-2]:
            continue
        out2.append(w)
    return " ".join(out2)


_HALLUCINATIONS = {"субтитры", "продолжение следует", "thank you", "thanks for watching", "спасибо за просмотр",
                   "редактор субтитров", "субтитры сделал", "субтитры создавал", "you"}


class SpeechToText:
    def __init__(self) -> None:
        from .cuda import setup_cuda_paths
        setup_cuda_paths()
        from faster_whisper import WhisperModel

        device = settings.STT_DEVICE
        compute = settings.STT_COMPUTE
        if device == "auto":
            device = "cuda" if self._cuda_ok() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(settings.STT_MODEL, device=device, compute_type=compute)
        except Exception as e:  # noqa: BLE001
            if device == "cuda":
                log.warning("CUDA STT failed (%s) — falling back to CPU int8", e)
                device, compute = "cpu", "int8"
                self.model = WhisperModel(settings.STT_MODEL, device=device, compute_type=compute)
            else:
                raise
        self.device = device
        self._lock = threading.Lock()
        log.info("STT %s on %s/%s loaded in %.1fs", settings.STT_MODEL, device, compute, time.perf_counter() - t0)
        self.warmup()

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def warmup(self) -> None:
        try:
            self.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            log.warning("STT warmup failed: %s", e)

    ALLOWED_AUTO = ("ru", "en")   # "auto" chooses between these two; anything else is noise heard as Japanese etc.

    def transcribe(self, audio16k: np.ndarray) -> str:
        lang = None if settings.STT_LANGUAGE in ("auto", "", None) else settings.STT_LANGUAGE
        kw = dict(beam_size=max(1, settings.STT_BEAM), vad_filter=False, condition_on_previous_text=False,
                  temperature=0.0, without_timestamps=True)
        with self._lock:
            segments, info = self.model.transcribe(audio16k, language=lang, **kw)
            text = " ".join(s.text.strip() for s in segments).strip()
            if lang is None and getattr(info, "language", None) not in self.ALLOWED_AUTO:
                # whisper "detected" Japanese/Korean/… on breathing or keyboard noise: redo as Russian and
                # let the hallucination filter decide
                log.info("STT auto picked %s (p=%.2f) — redoing as Russian", info.language, info.language_probability or 0)
                segments, info = self.model.transcribe(audio16k, language="ru", **kw)
                text = " ".join(s.text.strip() for s in segments).strip()
        text = _collapse_repeats(text)
        if text.strip(" .!?…").lower() in _HALLUCINATIONS:
            return ""
        if not re.search(r"[A-Za-zА-Яа-яЁё]", text):   # no Latin or Cyrillic letters at all: not speech we want
            return ""
        return text
