"""TeraTTSv2 engine + text preparation for streaming speech.

Speed tricks (all measured on an RTX 4070 Ti SUPER, see README):
  * the ONNX graphs are driven directly (no transformers AutoModel wrapper, no torch);
  * CUDAExecutionProvider when available (first audio ~0.2 s vs ~0.6 s on CPU), CPU fallback;
  * `generate_speech_stream` with small vocoder chunks — playback starts before the vocoder
    finishes the sentence;
  * the LLM reply is split into sentences while it streams, so the first sentence is being
    synthesised while the model is still writing the rest (see speech.py);
  * voices are picked per sentence by script: Cyrillic -> Russian voice, otherwise English voice;
    mixed sentences get <ru>/<en> tags per word run so both languages are pronounced correctly.
"""
from __future__ import annotations

import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .config import settings
from .translit import is_russian_context, transliterate_latin

log = logging.getLogger("tts")

SAMPLE_RATE = 44100
_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-z]")

# ----------------------------------------------------------------- text cleaning
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⬀-⯿️‍❤]+")


def clean_for_tts(text: str) -> str:
    """Strip everything that must not be read aloud (markdown, code, links, emoji)."""
    if not text:
        return ""
    t = text
    t = re.sub(r"```.*?```", " . ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[(?:citation|reference)[^\]]*\]", "", t, flags=re.I)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*[-*+][ \t]+", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*\d+[.)][ \t]+", "", t, flags=re.M)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"\*[^*\n]{1,60}\*", " ", t)          # *stage directions*
    t = re.sub(r"(?:^|(?<=\s))(?::3|;3|:[DPp]|[xX]D|\^\^+|<3+|:\)+|:\(+|;\)+)(?=[\s.,!?]|$)", " ", t)
    t = re.sub(r"\s+([.!?,;:])", r"\1", t)
    t = re.sub(r"[*_#>|~^]{1,}", " ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = t.replace("—", ", ").replace("–", ", ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def has_speech(text: str) -> bool:
    return bool(re.search(r"[\wЀ-ӿ]", text or ""))


# ----------------------------------------------------------------- vocabulary sanitizer
# TeraTTSv2 knows 134 characters: letters (й/ё via NFKD decomposition), space, + and
# . , ! ? : ; - ( ) « » " ' / < >. Everything else is silently skipped, which glues neighbouring
# words together ("это—тест" -> "этотест"). We map the usual typography to supported punctuation
# and turn the rest into spaces. Digits are kept: the runtime expands them with num2words.
_CHAR_MAP = {
    "—": ", ", "–": ", ", "―": ", ", "…": ". ", "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'",
    " ": " ", " ": " ", " ": " ", "\t": " ", "\n": ". ", "\r": " ",
    "%": " процентов ", "№": " номер ", "°": " градусов ", "€": " евро ", "₽": " рублей ", "$": " долларов ",
    "&": " и ", "+": "+", "/": "/",
}
_ALLOWED = set(" !\"'()+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz«»"
               "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзиклмнопрстуфхцчшщъыьэюяЁёЙй0123456789")


def sanitize_vocab(text: str) -> str:
    """Keep only what the TeraTTS character table can encode; fix typography instead of dropping it."""
    out: list[str] = []
    for ch in text:
        if ch in _CHAR_MAP:
            out.append(_CHAR_MAP[ch])
        elif ch in _ALLOWED:
            out.append(ch)
        elif ch.isalpha():
            out.append(ch)          # other scripts: the runtime skips them with a warning
        else:
            out.append(" ")
    t = "".join(out)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    # a space after punctuation glued to the next word, but keep decimals like 24,9 / 3.5 intact
    t = re.sub(r"(?<!\d)([.,!?;:])(?=[^\s.,!?;:)\"»'])|(?<=\d)([!?;:])(?=[^\s.,!?;:)\"»'])|(?<=\d)([.,])(?=[^\s\d.,!?;:)\"»'])",
               lambda m: (m.group(1) or m.group(2) or m.group(3)) + " ", t)
    t = re.sub(r"[ ]{2,}", " ", t).strip()
    return t


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё̆̈+'-]+")

# Things that must never reach the vocoder even if the model put them into `speech`: shell
# commands, code, paths, URLs. A sentence is cut from the first code-like token to its end and the
# cut is replaced by a short spoken note; the exact text stays on screen (display).
_CODE_TOKEN = re.compile(
    r"[|{}\[\]$\\<>=`~^#@]|://|::|->|=>|\s--?[a-zA-Z]+\b|\b\w+\.(?:exe|dll|py|js|ts|json|txt|docx?|pdf|xlsx?|csv|md|bat|ps1|sh|yaml|yml|ini|log|png|jpe?g|zip)\b"
    r"|\b[A-Za-z]:[\\/]|\b(?:Get|Set|Select|Sort|Where|New|Remove|Start|Stop|Invoke|Format|Out|Write|Test|Add)-[A-Z]\w+"
    r"|\b\w+-Object\b|\b(?:sudo|apt|pip|npm|git|docker|cmd|powershell|bash|python)\s+\S+\S*|\bwww\.\S+")


def scrub_code(sentence: str) -> str:
    m = _CODE_TOKEN.search(sentence)
    if not m:
        return sentence
    start = sentence.rfind(" ", 0, m.start()) + 1
    head = sentence[:start].rstrip(" :;,—-–(")
    ru = bool(_CYR.search(sentence))
    if not re.search(r"[A-Za-zА-Яа-яЁё]{3,}", head):
        return "Команда показана на экране." if ru else "The command is shown on screen."
    note = "команда на экране" if ru else "shown on screen"
    end = "." if not sentence.rstrip().endswith(("!", "?")) else sentence.rstrip()[-1]
    return f"{head}, {note}{end}"


def accentize_keep_manual(text: str, accentizer) -> str:
    """Run RUAccent over Russian text but keep the caller's `+` marks.

    The bundled runtime skips *all* automatic stressing of a <ru> span as soon as it sees one manual
    `+` (manual marks are "authoritative"), so a single homograph fixed by the model would leave every
    other word unstressed. Here the manual words are protected and the rest still gets marked.
    """
    if accentizer is None or not text:
        return text
    if "+" not in text:
        try:
            return accentizer.process_all(text)
        except Exception:  # noqa: BLE001
            return text
    original = _WORD_RE.findall(text)
    try:
        stressed = accentizer.process_all(text.replace("+", ""))
    except Exception:  # noqa: BLE001
        return text
    result = _WORD_RE.findall(stressed)
    if len(result) == len(original):
        # same word sequence: restore manual marks by position (two homographs with different
        # stress in one sentence — "з+амок … зам+ок" — must keep their own marks)
        idx = 0

        def by_position(m: re.Match) -> str:
            nonlocal idx
            word = m.group(0)
            src = original[idx] if idx < len(original) else word
            idx += 1
            return src if "+" in src else word

        return _WORD_RE.sub(by_position, stressed)
    # tokenisation drifted (rare): fall back to matching by spelling
    manual = {w.replace("+", "").lower(): w for w in original if "+" in w}
    return _WORD_RE.sub(lambda m: manual.get(m.group(0).replace("+", "").lower(), m.group(0)), stressed)


# ----------------------------------------------------------------- language tagging
def tag_languages(text: str, force: Optional[str] = None) -> tuple[str, str]:
    """Return (tagged_text, dominant_language) with <ru>/<en> spans per word run.

    force="ru" / "en" (the "Язык озвучки" setting) pins the voice: every Latin word and every
    number is read by that voice; Cyrillic words always stay Russian (the English voice cannot
    pronounce them)."""
    tokens = re.findall(r"\S+|\s+", text)
    runs: list[list[str]] = []   # [lang, text]
    # Russian sentences are full of Latin product names ("открой Visual Studio Code"): lean towards
    # the Russian voice unless the sentence is clearly English (same rule as the transliteration).
    # A sentence without any Latin letters (digits only, "24,9.") is Russian too.
    default = "ru" if (is_russian_context(text) or not _LAT.search(text)) else "en"
    if force in ("ru", "en"):
        default = force
    for tok in tokens:
        if tok.isspace():
            if runs:
                runs[-1][1] += tok
            continue
        if _CYR.search(tok):
            lang = "ru"
        elif _LAT.search(tok):
            lang = "ru" if force == "ru" else "en"
        else:
            # digits and punctuation follow the language of the SENTENCE, not of the previous word:
            # "RTX 4070" inside Russian speech must give "четыре тысячи семьдесят", not "forty seventy"
            lang = default
        if runs and runs[-1][0] == lang:
            runs[-1][1] += tok
        else:
            runs.append([lang, tok])
    if not runs:
        return "", default
    parts = []
    for lang, chunk in runs:
        chunk = chunk.strip()
        if not chunk:
            continue
        # The model vocabulary has no '<' or '>' outside tags; drop them to be safe.
        chunk = chunk.replace("<", " ").replace(">", " ")
        parts.append(f"<{lang}>{chunk}</{lang}>")
    return " ".join(parts), default


# ----------------------------------------------------------------- Russian dates and times
# The runtime expands plain numbers with num2words but skips "15.09.2026" and "15:02" (they are not
# numbers to it), so the vocoder would get raw digits — read aloud as English digits or dropped.
_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
_DATE_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?![\d.])")
_ISO_DATE_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d-])")
_TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")


def _ru_ordinal(n: int, form: str) -> str:
    """Ordinal in the form a date needs: 'n' neuter nominative (пятнадцатое), 'g' masculine genitive (шестого)."""
    from num2words import num2words
    w = num2words(n, lang="ru", to="ordinal")
    endings = (("ий", "ье"), ("ой", "ое"), ("ый", "ое")) if form == "n" else (("ий", "ьего"), ("ой", "ого"), ("ый", "ого"))
    for a, b in endings:
        if w.endswith(a):
            return w[:-2] + b
    return w


def expand_ru_dates_times(text: str) -> str:
    from num2words import num2words

    def date(d: int, mo: int, y: int, whole: str) -> str:
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return whole
        return f"{_ru_ordinal(d, 'n')} {_MONTHS_GEN[mo - 1]} {_ru_ordinal(y, 'g')} года"

    text = _DATE_RE.sub(lambda m: date(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0)), text)
    text = _ISO_DATE_RE.sub(lambda m: date(int(m.group(3)), int(m.group(2)), int(m.group(1)), m.group(0)), text)

    def clock(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group(0)
        minutes = "ноль ноль" if mi == 0 else ("ноль " if mi < 10 else "") + num2words(mi, lang="ru")
        return f"{num2words(h, lang='ru')} {minutes}"

    return _TIME_RE.sub(clock, text)


# The runtime's own number regex refuses a number that touches punctuation ("24,9.", "пункт 1.", "в 15."),
# and the vocoder then hums on the raw digit. All numbers in Russian text are expanded here instead:
# decimals ("двадцать четыре целых девять десятых"), then the remaining integers.
_DECIMAL_RE = re.compile(r"(?<![\w.,])(-?\d+)[.,](\d+)(?![\w])")
_INTEGER_RE = re.compile(r"(?<![\w.,])(-?\d+)(?![\w])")


def expand_ru_numbers(text: str) -> str:
    from num2words import num2words

    def dec(m: re.Match) -> str:
        try:
            return num2words(float(f"{m.group(1)}.{m.group(2)}"), lang="ru")
        except Exception:  # noqa: BLE001
            return m.group(0)

    def integer(m: re.Match) -> str:
        try:
            return num2words(int(m.group(1)), lang="ru")
        except Exception:  # noqa: BLE001
            return m.group(0)

    return _INTEGER_RE.sub(integer, _DECIMAL_RE.sub(dec, text))


# ----------------------------------------------------------------- sentence splitting
_SENT_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$|\n+")
_LIST_MARKER_RE = re.compile(r"(?:^|(?<=\n))[ \t]*(\d{1,3})[.)][ \t]+")


def _list_marker_word(m: re.Match) -> str:
    """'3. ' at a line start -> 'третье, ' (or 'third, ' in an English answer)."""
    n = int(m.group(1))
    try:
        if is_russian_context(m.string) or not _LAT.search(m.string):
            return _ru_ordinal(n, "n") + ", "
        from num2words import num2words
        return num2words(n, lang="en", to="ordinal") + ", "
    except Exception:  # noqa: BLE001
        return ""
_BREAKS = (", ", "; ", ": ", " — ", " - ", " ")


_PUNCT_BREAKS = (", ", "; ", ": ", " — ", " - ")


def _hardwrap(s: str, limit: int) -> list[str]:
    """Split an over-long sentence; prefer a punctuation break (natural pause) over a bare space so
    phrases like "две тысячи двадцать шестого года" are not cut in the middle."""
    out = []
    while len(s) > limit:
        window = s[:limit]
        cut = max((window.rfind(b) for b in _PUNCT_BREAKS), default=-1)
        if cut < int(limit * 0.25):
            cut = window.rfind(" ")
        if cut <= 10:
            cut = limit - 1
        out.append(s[:cut + 1].strip())
        s = s[cut + 1:].strip()
    if s:
        out.append(s)
    return out


class SentenceSplitter:
    """Feed streamed text; get complete sentences back as soon as they are complete.

    The first chunk is kept short (fast first audio); later chunks may merge short sentences
    up to `max_chars` so the prosody does not sound choppy.
    """

    def __init__(self, first_max: int = 70, max_chars: int = 180):
        self.first_max = first_max
        self.max_chars = max_chars
        self.buf = ""
        self.emitted = 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        # "1. Браузер…" at a line start: the sentence regex would cut "1." off as its own sentence and
        # the voice hums on the bare digit; say the item number as a word instead ("первое, Браузер…")
        self.buf = _LIST_MARKER_RE.sub(_list_marker_word, self.buf)
        out: list[str] = []
        while True:
            m = _SENT_END.search(self.buf)
            if not m:
                break
            sent = self.buf[:m.start()].strip()
            self.buf = self.buf[m.end():]
            if sent:
                out.extend(self._pack(sent))
        # very long run without punctuation: cut it so playback can start
        limit = self.first_max if self.emitted == 0 else self.max_chars
        if len(self.buf) > limit * 1.6:
            head = _hardwrap(self.buf, limit)
            if len(head) > 1:
                out.extend(self._pack(" ".join(head[:-1])))
                self.buf = head[-1]
        return out

    def _pack(self, sent: str) -> list[str]:
        limit = self.first_max if self.emitted == 0 else self.max_chars
        # a sentence slightly over the limit is synthesised whole (~0.6 s) rather than cut mid-phrase
        pieces = _hardwrap(sent, limit) if len(sent) > limit * 1.35 else [sent]
        self.emitted += len(pieces)
        return pieces

    def finish(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        if not rest:
            return []
        return self._pack(rest)

    def reset(self) -> None:
        self.buf = ""
        self.emitted = 0


# ----------------------------------------------------------------- engine
class TeraTTS:
    def __init__(self, model_dir: Path = settings.TTS_MODEL_DIR, provider: str = settings.TTS_PROVIDER,
                 threads: int = settings.TTS_THREADS):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"TeraTTSv2 model not found in {model_dir} — run download_models.py or set TTS_MODEL_DIR")
        from .cuda import setup_cuda_paths
        setup_cuda_paths()
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        import teratts  # the model's own runtime (numpy + onnxruntime), no torch
        self.rt = teratts
        self.model_dir = model_dir
        self.sample_rate = SAMPLE_RATE

        available = ort.get_available_providers()
        want_cuda = provider in ("auto", "cuda") and "CUDAExecutionProvider" in available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_cuda else ["CPUExecutionProvider"]
        t0 = time.perf_counter()
        self.loaded = self._load(providers, threads or None)
        self.provider = self.loaded.vocoder.get_providers()[0]
        if want_cuda and self.provider != "CUDAExecutionProvider":
            log.warning("CUDA provider requested but not active; using CPU")
        log.info("TeraTTSv2 loaded (%s) in %.1fs", self.provider, time.perf_counter() - t0)
        self.warmup()

    def _load(self, providers: list[str], threads: Optional[int]):
        rt = self.rt
        models = self.model_dir / "models"
        return rt.LoadedTTS(
            release=self.model_dir, model="distilled",
            text_encoder=rt.session(models, "text_encoder.onnx", providers, threads=threads),
            duration_predictor=rt.session(models, "duration_predictor.onnx", providers, threads=threads),
            sampler=rt.session(models, "sampler_distilled_cfg3_8step.onnx", providers, threads=threads),
            vocoder=rt.session(models, "vocoder.onnx", providers, threads=threads),
            indexer=rt.UnicodeIndexer(self.model_dir / "unicode_indexer.json"),
            accentizer=rt.load_ruaccent(workdir=self.model_dir / "ruaccent"),
        )

    def warmup(self) -> None:
        t0 = time.perf_counter()
        for text, voice in (("<ru>Привет, я готова.</ru>", settings.TTS_VOICE_RU), ("<en>Hello, ready.</en>", settings.TTS_VOICE_EN)):
            try:
                for _ in self.rt.generate_speech_stream(self.loaded, text, voice, chunk_frames=settings.TTS_STREAM_FRAMES):
                    pass
            except Exception as e:  # noqa: BLE001
                log.warning("warmup failed for %s: %s", voice, e)
        log.info("TTS warmup %.1fs", time.perf_counter() - t0)

    @staticmethod
    def voice_for(lang: str) -> str:
        return settings.TTS_VOICE_RU if lang == "ru" else settings.TTS_VOICE_EN

    def prepare(self, text: str) -> tuple[str, str]:
        """Sanitize -> language tags -> Russian stress (keeping manual marks). Returns (tagged, lang)."""
        force = settings.TTS_LANGUAGE if settings.TTS_LANGUAGE in ("ru", "en") else None
        russian = force == "ru" or (force is None and (is_russian_context(text) or not _LAT.search(text)))
        if russian:
            # before the vocabulary pass: it would put a space after the ':' of "15:02" and hide the time
            text = expand_ru_numbers(expand_ru_dates_times(text))
        text = sanitize_vocab(text)
        # Latin words inside Russian speech are read as noise by the Russian voice: say them in Cyrillic
        text = transliterate_latin(text, force=(force == "ru"))
        tagged, lang = tag_languages(text, force)
        if not tagged:
            return "", lang
        # Same order as the model's own normalize_text: spacing/vocabulary -> numbers to words ->
        # stress. Numbers must be words *before* RUAccent so they get stress marks too.
        try:
            tagged = self.rt.normalize_input_text(tagged, self.loaded.indexer)
            tagged = self.rt.expand_tagged_numbers(tagged)
        except Exception as e:  # noqa: BLE001
            log.debug("normalize failed, passing raw text: %s", e)
        if self.loaded.accentizer is not None and "<ru>" in tagged:
            tagged = re.sub(r"<ru>(.*?)</ru>",
                            lambda m: "<ru>" + accentize_keep_manual(m.group(1), self.loaded.accentizer) + "</ru>",
                            tagged, flags=re.S)
        return tagged, lang

    def synth_stream(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None,
                     chunk_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """Yield float32 44.1 kHz chunks for one sentence (already cleaned)."""
        tagged, lang = self.prepare(text)
        if not tagged or not has_speech(tagged):
            return
        voice = voice or self.voice_for(lang)
        scale = speed if speed else settings.TTS_SPEED
        # English voices read Russian better a bit faster (model card recommendation).
        if lang == "ru" and voice.startswith("eng"):
            scale *= 0.85
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield from self.rt.generate_speech_stream(self.loaded, tagged, voice, duration_scale=scale,
                                                      chunk_frames=chunk_frames or settings.TTS_STREAM_FRAMES)

    def synth(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        parts = list(self.synth_stream(text, voice))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
