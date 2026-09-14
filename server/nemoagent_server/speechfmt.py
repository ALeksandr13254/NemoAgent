"""Speech text post-processing shared by both speech modes.

* `strip_filler` removes the closing offers the model loves to append ("Чем могу помочь?",
  "Если нужно что-то ещё, скажите", "Let me know if…") — they are read aloud on every answer
  otherwise, and once in the history/memory they reinforce themselves.
* `ProseSpeechRouter` streams a plain-text answer as speech while holding back a short tail, so
  the filler can be cut before it is spoken, and splits off the screen-only part after a `===` line.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional

# Only the literal closing formulas. Anything that can also be a real answer ("Готова помочь с
# задачами", "Если хотите, могу…", "Обращайтесь к врачу") must NOT be here — an earlier, broader
# list ate the second half of "как дела, чем хочешь заняться".
_FILLER_SENTENCES = [
    r"(?:чем|как) (?:ещё |еще )?(?:я )?(?:могу|смогу) (?:вам |тебе )?(?:помочь|быть полезен|быть полезна|быть полезным|быть полезной)(?: сегодня)?",
    r"(?:чем|что) (?:ещё |еще )?(?:вам |тебе )?(?:помочь|подсказать)",
    r"(?:если|когда) (?:вам |тебе )?(?:понадобится|нужно|нужна|надо|захотите|хотите)(?: будет)? (?:ещё |еще )?"
    r"(?:что-то|что-нибудь|что-либо|помощь|моя помощь)(?: ещё| еще)?[^.!?\n]{0,12}?"
    r"(?:скажите|сообщите|дайте знать|обращайтесь|напишите|пишите|говорите|спрашивайте)[^.!?\n]{0,12}",
    r"(?:дайте знать|сообщите|скажите|пишите), если (?:вам |тебе )?(?:понадобится|нужно|нужна|надо|захотите|хотите)"
    r"[^.!?\n]{0,10}?(?:ещё|еще|что-то|что-нибудь|помощь)[^.!?\n]{0,20}",
    r"обращайтесь,? если (?:что|понадобится|нужно|захотите)[^.!?\n]{0,20}",
    r"(?:всегда )?(?:рад|рада|готов|готова|буду рад|буду рада) (?:помочь|быть полезным|быть полезной)(?: ещё| еще)?(?: чем-нибудь| чем-то)?",
    r"how (?:else )?(?:can|may) i (?:help|assist)(?: you)?(?: today| further| with that)?",
    r"is there anything else(?: i can (?:help|do)(?: you)?(?: with)?)?",
    r"(?:just |please )?let me know if (?:you need|you have|there(?:'s| is)) (?:any(?:thing)?|more|other)[^.!?\n]{0,25}",
    r"feel free to ask(?: if you have (?:any )?(?:other |more )?questions)?",
]
_FILLER_RE = re.compile(
    r"(?:^|(?<=[.!?…\n])\s*)(?:" + "|".join(_FILLER_SENTENCES) + r")\s*[.!?…]*\s*$",
    re.I | re.U,
)


def strip_filler(text: str) -> str:
    """Cut trailing offer-of-help sentences (repeatedly); never leaves the answer empty."""
    if not text:
        return text
    out = text.rstrip()
    for _ in range(4):
        new = _FILLER_RE.sub("", out).rstrip()
        if new == out:
            break
        out = new
    out = out.rstrip(" \n\t,;:—-")
    if out.strip():
        return out
    # the whole answer was filler: keep its first sentence unless that is itself an offer of help
    first = re.split(r"(?<=[.!?…])\s+", text.strip(), maxsplit=1)[0]
    if _FILLER_RE.search(first):
        return "Слушаю." if re.search(r"[Ѐ-ӿ]", text) else "Okay."
    return first or text.strip()


DISPLAY_MARKER_RE = re.compile(r"(?:^|\n)\s*={3,}\s*(?:\n|$)")

# how much of the spoken stream is held back so a filler ending can be removed before it is heard
HOLD_CHARS = 70


class ProseSpeechRouter:
    """Route a streamed plain-text answer: spoken part -> speech pieces, `===` part -> display.

    feed() yields ("speech", text) or ("display", text); finish() flushes the held tail with the
    filler stripped and returns the final (speech, display) strings.
    """

    def __init__(self) -> None:
        self.buf = ""            # spoken text not yet released
        self.speech = ""         # everything released as speech
        self.display = ""        # screen-only part
        self.in_display = False

    def feed(self, chunk: str) -> Iterator[tuple[str, str]]:
        if self.in_display:
            self.display += chunk
            yield ("display", chunk)
            return
        self.buf += chunk
        m = DISPLAY_MARKER_RE.search(self.buf)
        if m:
            spoken, rest = self.buf[:m.start()], self.buf[m.end():]
            self.buf = ""
            spoken = strip_filler(spoken)
            if spoken.strip():
                self.speech += spoken
                yield ("speech", spoken)
            self.in_display = True
            if rest:
                self.display += rest
                yield ("display", rest)
            return
        # release all but a tail: enough to still remove a trailing "Чем могу помочь?" and to
        # catch a marker that arrives split across chunks
        if len(self.buf) > HOLD_CHARS:
            cut = len(self.buf) - HOLD_CHARS
            # prefer releasing whole sentences/words
            nl = self.buf.rfind(" ", 0, cut)
            if nl > 0:
                cut = nl + 1
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            if piece:
                self.speech += piece
                yield ("speech", piece)

    def finish(self) -> Iterator[tuple[str, str]]:
        if self.buf:
            whole = strip_filler(self.speech + self.buf)
            tail = whole[len(self.speech):] if whole.startswith(self.speech) else self.buf
            self.buf = ""
            if tail.strip():
                self.speech += tail
                yield ("speech", tail)
        self.display = self.display.strip()

    @property
    def result(self) -> tuple[str, Optional[str]]:
        return self.speech.strip(), (self.display or None)
