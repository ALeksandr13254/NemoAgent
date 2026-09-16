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


_DEGEN_TAIL_RE = re.compile(r"(.{2,16}?)\1{5,}\s*$", re.S)
DISPLAY_MARKER_RE = re.compile(r"(?:^|\n)\s*={3,}\s*(?:\n|$)")
TASK_MARKER_RE = re.compile(r"(?:^|\n)\s*>{3,}\s*")

# how much of the spoken stream is held back so a filler ending can be removed before it is heard
HOLD_CHARS = 70

# "Сейчас проверю." / "Let me check" with no task attached: the dialogue agent promised an action.
# Only first-person future forms count ("проверю", "посмотрим"), so a recap of past actions
# ("я проверил и сказал…") is not mistaken for a promise.
_PROMISE_RE = re.compile(
    r"(?:\b(?:сейчас|секунд\w*|минут\w*|давай\w*|попробую|ладно|хорошо|окей)\b[^.!?\n]{0,40}?"
    r"\b(?:провер|посмотр|глян|сдела|запущ|запуст|найд|поищ|откро|узна|выполн|измер|прочита|прочт|скача|установ|"
    r"зайд|переключ|закро|включ|выключ|удал|созда|сохран|отправ|скопир|перемещ|напиш|собер|посчита|подключ|запомн|"
    r"введ|напечата|набер|нажм|кликн|перейд|покаж|обнов|очист|очищ|перезагруз|перезагруж|перезапущ|перезапуст|выдел|"
    r"вставл|прокруч|свер|смен|помен|переимен|распаку|запиш|отмен|останов|верн|постав|убер|перенес|загруж|выгруж|"
    r"переда|скач|пересчита|перечита|разбер|отправл|отвеч|позвон|запрош|попрош|подожд|подготов|настро|отключ|активир)"
    r"\w{0,3}(?:ю|у|им|ем|ём)\b"
    r"|\b(?:let me|i(?:'ll| will)|going to)\s+(?:check|look|see|run|open|find|search|read|try|do|execute|take a look|"
    r"type|enter|click|press|write|install|launch|start|close|switch|create|delete|save|send|set|turn))",
    re.I)
# A whole short answer that just starts with "Сейчас …" / "Секунду …" is an announcement whatever the verb
_PROMISE_OPENER_RE = re.compile(r"^\s*(?:сейчас|секунду|секундочку|минуту|минутку|один момент|момент)\b", re.I)


def looks_like_promise(text: str) -> bool:
    """True when a SHORT spoken text announces an action ('Сейчас проверю место на диске.')."""
    text = (text or "").strip()
    if not text or len(text) > 220:
        return False
    first = re.split(r"(?<=[.!?…])\s+", text, maxsplit=1)[0]
    if first.rstrip().endswith("?"):
        return False   # "Сейчас перезагружу компьютер?" asks for confirmation, it is not an announcement
    if _PROMISE_RE.search(first):
        return True
    return bool(_PROMISE_OPENER_RE.match(first)) and len(first) <= 90


class ProseSpeechRouter:
    """Route a streamed plain-text answer of the dialogue agent:
      spoken part -> speech pieces;  `===` part -> display (screen only);  `>>>` part -> task for the executor.

    feed() yields ("speech", text) or ("display", text); the task is never emitted as text.
    finish() flushes the held tail with the filler stripped; `result` = (speech, display, task).
    """

    def __init__(self) -> None:
        self.buf = ""            # spoken text not yet released
        self.speech = ""         # everything released as speech
        self.display = ""        # screen-only part
        self.task = ""           # task for the executor
        self.mode = "speech"     # speech | display | task

    def _enter_task(self, rest: str) -> None:
        self.mode = "task"
        self.task += rest

    def feed(self, chunk: str) -> Iterator[tuple[str, str]]:
        if self.mode == "task":
            self.task += chunk
            return
        if self.mode == "display":
            self.buf += chunk
            m = TASK_MARKER_RE.search(self.buf)
            if m:
                before, rest = self.buf[:m.start()], self.buf[m.end():]
                self.buf = ""
                if before:
                    self.display += before
                    yield ("display", before)
                self._enter_task(rest)
                return
            # keep a small tail so a ">>>" split across chunks is still caught
            if len(self.buf) > 8:
                piece, self.buf = self.buf[:-8], self.buf[-8:]
                self.display += piece
                yield ("display", piece)
            return
        self.buf += chunk
        md = DISPLAY_MARKER_RE.search(self.buf)
        mt = TASK_MARKER_RE.search(self.buf)
        m = min((x for x in (md, mt) if x), key=lambda x: x.start(), default=None)
        if m:
            spoken, rest = self.buf[:m.start()], self.buf[m.end():]
            self.buf = ""
            spoken = strip_filler(spoken)
            if spoken.strip():
                self.speech += spoken
                yield ("speech", spoken)
            if m is mt:
                self._enter_task(rest)
                return
            self.mode = "display"
            if rest:
                yield from self.feed(rest)
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
        # a repetition loop the model fell into is held in the tail: never speak it
        self.buf = _DEGEN_TAIL_RE.sub("", self.buf)
        if self.mode == "speech" and self.buf:
            whole = strip_filler(self.speech + self.buf)
            tail = whole[len(self.speech):] if whole.startswith(self.speech) else self.buf
            self.buf = ""
            if tail.strip():
                self.speech += tail
                yield ("speech", tail)
        elif self.mode == "display" and self.buf:
            piece, self.buf = self.buf, ""
            self.display += piece
            yield ("display", piece)
        self.display = self.display.strip()
        # the model sometimes appends a stray === block or another >>> after the task: keep the task only
        task = self.task.split("===")[0]
        task = re.split(r"\n\s*>{3,}", task)[0]
        # an imagined "Результат исполнителя …" glued to the task is not part of the instruction
        task = re.split(r"Результат исполнителя", task, maxsplit=1)[0]
        self.task = " ".join(task.split()).strip(" >")

    @property
    def result(self) -> tuple[str, Optional[str], Optional[str]]:
        return self.speech.strip(), (self.display or None), (self.task or None)
