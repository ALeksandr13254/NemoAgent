"""Speaker: LLM text stream -> sentences -> TeraTTS -> gapless player, with instant cancel."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

from .player import StreamPlayer
from .tts import SentenceSplitter, TeraTTS, clean_for_tts, has_speech, scrub_code

log = logging.getLogger("speech")


class Speaker:
    def __init__(self, tts: TeraTTS, player: StreamPlayer, on_state: Optional[Callable[[dict], None]] = None):
        self.tts = tts
        self.player = player
        self.on_state = on_state or (lambda d: None)
        self.splitter = SentenceSplitter()
        self._q: "queue.Queue[tuple[int, str] | None]" = queue.Queue()
        self._gen = 0
        self._busy = threading.Event()
        self._utt_started = 0.0
        self._first_audio_at = 0.0
        self.recent: deque[str] = deque(maxlen=12)   # last spoken sentences (for echo rejection)
        self._prepared = False
        self._last_enqueued = ""
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()

    # ----------------------------------------------------------------- state
    @property
    def is_speaking(self) -> bool:
        return self._busy.is_set() or self.player.is_playing or not self._q.empty()

    # ----------------------------------------------------------------- input
    def begin(self) -> int:
        """Start a new utterance (new LLM answer). Cancels anything still playing."""
        self.cancel()
        self._utt_started = time.time()
        self._first_audio_at = 0.0
        self._last_enqueued = ""
        self.splitter.reset()
        return self._gen

    def feed(self, delta: str, prepared: bool = False) -> None:
        """Stream text in. prepared=True means the model wrote TTS-ready text via the `speak` tool:
        only the vocabulary sanitizer runs, the markdown cleaner is skipped."""
        for sent in self.splitter.feed(delta):
            self._enqueue(sent, prepared)

    def end(self) -> None:
        for sent in self.splitter.finish():
            self._enqueue(sent, self._prepared)

    def say(self, text: str) -> None:
        self.begin()
        self.feed(text)
        self.end()

    def _enqueue(self, sentence: str, prepared: bool = False) -> None:
        self._prepared = prepared
        # markdown cleaner is cheap and harmless on clean text, so it always runs: the model is asked
        # to write speech, but a stray list marker or **bold** must never reach the vocoder
        cleaned = scrub_code(clean_for_tts(sentence))
        if not has_speech(cleaned):
            return
        if cleaned == self._last_enqueued:      # a long command split into chunks -> one note, not three
            return
        self._last_enqueued = cleaned
        self._q.put((self._gen, cleaned))

    def cancel(self) -> None:
        old = self._gen
        self._gen += 1
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self.player.flush()
        self.splitter.reset()
        self.on_state({"type": "speech_cancelled", "gen": old})

    # ----------------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            gen, sentence = item
            if gen != self._gen:
                continue
            self._busy.set()
            try:
                t0 = time.perf_counter()
                first = True
                samples = 0
                # progress for the UI (the read-aloud tab shows which sentence is being spoken)
                self.on_state({"type": "speaking", "gen": gen, "text": sentence, "queued": self._q.qsize()})
                for chunk in self.tts.synth_stream(sentence):
                    if gen != self._gen:
                        log.debug("speech cancelled mid-sentence: %r", sentence[:50])
                        break
                    if first:
                        first = False
                        if not self._first_audio_at:
                            self._first_audio_at = time.time()
                            self.on_state({"type": "tts_first_audio", "ms": int((self._first_audio_at - self._utt_started) * 1000),
                                           "synth_ms": int((time.perf_counter() - t0) * 1000)})
                    samples += len(chunk)
                    self.player.feed(chunk, self.player.generation)
                self.recent.append(sentence)
                log.info("tts: %.1fs audio in %.2fs | %r | played %.1fs of %.1fs fed",
                         samples / self.player.samplerate, time.perf_counter() - t0, sentence[:60],
                         self.player.samples_played / self.player.samplerate, self.player.samples_fed / self.player.samplerate)
            except Exception as e:  # noqa: BLE001
                log.exception("TTS failed for %r: %s", sentence[:60], e)
            finally:
                self._busy.clear()
                if gen == self._gen and self._q.empty():
                    self.on_state({"type": "speech_drained", "gen": gen})   # all sentences synthesised

    def close(self) -> None:
        self._q.put(None)
