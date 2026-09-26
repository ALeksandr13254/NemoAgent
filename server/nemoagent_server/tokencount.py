"""Exact prompt token count for Nemotron 3 Nano Omni, computed locally: no request to NVIDIA.

The hosted model (vLLM behind integrate.api.nvidia.com) builds a request like this:
  1. the chat template (chat_template.jinja from the model repo) renders the messages into text; an attachment
     becomes a placeholder: one image "<image>", several "<image 1><image> <image 2><image>", audio
     "<so_embedding>", video "<video>";
  2. every placeholder is expanded into the media's own tokens: an image into "<img>" + N x "<image>" + "</img>",
     audio into "<so_start>" + N x "<so_embedding>" + "<so_end>", a video into frame groups;
  3. the text is tokenized with tokenizer.json (no BOS added).
This module does the same with the model's own files (models/omni-tokenizer/, fetched once from Hugging Face)
and the model's own formulas for N (ported from image_processing.py / processing.py of the model repo). The one
exception is video: the hosted server samples frames in a way the published code does not reproduce, so a clip is
measured once with a request of its own (see MEASURED_VIDEO) and its tokens are known exactly from then on.
Checked against NVIDIA's prompt_tokens: text in both languages, images of several sizes, audio, a dialogue with
history and memories, the executor with tools, tool calls and a screenshot, and prompts of 1,000,000 tokens.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import SERVER_DIR, settings

log = logging.getLogger("tokens")

REPO = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
DIR = SERVER_DIR / "models" / "omni-tokenizer"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"     # the hosted model these files belong to
# The hosted model refuses a prompt of CONTEXT_LIMIT tokens already ("maximum context length is 1000000 tokens.
# However, your messages resulted in 1000000 tokens"), so a prompt fits with at most MAX_PROMPT tokens; the answer
# gets what is left (max_tokens does not limit this model on the hosted API).
CONTEXT_LIMIT = 1_000_000
MAX_PROMPT = CONTEXT_LIMIT - 1


def supports(model: str) -> bool:
    return model == MODEL


# Speed: the prompt is cut at every "<|im_start|>" (a special token: the tokenizer never lets text on its two sides
# merge), so each message block is tokenized on its own and remembered; the next count of the same chat tokenizes only
# the blocks that changed. A long block is cut further at safe points and the pieces are tokenized in parallel.
BLOCK = "<|im_start|>"
PIECE_CHARS = 50_000
CACHE_CHARS = 64_000_000


def _safe_pieces(text: str) -> list[tuple[int, str]]:
    """`text` as (start, piece) parts whose tokens add up to the tokens of the whole. A cut goes right before a space
    that stands between two letters: the pre-tokenizer of tokenizer.json always ends a piece there (a run of letters
    stops at the space and the space starts the next word), so no token can span the cut."""
    n = len(text)
    if n <= 2 * PIECE_CHARS:
        return [(0, text)]
    out, start, pos = [], 0, PIECE_CHARS
    while pos < n - PIECE_CHARS // 2:
        limit = min(n - 1, pos + 10_000)
        i = text.find(" ", pos, limit)
        while i != -1 and not (text[i - 1].isalpha() and text[i + 1].isalpha()):
            i = text.find(" ", i + 1, limit)
        if i == -1:
            pos = limit
            continue
        out.append((start, text[start:i]))
        start, pos = i, i + PIECE_CHARS
    out.append((start, text[start:]))
    return out


# image processor (preprocessor_config.json of the model)
PATCH = 16
DOWNSAMPLE = 2                 # pixel shuffle: 2 x 2 patches make one token
MIN_PATCHES, MAX_PATCHES = 1024, 13312
# audio (processing.py): 16 kHz, 10 ms hop, three stride-2 conv stages
AUDIO_RATE, AUDIO_HOP, AUDIO_STAGES, AUDIO_KERNEL, AUDIO_STRIDE = 16000, 160, 3, 3, 2


@dataclass
class Count:
    """Tokens of one request: `total` is what the model receives; the rest explains it."""
    total: int = 0
    text: int = 0                       # template + text, placeholders excluded
    media: int = 0                      # all expanded media tokens
    per_message: list[int] = field(default_factory=list)   # tokens of each message, in order (media included)
    items: list[dict] = field(default_factory=list)        # one entry per attachment: kind, tokens, detail
    exact: bool = True                  # False when a part could not be measured (e.g. an unreadable video)
    ms: float = 0.0


class OmniTokenCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tok = None
        self._template = None
        self._media_cache: dict[str, tuple[int, str]] = {}
        self._blocks: OrderedDict[str, int] = OrderedDict()     # tokens of message blocks already seen
        self._blocks_chars = 0
        self._blocks_lock = threading.Lock()
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ setup
    def ensure_files(self) -> bool:
        """Fetch the three tokenizer files once (about 17 MB); the server stores nothing else on disk."""
        missing = [f for f in FILES if not (DIR / f).exists() or (DIR / f).stat().st_size == 0]
        if not missing:
            return True
        DIR.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=300, follow_redirects=True) as c:
                for f in missing:
                    r = c.get(f"https://huggingface.co/{REPO}/resolve/main/{f}")
                    r.raise_for_status()
                    (DIR / f).write_bytes(r.content)
                    log.info("token counter: downloaded %s (%.1f MB)", f, len(r.content) / 1e6)
            return True
        except Exception as e:  # noqa: BLE001
            self.error = f"tokenizer files unavailable: {e}"
            log.warning("token counter: %s", self.error)
            return False

    def _load(self) -> bool:
        if self._tok is not None:
            return True
        with self._lock:
            if self._tok is not None:
                return True
            if not self.ensure_files():
                return False
            try:
                from jinja2.ext import loopcontrols
                from jinja2.sandbox import ImmutableSandboxedEnvironment
                from tokenizers import Tokenizer
            except ImportError as e:
                self.error = f"missing library: {e} (pip install tokenizers jinja2)"
                return False
            tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))

            # the same Jinja environment transformers uses for apply_chat_template (and so does vLLM)
            def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

            def raise_exception(message):
                raise ValueError(message)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
            env.filters["tojson"] = tojson
            env.globals["raise_exception"] = raise_exception
            env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
            self._template = env.from_string((DIR / "chat_template.jinja").read_text("utf-8"))
            self._tok = tok
            log.info("token counter ready (%s)", REPO)
            return True

    @property
    def ready(self) -> bool:
        return self._load()

    # ------------------------------------------------------------------ media formulas (ported from the model repo)
    @staticmethod
    def image_tokens(width: int, height: int) -> int:
        """image_processing.py: _compute_target_patches with the per-image budget of MAX_PATCHES."""
        available = MAX_PATCHES
        ph = round(height / PATCH + 0.5)
        pw = round(width / PATCH + 0.5)
        factor = min(math.sqrt(available / (ph * pw)), 1.0)
        th, tw = math.floor(factor * ph), math.floor(factor * pw)
        if available > MIN_PATCHES and th * tw < MIN_PATCHES:
            up = math.sqrt(MIN_PATCHES / (th * tw))
            th, tw = math.ceil(up * th), math.ceil(up * tw)
        d = DOWNSAMPLE
        if th % d:
            if (th + d - th % d) * tw <= available:
                th += d - th % d
            else:
                th = max(d, th - th % d)
        if tw % d:
            if th * (tw + d - tw % d) <= available:
                tw += d - tw % d
            else:
                tw = max(d, tw - tw % d)
        return (tw * th) // (d * d)

    @staticmethod
    def audio_tokens(samples_16k: int) -> int:
        """processing.py: _estimate_audio_num_embeddings (mel frames, then three stride-2 conv stages)."""
        n = 1 + samples_16k // AUDIO_HOP
        pad = (AUDIO_KERNEL - 1) // 2
        for _ in range(AUDIO_STAGES):
            n = (n + 2 * pad - AUDIO_KERNEL) // AUDIO_STRIDE + 1
        return max(1, n)

    def _media_part_tokens(self, part: dict) -> tuple[int, str, bool]:
        """Expanded tokens of one media part (placeholder token excluded), a short description, exactness."""
        kind = part.get("type")
        key_src = part.get(kind) or {}
        url = key_src.get("url") if isinstance(key_src, dict) else None
        if not url:
            return 0, kind or "?", False
        key = media_key(part)
        if key in self._media_cache:
            n, detail = self._media_cache[key]
            return n, detail, n >= 0
        n, detail = -1, kind
        try:
            data = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else None
            if data is None:
                raise ValueError("not a data URI")
            if kind == "image_url":
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
                n = self.image_tokens(w, h) + 2 - 1          # <img> + N x <image> + </img> replaces one <image>
                detail = f"изображение {w}×{h}"
            elif kind == "audio_url":
                samples = _decode_samples(data)
                n = self.audio_tokens(samples) + 2 - 1        # <so_start> + N + <so_end> replaces <so_embedding>
                detail = f"аудио {samples / AUDIO_RATE:.1f} с"
            else:                                             # video: measured once per clip, see measured_video()
                measured = MEASURED_VIDEO.get(key)
                if measured is None:
                    return 0, "видео (измеряется)", False
                n, detail = measured, "видео"
        except Exception as e:  # noqa: BLE001
            log.warning("token counter: cannot measure %s: %s", kind, e)
        self._media_cache[key] = (n, detail)
        if len(self._media_cache) > 512:
            self._media_cache.pop(next(iter(self._media_cache)))
        return n, detail, n >= 0

    # ------------------------------------------------------------------ the count
    def render(self, messages: list[dict], tools: Optional[list] = None, enable_thinking: bool = False) -> str:
        return self._template.render(messages=messages, tools=tools or [], add_generation_prompt=True,
                                     enable_thinking=enable_thinking, bos_token="<s>", eos_token="</s>")

    @staticmethod
    def _as_template_input(messages: list[dict]) -> list[dict]:
        """What vLLM hands to the template: tool-call arguments as objects, not JSON strings."""
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    fn = dict(c.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except json.JSONDecodeError:
                            pass
                    calls.append(dict(c, function=fn))
                m = dict(m, tool_calls=calls)
            out.append(m)
        return out

    def _block_tokens(self, blocks: list[str]) -> list[int]:
        """Tokens of each text block: remembered ones at once, the new ones tokenized together in parallel."""
        out = [0] * len(blocks)
        todo: list[str] = []
        owner: list[int] = []
        fresh: dict[int, str] = {}
        with self._blocks_lock:
            for k, b in enumerate(blocks):
                hit = self._blocks.get(b)
                if hit is not None:
                    self._blocks.move_to_end(b)
                    out[k] = hit
                    continue
                fresh[k] = b
                # "<video>" is plain text (3 tokens) that merges with the newline after it: the server cuts the
                # prompt at every such placeholder, tokenizes the pieces separately and puts the clip in between
                for part in b.split("<video>"):
                    for _, piece in _safe_pieces(part):
                        todo.append(piece)
                        owner.append(k)
        if todo:
            for k, enc in zip(owner, self._tok.encode_batch_fast(todo, add_special_tokens=False)):
                out[k] += len(enc.ids)
            with self._blocks_lock:
                for k, b in fresh.items():
                    self._blocks[b] = out[k]
                    self._blocks_chars += len(b)
                while self._blocks_chars > CACHE_CHARS and self._blocks:
                    old, _ = self._blocks.popitem(last=False)
                    self._blocks_chars -= len(old)
        return out

    def count(self, messages: list[dict], tools: Optional[list] = None, enable_thinking: bool = False) -> Count:
        """Exact prompt tokens of a chat request as the hosted Omni will see it."""
        t0 = time.perf_counter()
        if not self._load():
            raise RuntimeError(self.error or "token counter unavailable")
        res = Count()
        text = self.render(self._as_template_input(messages), tools, enable_thinking)
        # Each message opens with <|im_start|> (a run of tool results shares one). Before the first message there is
        # always a system block (empty when there is no system message); after the last one, the generation prompt.
        head, *blocks = text.split(BLOCK)
        sizes = self._block_tokens([head] + blocks)
        span_len = [1 + n for n in sizes[1:]]               # the <|im_start|> token itself and its block
        if span_len:
            span_len[0] += sizes[0]
        res.text = sum(sizes) + len(blocks)
        span_of_msg: list[int] = []
        k = 0 if messages and messages[0].get("role") == "system" else 1   # no system message: span 0 is the empty one
        for mi, m in enumerate(messages):
            if mi > 0 and m.get("role") == "tool" and messages[mi - 1].get("role") == "tool":
                span_of_msg.append(-1)                   # shares the span of the first tool result of the run
                continue
            span_of_msg.append(k)
            k += 1
        res.per_message = [span_len[s] if 0 <= s < len(span_len) else 0 for s in span_of_msg]
        # media: expanded token counts, attributed to the message that carries them
        for mi, m in enumerate(messages):
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                kind = part.get("type")
                if kind not in ("image_url", "audio_url", "video_url"):
                    continue
                n, detail, ok = self._media_part_tokens(part)
                if not ok:
                    res.exact = False
                    n = max(n, 0)
                shown = n if kind == "video_url" else n + 1           # the placeholder token of an image / audio
                res.items.append({"message": mi, "kind": kind.split("_")[0], "tokens": shown, "detail": detail, "exact": ok,
                                  "key": media_key(part) if kind == "video_url" else None})
                res.media += n
                res.per_message[mi] += n
        res.total = res.text + res.media
        res.ms = (time.perf_counter() - t0) * 1000
        return res

    def count_text(self, text: str) -> int:
        """Tokens of a bare text (no template around it)."""
        if not self._load():
            raise RuntimeError(self.error or "token counter unavailable")
        return sum(len(e.ids) for e in self._tok.encode_batch_fast([p for _, p in _safe_pieces(text or "")],
                                                                     add_special_tokens=False))

    def prefix_chars(self, text: str, tokens: int) -> int:
        """Length in characters of the longest start of `text` that is at most `tokens` tokens long."""
        if not self._load():
            raise RuntimeError(self.error or "token counter unavailable")
        if tokens <= 0:
            return 0
        pieces = _safe_pieces(text or "")
        left = tokens
        for (start, _), enc in zip(pieces, self._tok.encode_batch([p for _, p in pieces], add_special_tokens=False)):
            if left <= len(enc.ids):
                return start + enc.offsets[left - 1][1]
            left -= len(enc.ids)
        return len(text or "")


def _decode_samples(data: bytes) -> int:
    """Samples of an audio clip at 16 kHz mono, decoded with ffmpeg exactly as a server-side loader would."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        out = subprocess.run([settings.FFMPEG, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(AUDIO_RATE), "-"],
                             capture_output=True, timeout=120, check=True).stdout
        return len(out) // 2
    finally:
        Path(path).unlink(missing_ok=True)


def media_key(part: dict) -> str:
    kind = part.get("type") or ""
    url = ((part.get(kind) or {}).get("url") if isinstance(part.get(kind), dict) else "") or ""
    return hashlib.sha1(url[:4096].encode() + url[-4096:].encode() + str(len(url)).encode()).hexdigest()


# Video: the hosted server samples frames its own way (not the published vLLM code), so a clip is measured once, with
# one short request that holds just the clip (NIMClient.prompt_tokens), or from the usage of a real request in which
# the clip was the only unknown (AgentSession._check_count).
MEASURED_VIDEO: dict[str, int] = {}


def video_probe_messages(part: dict) -> list[dict]:
    # a one-word answer: the hosted model ignores max_tokens, and the exact size comes with the end of the stream
    return [{"role": "user", "content": [{"type": "text", "text": "Answer with the single word: ok"}, part]}]


def set_video(key: str, tokens: int) -> int:
    MEASURED_VIDEO[key] = tokens
    if len(MEASURED_VIDEO) > 256:
        MEASURED_VIDEO.pop(next(iter(MEASURED_VIDEO)))
    return tokens


def remember_video(part: dict, prompt_tokens: int) -> int:
    """Store the clip's own tokens: the measured prompt minus everything around the placeholder."""
    return set_video(media_key(part), prompt_tokens - COUNTER.count(video_probe_messages(part)).text)


COUNTER = OmniTokenCounter()
