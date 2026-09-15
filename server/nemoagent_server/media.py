"""Attachments -> content parts for the omni model (images, audio, video as media; documents as text).

The model takes `image_url`, `audio_url` and `video_url` parts with base64 data URIs — checked on
integrate.api.nvidia.com: a 19 MB request with a 40-second 720p mp4 went through; a 1600 px image
costs ~1600 prompt tokens, 90 s of audio ~1100, 40 s of video ~6800. Everything is normalised here so
that one attachment never blows up the request or the context:
  * images  -> JPEG, longest side MEDIA_IMAGE_MAX_SIDE (GIF: first frame);
  * audio   -> mono mp3 64 kbit/s via ffmpeg, cut to MEDIA_AUDIO_MAX_S (wav/mp3 pass as they are without ffmpeg);
  * video   -> mp4 re-encoded by ffmpeg (<= MEDIA_VIDEO_MAX_HEIGHT p, 4 fps, mono audio), cut to MEDIA_VIDEO_MAX_S;
  * pdf     -> text per page (PyMuPDF); pages without a text layer are rendered as images;
  * docx / pptx / xlsx -> text pulled out of the XML;  plain text and code -> text.
Built parts are cached per attachment id, so an attachment repeated in later turns costs nothing.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import html
import io
import logging
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .attachments import Attachment
from .config import settings

log = logging.getLogger("media")

MEDIA_KEYS = ("image_url", "audio_url", "video_url")
_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)


@dataclass
class Built:
    parts: list[dict] = field(default_factory=list)   # content parts to put into the message
    note: str = ""                                     # one-line description for the text listing
    tokens: int = 0                                    # rough prompt-token estimate


_cache: dict[str, Built] = {}


def _data_uri(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _ffmpeg() -> Optional[str]:
    return shutil.which(settings.FFMPEG)


def _ffprobe_seconds(path: str) -> Optional[float]:
    probe = shutil.which("ffprobe") or (str(Path(_ffmpeg()).with_name("ffprobe.exe" if Path(_ffmpeg()).suffix else "ffprobe"))
                                        if _ffmpeg() else None)
    if not probe or not Path(probe).exists():
        return None
    try:
        out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _run_ffmpeg(args: list[str], out_path: Path, timeout: int = 300) -> bytes:
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args, str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[:300])
    return out_path.read_bytes()


def _too_big(data: bytes) -> bool:
    return len(data) > settings.MEDIA_MAX_INLINE_MB * 1048576


def _cut_text(text: str, name: str) -> str:
    limit = settings.MEDIA_TEXT_MAX_CHARS
    if len(text) > limit:
        return text[:limit] + f"\n… [файл '{name}' обрезан: показано {limit} из {len(text)} символов]"
    return text


# ---------------------------------------------------------------------------- images
def _image_jpeg(data: bytes, max_side: Optional[int] = None, quality: int = 85) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    try:
        im.seek(0)  # animated formats: first frame
    except Exception:  # noqa: BLE001
        pass
    im = im.convert("RGB")
    im.thumbnail((max_side or settings.MEDIA_IMAGE_MAX_SIDE,) * 2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), im.size


def image_part_from_bytes(data: bytes, max_side: Optional[int] = None) -> tuple[dict, tuple[int, int]]:
    """A ready `image_url` part (JPEG data URI) + the size it was sent at."""
    jpeg, size = _image_jpeg(data, max_side)
    return {"type": "image_url", "image_url": {"url": _data_uri("image/jpeg", jpeg)}}, size


def _build_image(att: Attachment) -> Built:
    part, (w, h) = image_part_from_bytes(att.read())
    return Built([part], f"image '{att.name}' ({w}x{h} px)", 1700)


# ---------------------------------------------------------------------------- audio / video
def _build_audio(att: Attachment) -> Built:
    data = att.read()
    ext = att.ext
    ff = _ffmpeg()
    if ff:
        out = Path(att.path + ".omni.mp3")
        if not out.exists():
            _run_ffmpeg(["-i", att.path, "-t", str(settings.MEDIA_AUDIO_MAX_S), "-vn", "-ac", "1", "-ar", "16000",
                         "-codec:a", "libmp3lame", "-b:a", "64k"], out)
        data, mime = out.read_bytes(), "audio/mpeg"
        seconds = _ffprobe_seconds(str(out)) or len(data) / 8000
    elif ext in ("wav", "mp3"):
        mime = "audio/wav" if ext == "wav" else "audio/mpeg"
        seconds = len(data) / (32000 if ext == "wav" else 16000)
    else:
        raise RuntimeError(f"формат .{ext} требует ffmpeg (FFMPEG в server/.env), модель принимает wav и mp3")
    if _too_big(data):
        raise RuntimeError(f"аудио слишком большое для одного запроса ({len(data)/1048576:.0f} MB)")
    part = {"type": "audio_url", "audio_url": {"url": _data_uri(mime, data)}}
    return Built([part], f"audio '{att.name}' (~{seconds:.0f} s)", int(14 * seconds) + 50)


def _build_video(att: Attachment) -> Built:
    ff = _ffmpeg()
    if ff:
        out = Path(att.path + ".omni.mp4")
        if not out.exists():
            _run_ffmpeg(["-i", att.path, "-t", str(settings.MEDIA_VIDEO_MAX_S),
                         "-vf", f"scale=-2:'min({settings.MEDIA_VIDEO_MAX_HEIGHT},ih)'", "-r", "4",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart"], out, timeout=600)
        data = out.read_bytes()
        seconds = _ffprobe_seconds(str(out)) or settings.MEDIA_VIDEO_MAX_S
    elif att.ext in ("mp4", "m4v"):
        data = att.read()
        seconds = settings.MEDIA_VIDEO_MAX_S
    else:
        raise RuntimeError(f"формат .{att.ext} требует ffmpeg (FFMPEG в server/.env), модель принимает mp4")
    if _too_big(data):
        raise RuntimeError(f"видео слишком большое для одного запроса ({len(data)/1048576:.0f} MB)")
    part = {"type": "video_url", "video_url": {"url": _data_uri("video/mp4", data)}}
    return Built([part], f"video '{att.name}' (~{seconds:.0f} s)", int(120 * seconds) + 100)


# ---------------------------------------------------------------------------- documents
def _build_pdf(att: Attachment) -> Built:
    import fitz  # PyMuPDF

    doc = fitz.open(att.path)
    parts: list[dict] = []
    texts: list[str] = []
    images = 0
    tokens = 0
    limit = settings.MEDIA_PDF_MAX_PAGES
    for i, page in enumerate(doc):
        if i >= limit:
            break
        text = page.get_text("text").strip()
        if len(text) >= 40:
            texts.append(f"--- {att.name}, страница {i + 1} ---\n{text}")
            tokens += len(text) // 3
        else:  # scanned page: show it as an image
            rect = page.rect
            zoom = settings.MEDIA_IMAGE_MAX_SIDE / max(rect.width, rect.height, 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            part, _ = image_part_from_bytes(pix.tobytes("png"))
            parts.append(_text_part(f"[{att.name}, страница {i + 1} — без текстового слоя, показана картинкой]"))
            parts.append(part)
            images += 1
            tokens += 1700
    total = doc.page_count
    doc.close()
    if texts:
        parts.insert(0, _text_part(_cut_text("\n\n".join(texts), att.name)))
    note = f"pdf '{att.name}' ({total} стр.; текст {len(texts)} стр." + (f", как картинки {images} стр." if images else "") + \
           (f"; показаны первые {limit}" if total > limit else "") + ")"
    return Built(parts, note, tokens)


def _xml_text(xml: str, para_tags: tuple[str, ...] = ("</w:p>", "</a:p>")) -> str:
    for tag in para_tags:
        xml = xml.replace(tag, "\n")
    xml = xml.replace("<w:tab/>", "\t").replace("<w:br/>", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(text)


def _build_office(att: Attachment) -> Built:
    ext = att.ext
    with zipfile.ZipFile(att.path) as z:
        names = z.namelist()
        if ext == "docx":
            text = _xml_text(z.read("word/document.xml").decode("utf-8", "replace"))
        elif ext == "pptx":
            slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                            key=lambda n: int(re.search(r"(\d+)", n).group(1)))
            chunks = []
            for n in slides:
                body = re.findall(r"<a:t>(.*?)</a:t>", z.read(n).decode("utf-8", "replace"), re.S)
                chunks.append(f"--- слайд {len(chunks) + 1} ---\n" + "\n".join(html.unescape(b) for b in body))
            text = "\n\n".join(chunks)
        else:  # xlsx
            shared = []
            if "xl/sharedStrings.xml" in names:
                shared = [html.unescape(re.sub(r"<[^>]+>", "", s)) for s in
                          re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8", "replace"), re.S)]
            sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
            chunks = []
            for n in sheets:
                rows = []
                for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read(n).decode("utf-8", "replace"), re.S)[:500]:
                    cells = []
                    for attrs, inner in re.findall(r"<c([^>]*)>(.*?)</c>", row, re.S):
                        v = re.search(r"<v>(.*?)</v>", inner, re.S)
                        t = re.search(r"<t[^>]*>(.*?)</t>", inner, re.S)
                        val = v.group(1) if v else (t.group(1) if t else "")
                        if 't="s"' in attrs and val.isdigit() and int(val) < len(shared):
                            val = shared[int(val)]
                        cells.append(html.unescape(val))
                    rows.append("\t".join(cells))
                chunks.append(f"--- лист {Path(n).stem} ---\n" + "\n".join(rows))
            text = "\n\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return Built([_text_part(f"=== {att.name} ===\n{_cut_text(text, att.name)}")], f"document '{att.name}' ({len(text)} chars)",
                 len(text) // 3 + 20)


def _build_text(att: Attachment) -> Built:
    raw = att.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251", "replace")
    return Built([_text_part(f"=== {att.name} ===\n{_cut_text(text, att.name)}")], f"text '{att.name}' ({len(text)} chars)",
                 len(text) // 3 + 20)


_BUILDERS = {"image": _build_image, "audio": _build_audio, "video": _build_video, "pdf": _build_pdf,
             "office": _build_office, "text": _build_text}


def build(att: Attachment) -> Built:
    """Content parts for one attachment (cached)."""
    hit = _cache.get(att.id)
    if hit:
        return hit
    builder = _BUILDERS.get(att.kind)
    if not builder:
        b = Built([_text_part(f"[файл '{att.name}' ({att.size // 1024} KB, {att.mime}): формат не поддерживается, "
                              "содержимое не передано]")], f"file '{att.name}' (unsupported)", 40)
    else:
        try:
            b = builder(att)
        except Exception as e:  # noqa: BLE001
            log.warning("attachment %s (%s) failed: %s", att.name, att.kind, e)
            b = Built([_text_part(f"[файл '{att.name}': не удалось подготовить для модели: {str(e)[:200]}]")],
                      f"{att.kind} '{att.name}' (failed: {str(e)[:80]})", 40)
    if len(_cache) > 64:
        _cache.pop(next(iter(_cache)))
    _cache[att.id] = b
    return b


async def build_parts(atts: list[Attachment]) -> tuple[list[dict], list[str], int]:
    """Parts for all attachments (media first, then document texts) + notes + token estimate."""
    if not atts:
        return [], [], 0
    built = await asyncio.gather(*(asyncio.to_thread(build, a) for a in atts))
    media_parts = [p for b in built for p in b.parts if p.get("type") != "text"]
    text_parts = [p for b in built for p in b.parts if p.get("type") == "text"]
    return media_parts + text_parts, [b.note for b in built], sum(b.tokens for b in built)


# ---------------------------------------------------------------------------- history helpers
def has_media(content) -> bool:
    return isinstance(content, list) and any(p.get("type") in MEDIA_KEYS for p in content)


def text_of(content) -> str:
    """Plain text of a message content (parts of any media are dropped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text").strip()
    return ""


def estimate_tokens(content) -> int:
    if isinstance(content, str):
        return int(len(content) / 3.2)
    n = 0
    for p in content or []:
        t = p.get("type")
        if t == "text":
            n += int(len(p.get("text", "")) / 3.2)
        elif t == "image_url":
            n += 1700
        elif t == "audio_url":
            n += 1200
        elif t == "video_url":
            n += 7000
    return n


def prune_old_media(messages: list[dict], keep_turns: int) -> list[dict]:
    """Copy of the history where media parts survive only in the last `keep_turns` user messages
    (counting every user message, with or without media); older ones become plain text — the answer
    about them is already in the history, and a request without media can go to the fast text model."""
    out = list(messages)
    seen = 0
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if m.get("role") != "user":
            continue
        seen += 1
        if seen > keep_turns and has_media(m.get("content")):
            n = sum(1 for p in m["content"] if p.get("type") in MEDIA_KEYS)
            out[i] = dict(m, content=text_of(m["content"]) + f"\n[медиа этого сообщения ({n} шт.) уже были показаны и убраны из контекста]")
    return out


def redact(messages: list[dict]) -> list[dict]:
    """Messages for the UI trace: base64 payloads replaced by their size."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                p = copy.copy(p)
                for key in MEDIA_KEYS:
                    if p.get("type") == key and isinstance(p.get(key), dict):
                        url = p[key].get("url", "")
                        mm = _DATA_URI_RE.match(url)
                        if mm:
                            p[key] = {"url": f"data:{mm.group(1)};base64,<{len(mm.group(2)) * 3 // 4 // 1024} KB>"}
                parts.append(p)
            out.append(dict(m, content=parts))
        else:
            out.append(m)
    return out
