"""Uploaded files (images, audio, video, documents, screenshots) kept in memory only.

The server writes nothing to disk. An upload lives in RAM for UPLOAD_TTL_S seconds — long enough for the
conversation turns that use it (each use refreshes the timer) — and is dropped afterwards. The client
keeps the originals in its own chat history.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import settings

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "avif", "apng", "jfif"}
AUDIO_EXTS = {"wav", "mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wma", "amr", "aiff", "aif"}
VIDEO_EXTS = {"mp4", "m4v", "mov", "mkv", "avi", "webm", "wmv", "mpg", "mpeg", "3gp", "ts", "flv"}
TEXT_EXTS = {"txt", "md", "markdown", "rst", "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "toml", "ini", "cfg",
             "conf", "csv", "tsv", "log", "xml", "html", "htm", "css", "scss", "sql", "sh", "bat", "ps1", "cmd", "c", "h",
             "cpp", "hpp", "cs", "java", "kt", "go", "rs", "rb", "php", "swift", "lua", "r", "tex", "bib", "env", "srt",
             "vtt", "diff", "patch", "gradle", "properties", "dockerfile", "makefile"}
OFFICE_EXTS = {"docx", "pptx", "xlsx"}


@dataclass
class Attachment:
    id: str
    name: str
    data: bytes = field(repr=False)
    size: int = 0
    mime: str = "application/octet-stream"
    is_image: bool = False
    uploaded_at: float = 0.0
    last_used: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def ext(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""

    @property
    def kind(self) -> str:
        """image | audio | video | pdf | text | office | other — decides how the model gets the file."""
        ext, mime = self.ext, (self.mime or "")
        if self.is_image or mime.startswith("image/"):
            return "image"
        if ext in VIDEO_EXTS or mime.startswith("video/"):
            return "video"
        if ext in AUDIO_EXTS or mime.startswith("audio/"):
            return "audio"
        if ext == "pdf" or mime == "application/pdf":
            return "pdf"
        if ext in OFFICE_EXTS:
            return "office"
        if ext in TEXT_EXTS or mime.startswith("text/") or mime in ("application/json", "application/xml"):
            return "text"
        return "other"

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "size": self.size, "mime": self.mime, "is_image": self.is_image,
                "uploaded_at": self.uploaded_at, "meta": self.meta, "kind": self.kind}

    def read(self) -> bytes:
        return self.data


class AttachmentStore:
    def __init__(self, ttl_s: float = settings.UPLOAD_TTL_S):
        self.ttl = ttl_s
        self._items: dict[str, Attachment] = {}
        self._lock = threading.Lock()

    @staticmethod
    def guess_mime(name: str, fallback: str = "application/octet-stream") -> str:
        mime, _ = mimetypes.guess_type(name)
        return mime or fallback

    def add(self, name: str, data: bytes, mime: Optional[str] = None, meta: Optional[dict] = None) -> Attachment:
        if len(data) > settings.UPLOAD_MAX_MB * 1024 * 1024:
            raise ValueError(f"file too large: {len(data)/1048576:.1f} MB > {settings.UPLOAD_MAX_MB} MB")
        safe = "".join(ch for ch in (name or "file") if ch not in '\\/:*?"<>|').strip() or "file"
        ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
        mime = mime or self.guess_mime(safe)
        if mime == "application/octet-stream":
            mime = self.guess_mime(safe)
        now = time.time()
        att = Attachment(id=uuid.uuid4().hex[:12], name=safe, data=data, size=len(data), mime=mime,
                         is_image=ext in IMAGE_EXTS or mime.startswith("image/"), uploaded_at=now, last_used=now, meta=meta or {})
        with self._lock:
            self._items[att.id] = att
        self.sweep()
        return att

    def get(self, aid: str) -> Optional[Attachment]:
        with self._lock:
            att = self._items.get(aid)
            if att:
                att.last_used = time.time()
            return att

    def sweep(self) -> int:
        """Drop uploads nobody touched for UPLOAD_TTL_S seconds."""
        cutoff = time.time() - self.ttl
        with self._lock:
            dead = [k for k, a in self._items.items() if a.last_used < cutoff]
            for k in dead:
                del self._items[k]
        if dead:
            from . import media
            media.forget(dead)
        return len(dead)

    def count(self) -> int:
        return len(self._items)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(a.size for a in self._items.values())

    def image_data_uri(self, att: Attachment, max_side: int = 768, fmt: str = "JPEG", quality: int = 82) -> Optional[str]:
        """Downscaled data-URI (used by the client's memory for image similarity)."""
        if not att.is_image:
            return None
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(att.read()))
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format=fmt, quality=quality)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception:
            return None

    @staticmethod
    def describe(att: Attachment) -> str:
        size = f"{att.size/1024:.0f} KB" if att.size < 1048576 else f"{att.size/1048576:.1f} MB"
        return f"{att.kind} '{att.name}' ({size}, id={att.id})"
