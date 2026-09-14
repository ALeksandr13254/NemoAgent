"""Uploaded files registry (images, audio, video, documents, screenshots) living in UPLOAD_DIR."""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import mimetypes
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    path: str
    size: int
    mime: str
    is_image: bool
    uploaded_at: float
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
        d = asdict(self)
        d.pop("path", None)
        d["kind"] = self.kind
        return d

    def read(self) -> bytes:
        return Path(self.path).read_bytes()


_FIELDS = {f.name for f in dataclasses.fields(Attachment)}


class AttachmentStore:
    def __init__(self, root: Path = settings.UPLOAD_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Attachment] = {}
        self._load_index()

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> None:
        try:
            data = json.loads(self._index_path().read_text("utf-8"))
            for d in data:
                a = Attachment(**{k: v for k, v in d.items() if k in _FIELDS})  # tolerate fields of older versions
                if Path(a.path).exists():
                    self._items[a.id] = a
        except Exception:
            pass

    def _save_index(self) -> None:
        try:
            self._index_path().write_text(json.dumps([asdict(a) for a in self._items.values()], ensure_ascii=False), "utf-8")
        except Exception:
            pass

    @staticmethod
    def guess_mime(name: str, fallback: str = "application/octet-stream") -> str:
        mime, _ = mimetypes.guess_type(name)
        return mime or fallback

    def add(self, name: str, data: bytes, mime: Optional[str] = None, meta: Optional[dict] = None) -> Attachment:
        if len(data) > settings.UPLOAD_MAX_MB * 1024 * 1024:
            raise ValueError(f"file too large: {len(data)/1048576:.1f} MB > {settings.UPLOAD_MAX_MB} MB")
        safe = "".join(ch for ch in (name or "file") if ch not in '\\/:*?"<>|').strip() or "file"
        ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
        aid = uuid.uuid4().hex[:12]
        path = self.root / f"{aid}_{safe}"
        path.write_bytes(data)
        mime = mime or self.guess_mime(safe)
        if mime == "application/octet-stream":
            mime = self.guess_mime(safe)
        att = Attachment(id=aid, name=safe, path=str(path), size=len(data), mime=mime,
                         is_image=ext in IMAGE_EXTS or mime.startswith("image/"),
                         uploaded_at=time.time(), meta=meta or {})
        self._items[aid] = att
        self._save_index()
        return att

    def get(self, aid: str) -> Optional[Attachment]:
        return self._items.get(aid)

    def image_data_uri(self, att: Attachment, max_side: int = 768, fmt: str = "JPEG", quality: int = 82) -> Optional[str]:
        """Downscaled data-URI for the vision-language embedding model (~780 tokens per image)."""
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
