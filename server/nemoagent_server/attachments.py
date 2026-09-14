"""Uploaded files registry (images, documents, screenshots) living in UPLOAD_DIR."""
from __future__ import annotations

import base64
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


@dataclass
class Attachment:
    id: str
    name: str
    path: str
    size: int
    mime: str
    is_image: bool
    uploaded_at: float
    deepseek_file_id: Optional[str] = None
    analyzed: bool = False
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d

    def read(self) -> bytes:
        return Path(self.path).read_bytes()


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
                a = Attachment(**d)
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
        att = Attachment(id=aid, name=safe, path=str(path), size=len(data), mime=mime,
                         is_image=ext in IMAGE_EXTS or mime.startswith("image/"),
                         uploaded_at=time.time(), meta=meta or {})
        self._items[aid] = att
        self._save_index()
        return att

    def get(self, aid: str) -> Optional[Attachment]:
        return self._items.get(aid)

    def set_deepseek_id(self, aid: str, file_id: str) -> None:
        a = self._items.get(aid)
        if a:
            a.deepseek_file_id = file_id
            self._save_index()

    def mark_analyzed(self, aid: str) -> None:
        a = self._items.get(aid)
        if a:
            a.analyzed = True
            self._save_index()

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
        kind = "image" if att.is_image else "file"
        size = f"{att.size/1024:.0f} KB" if att.size < 1048576 else f"{att.size/1048576:.1f} MB"
        return f"{kind} '{att.name}' ({size}, id={att.id})"
