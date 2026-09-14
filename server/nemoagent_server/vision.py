"""Vision / documents / web search through the DeepSeek reverse API.

Nemotron is text-only, so whenever it needs to *see* something (user attachments, the screen)
or needs fresh facts from the web, it calls a tool that lands here. Each analysis is also written
to long-term memory (VL collection) so it can be recalled later by text or by image similarity.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from .attachments import Attachment, AttachmentStore
from .config import settings
from .deepseek import DeepSeekAPI
from .memory import MemoryStore

log = logging.getLogger("vision")

_CITATION_RE = re.compile(r"\[(?:citation|reference)[^\]]{0,60}\]", re.I)


def _load_cookies() -> dict:
    p = settings.DEEPSEEK_COOKIES_FILE
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("cannot read %s: %s", p, e)
    return {}


class VisionService:
    def __init__(self, attachments: AttachmentStore, memory: MemoryStore):
        self.attachments = attachments
        self.memory = memory
        self.api: Optional[DeepSeekAPI] = None
        self.error: Optional[str] = None
        self._sem = asyncio.Semaphore(2)  # DeepSeek is a browser account, not an API — be gentle
        if settings.DEEPSEEK_AUTH_TOKEN:
            try:
                self.api = DeepSeekAPI(
                    auth_token=settings.DEEPSEEK_AUTH_TOKEN, cookies=_load_cookies(),
                    timezone_offset=settings.DEEPSEEK_TIMEZONE_OFFSET, proxy=settings.DEEPSEEK_PROXY,
                )
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                log.error("DeepSeek client init failed: %s", e)
        else:
            self.error = "DEEPSEEK_AUTH_TOKEN not configured"

    @property
    def enabled(self) -> bool:
        return self.api is not None

    def reload_credentials(self) -> None:
        """Re-read token/cookies from .env + cookies file without restarting the server."""
        from importlib import reload
        from . import config as _cfg
        reload(_cfg)
        if _cfg.settings.DEEPSEEK_AUTH_TOKEN:
            self.api = DeepSeekAPI(auth_token=_cfg.settings.DEEPSEEK_AUTH_TOKEN, cookies=_load_cookies(),
                                   timezone_offset=_cfg.settings.DEEPSEEK_TIMEZONE_OFFSET, proxy=_cfg.settings.DEEPSEEK_PROXY)
            self.error = None

    # ------------------------------------------------------------------ helpers
    async def _ensure_uploaded(self, att: Attachment) -> str:
        if att.deepseek_file_id:
            return att.deepseek_file_id
        if att.size > settings.DEEPSEEK_MAX_FILE_MB * 1048576:
            raise ValueError(f"{att.name}: {att.size/1048576:.0f} MB exceeds the {settings.DEEPSEEK_MAX_FILE_MB} MB limit")
        data = await asyncio.to_thread(att.read)
        fid = await asyncio.to_thread(self.api.upload_file, data, att.name, att.mime,
                                      model_type="default", thinking=settings.DEEPSEEK_THINKING)
        self.attachments.set_deepseek_id(att.id, fid)
        return fid

    async def _ask(self, session: Any, prompt: str, ref_file_ids: list[str], *, search: bool = False, fresh: bool = False) -> dict:
        """Send one prompt to DeepSeek, chained to the agent session's DeepSeek chat when possible."""
        if not self.api:
            return {"error": f"vision backend unavailable: {self.error}"}
        async with self._sem:
            for attempt in range(2):
                ds_sid = None if fresh else session.deepseek_session_id
                parent = None if fresh else session.deepseek_parent_id
                try:
                    if ds_sid is None:
                        ds_sid = await asyncio.to_thread(self.api.create_session)
                        parent = None
                        if not fresh:
                            session.deepseek_session_id = ds_sid
                            session.deepseek_parent_id = None
                    t0 = time.time()
                    result = await asyncio.to_thread(
                        self.api.chat, prompt, chat_session_id=ds_sid, parent_message_id=parent,
                        model_type="default", thinking_enabled=settings.DEEPSEEK_THINKING and not search,
                        search_enabled=search, ref_file_ids=ref_file_ids,
                    )
                    if not fresh and result.get("response_message_id") is not None:
                        session.deepseek_parent_id = result["response_message_id"]
                    text = _CITATION_RE.sub("", result.get("text") or "").strip()
                    log.info("deepseek %s: %d chars in %.1fs", "search" if search else "vision", len(text), time.time() - t0)
                    return {"text": text, "search_log": result.get("search") or "", "elapsed": round(time.time() - t0, 1)}
                except Exception as e:  # noqa: BLE001
                    log.warning("deepseek request failed (attempt %d): %s", attempt + 1, e)
                    session.deepseek_session_id = None
                    session.deepseek_parent_id = None
                    if attempt == 1:
                        return {"error": f"DeepSeek request failed: {e}"}
        return {"error": "unreachable"}

    # ------------------------------------------------------------------ public
    async def analyze(self, session: Any, attachment_ids: list[str], question: str) -> dict:
        if not self.api:
            return {"error": f"vision backend unavailable: {self.error}"}
        ids = [i for i in attachment_ids if i] or list(session.last_attachment_ids)
        atts = [self.attachments.get(i) for i in ids]
        missing = [i for i, a in zip(ids, atts) if a is None]
        atts = [a for a in atts if a is not None]
        if not atts:
            return {"error": "no attachments found" + (f" (unknown ids: {missing})" if missing else "")}
        if len(atts) > settings.DEEPSEEK_MAX_FILES:
            return {"error": f"too many files: {len(atts)} > {settings.DEEPSEEK_MAX_FILES}"}
        try:
            file_ids = await asyncio.gather(*(self._ensure_uploaded(a) for a in atts))
        except Exception as e:  # noqa: BLE001
            return {"error": f"upload to vision backend failed: {e}"}
        names = ", ".join(a.name for a in atts)
        prompt = (f"The user attached: {names}.\n{question}\n"
                  "Be precise and complete; transcribe important text verbatim; mention positions of elements for images. "
                  "Answer in the same language as the question.")
        res = await self._ask(session, prompt, list(file_ids))
        if "error" in res:
            return res
        for a in atts:
            self.attachments.mark_analyzed(a.id)
        uris = [u for u in (self.attachments.image_data_uri(a) for a in atts if a.is_image) if u]
        asyncio.create_task(self.memory.remember_analysis(
            session.id, question, res["text"], [a.public() for a in atts], uris))
        return {"files": [a.name for a in atts], "analysis": res["text"], "elapsed_s": res["elapsed"]}

    async def analyze_screenshot(self, session: Any, shot: dict, question: str) -> dict:
        """shot = {"png_base64": ..., "width":..., "height":..., "monitor":...} from the client."""
        if not self.api:
            return {"error": f"vision backend unavailable: {self.error}"}
        try:
            data = base64.b64decode(shot["png_base64"])
        except Exception as e:  # noqa: BLE001
            return {"error": f"bad screenshot payload: {e}"}
        att = self.attachments.add(f"screen_{time.strftime('%Y%m%d_%H%M%S')}.png", data, "image/png",
                                   meta={"screenshot": True, "width": shot.get("width"), "height": shot.get("height")})
        try:
            fid = await self._ensure_uploaded(att)
        except Exception as e:  # noqa: BLE001
            return {"error": f"screenshot upload failed: {e}"}
        w, h = shot.get("width"), shot.get("height")
        prompt = (f"This is a screenshot of the user's screen ({w}x{h} px, origin top-left).\n{question}\n"
                  "Describe windows, visible text (verbatim where relevant) and interactive elements. "
                  "Give approximate pixel coordinates (x, y) of the elements you mention so they can be clicked. "
                  "Answer in the same language as the question.")
        res = await self._ask(session, prompt, [fid])
        if "error" in res:
            return res
        uri = self.attachments.image_data_uri(att)
        asyncio.create_task(self.memory.remember_analysis(
            session.id, f"[screen] {question}", res["text"], [att.public()], [uri] if uri else None))
        session.note_screenshot(att.id)
        return {"screenshot_id": att.id, "width": w, "height": h, "analysis": res["text"], "elapsed_s": res["elapsed"]}

    async def web_search(self, session: Any, query: str) -> dict:
        if not self.api:
            return {"error": f"search backend unavailable: {self.error}"}
        prompt = (f"{query}\n\nSearch the web and answer concisely with the key facts, dates and numbers. "
                  "List the sources (site and URL) you relied on at the end.")
        res = await self._ask(session, prompt, [], search=True, fresh=True)
        if "error" in res:
            return res
        return {"answer": res["text"], "search_log": res["search_log"][:2000], "elapsed_s": res["elapsed"]}
