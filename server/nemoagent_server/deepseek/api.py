"""Unofficial chat.deepseek.com client (updated for the September 2026 frontend, client v2.5.0).

What changed vs. the older reverse-engineered client:
  * there is a single user-facing model now: model_type="default" (Instant). "expert" and "vision"
    still exist in model_configs but are disabled; the default model has file_feature.vision=true,
    so images and documents are simply attached to normal completions;
  * /chat_session/fetch_page became GET with cursor query params;
  * x-client-version is 2.5.0; file limits come from /client/settings?scope=model
    (50 files per message, 100 MB each);
  * the completion payload gained `source` / `action` (both may be omitted / null).
The PoW algorithm and WASM module are unchanged.
"""
from __future__ import annotations

import json
import random
import string
import time
from typing import Any, Dict, Generator, List, Optional

from curl_cffi import requests as curl_requests, CurlMime

from .pow_solver import DeepSeekPOW


class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"
    CLIENT_VERSION = "2.5.0"

    DEFAULT_HEADERS = {
        "accept": "*/*",
        "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/a/chat",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-locale": "en_US",
        "x-client-platform": "web",
        "x-client-version": CLIENT_VERSION,
    }

    def __init__(
        self,
        auth_token: str,
        cookies: Optional[Dict[str, str]] = None,
        timezone_offset: int = 10800,
        impersonate: str = "chrome120",
        proxy: Optional[str] = None,
        pow_solver: Optional[DeepSeekPOW] = None,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token is required (localStorage.userToken on chat.deepseek.com)")
        self.auth_token = auth_token
        self.cookies = dict(cookies or {})
        self.timezone_offset = str(timezone_offset)
        self.impersonate = impersonate
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        self.pow_solver = pow_solver or DeepSeekPOW()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _random_hif_leim() -> str:
        body = "".join(random.choices(string.ascii_letters + string.digits + "+/", k=43))
        suffix = "".join(random.choices(string.ascii_letters + string.digits, k=16))
        return f"{body}=.{suffix}"

    def _headers(self, *, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = dict(self.DEFAULT_HEADERS)
        headers["authorization"] = f"Bearer {self.auth_token}"
        headers["x-client-timezone-offset"] = self.timezone_offset
        headers["x-hif-leim"] = self._random_hif_leim()
        if pow_response:
            headers["x-ds-pow-response"] = pow_response
        return headers

    def _common(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"cookies": self.cookies, "impersonate": self.impersonate}
        if self.proxies:
            kw["proxies"] = self.proxies
        return kw

    def _post(self, path: str, *, json_body: Any, pow_response: Optional[str] = None,
              stream: bool = False, timeout: Optional[int] = 30):
        return curl_requests.post(
            f"{self.BASE_URL}{path}",
            headers=self._headers(pow_response=pow_response),
            json=json_body, stream=stream, timeout=timeout, **self._common(),
        )

    def _get(self, path: str, *, params: Optional[dict] = None, timeout: int = 30):
        return curl_requests.get(
            f"{self.BASE_URL}{path}", headers=self._headers(), params=params,
            timeout=timeout, **self._common(),
        )

    @staticmethod
    def _biz(resp) -> Dict[str, Any]:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        d = data.get("data") or {}
        if d.get("biz_code") not in (0, None):
            raise RuntimeError(f"DeepSeek biz error {d.get('biz_code')}: {d.get('biz_msg')}")
        return d.get("biz_data") or {}

    # ----------------------------------------------------------- sessions
    def create_session(self) -> str:
        biz = self._biz(self._post("/chat_session/create", json_body={}))
        return biz["chat_session"]["id"]

    def list_sessions(self, count: int = 20) -> Dict[str, Any]:
        return self._biz(self._get("/chat_session/fetch_page", params={"count": count}))

    def delete_session(self, chat_session_id: str) -> bool:
        resp = self._post("/chat_session/delete", json_body={"chat_session_id": chat_session_id})
        return resp.status_code == 200

    def model_configs(self) -> List[Dict[str, Any]]:
        """Remote model list with file limits (scope=model settings)."""
        import uuid
        biz = self._biz(self._get("/client/settings", params={"did": str(uuid.uuid4()), "scope": "model"}))
        return (biz.get("settings") or {}).get("model_configs", {}).get("value", [])

    def _get_pow_challenge(self, target_path: str = "/api/v0/chat/completion") -> Dict[str, Any]:
        biz = self._biz(self._post("/chat/create_pow_challenge", json_body={"target_path": target_path}))
        return biz["challenge"]

    # ------------------------------------------------------- files
    def upload_file(self, data: bytes, filename: str, mimetype: str = "application/octet-stream",
                    *, model_type: str = "default", thinking: bool = False,
                    wait: bool = True, timeout: int = 90) -> str:
        """Upload an image or a document; returns DeepSeek file id (waits until parsed)."""
        challenge = self._get_pow_challenge("/api/v0/file/upload_file")
        pow_response = self.pow_solver.solve_and_encode(challenge)
        headers = self._headers(pow_response=pow_response)
        headers.pop("content-type", None)
        headers["x-file-size"] = str(len(data))
        headers["x-model-type"] = model_type
        headers["x-thinking-enabled"] = "1" if thinking else "0"
        mp = CurlMime()
        mp.addpart(name="file", filename=filename, content_type=mimetype, data=data)
        resp = curl_requests.post(
            f"{self.BASE_URL}/file/upload_file", headers=headers, multipart=mp, timeout=300,
            **self._common(),
        )
        file_id = self._biz(resp)["id"]
        if wait:
            self.wait_file_ready(file_id, timeout=timeout)
        return file_id

    def wait_file_ready(self, file_id: str, timeout: int = 90) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._get("/file/fetch_files", params={"file_ids": file_id}, timeout=20)
            status = None
            try:
                files = resp.json()["data"]["biz_data"]["files"]
                status = files[0]["status"] if files else None
                err = files[0].get("error_code") if files else None
            except Exception:
                err = None
            if status == "SUCCESS":
                return True
            if status in ("FAILED", "ERROR"):
                raise RuntimeError(f"file {file_id} parse failed (status={status}, error={err})")
            time.sleep(0.4)
        raise RuntimeError(f"file {file_id} not ready within {timeout}s")

    # ------------------------------------------------------- completion
    def send_message(
        self,
        chat_session_id: str,
        prompt: str,
        *,
        parent_message_id: Optional[int] = None,
        model_type: str = "default",
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: Optional[list] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream a reply. Yields dicts {"type": text|thinking|search|meta, "content": str, ...}.

        The first "meta" chunk carries request_message_id / response_message_id (for chaining
        follow-up messages with parent_message_id).
        """
        challenge = self._get_pow_challenge()
        pow_response = self.pow_solver.solve_and_encode(challenge)
        body = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "source": None,
            "action": None,
            "preempt": False,
        }
        resp = self._post("/chat/completion", json_body=body, pow_response=pow_response,
                          stream=True, timeout=None)
        if resp.status_code != 200:
            error = next(resp.iter_lines(), b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"chat/completion failed: {resp.status_code} {error[:300]}")

        TYPE_MAP = {"THINK": "thinking", "RESPONSE": "text", "TOOL_SEARCH": "search", "TOOL_OPEN": "search"}
        state = {"current_fragment_type": "text", "current_text_path": None, "finished": False}

        def emit(content: str, event_type: str = "text", **extra):
            return {"content": content, "type": event_type, "finish_reason": None, **extra}

        def emit_search_results(results: list):
            yield emit(f"Found {len(results)} pages:\n", "search")
            for r in results[:10]:
                title = r.get("title") or r.get("url", "")
                yield emit(f"  - {title} {r.get('url', '')}\n", "search")

        def handle_new_fragment(frag: dict):
            raw_type = frag.get("type", "RESPONSE")
            mapped = TYPE_MAP.get(raw_type, raw_type.lower())
            state["current_fragment_type"] = mapped
            if raw_type == "TOOL_SEARCH":
                queries = [q.get("query", "") for q in frag.get("queries", [])]
                if queries:
                    yield emit(f"Search: {' | '.join(queries)}\n", "search")
                if frag.get("results"):
                    yield from emit_search_results(frag["results"])
                return
            if raw_type == "TOOL_OPEN":
                result = frag.get("result", {}) or {}
                yield emit(f"Reading: {result.get('title') or result.get('url', '')}\n", "search")
                return
            fcontent = frag.get("content") or ""
            if fcontent:
                yield emit(fcontent, mapped)

        def process_patch(p, o, v):
            o = o or "APPEND"
            if o == "BATCH" and isinstance(v, list):
                for sub in v:
                    sub_p = sub.get("p") or ""
                    full_p = f"{p}/{sub_p}" if (p and sub_p) else (p or sub_p) or ""
                    yield from process_patch(full_p, sub.get("o"), sub.get("v"))
                return
            if not p:
                if isinstance(v, str) and state["current_text_path"]:
                    yield emit(v, state["current_fragment_type"])
                return
            if p == "response/status" and v == "FINISHED":
                state["finished"] = True
                yield {"content": "", "type": state["current_fragment_type"], "finish_reason": "stop"}
                return
            if p == "response/fragments" and o == "APPEND" and isinstance(v, list):
                for frag in v:
                    yield from handle_new_fragment(frag)
                return
            if p.endswith("/results") and o == "SET" and isinstance(v, list):
                yield from emit_search_results(v)
                return
            if p.endswith("/content") and isinstance(v, str):
                state["current_text_path"] = p
                yield emit(v, state["current_fragment_type"])
                return

        current_event: Optional[str] = None
        for raw_line in resp.iter_lines():
            if state["finished"]:
                break
            if raw_line is None:
                continue
            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
            if not line or line.startswith(":"):
                current_event = None
                continue
            if line.startswith("event: "):
                current_event = line[7:].strip()
                if current_event == "close":
                    state["finished"] = True
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if current_event == "error":
                raise RuntimeError(f"DeepSeek server error: {evt}")
            if current_event == "ready":
                yield {"type": "meta", "content": "", "finish_reason": None,
                       "request_message_id": evt.get("request_message_id"),
                       "response_message_id": evt.get("response_message_id")}
                continue
            if current_event in ("update_session", "title"):
                continue
            v, p, o = evt.get("v"), evt.get("p"), evt.get("o")
            if p is None and isinstance(v, dict) and "response" in v:
                for frag in v["response"].get("fragments", []):
                    yield from handle_new_fragment(frag)
                continue
            yield from process_patch(p, o, v)

    def chat(self, prompt: str, *, chat_session_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """One request -> full answer. Returns text, thinking, search log and message ids."""
        if chat_session_id is None:
            chat_session_id = self.create_session()
        text, thinking, search = [], [], []
        meta: Dict[str, Any] = {}
        for chunk in self.send_message(chat_session_id, prompt, **kwargs):
            t = chunk["type"]
            if t == "meta":
                meta = chunk
            elif t == "thinking":
                thinking.append(chunk["content"])
            elif t == "search":
                search.append(chunk["content"])
            else:
                text.append(chunk["content"])
        return {
            "session_id": chat_session_id,
            "text": "".join(text),
            "thinking": "".join(thinking),
            "search": "".join(search),
            "response_message_id": meta.get("response_message_id"),
        }
