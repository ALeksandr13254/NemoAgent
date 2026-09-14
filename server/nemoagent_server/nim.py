"""Async client for NVIDIA NIM (integrate.api.nvidia.com): streaming chat with tools + embeddings.

Ported from the Node playground: keeps the retry-on-overload, optional-parameter fallback and
immutable-parameter handling that the free pool needs, but on httpx + asyncio.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

from .config import settings

log = logging.getLogger("nim")

OPTIONAL_PARAMS = ("chat_template_kwargs", "reasoning_effort", "stream_options")
# The free pool is flaky rather than slow: a 500/503 usually succeeds on the very next try,
# so start with short waits (this is a voice assistant — every second counts).
RETRY_DELAYS = (0.8, 1.5, 3.0, 6.0, 10.0)
_IMMUTABLE_RE = re.compile(r"`?([a-z_]+)`? is immutable for this model and must be ([-\d.]+)", re.I)


class UpstreamError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    first_token_at: float = 0.0


def _error_detail(text: str) -> str:
    try:
        j = json.loads(text)
        d = j.get("detail") or (j.get("error") or {}).get("message") or j.get("error") or j.get("message") or j.get("title") or text
        return d if isinstance(d, str) else json.dumps(d)
    except Exception:
        return text


def _classify(err: Exception) -> Optional[str]:
    msg = str(err)
    if isinstance(err, UpstreamError):
        if _IMMUTABLE_RE.search(msg):
            return "immutable"
        if re.search(r"request limit reached|overloaded|ResourceExhausted|temporarily unavailable|capacity|"
                     r"Agent failed|API failed|timed out after", msg, re.I):
            return "overloaded"
        if err.status in (429, 500, 502, 503, 504, 529):
            return "overloaded"
        return "fatal"
    if isinstance(err, (httpx.TransportError, httpx.RemoteProtocolError)):
        return "dropped"
    return None


class NIMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.NIM_BASE_URL,
            headers={"Authorization": f"Bearer {settings.NVIDIA_API_KEY}", "Accept": "application/json"},
            timeout=httpx.Timeout(connect=20.0, read=float(settings.UPSTREAM_TIMEOUT), write=60.0, pool=60.0),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32, keepalive_expiry=60),
        )
        self._fixed_params: dict[str, dict] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ chat
    async def chat_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        *,
        model: Optional[str] = None,
        thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Any = "auto",
        on_event: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> Completion:
        """Stream one assistant turn. on_event(kind, data) gets 'delta' / 'reasoning' / 'wait' events."""
        model = model or settings.LLM_MODEL
        thinking = settings.LLM_THINKING if thinking is None else thinking
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
            "stream_options": {"include_usage": True},
        }
        if settings.LLM_TOP_P:
            payload["top_p"] = float(settings.LLM_TOP_P)
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        payload.update(self._fixed_params.get(model, {}))
        if log.isEnabledFor(logging.DEBUG):
            log.debug("NIM request: model=%s tool_choice=%s tools=%s keys=%s", model, payload.get("tool_choice"),
                      [t["function"]["name"] for t in (tools or [])], sorted(payload))

        attempt = 0
        while True:
            emitted = False
            started = time.time()
            try:
                return await self._stream_once(payload, on_event, lambda: emitted, started)
            except _Emitted as e:
                emitted = True
                raise e.inner
            except Exception as err:  # noqa: BLE001
                kind = _classify(err)
                text = str(err)
                if kind == "immutable" and attempt < len(RETRY_DELAYS):
                    m = _IMMUTABLE_RE.search(text)
                    name, value = m.group(1), float(m.group(2))
                    self._fixed_params.setdefault(model, {})[name] = value
                    payload[name] = value
                    attempt += 1
                    log.warning("%s requires %s=%s; retrying", model, name, value)
                    continue
                if kind in ("overloaded", "dropped") and attempt < len(RETRY_DELAYS):
                    delay = 1.5 if kind == "dropped" else RETRY_DELAYS[attempt]
                    attempt += 1
                    log.warning("NIM %s (%s) — retry %d in %.1fs", kind, text[:120], attempt, delay)
                    if on_event:
                        await on_event("wait", {"stage": "retry", "reason": kind, "attempt": attempt, "delay": delay})
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _stream_once(self, payload: dict, on_event, emitted_flag, started: float) -> Completion:
        body = dict(payload)
        # optional-parameter fallback: 400 mentioning one of OPTIONAL_PARAMS -> drop it and retry
        for _ in range(len(OPTIONAL_PARAMS) + 1):
            req = self._client.build_request("POST", "chat/completions", json=body,
                                             headers={"Accept": "text/event-stream"})
            resp = await self._client.send(req, stream=True)
            if resp.status_code == 400:
                text = (await resp.aread()).decode("utf-8", "ignore")
                await resp.aclose()
                key = next((k for k in OPTIONAL_PARAMS if k in body and k in text), None)
                if key:
                    log.warning("backend rejected %s — retrying without it (%s)", key, _error_detail(text)[:100])
                    del body[key]
                    continue
                raise UpstreamError(400, _error_detail(text))
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "ignore")
                await resp.aclose()
                raise UpstreamError(resp.status_code, _error_detail(text))
            try:
                return await self._consume(resp, on_event)
            finally:
                await resp.aclose()
        raise UpstreamError(400, "backend kept rejecting optional parameters")

    async def _consume(self, resp: httpx.Response, on_event) -> Completion:
        acc = Completion()
        tool_slots: dict[int, dict] = {}
        emitted = False
        # The free pool sometimes returns its own failure *as the assistant text*:
        #   "[ERROR: Agent failed (... timed out after 90.0 seconds), API failed (...)]"
        # Hold the first characters back until we know the answer is not such an error, so the
        # retry logic can kick in without the client ever seeing it.
        held = ""
        holding = True
        err_prefix = "[ERROR:"

        async def deliver(text: str) -> None:
            nonlocal held, holding, emitted
            if holding:
                held += text
                probe = held.lstrip()
                if probe.startswith(err_prefix):
                    return  # keep holding; decided at the end
                if err_prefix.startswith(probe[:len(err_prefix)]) and len(probe) < len(err_prefix):
                    return  # still ambiguous
                holding = False
                text, held = held, ""
            emitted = True
            if on_event:
                await on_event("delta", {"content": text})

        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    chunk = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    e = chunk["error"]
                    raise UpstreamError(int(e.get("code") or 0), e.get("message") or json.dumps(e))
                if chunk.get("usage"):
                    acc.usage = chunk["usage"]
                choice = (chunk.get("choices") or [None])[0]
                if not choice:
                    continue
                if choice.get("finish_reason"):
                    acc.finish_reason = choice["finish_reason"]
                d = choice.get("delta") or {}
                reasoning = d.get("reasoning_content") or d.get("reasoning")
                if reasoning:
                    if not acc.first_token_at:
                        acc.first_token_at = time.time()
                    acc.reasoning += reasoning
                    emitted = True
                    if on_event:
                        await on_event("reasoning", {"content": reasoning})
                if d.get("content"):
                    if not acc.first_token_at:
                        acc.first_token_at = time.time()
                    acc.content += d["content"]
                    await deliver(d["content"])
                for tc in d.get("tool_calls") or []:
                    if not acc.first_token_at:
                        acc.first_token_at = time.time()
                    i = tc.get("index", 0)
                    slot = tool_slots.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                    if on_event and (fn.get("name") or fn.get("arguments")):
                        # streamed so the agent can start speaking a `speak` call before it is complete
                        await on_event("tool_delta", {"index": i, "name": slot["function"]["name"],
                                                      "arguments": fn.get("arguments") or ""})
        except Exception as e:
            if emitted:
                raise _Emitted(e)
            raise
        acc.tool_calls = [tool_slots[i] for i in sorted(tool_slots)]
        if holding and held:
            if held.lstrip().startswith(err_prefix) and not acc.tool_calls:
                raise UpstreamError(502, f"backend error returned as content: {held.strip()[:300]}")
            emitted = True
            if on_event:
                await on_event("delta", {"content": held})
        return acc

    # ------------------------------------------------------------ embeddings
    async def embed(self, model: str, inputs: list[str], input_type: str = "passage") -> list[list[float]]:
        """Embed texts (or data:image/... URIs for the VL model). Batches of 64."""
        out: list[list[float]] = []
        for i in range(0, len(inputs), 64):
            batch = inputs[i:i + 64]
            for attempt in range(4):
                try:
                    r = await self._client.post("embeddings", json={
                        "model": model, "input": batch, "input_type": input_type, "encoding_format": "float",
                    }, timeout=120)
                    if r.status_code in (429, 502, 503, 504, 529):
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    if r.status_code != 200:
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception as err:  # noqa: BLE001
                    if _classify(err) in ("overloaded", "dropped") and attempt < 3:
                        await asyncio.sleep(RETRY_DELAYS[attempt])
                        continue
                    raise
        return out


class _Emitted(Exception):
    """Wraps an error raised after tokens were already streamed — must not be retried silently."""

    def __init__(self, inner: Exception):
        super().__init__(str(inner))
        self.inner = inner
