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
# The free pool fails in three ways: a fast 503 (all worker slots taken), a stream closed with nothing in it, and a
# request accepted but left in a queue for tens of seconds. A failed request is retried at once by default
# (settings.NIM_RETRY_DELAYS); a 429 (the key's rate limit) makes every request wait (NIMClient._rate_limited).
# Nothing can be done about the third case from here: such a request does not fail, it answers late.
_IMMUTABLE_RE = re.compile(r"`?([a-z_]+)`? is immutable for this model and must be ([-\d.]+)", re.I)
# A repetition loop ("ellsellsellsells…" until max_tokens): a short piece repeated ten or more times at the end
_DEGENERATE_RE = re.compile(r"(.{2,16}?)\1{9,}\s*$", re.S)


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _status_text(resp: httpx.Response) -> str:
    """The message of an error whose body is empty: its status line, and what 451 means here."""
    text = f"HTTP {resp.status_code} {resp.reason_phrase or ''}".strip()
    if resp.status_code == 451:
        text += " (NVIDIA refuses requests from this network: the VPN looks off)"
    return text


def _retry_after(resp: httpx.Response) -> Optional[float]:
    try:
        return float(resp.headers.get("retry-after") or "")
    except ValueError:
        return None


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
        if err.status == 451:
            return "blocked"      # NVIDIA refuses this network (VPN off): it passes as soon as the VPN is back
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
        self._blocked_until = 0.0     # after a 429: no request of this server goes out before then
        self._rl_streak = 0           # 429s in a row (reset after a minute without one)
        self._rl_last = 0.0

    def _rate_limited(self, err: "UpstreamError") -> float:
        """The key's per-minute limit is spent: block every request of this server for a while and return how long
        from now. Retry-After when the gateway sends it, otherwise NIM_RATE_LIMIT_DELAYS in turn."""
        now = time.time()
        if now - self._rl_last > 60.0:
            self._rl_streak = 0
        self._rl_last = now
        if err.retry_after is not None:
            wait = min(max(err.retry_after, 0.5), 120.0)
        else:
            delays = settings.NIM_RATE_LIMIT_DELAYS
            wait = delays[min(self._rl_streak, len(delays) - 1)]
        self._rl_streak += 1
        self._blocked_until = max(self._blocked_until, now + wait)
        return self._blocked_until - now

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

        try:
            return await self._with_retries(payload, on_event, model)
        except _Emitted as e:
            raise e.inner     # tokens already reached the client: no silent retry

    @staticmethod
    def _retry_delay(err: Exception, failures: int) -> float:
        """Pause before retrying any error except a 429 (see _rate_limited): none by default; a host that cannot be
        reached at all, or refuses this network (451, VPN off), gets NIM_CONNECT_RETRY_DELAY_S."""
        delays = settings.NIM_RETRY_DELAYS
        delay = delays[min(failures, len(delays) - 1)]
        if isinstance(err, (httpx.ConnectError, httpx.ConnectTimeout)) or (isinstance(err, UpstreamError) and err.status == 451):
            delay = max(delay, settings.NIM_CONNECT_RETRY_DELAY_S)
        return delay

    async def _wait(self, seconds: float, on_event, reason: str, attempt: int) -> None:
        if on_event:
            await on_event("wait", {"stage": "retry", "reason": reason, "attempt": attempt, "delay": seconds})
        if seconds > 0:
            await asyncio.sleep(seconds)

    async def _with_retries(self, payload: dict, on_event, model: str) -> Completion:
        """One call to the pool. A request that fails (503 and other 5xx, a stream closed with nothing in it, an error
        returned as text, a dropped connection) is sent again at once, until NIM_RETRY_BUDGET_S runs out. A 429 makes
        every request of this server wait (see _rate_limited); that wait does not count against the budget and is
        capped at NIM_RATE_LIMIT_MAX_WAIT_S per call. A failed attempt shows nothing to the client: the stream holds a
        leading error text back, and a 503 or an empty stream carries no tokens."""
        start = time.time()
        failures = 0
        rl_waited = 0.0
        if self._blocked_until > start:              # another call hit the rate limit: wait with it
            rl_waited = self._blocked_until - start
            await self._wait(rl_waited, on_event, "rate_limit", 0)
        while True:
            try:
                return await self._stream_once(dict(payload), on_event)
            except _Emitted:
                raise
            except Exception as err:  # noqa: BLE001
                kind = _classify(err)
                if kind == "immutable":
                    m = _IMMUTABLE_RE.search(str(err))
                    name, value = m.group(1), float(m.group(2))
                    if self._fixed_params.setdefault(model, {}).get(name) == value:
                        raise                               # already fixed and still refused: give up
                    self._fixed_params[model][name] = value
                    payload[name] = value
                    log.warning("%s requires %s=%s; retrying", model, name, value)
                    continue
                if kind not in ("overloaded", "dropped", "blocked"):
                    raise
                failures += 1
                if isinstance(err, UpstreamError) and err.status == 429:
                    wait = self._rate_limited(err)
                    if rl_waited + wait > settings.NIM_RATE_LIMIT_MAX_WAIT_S:
                        log.warning("NIM rate limit (429): gave up after waiting %.0fs", rl_waited)
                        raise
                    rl_waited += wait
                    log.warning("NIM rate limit (429%s): every request waits %.1fs",
                                ", Retry-After" if err.retry_after is not None else "", wait)
                    await self._wait(wait, on_event, "rate_limit", failures)
                    continue
                if time.time() >= start + rl_waited + settings.NIM_RETRY_BUDGET_S:
                    raise
                delay = self._retry_delay(err, failures - 1)
                held = max(0.0, self._blocked_until - (time.time() + delay))   # a 429 elsewhere meanwhile
                rl_waited += held
                log.warning("NIM %s (%s): retry %d in %.1fs", kind, str(err)[:120], failures, delay + held)
                await self._wait(delay + held, on_event, "rate_limit" if held else kind, failures)

    async def _stream_once(self, payload: dict, on_event) -> Completion:
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
                raise UpstreamError(400, _error_detail(text) or _status_text(resp))
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "ignore")
                await resp.aclose()
                raise UpstreamError(resp.status_code, _error_detail(text) or _status_text(resp), _retry_after(resp))
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
                    if len(acc.content) >= 60:
                        loop = _DEGENERATE_RE.search(acc.content[-400:])
                        if loop:
                            # the model fell into a repetition loop (seen on the free pool under load):
                            # cut it here instead of waiting for max_tokens
                            acc.content = acc.content[:-len(loop.group(0))]
                            acc.finish_reason = "degenerate"
                            log.warning("degenerate output after %d chars — stream aborted", len(acc.content))
                            break
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
                        await on_event("tool_delta", {"index": i, "name": slot["function"]["name"],
                                                      "arguments": fn.get("arguments") or ""})
        except Exception as e:
            if emitted:
                raise _Emitted(e)
            raise
        acc.tool_calls = [tool_slots[i] for i in sorted(tool_slots)]
        if acc.finish_reason is None and not acc.content.strip() and not acc.tool_calls and not acc.reasoning:
            # the pool sometimes closes the stream right away with nothing in it (seen with media inputs)
            raise UpstreamError(502, "stream ended without a result")
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
                    if r.status_code != 200:
                        raise UpstreamError(r.status_code, _error_detail(r.text) or _status_text(r), _retry_after(r))
                    data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception as err:  # noqa: BLE001
                    if _classify(err) in ("overloaded", "dropped", "blocked") and attempt < 3:
                        rate_limited = isinstance(err, UpstreamError) and err.status == 429
                        await asyncio.sleep(self._rate_limited(err) if rate_limited else self._retry_delay(err, attempt))
                        continue
                    raise
        return out


class _Emitted(Exception):
    """Wraps an error raised after tokens were already streamed — must not be retried silently."""

    def __init__(self, inner: Exception):
        super().__init__(str(inner))
        self.inner = inner
