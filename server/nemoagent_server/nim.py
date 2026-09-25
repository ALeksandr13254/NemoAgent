"""Async client for NVIDIA NIM (integrate.api.nvidia.com): streaming chat with tools + embeddings.

Ported from the Node playground: keeps the retry-on-overload, optional-parameter fallback and
immutable-parameter handling that the free pool needs, but on httpx + asyncio.
"""
from __future__ import annotations

import asyncio
import collections
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
_IMMUTABLE_RE = re.compile(r"`?([a-z_]+)`? is immutable for this model and must be ([-\d.]+)", re.I)
# A repetition loop ("ellsellsellsells…" until max_tokens): a short piece repeated ten or more times at the end
_DEGENERATE_RE = re.compile(r"(.{2,16}?)\1{9,}\s*$", re.S)


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


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
        self._sent: collections.deque = collections.deque()   # send times of the last minute (the key's rate budget)
        self._calm_until = 0.0                                 # after a 429: no extra parallel requests until then
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

    def _note_sent(self) -> None:
        now = time.time()
        self._sent.append(now)
        while self._sent and self._sent[0] < now - 60.0:
            self._sent.popleft()

    def _headroom(self) -> bool:
        """May an extra (parallel) request go out now without eating into the key's per-minute limit?"""
        now = time.time()
        while self._sent and self._sent[0] < now - 60.0:
            self._sent.popleft()
        return now >= self._calm_until and len(self._sent) < settings.NIM_RPM_BUDGET

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
            return await self._race(payload, on_event, model)
        except _Emitted as e:
            raise e.inner     # tokens already reached the client: no silent retry

    @staticmethod
    def _retry_delay(err: Exception, failures: int) -> float:
        """Pause before retrying any error except a 429 (see _rate_limited): none by default; a host that cannot be
        reached at all gets NIM_CONNECT_RETRY_DELAY_S."""
        delays = settings.NIM_RETRY_DELAYS
        delay = delays[min(failures, len(delays) - 1)]
        if isinstance(err, (httpx.ConnectError, httpx.ConnectTimeout)):
            delay = max(delay, settings.NIM_CONNECT_RETRY_DELAY_S)
        return delay

    async def _race(self, payload: dict, on_event, model: str) -> Completion:
        """One call to the pool, with retries. By default a single request is in flight and a failed one is replaced
        at once (see _retry_delay). With NIM_PARALLEL > 1 identical streams run side by side and the first that
        produces real output (a checked text token, reasoning or a tool call) wins, the others are closed at once;
        with NIM_HEDGE_AFTER_S > 0 one more joins when the oldest live stream stays silent that long. Nothing reaches
        on_event before a stream wins, so the client never sees a failed attempt."""
        start = time.time()
        budget_end = start + settings.NIM_RETRY_BUDGET_S
        calm = start < self._calm_until            # a recent 429: one request at a time
        n0 = 1 if calm else max(1, settings.NIM_PARALLEL)
        n_max = n0 if calm else max(n0, settings.NIM_MAX_PARALLEL)
        hedge_after = 0.0 if calm else max(0.0, settings.NIM_HEDGE_AFTER_S)
        tasks: dict[asyncio.Task, int] = {}
        born: dict[asyncio.Task, float] = {}
        first_at = max(start, self._blocked_until)   # another call hit the rate limit: wait with it
        spawn_at: list[float] = [first_at] * n0      # due times of streams still to start
        winner: Optional[int] = None
        hedges = failures = 0
        rl_waited = first_at - start                 # rate-limit waiting: not part of the retry budget
        last_err: Optional[Exception] = None
        if rl_waited > 0 and on_event:
            await on_event("wait", {"stage": "retry", "reason": "rate_limit", "attempt": 0, "delay": rl_waited})

        def claim(i: int) -> None:
            nonlocal winner
            winner = i
            spawn_at.clear()
            for t, j in tasks.items():
                if j != i and not t.done():
                    t.cancel()
            if len(tasks) > 1:
                log.info("NIM %s: request %d of %d answered first after %.1fs", model, i + 1, len(tasks), time.time() - start)

        def gate(i: int):
            async def on(kind: str, data: dict) -> None:
                if winner is None:
                    claim(i)
                if winner != i:
                    raise asyncio.CancelledError()   # a loser that got its first token just as it was being closed
                if on_event:
                    await on_event(kind, data)
            return on

        def spawn() -> bool:
            """Start one more stream. The first live one always goes; extra ones only within the rate budget."""
            if any(not t.done() for t in tasks) and not self._headroom():
                return False
            i = len(tasks)
            t = asyncio.create_task(self._stream_once(dict(payload), gate(i), None, time.time()))
            tasks[t] = i
            born[t] = time.time()
            self._note_sent()
            return True

        try:
            while True:
                now = time.time()
                while winner is None and spawn_at and spawn_at[0] <= now:
                    if now < self._blocked_until:    # a 429 elsewhere meanwhile: every request waits it out
                        spawn_at[:] = sorted(max(x, self._blocked_until) for x in spawn_at)
                        break
                    spawn_at.pop(0)
                    spawn()
                live = [t for t in tasks if not t.done()]
                if not live and not spawn_at:
                    raise last_err or UpstreamError(502, "no response")
                hedge_due = (min(born[t] for t in live) + hedge_after) if live else 0.0
                can_hedge = (winner is None and hedge_after > 0 and bool(live) and len(live) < n_max
                             and hedges < n_max - n0)
                wake = ([spawn_at[0]] if spawn_at else []) + ([hedge_due] if can_hedge else [])
                timeout = max(0.0, min(wake) - time.time()) if wake else None
                if live:
                    done, _ = await asyncio.wait(live, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(timeout or 0.0)
                    done = set()
                if not done:
                    if can_hedge and time.time() >= hedge_due:
                        hedges += 1      # counted even when the budget says no: one decision per call
                        if spawn():
                            log.info("NIM %s: nothing after %.1fs, one more parallel request", model, time.time() - start)
                    continue
                for t in done:
                    i = tasks[t]
                    if t.cancelled():
                        continue
                    err = t.exception()
                    if err is None:
                        if winner is None:
                            claim(i)
                        if winner == i:
                            return t.result()
                        continue
                    if winner == i:
                        if isinstance(err, _Emitted):
                            raise err
                        winner = None     # failed before any text reached the client: the race goes on
                    kind = _classify(err)
                    last_err = err
                    if kind == "immutable":
                        m = _IMMUTABLE_RE.search(str(err))
                        name, value = m.group(1), float(m.group(2))
                        if self._fixed_params.setdefault(model, {}).get(name) != value:
                            self._fixed_params[model][name] = value
                            payload[name] = value
                            log.warning("%s requires %s=%s; retrying", model, name, value)
                        spawn_at.append(time.time())
                        continue
                    if kind not in ("overloaded", "dropped"):
                        raise err
                    in_flight = sum(1 for x in tasks if not x.done())
                    if isinstance(err, UpstreamError) and err.status == 429:
                        # the key's per-minute limit: every request of the server waits, however long it takes
                        self._calm_until = time.time() + settings.NIM_RATE_LIMIT_COOLDOWN_S
                        n0 = n_max = 1
                        hedge_after = 0.0
                        wait = self._rate_limited(err)
                        spawn_at[:] = sorted(max(x, self._blocked_until) for x in spawn_at)
                        if in_flight + len(spawn_at) >= n0:
                            continue      # another request is still trying, or one is already waiting its turn
                        if rl_waited + wait > settings.NIM_RATE_LIMIT_MAX_WAIT_S:
                            log.warning("NIM rate limit (429): gave up after waiting %.0fs", rl_waited)
                            continue
                        rl_waited += wait
                        failures += 1
                        log.warning("NIM rate limit (429%s): every request waits %.1fs",
                                    ", Retry-After" if err.retry_after is not None else "", wait)
                        if on_event:
                            await on_event("wait", {"stage": "retry", "reason": "rate_limit", "attempt": failures, "delay": wait})
                        spawn_at.append(self._blocked_until)
                        spawn_at.sort()
                        continue
                    if time.time() >= budget_end + rl_waited or in_flight + len(spawn_at) >= n0:
                        continue      # out of time, or enough requests are still trying
                    delay = self._retry_delay(err, failures)
                    failures += 1
                    log.warning("NIM %s (%s): retry %d in %.1fs, %d still in flight", kind, str(err)[:120], failures,
                                delay, in_flight)
                    if on_event and in_flight == 0:
                        await on_event("wait", {"stage": "retry", "reason": kind, "attempt": failures, "delay": delay})
                    spawn_at.append(max(time.time() + delay, self._blocked_until))
                    spawn_at.sort()
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
                raise UpstreamError(resp.status_code, _error_detail(text), _retry_after(resp))
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
                    if r.status_code in (429, 502, 503, 504, 529):
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    if r.status_code != 200:
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception as err:  # noqa: BLE001
                    if _classify(err) in ("overloaded", "dropped") and attempt < 3:
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
