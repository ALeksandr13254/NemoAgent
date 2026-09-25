"""Parallel ("hedged") NIM requests: NIMClient._race against a fake NIM (httpx.MockTransport with timed SSE
streams). No network access and no API key are needed; each scenario prints PASS or FAIL.

Run from the server directory:  .venv\Scripts\python.exe tests\test_nim_race.py
"""
import asyncio, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx
from nemoagent_server import nim
from nemoagent_server.config import settings


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def text_chunks(text, finish="stop"):
    out = [sse({"choices": [{"index": 0, "delta": {"content": piece}}]}) for piece in text.split("|")]
    out.append(sse({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    out.append(b"data: [DONE]\n\n")
    return out


class Timed(httpx.AsyncByteStream):
    """SSE body: waits `first` seconds, then yields chunks `gap` apart; records whether it was closed early."""
    def __init__(self, log, name, chunks, first=0.0, gap=0.02, fail_after=None):
        self.log, self.name, self.chunks, self.first, self.gap, self.fail_after = log, name, chunks, first, gap, fail_after
        self.done = False

    async def __aiter__(self):
        await asyncio.sleep(self.first)
        for n, c in enumerate(self.chunks):
            if self.fail_after is not None and n == self.fail_after:
                raise httpx.RemoteProtocolError("peer closed connection")
            yield c
            await asyncio.sleep(self.gap)
        self.done = True

    async def aclose(self):
        self.log.append((self.name, "closed" if not self.done else "finished"))


def make_client(plan):
    """plan: list of callables(log) -> httpx.Response, one per request in arrival order."""
    log, calls = [], []

    async def handler(request: httpx.Request) -> httpx.Response:
        n = len(calls)
        calls.append(time.time())
        if n >= len(plan):
            raise AssertionError(f"unexpected request #{n + 1}")
        return plan[n](log, f"r{n + 1}")

    c = nim.NIMClient()
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://nim.test/v1/")
    return c, log, calls


def ok(text, first=0.0, gap=0.02):
    return lambda log, name: httpx.Response(200, stream=Timed(log, name, text_chunks(text), first, gap),
                                            headers={"content-type": "text/event-stream"})


def empty(first=0.05):
    return lambda log, name: httpx.Response(200, stream=Timed(log, name, [b"data: [DONE]\n\n"], first),
                                            headers={"content-type": "text/event-stream"})


def status(code, body='{"detail": "Worker local total request limit reached (16/16)"}', headers=None):
    return lambda log, name: httpx.Response(code, content=body.encode(), headers=headers or {})


def breaks(text, first=0.0, fail_after=1):
    return lambda log, name: httpx.Response(200, stream=Timed(log, name, text_chunks(text), first, fail_after=fail_after),
                                            headers={"content-type": "text/event-stream"})


async def run(plan, *, parallel=2, max_parallel=3, hedge=6.0, budget=30.0):
    settings.NIM_PARALLEL, settings.NIM_MAX_PARALLEL, settings.NIM_HEDGE_AFTER_S, settings.NIM_RETRY_BUDGET_S = parallel, max_parallel, hedge, budget
    c, log, calls = make_client(plan)
    events = []

    async def on_event(kind, data):
        events.append((kind, data.get("content", data.get("attempt"))))

    t0 = time.time()
    err = res = None
    try:
        res = await c.chat_stream([{"role": "user", "content": "hi"}], None, model="m", on_event=on_event)
    except Exception as e:  # noqa: BLE001
        err = e
    await asyncio.sleep(0.05)
    return {"res": res, "err": err, "t": time.time() - t0, "events": events, "log": log, "calls": len(calls)}


def check(name, cond, info):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {info}"))
    return cond


async def main():
    results = []
    # 1. the faster of two streams wins; the slow one is closed; only the winner's text reaches on_event
    r = await run([ok("Мед|ленный", first=2.0), ok("Быст|рый", first=0.2)])
    deltas = "".join(c for k, c in r["events"] if k == "delta")
    results.append(check("fastest of two wins, loser closed, no leaked text",
                         r["res"] and r["res"].content == "Быстрый" and deltas == "Быстрый" and r["t"] < 0.8
                         and ("r1", "closed") in r["log"], r))
    # 2. one fails fast with 503, the other answers: no retry pause, no 'wait' event
    r = await run([status(503), ok("Ответ", first=0.3)])
    results.append(check("503 on one, the other answers, no wait event",
                         r["res"] and r["res"].content == "Ответ" and not any(k == "wait" for k, _ in r["events"]) and r["t"] < 0.8, r))
    # 3. both come back empty; replacements start after 0.3 s / 0.6 s and one of them answers
    r = await run([empty(), empty(), ok("Третий", first=0.1), ok("Четвёртый", first=0.5)])
    results.append(check("both empty -> replaced quickly, answer within ~1 s",
                         r["res"] and r["res"].content == "Третий" and r["t"] < 1.2 and r["calls"] >= 3, r))
    # 4. both stuck in a queue: a third request joins after NIM_HEDGE_AFTER_S and wins
    r = await run([ok("Первый", first=5.0), ok("Второй", first=5.0), ok("Третий", first=0.1)], hedge=0.5)
    results.append(check("hedge: third request after 0.5 s wins, stuck ones closed",
                         r["res"] and r["res"].content == "Третий" and 0.5 < r["t"] < 1.2
                         and ("r1", "closed") in r["log"] and ("r2", "closed") in r["log"], r))
    # 5. a fatal 400 is raised at once (the same payload would fail everywhere)
    r = await run([status(400, '{"detail": "bad request: unknown field"}'), ok("x", first=3.0)])
    results.append(check("fatal 400 raised without retry",
                         isinstance(r["err"], nim.UpstreamError) and r["err"].status == 400 and r["t"] < 0.5, r))
    # 6. the winner breaks after streaming text: the error surfaces (no silent retry that would duplicate text)
    r = await run([breaks("Нача|ло|конец", first=0.05, fail_after=2), ok("Другой", first=3.0)])
    results.append(check("error after text reached the client is raised, not retried",
                         r["err"] is not None and not isinstance(r["err"], nim._Emitted) and r["calls"] == 2 and r["t"] < 0.8, r))
    # 7. 429 with Retry-After: the replacement waits for it
    r = await run([status(429, '{"detail": "Too Many Requests"}', {"retry-after": "1"}), ok("Поздно", first=0.1)], parallel=1, hedge=0)
    results.append(check("429 honours Retry-After (1 s)", r["res"] and r["res"].content == "Поздно" and 1.0 <= r["t"] < 1.6
                         and any(k == "wait" for k, _ in r["events"]), r))
    # 8. parallel=1, hedge off: plain sequential retries with the short schedule 0.3, 0.6
    r = await run([status(503), status(503), ok("С третьей", first=0.0)], parallel=1, hedge=0)
    results.append(check("single mode: 503, 503, ok after 0.3 + 0.6 s", r["res"] and r["res"].content == "С третьей"
                         and 0.85 <= r["t"] < 1.3 and r["calls"] == 3, r))
    # 9. retry budget: all 503 -> gives up after the budget with the last error
    r = await run([status(503)] * 40, budget=1.2)
    results.append(check("gives up after the retry budget with 503", isinstance(r["err"], nim.UpstreamError)
                         and r["err"].status == 503 and r["t"] < 2.5, r))
    # 10. cancellation (user interrupts): every stream is closed
    settings.NIM_PARALLEL, settings.NIM_MAX_PARALLEL, settings.NIM_HEDGE_AFTER_S = 2, 3, 6.0
    c, log, calls = make_client([ok("a", first=5.0), ok("b", first=5.0)])
    task = asyncio.create_task(c.chat_stream([{"role": "user", "content": "hi"}], None, model="m"))
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)
    results.append(check("interrupt closes every stream", sorted(log) == [("r1", "closed"), ("r2", "closed")], log))
    # 11. tool calls: the winner's tool call is returned intact
    tool = [sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "run_command", "arguments": ""}}]}}]}),
            sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"command\": \"dir\"}"}}]}}]}),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}), b"data: [DONE]\n\n"]
    r = await run([lambda log, name: httpx.Response(200, stream=Timed(log, name, tool, 0.1)), empty()])
    tc = r["res"].tool_calls if r["res"] else []
    results.append(check("tool call from the winner", len(tc) == 1 and tc[0]["function"]["name"] == "run_command"
                         and json.loads(tc[0]["function"]["arguments"]) == {"command": "dir"}, r))
    # 12. error returned as content ('[ERROR: ...') by one stream is not shown; the other stream answers
    r = await run([ok("[ERROR: Agent failed (timed out after 90.0 seconds)]", first=0.05), ok("Нормально", first=0.4)])
    deltas = "".join(c for k, c in r["events"] if k == "delta")
    results.append(check("error-as-content ignored, the good stream wins", r["res"] and r["res"].content == "Нормально"
                         and deltas == "Нормально", r))
    # 13. a 429 turns parallel requests off for the whole client: the next call goes out alone
    settings.NIM_PARALLEL, settings.NIM_MAX_PARALLEL, settings.NIM_HEDGE_AFTER_S, settings.NIM_RETRY_BUDGET_S = 2, 3, 6.0, 30.0
    settings.NIM_RATE_LIMIT_COOLDOWN_S = 60.0
    c, log, calls = make_client([status(429, '{"detail": "Too Many Requests"}', {"retry-after": "0.2"}), ok("Первый", first=0.4),
                                 ok("Второй", first=0.1), ok("лишний", first=0.1)])
    r1 = await c.chat_stream([{"role": "user", "content": "hi"}], None, model="m")
    n_after_first = len(calls)
    r2 = await c.chat_stream([{"role": "user", "content": "hi"}], None, model="m")
    results.append(check("after a 429 the next call sends a single request",
                         r1.content == "Первый" and r2.content == "Второй" and len(calls) == n_after_first + 1, (n_after_first, len(calls))))
    # 14. rate budget: with no headroom left only the first request goes out
    settings.NIM_RPM_BUDGET = 1
    r = await run([ok("Один", first=0.2), ok("лишний", first=0.1)])
    results.append(check("rate budget exhausted -> no parallel request", r["res"] and r["res"].content == "Один" and r["calls"] == 1, r))
    settings.NIM_RPM_BUDGET = 30
    # 15. the extra request is timed from the oldest live request, not from the start of the call, and sent once
    r = await run([empty(0.05), ok("Медленный", first=5.0), ok("Замена", first=5.0), ok("Запасной", first=0.1),
                   ok("лишний", first=0.1)], hedge=0.6)
    results.append(check("one extra request, timed from the oldest live one", r["res"] and r["res"].content == "Запасной"
                         and r["calls"] == 4 and 0.55 < r["t"] < 1.2, r))
    print(f"\n{sum(results)}/{len(results)} passed")
    if sum(results) != len(results):
        sys.exit(1)


asyncio.run(main())
