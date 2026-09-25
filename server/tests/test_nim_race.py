"""Requests to NIM with retries: NIMClient._race against a fake NIM (httpx.MockTransport with timed SSE streams).

The first group checks the default mode (one request at a time, a failed one retried at once); the second group
checks the optional parallel mode (NIM_PARALLEL > 1). No network access and no API key are needed; each scenario
prints PASS or FAIL.

Run from the server directory:  .venv\\Scripts\\python.exe tests\\test_nim_race.py
"""
import asyncio, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx
from nemoagent_server import nim
from nemoagent_server.config import settings

RTT = 0.01          # every fake request takes this long before the gateway answers
SCHEDULE = (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0)


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
    """plan: callables(log, name) -> httpx.Response, one per request in arrival order; the last one repeats."""
    log, calls = [], []

    async def handler(request: httpx.Request) -> httpx.Response:
        n = len(calls)
        calls.append(time.time())
        await asyncio.sleep(RTT)
        return plan[min(n, len(plan) - 1)](log, f"r{n + 1}")

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


def refused():
    def f(log, name):
        raise httpx.ConnectError("connection refused")
    return f


def tool_call(first=0.1):
    chunks = [sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "run_command", "arguments": ""}}]}}]}),
              sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"command\": \"dir\"}"}}]}}]}),
              sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}), b"data: [DONE]\n\n"]
    return lambda log, name: httpx.Response(200, stream=Timed(log, name, chunks, first))


def configure(parallel=1, max_parallel=3, hedge=0.0, budget=30.0, delays=(0.0,), rate_limit=(5.0, 10.0, 20.0, 30.0),
              rate_limit_max=120.0, connect=0.5, rpm=30):
    settings.NIM_PARALLEL, settings.NIM_MAX_PARALLEL, settings.NIM_HEDGE_AFTER_S = parallel, max_parallel, hedge
    settings.NIM_RETRY_BUDGET_S, settings.NIM_RETRY_DELAYS = budget, delays
    settings.NIM_RATE_LIMIT_DELAYS, settings.NIM_RATE_LIMIT_MAX_WAIT_S = rate_limit, rate_limit_max
    settings.NIM_CONNECT_RETRY_DELAY_S = connect
    settings.NIM_RPM_BUDGET, settings.NIM_RATE_LIMIT_COOLDOWN_S = rpm, 60.0


async def run(plan, **cfg):
    """One chat_stream call; cfg overrides the defaults of configure() (which are the production defaults)."""
    configure(**cfg)
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
    return {"res": res, "err": err, "t": time.time() - t0, "events": events, "log": log, "calls": len(calls),
            "deltas": "".join(c for k, c in events if k == "delta")}


async def interrupted(plan, **cfg):
    configure(**cfg)
    c, log, calls = make_client(plan)
    task = asyncio.create_task(c.chat_stream([{"role": "user", "content": "hi"}], None, model="m"))
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)
    return log


def check(name, cond, info):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {info}"))
    return cond


def text_is(r, text):
    return r["res"] is not None and r["res"].content == text


async def main():
    results = []
    defaults = (nim.settings.NIM_PARALLEL, nim.settings.NIM_HEDGE_AFTER_S, nim.settings.NIM_RETRY_DELAYS,
                nim.settings.NIM_RATE_LIMIT_DELAYS)
    results.append(check("production defaults: one request at a time, retry at once, 429 waits 5/10/20/30 s",
                         defaults == (1, 0.0, (0.0,), (5.0, 10.0, 20.0, 30.0)), defaults))

    print("-- default mode: sequential requests, immediate retry")
    r = await run([ok("Один", first=0.5), ok("лишний")])
    results.append(check("a single request while the first one is still waiting", text_is(r, "Один") and r["calls"] == 1, r))
    r = await run([status(503), empty(0.05), ok("Третий")])
    results.append(check("503, then an empty stream, then the answer: retried at once", text_is(r, "Третий")
                         and r["calls"] == 3 and r["t"] < 0.35, r))
    r = await run([ok("[ERROR: Agent failed (timed out after 90.0 seconds)]", first=0.02), ok("Нормально")])
    results.append(check("an error returned as text is never shown and retried at once", text_is(r, "Нормально")
                         and r["deltas"] == "Нормально" and r["t"] < 0.35, r))
    r = await run([status(429, '{"detail": "Too Many Requests"}', {"retry-after": "0.5"}), ok("Поздно")])
    results.append(check("429 waits for its Retry-After (0.5 s)", text_is(r, "Поздно") and 0.5 <= r["t"] < 0.9, r))
    too_many = '{"detail": "Too Many Requests"}'
    r = await run([status(429, too_many), ok("Поздно")], rate_limit=(0.4, 0.8))
    results.append(check("429 without Retry-After waits the first pause (0.4 s)", text_is(r, "Поздно")
                         and 0.4 <= r["t"] < 0.8 and ("wait", 1) in r["events"], r))
    r = await run([status(429, too_many), status(429, too_many), ok("Третий")], rate_limit=(0.4, 0.8))
    results.append(check("a second 429 in a row waits longer (0.4 + 0.8 s)", text_is(r, "Третий") and 1.2 <= r["t"] < 1.7, r))
    r = await run([status(429, too_many), ok("Дождался")], budget=0.2, rate_limit=(0.5,))
    results.append(check("rate-limit waiting does not eat the retry budget", text_is(r, "Дождался") and r["t"] >= 0.5, r))
    r = await run([status(429, too_many)], rate_limit=(0.2,), rate_limit_max=0.5)
    results.append(check("gives up with 429 after NIM_RATE_LIMIT_MAX_WAIT_S", isinstance(r["err"], nim.UpstreamError)
                         and r["err"].status == 429 and 0.35 < r["t"] < 1.0 and r["calls"] == 3, r))
    configure(rate_limit=(0.6,))
    c, log, calls = make_client([status(429, too_many), ok("Первый"), ok("Второй")])
    first = asyncio.create_task(c.chat_stream([{"role": "user", "content": "a"}], None, model="m"))
    await asyncio.sleep(0.15)                                    # the first call has its 429 by now
    t_second = time.time()
    second = await c.chat_stream([{"role": "user", "content": "b"}], None, model="m")
    await first
    results.append(check("a 429 in one call holds back the other calls too", second.content in ("Первый", "Второй")
                         and calls[1] - calls[0] >= 0.55 and calls[2] - calls[0] >= 0.55 and time.time() - t_second >= 0.4,
                         [round(x - calls[0], 2) for x in calls]))
    r = await run([refused(), refused(), ok("Сеть")])
    results.append(check("an unreachable host is retried every 0.5 s, not in a tight loop", text_is(r, "Сеть")
                         and 1.0 <= r["t"] < 1.4 and r["calls"] == 3, r))
    r = await run([refused(), ok("Сразу")], connect=0.0)
    results.append(check("NIM_CONNECT_RETRY_DELAY_S=0 retries an unreachable host at once too", text_is(r, "Сразу") and r["t"] < 0.3, r))
    r = await run([status(400, '{"detail": "bad request: unknown field"}'), ok("x")])
    results.append(check("a fatal 400 is raised without a retry", isinstance(r["err"], nim.UpstreamError)
                         and r["err"].status == 400 and r["calls"] == 1, r))
    r = await run([breaks("Нача|ло|конец", first=0.05, fail_after=2), ok("Другой")])
    results.append(check("an error after text reached the client is raised, not retried",
                         r["err"] is not None and not isinstance(r["err"], nim._Emitted) and r["calls"] == 1, r))
    r = await run([status(503)], budget=0.5)
    results.append(check("gives up with the last 503 when the retry budget runs out", isinstance(r["err"], nim.UpstreamError)
                         and r["err"].status == 503 and r["t"] < 1.0 and r["calls"] > 5, r))
    r = await run([tool_call()])
    tc = r["res"].tool_calls if r["res"] else []
    results.append(check("a tool call comes back intact", len(tc) == 1 and tc[0]["function"]["name"] == "run_command"
                         and json.loads(tc[0]["function"]["arguments"]) == {"command": "dir"}, r))
    r = await run([status(503), status(503), ok("С третьей")], delays=SCHEDULE)
    results.append(check("a configured schedule is honoured: 503, 503, answer after 0.3 + 0.6 s", text_is(r, "С третьей")
                         and 0.9 <= r["t"] < 1.3 and r["calls"] == 3, r))
    log = await interrupted([ok("a", first=5.0)])
    results.append(check("an interrupt closes the stream", log == [("r1", "closed")], log))

    print("-- optional parallel mode (NIM_PARALLEL=2)")
    par = {"parallel": 2, "hedge": 6.0, "delays": SCHEDULE}
    r = await run([ok("Мед|ленный", first=2.0), ok("Быст|рый", first=0.2)], **par)
    results.append(check("the faster of two wins, the loser is closed, no leaked text", text_is(r, "Быстрый")
                         and r["deltas"] == "Быстрый" and r["t"] < 0.8 and ("r1", "closed") in r["log"], r))
    r = await run([status(503), ok("Ответ", first=0.3)], **par)
    results.append(check("503 on one, the other answers, no wait event", text_is(r, "Ответ")
                         and not any(k == "wait" for k, _ in r["events"]) and r["t"] < 0.8, r))
    r = await run([empty(), empty(), ok("Третий", first=0.1), ok("Четвёртый", first=0.5)], **par)
    results.append(check("both empty: replaced, answer within ~1 s", text_is(r, "Третий") and r["t"] < 1.2 and r["calls"] >= 3, r))
    r = await run([ok("Первый", first=5.0), ok("Второй", first=5.0), ok("Третий", first=0.1)], **dict(par, hedge=0.5))
    results.append(check("a third request joins after 0.5 s of silence and wins", text_is(r, "Третий") and 0.5 < r["t"] < 1.2
                         and ("r1", "closed") in r["log"] and ("r2", "closed") in r["log"], r))
    r = await run([ok("[ERROR: Agent failed (timed out after 90.0 seconds)]", first=0.05), ok("Нормально", first=0.4)], **par)
    results.append(check("error as text on one stream, the good stream wins", text_is(r, "Нормально") and r["deltas"] == "Нормально", r))
    log = await interrupted([ok("a", first=5.0), ok("b", first=5.0)], **par)
    results.append(check("an interrupt closes every stream", sorted(log) == [("r1", "closed"), ("r2", "closed")], log))
    configure(**par)
    c, log, calls = make_client([status(429, '{"detail": "Too Many Requests"}', {"retry-after": "0.2"}), ok("Первый", first=0.4),
                                 ok("Второй", first=0.1), ok("лишний", first=0.1)])
    r1 = await c.chat_stream([{"role": "user", "content": "hi"}], None, model="m")
    n_first = len(calls)
    r2 = await c.chat_stream([{"role": "user", "content": "hi"}], None, model="m")
    results.append(check("after a 429 the next call sends a single request", r1.content == "Первый" and r2.content == "Второй"
                         and len(calls) == n_first + 1, (n_first, len(calls))))
    r = await run([ok("Один", first=0.2), ok("лишний", first=0.1)], **dict(par, rpm=1))
    results.append(check("with the minute budget spent no extra request goes out", text_is(r, "Один") and r["calls"] == 1, r))
    r = await run([empty(0.05), ok("Медленный", first=5.0), ok("Замена", first=5.0), ok("Запасной", first=0.1),
                   ok("лишний", first=0.1)], **dict(par, hedge=0.6))
    results.append(check("one extra request, timed from the oldest live one", text_is(r, "Запасной")
                         and r["calls"] == 4 and 0.55 < r["t"] < 1.2, r))

    print(f"\n{sum(results)}/{len(results)} passed")
    if sum(results) != len(results):
        sys.exit(1)


asyncio.run(main())
