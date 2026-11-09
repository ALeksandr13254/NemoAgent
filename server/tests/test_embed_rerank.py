"""Embeddings and reranking against a fake NIM that fails the way the free pool does (httpx.MockTransport).

nemotron-3.5-embed-8b on the pool answers 500 "service inference failed" to most batches of three or more texts
and to every request with two or more images; a single image fails on every other request (one of two replicas is
broken). These scenarios check that the client still gets every vector, in order, and that a failure stays with
its own input when partial results are asked for. No network access and no API key are needed.

Run from the server directory:  .venv\\Scripts\\python.exe tests\\test_embed_rerank.py
"""
import asyncio, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx
from nemoagent_server import nim
from nemoagent_server.config import settings

MM8B = "nvidia/nemotron-3.5-embed-8b"
FAIL = httpx.Response(500, json={"object": "error", "message": "service inference failed", "type": "server_error"})


def vec(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


def make_client(behaviour):
    """behaviour(body, state) -> httpx.Response | None (None = answer with a vector per input)."""
    state = {"calls": [], "image_calls": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["calls"].append(body)
        r = behaviour(body, state)
        if r is not None:
            return r
        if "rankings" in request.url.path or request.url.path.endswith("/reranking"):
            return httpx.Response(200, json={"rankings": [{"index": i, "logit": -float(i)} for i in range(len(body["passages"]))][::-1]})
        return httpx.Response(200, json={"data": [{"index": i, "embedding": vec(x)} for i, x in enumerate(body["input"])][::-1]})

    c = nim.NIMClient()
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://nim.test/v1/")
    return c, state


def broken_8b(body, state):
    """The pool's nemotron-3.5-embed-8b: 3+ inputs fail, 2+ images fail, a single image fails every other time."""
    images = sum(1 for m in body.get("modality") or [] if m != "text")
    if len(body["input"]) >= 3 or images >= 2:
        return FAIL
    if images == 1:
        state["image_calls"] += 1
        if state["image_calls"] % 2 == 1:
            return FAIL
    return None


def check(name, cond, info):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {info}"))
    return bool(cond)


async def main():
    settings.NIM_RETRY_DELAYS = [0.0]
    settings.NIM_RETRY_BUDGET_S = 5.0
    results = []
    texts = [f"текст {i}" for i in range(6)]

    c, st = make_client(broken_8b)
    out = await c.embed(MM8B, texts, "passage", modality="text")
    results.append(check("8b: six texts come back in order although every batch of 3+ fails",
                         out == [vec(t) for t in texts], out))
    results.append(check("8b: sent one input per request from the start (no wasted batch)",
                         all(len(b["input"]) == 1 for b in st["calls"]), [len(b["input"]) for b in st["calls"]]))

    c, st = make_client(broken_8b)
    imgs = ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB", "data:image/jpeg;base64,CCC"]
    out = await c.embed(MM8B, ["кот"] + imgs, "passage", modality=["text", "image", "image", "image"])
    results.append(check("8b: three images and a text pass, each image on a retry at worst",
                         out == [vec(x) for x in ["кот"] + imgs], out))
    results.append(check("8b: modality goes with each single input", all(len(b["modality"]) == 1 for b in st["calls"]),
                         [b.get("modality") for b in st["calls"]]))

    model = "vendor/other-embed"   # a model not known to be broken: a failed batch teaches it one input per request
    c, st = make_client(broken_8b)
    out = await c.embed(model, texts, "query")
    results.append(check("other model: a failing batch falls back to one input per request",
                         out == [vec(t) for t in texts] and model in c._embed_single, out))
    n = len(st["calls"])
    await c.embed(model, texts[:4], "query")
    results.append(check("other model: the next call goes one input per request at once",
                         all(len(b["input"]) == 1 for b in st["calls"][n:]) and len(st["calls"]) - n == 4,
                         [len(b["input"]) for b in st["calls"][n:]]))

    c, st = make_client(lambda body, s: None)
    out = await c.embed("nvidia/nemotron-3-embed-1b", texts * 12, "passage")
    results.append(check("a healthy model keeps batches of 64", out == [vec(t) for t in texts * 12]
                         and [len(b["input"]) for b in st["calls"]] == [64, 8], [len(b["input"]) for b in st["calls"]]))

    def one_bad(body, s):
        return FAIL if "плохой" in body["input"] else None
    settings.NIM_RETRY_BUDGET_S = 0.2
    c, st = make_client(one_bad)
    errors: dict = {}
    out = await c.embed(MM8B, ["хороший", "плохой", "тоже хороший"], "passage", modality="text", errors=errors)
    results.append(check("partial: the input the pool keeps refusing gets None and an error, the others their vectors",
                         out[0] == vec("хороший") and out[1] is None and out[2] == vec("тоже хороший") and list(errors) == [1],
                         (out, errors)))
    c, st = make_client(one_bad)
    try:
        await c.embed(MM8B, ["хороший", "плохой"], "passage", modality="text")
        raised = None
    except nim.UpstreamError as e:
        raised = e
    results.append(check("without partial the failure is raised", raised is not None and raised.status == 500, raised))
    settings.NIM_RETRY_BUDGET_S = 5.0

    c, st = make_client(lambda body, s: httpx.Response(422, json={"detail": "Your request cannot be validated"}))
    try:
        await c.embed(MM8B, ["a"], "passage", modality="text")
        raised = None
    except nim.UpstreamError as e:
        raised = e
    results.append(check("a 422 is final: raised after one request", raised is not None and raised.status == 422
                         and len(st["calls"]) == 1, (raised, len(st["calls"]))))

    flaky = {"n": 0}
    def flaky_rerank(body, s):
        flaky["n"] += 1
        return FAIL if flaky["n"] <= 2 else None
    c, st = make_client(flaky_rerank)
    settings.RERANK_BASE_URL = "https://rerank.test/v1/"
    r = await c.rerank("nvidia/nemotron-3.5-rerank-8b", "запрос", [{"text": "a"}, {"text": "b"}, {"image": "data:image/png;base64,A"}])
    results.append(check("rerank: two 500s are retried, rankings come sorted by logit",
                         [x["index"] for x in r] == [0, 1, 2] and len(st["calls"]) == 3, (r, len(st["calls"]))))

    print(f"\n{sum(results)}/{len(results)} passed")
    if sum(results) != len(results):
        sys.exit(1)


asyncio.run(main())
