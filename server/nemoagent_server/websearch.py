"""Web search for the executor: a search API when a key is configured, otherwise Bing's RSS feed.

Search engines' HTML pages are useless from a script (DuckDuckGo and Yandex show a captcha, Bing and
Google serve decoy results to non-browsers, Brave and Qwant answer 403/429), so without a key the only
thing that still returns real links is Bing's RSS endpoint — acceptable for English queries, weak for
Russian ones. With a free key from tavily.com, brave.com/search/api or serper.dev the results are
proper; the first configured provider wins, the next one is tried when it fails.
"""
from __future__ import annotations

import html
import logging
import re
import time
from typing import Awaitable, Callable, Optional

import httpx

from .config import settings

log = logging.getLogger("websearch")

Result = dict
Provider = Callable[[str, int], Awaitable[tuple[list[Result], Optional[str]]]]


def _client(timeout: float = 20) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True, proxy=settings.WEB_PROXY, trust_env=False,
                             headers={"User-Agent": settings.WEB_USER_AGENT, "Accept-Language": "ru,en;q=0.8"})


def _clean(fragment: Optional[str]) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment or "")).split())


def _cyrillic(q: str) -> bool:
    return bool(re.search(r"[Ѐ-ӿ]", q))


async def _tavily(q: str, n: int) -> tuple[list[Result], Optional[str]]:
    async with _client(25) as c:
        r = await c.post("https://api.tavily.com/search", json={
            "api_key": settings.TAVILY_API_KEY, "query": q, "max_results": n, "include_answer": True, "search_depth": "basic"})
    r.raise_for_status()
    j = r.json()
    res = [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": (x.get("content") or "")[:500]}
           for x in j.get("results", [])]
    return res, j.get("answer") or None


async def _brave(q: str, n: int) -> tuple[list[Result], Optional[str]]:
    async with _client() as c:
        r = await c.get("https://api.search.brave.com/res/v1/web/search", params={"q": q, "count": n},
                        headers={"X-Subscription-Token": settings.BRAVE_SEARCH_API_KEY, "Accept": "application/json"})
    r.raise_for_status()
    res = [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": _clean(x.get("description"))[:500]}
           for x in r.json().get("web", {}).get("results", [])]
    return res, None


async def _serper(q: str, n: int) -> tuple[list[Result], Optional[str]]:
    async with _client() as c:
        r = await c.post("https://google.serper.dev/search", json={"q": q, "num": n, "hl": "ru" if _cyrillic(q) else "en"},
                         headers={"X-API-KEY": settings.SERPER_API_KEY})
    r.raise_for_status()
    j = r.json()
    res = [{"title": x.get("title", ""), "url": x.get("link", ""), "snippet": (x.get("snippet") or "")[:500]}
           for x in j.get("organic", [])]
    box = j.get("answerBox") or {}
    return res, box.get("answer") or box.get("snippet") or None


async def _bing_rss(q: str, n: int) -> tuple[list[Result], Optional[str]]:
    async with _client() as c:
        r = await c.get("https://www.bing.com/search", params={"q": q, "format": "rss", "count": str(n)})
    r.raise_for_status()
    out: list[Result] = []
    for item in re.findall(r"<item>(.*?)</item>", r.text, re.S):
        link = re.search(r"<link>(.*?)</link>", item, re.S)
        if not link:
            continue
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        desc = re.search(r"<description>(.*?)</description>", item, re.S)
        out.append({"title": _clean(title.group(1) if title else ""), "url": html.unescape(link.group(1).strip()),
                    "snippet": _clean(desc.group(1) if desc else "")[:400]})
    return out[:n], None


PROVIDERS: list[tuple[str, Callable[[], bool], Provider]] = [
    ("tavily", lambda: bool(settings.TAVILY_API_KEY), _tavily),
    ("brave", lambda: bool(settings.BRAVE_SEARCH_API_KEY), _brave),
    ("serper", lambda: bool(settings.SERPER_API_KEY), _serper),
    ("bing-rss", lambda: True, _bing_rss),
]


def configured_provider() -> str:
    return next(name for name, enabled, _ in PROVIDERS if enabled())


async def search(query: str, limit: int = 8) -> dict:
    errors: list[str] = []
    for name, enabled, fn in PROVIDERS:
        if not enabled():
            continue
        t0 = time.time()
        try:
            results, answer = await fn(query, limit)
        except Exception as e:  # noqa: BLE001
            log.warning("web search via %s failed: %s", name, e)
            errors.append(f"{name}: {str(e)[:120]}")
            continue
        if results:
            out = {"query": query, "provider": name, "results": results, "elapsed_s": round(time.time() - t0, 1)}
            if answer:
                out["answer"] = answer
            if name == "bing-rss":
                out["hint"] = ("keyless fallback: results are often only loosely related — open the promising ones with "
                               "fetch_page, or fetch a page you can guess (Wikipedia, the vendor's site, docs) directly")
            else:
                out["hint"] = "use fetch_page(url) to read a result in full"
            return out
        errors.append(f"{name}: no results")
    return {"error": "no search results", "details": errors, "query": query,
            "hint": "try fetch_page on a site you know (Wikipedia, the vendor's site, documentation)"}
