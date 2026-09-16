"""Tool catalogue for the executor agent.

Two kinds of tools:
  * server tools — executed here: the screen and the attachments (handed to the omni model as
    images / audio / video / text), web search and page fetching, memory, small utilities;
  * client tools — executed on the user's machine by the client (shell, python, files, GUI,
    clipboard…). The server only ships their schemas and proxies the calls over the WebSocket.

A server tool may return `_media` (a list of content parts): the agent loop puts them into the
conversation as a user message right after the tool result — that is how the executor gets to
*see* a screenshot or an attachment (tool results themselves can only be text).

`ToolContext` is what a tool implementation gets: the agent session and the client link. Keep
tool results JSON-serialisable and reasonably small (they go back into the model context).
"""
from __future__ import annotations

import ast
import asyncio
import base64
import datetime as dt
import html
import json
import logging
import math
import operator
import re
import time
import zoneinfo
from typing import Any, Awaitable, Callable, Optional

import httpx

from . import media, websearch
from .config import settings

log = logging.getLogger("tools")

# --------------------------------------------------------------------------- client tools
# Schemas of tools the *client* executes. Descriptions are written for the model.

CLIENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command on the user's computer and return stdout/stderr/exit code. "
                "Default shell is PowerShell on Windows and bash/sh on Linux/macOS (see the OS in the system prompt). "
                "Use it for anything the user asks to do on their machine: open programs, inspect files, install software, "
                "manage processes, change settings, automate tasks. Prefer one well-formed command over many small ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to execute."},
                    "shell": {"type": "string", "enum": ["auto", "powershell", "cmd", "bash", "sh", "zsh"],
                              "description": "Shell to use. 'auto' picks the OS default."},
                    "timeout": {"type": "integer", "description": "Seconds to wait before killing the command (default 60, max 600)."},
                    "cwd": {"type": "string", "description": "Working directory (optional)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python 3 script on the user's computer (the client's own interpreter) and return its output. "
                "Cross-platform alternative to shell commands for file processing, calculations, automation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to run."},
                    "timeout": {"type": "integer", "description": "Seconds before the script is killed (default 60, max 600)."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the user's computer (UTF-8, truncated to max_chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "Default 20000."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or append) UTF-8 text to a file on the user's computer, creating parent folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory on the user's computer (default: user's home).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gui_action",
            "description": (
                "Control the mouse and keyboard on the user's screen: click, double_click, right_click, move, drag, "
                "type (unicode text), hotkey (e.g. ['ctrl','c']), press (single key), scroll. Coordinates are pixels of the "
                "screenshot frame returned by look_at_screen (origin top-left); call look_at_screen first to see where things are."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "double_click", "right_click", "move", "drag", "type", "hotkey", "press", "scroll"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "to_x": {"type": "integer", "description": "For drag: destination."},
                    "to_y": {"type": "integer"},
                    "text": {"type": "string", "description": "For type: the text to type."},
                    "keys": {"type": "array", "items": {"type": "string"}, "description": "For hotkey/press: key names (pyautogui names, e.g. 'enter', 'ctrl', 'win', 'f5')."},
                    "amount": {"type": "integer", "description": "For scroll: positive = up, negative = down (clicks)."},
                    "monitor": {"type": "integer", "description": "Which monitor's screenshot the coordinates refer to (its number from look_at_screen). Required when several monitors were captured."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_target",
            "description": "Open a URL in the default browser, or a file/folder/application with its default program on the user's computer.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "URL, file path, folder path or program name."}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clipboard",
            "description": "Read or replace the user's clipboard text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["get", "set"]},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_windows",
            "description": "List the titles of open windows on the user's desktop (and which one is active), or focus a window by (part of) its title.",
            "parameters": {
                "type": "object",
                "properties": {"focus": {"type": "string", "description": "If given, bring the first window whose title contains this text to the front."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "OS, user, CPU/RAM/disk usage, screen size, uptime and the top processes on the user's computer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

CLIENT_TOOL_NAMES = {t["function"]["name"] for t in CLIENT_TOOLS}


# --------------------------------------------------------------------------- server tools
class ToolContext:
    """Everything a server-side tool may need."""

    def __init__(self, session: "Any", client_call: Callable[[str, dict, float], Awaitable[dict]]):
        self.session = session
        self.client_call = client_call


ServerTool = Callable[[ToolContext, dict], Awaitable[dict]]


def _safe_calc(expression: str) -> float:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
           ast.UAdd: operator.pos}
    funcs = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log",
                                             "log10", "log2", "floor", "ceil", "fabs", "pow", "hypot", "degrees", "radians")}
    funcs.update({"abs": abs, "round": round, "min": min, "max": max, "ln": math.log})
    consts = {"pi": math.pi, "e": math.e, "tau": math.tau}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in consts:
            return consts[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            return funcs[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    return ev(ast.parse(expression.replace("^", "**"), mode="eval"))


async def tool_calculate(ctx: ToolContext, args: dict) -> dict:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return {"error": "expression is empty"}
    try:
        return {"expression": expr, "result": _safe_calc(expr)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot evaluate: {e}"}


def _client_tz(client_info: dict) -> tuple[dt.tzinfo, str]:
    """Timezone of the user's machine: IANA name if valid, else the UTC offset the client reported."""
    name = client_info.get("timezone") or ""
    try:
        if name and "/" in name:
            return zoneinfo.ZoneInfo(name), name
    except Exception:
        pass
    off = client_info.get("utc_offset")  # "+03:00"
    if isinstance(off, str) and len(off) == 6 and off[0] in "+-":
        sign = 1 if off[0] == "+" else -1
        delta = dt.timedelta(hours=int(off[1:3]), minutes=int(off[4:6])) * sign
        return dt.timezone(delta), f"UTC{off}" + (f" ({name})" if name else "")
    return dt.timezone.utc, "UTC"


async def tool_get_current_time(ctx: ToolContext, args: dict) -> dict:
    tz_name = (args.get("timezone") or "").strip()
    if tz_name:
        try:
            tz: dt.tzinfo = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz, tz_name = _client_tz(ctx.session.client_info)
            tz_name = f"{tz_name} (requested zone unknown)"
    else:
        tz, tz_name = _client_tz(ctx.session.client_info)
    now = dt.datetime.now(tz)
    return {"timezone": tz_name, "iso": now.isoformat(), "human": now.strftime("%A, %d %B %Y, %H:%M:%S")}


_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog", 51: "light drizzle",
        53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
        75: "heavy snow", 77: "snow grains", 80: "showers", 81: "heavy showers", 82: "violent showers", 95: "thunderstorm",
        96: "thunderstorm with hail", 99: "severe thunderstorm with hail"}


async def tool_get_weather(ctx: ToolContext, args: dict) -> dict:
    city = str(args.get("city", "")).strip()
    if not city:
        return {"error": "city is required"}
    async with httpx.AsyncClient(timeout=15) as c:
        g = (await c.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1, "language": "ru", "format": "json"})).json()
        place = (g.get("results") or [None])[0]
        if not place:
            return {"error": f"city not found: {city}"}
        w = (await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto"})).json()
    cur = w.get("current", {})
    return {"city": place.get("name"), "country": place.get("country"), "temperature_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"), "humidity_pct": cur.get("relative_humidity_2m"),
            "wind_kmh": cur.get("wind_speed_10m"), "conditions": _WMO.get(cur.get("weather_code"), str(cur.get("weather_code"))),
            "observed_at": cur.get("time")}


async def tool_search_memory(ctx: ToolContext, args: dict) -> dict:
    """Long-term memory lives on the client: the search runs there (embeddings still go through this server)."""
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is empty"}
    limit = max(1, min(int(args.get("limit") or 5), 12))
    try:
        return await ctx.client_call("__search_memory", {"query": query, "limit": limit}, 30.0)
    except Exception as e:  # noqa: BLE001
        return {"error": f"memory search failed: {e}"}


# ---- seeing things: attachments and the screen go to the model as media parts (`_media`)
async def tool_view_attachments(ctx: ToolContext, args: dict) -> dict:
    ids = args.get("attachment_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    ids = [i for i in ids if i] or list(ctx.session.last_attachment_ids)
    store = ctx.session.services.attachments
    atts = [store.get(i) for i in ids]
    missing = [i for i, a in zip(ids, atts) if a is None]
    atts = [a for a in atts if a is not None][:12]
    if not atts:
        return {"error": "no attachments found" + (f" (unknown ids: {missing})" if missing else "")}
    parts, notes, tokens = await media.build_parts(atts)
    return {"files": notes, "note": "the attachments follow this result as images / audio / video / text — look at them",
            "_media": parts}


async def tool_look_at_screen(ctx: ToolContext, args: dict) -> dict:
    """One image per monitor: the requested monitor, or the monitors the user picked in the client settings."""
    try:
        shot = await ctx.client_call("__screenshot", {"monitor": args.get("monitor")}, 30.0)
    except Exception as e:  # noqa: BLE001
        return {"error": f"screenshot failed: {e}"}
    if shot.get("error"):
        return shot
    shots = shot.get("shots") or ([shot] if shot.get("png_base64") else [])
    if not shots:
        return {"error": "the client returned no screenshot"}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts, infos = [], []
    for s in shots:
        try:
            data = base64.b64decode(s["png_base64"])
        except Exception as e:  # noqa: BLE001
            return {"error": f"bad screenshot payload: {e}"}
        w, h, mon = s.get("width"), s.get("height"), s.get("monitor")
        att = ctx.session.services.attachments.add(f"screen_{stamp}_m{mon}.png", data, "image/png",
                                                   meta={"screenshot": True, "width": w, "height": h, "monitor": mon})
        ctx.session.note_screenshot(att.id)
        # keep the client's pixel frame: the client maps these coordinates back onto the real screen
        part, (pw, ph) = await asyncio.to_thread(media.image_part_from_bytes, data, max(int(w or 0), int(h or 0)) or None)
        parts.append(part)
        infos.append({"screenshot_id": att.id, "monitor": mon, "width": pw, "height": ph})
    if len(infos) == 1:
        i = infos[0]
        return {**i, "note": f"the screenshot of monitor {i['monitor']} follows this result as an image "
                             f"({i['width']}x{i['height']} px, origin top-left); give gui_action coordinates in that frame",
                "_media": parts}
    listing = ", ".join(f"monitor {i['monitor']} = {i['width']}x{i['height']} px" for i in infos)
    return {"screenshots": infos, "monitors": shot.get("monitors"),
            "note": f"{len(infos)} screenshots follow this result as images, in this order: {listing} "
                    "(origin top-left in each); give gui_action coordinates in the frame of ONE of them and pass its "
                    "monitor number",
            "_media": parts}


# ---- the web: a search engine's result page + readable page text
def _web_headers() -> dict:
    return {"User-Agent": settings.WEB_USER_AGENT, "Accept-Language": "ru,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"}


def _clean_html(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


async def tool_web_search(ctx: ToolContext, args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is empty"}
    limit = max(1, min(int(args.get("count") or 8), 15))
    return await websearch.search(query, limit)


def html_to_text(body: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page — scripts/styles dropped, block tags become line breaks."""
    title = _clean_html((re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I) or [None, ""])[1])[:200]
    body = re.sub(r"<(script|style|noscript|svg|head|template)[^>]*>.*?</\1>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    body = re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6]|/section|/article|/blockquote|/pre|/td|/th)[^>]*>", "\n", body, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


async def tool_fetch_page(ctx: ToolContext, args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url:
        return {"error": "url is empty"}
    if not re.match(r"https?://", url, re.I):
        url = "https://" + url
    max_chars = max(500, min(int(args.get("max_chars") or 12000), 40000))
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True, proxy=settings.WEB_PROXY, headers=_web_headers()) as c:
            r = await c.get(url)
    except Exception as e:  # noqa: BLE001
        return {"error": f"fetch failed: {e}", "url": url}
    ctype = (r.headers.get("content-type") or "").lower()
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "url": str(r.url)}
    if "html" in ctype or "xml" in ctype:
        title, text = html_to_text(r.text)
    elif "json" in ctype or ctype.startswith("text/"):
        title, text = "", r.text
    else:
        return {"error": f"unsupported content type {ctype} ({len(r.content)} bytes)", "url": str(r.url)}
    return {"url": str(r.url), "title": title, "text": text[:max_chars], "chars": len(text), "truncated": len(text) > max_chars}


SERVER_TOOLS: dict[str, tuple[dict, ServerTool]] = {
    "view_attachments": ({
        "type": "function",
        "function": {
            "name": "view_attachments",
            "description": (
                "Show yourself the files the user attached (listed in the conversation as attachment ids): images, "
                "screenshots, audio, video and document text come back as content you can see, hear and read. "
                "Empty ids = the attachments of the latest user message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_ids": {"type": "array", "items": {"type": "string"},
                                       "description": "Attachment ids to show. Empty = all attachments of the latest user message."},
                },
                "required": [],
            },
        },
    }, tool_view_attachments),
    "look_at_screen": ({
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": (
                "Take a screenshot of the user's screen and see it yourself (it comes back as an image with its pixel size). "
                "Use it to see what the user sees, to find UI elements before gui_action, and to verify a result afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {"monitor": {"type": "integer", "description": "Monitor number (1, 2, ...) to capture just that one. Omit to get the monitors the user selected in the settings (all of them by default, one image each)."}},
                "required": [],
            },
        },
    }, tool_look_at_screen),
    "web_search": ({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (titles, links, snippets) for current information: news, prices, docs, facts after your training data. Follow up with fetch_page to read a result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A search query (keywords work better than long sentences)."},
                    "count": {"type": "integer", "description": "Results wanted (default 8, max 15)."},
                },
                "required": ["query"],
            },
        },
    }, tool_web_search),
    "fetch_page": ({
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Download a web page and return its readable text (HTML stripped) — for reading articles, docs, search results in full.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "Default 12000, max 40000."},
                },
                "required": ["url"],
            },
        },
    }, tool_fetch_page),
    "search_memory": ({
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search your long-term memory of past conversations with this user (earlier sessions, archived parts of this one, previously discussed files/screens).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max results (default 5)."},
                },
                "required": ["query"],
            },
        },
    }, tool_search_memory),
    "get_current_time": ({
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Current date and time (IANA timezone, default = user's timezone).",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []},
        },
    }, tool_get_current_time),
    "calculate": ({
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression precisely (+ - * / ^ %, parentheses, sqrt, sin, cos, log, pi, e…).",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    }, tool_calculate),
    "get_weather": ({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather in a city (Open-Meteo).",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }, tool_get_weather),
}


def all_schemas(client_tools_enabled: bool, memory_enabled: bool = True) -> list[dict]:
    """Tools are for actions only; talking is plain streamed text (see agent.py)."""
    schemas = []
    for name, (schema, _) in SERVER_TOOLS.items():
        if name == "look_at_screen" and not client_tools_enabled:
            continue
        if name == "search_memory" and not memory_enabled:
            continue
        schemas.append(schema)
    if client_tools_enabled:
        schemas.extend(CLIENT_TOOLS)
    return schemas


async def run_server_tool(name: str, ctx: ToolContext, args: dict) -> dict:
    entry = SERVER_TOOLS.get(name)
    if not entry:
        return {"error": f"unknown tool {name}"}
    try:
        return await entry[1](ctx, args or {})
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}


def compact_result(result: Any, limit: int = 24000) -> str:
    s = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
    if len(s) > limit:
        s = s[:limit] + f"… [truncated, {len(s) - limit} more chars]"
    return s
