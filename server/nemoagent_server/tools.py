"""Tool catalogue for the Nemotron agent.

Two kinds of tools:
  * server tools — executed here (DeepSeek vision/documents/web search, memory, small utilities);
  * client tools — executed on the user's machine by the client (shell, python, files, GUI,
    clipboard…). The server only ships their schemas and proxies the calls over the WebSocket.

`ToolContext` is what a tool implementation gets: the agent session, the client link and the
shared services. Keep tool results JSON-serialisable and reasonably small (they go back into
the model context).
"""
from __future__ import annotations

import ast
import asyncio
import datetime as dt
import json
import logging
import math
import operator
import zoneinfo
from typing import Any, Awaitable, Callable, Optional

import httpx

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
                "type (unicode text), hotkey (e.g. ['ctrl','c']), press (single key), scroll. Coordinates are screen pixels; "
                "use look_at_screen first to see where things are. The screen size is in the system prompt."
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
    query = str(args.get("query", "")).strip()
    limit = int(args.get("limit") or 5)
    items = await ctx.session.services.memory.search(query, top_k=max(1, min(limit, 12)), exclude_session=None, min_score=0.3)
    return {"count": len(items), "results": [{"when": dt.datetime.fromtimestamp(i["ts"]).strftime("%Y-%m-%d %H:%M"),
                                              "kind": i["kind"], "score": round(i["score"], 3), "text": i["text"][:2500]} for i in items]}


async def tool_analyze_attachments(ctx: ToolContext, args: dict) -> dict:
    ids = args.get("attachment_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    question = str(args.get("question") or "Describe the content in detail. For documents, summarise the key information.").strip()
    return await ctx.session.services.vision.analyze(ctx.session, ids, question)


async def tool_look_at_screen(ctx: ToolContext, args: dict) -> dict:
    question = str(args.get("question") or "Describe what is on the screen in detail: windows, text, UI elements and their approximate positions.").strip()
    try:
        shot = await ctx.client_call("__screenshot", {"monitor": args.get("monitor")}, 30.0)
    except Exception as e:  # noqa: BLE001
        return {"error": f"screenshot failed: {e}"}
    if shot.get("error"):
        return shot
    return await ctx.session.services.vision.analyze_screenshot(ctx.session, shot, question)


async def tool_web_search(ctx: ToolContext, args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is empty"}
    return await ctx.session.services.vision.web_search(ctx.session, query)


SERVER_TOOLS: dict[str, tuple[dict, ServerTool]] = {
    "analyze_attachments": ({
        "type": "function",
        "function": {
            "name": "analyze_attachments",
            "description": (
                "Look at the files and images the user attached (listed in the user message as attachment ids). "
                "A vision + document model reads them and answers your question about them (describe, extract text, "
                "summarise, answer a specific question). Up to 50 files, 100 MB each. Call it whenever the user attaches "
                "something — you cannot see attachments yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_ids": {"type": "array", "items": {"type": "string"},
                                       "description": "Attachment ids to analyse. Empty = all attachments of the latest user message."},
                    "question": {"type": "string", "description": "What you need to know about them, in the user's language."},
                },
                "required": ["question"],
            },
        },
    }, tool_analyze_attachments),
    "look_at_screen": ({
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": (
                "Take a screenshot of the user's screen and have a vision model answer a question about it. "
                "Use it to see what the user sees, to find UI elements before gui_action, or to verify a result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "What to look for / describe."},
                    "monitor": {"type": "integer", "description": "Monitor index (1 = primary). Optional."},
                },
                "required": ["question"],
            },
        },
    }, tool_look_at_screen),
    "web_search": ({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information (news, prices, docs, facts after your training data) and get a sourced summary.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A complete question or search query."}},
                "required": ["query"],
            },
        },
    }, tool_web_search),
    "search_memory": ({
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search your long-term memory of past conversations with this user (earlier sessions, archived parts of this one, previously analysed files/screens).",
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


def all_schemas(client_tools_enabled: bool, vision_enabled: bool) -> list[dict]:
    """Tools are for actions only; talking is plain streamed text (see agent.py)."""
    schemas = []
    for name, (schema, _) in SERVER_TOOLS.items():
        if not vision_enabled and name in ("analyze_attachments", "look_at_screen", "web_search"):
            continue
        if name == "look_at_screen" and not client_tools_enabled:
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
