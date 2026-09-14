"""Agent session: conversation state, streaming tool loop, memory recall and context trimming."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .attachments import AttachmentStore
from .config import settings
from .memory import MemoryStore
from .nim import Completion, NIMClient
from .tools import CLIENT_TOOL_NAMES, ToolContext, all_schemas, compact_result, run_server_tool
from .vision import VisionService

log = logging.getLogger("agent")

SendFn = Callable[[dict], Awaitable[None]]
ClientCallFn = Callable[[str, dict, float], Awaitable[dict]]


@dataclass
class Services:
    nim: NIMClient
    memory: MemoryStore
    attachments: AttachmentStore
    vision: VisionService


SYSTEM_PROMPT = """You are NemoAgent, a fast voice-and-text assistant that lives on the user's computer.

Capabilities:
- You can run commands, scripts and GUI actions on the user's machine through tools (run_command, run_python, gui_action, open_target, clipboard, list_windows, read_file/write_file, system_info). Use them proactively whenever the user asks to do something on the computer; do not just explain how — do it, then report the result briefly.
- You cannot see images or files yourself. When the user attaches files or images (they appear as "[attachments: ...]" with ids in the message) call analyze_attachments with a precise question. To see the screen call look_at_screen. Results come from a separate vision/document model — treat them as observations.
- web_search gives you fresh information from the internet with sources.
- search_memory searches earlier conversations with this user; relevant memories may also be injected automatically. Memories are facts from the PAST: use them for preferences, names, decisions and context, never for anything time-sensitive (current time, weather, prices, system state, file contents) — for those always call the tool again.

Environment: {env}

Style:
- Reply in the user's language (Russian if the user writes/speaks Russian).
- {voice_style}
- Before dangerous or irreversible actions (deleting data, changing system settings, sending anything, payments) ask for explicit confirmation first.
- Never claim you did something you did not do. If a tool fails, say so and suggest the next step.
- When the user asks to check, run, execute or verify something, actually call the tool in this turn — even if a memory or an earlier answer already contains a plausible result.
- When you use tools, keep the final answer focused on the outcome, not the mechanics."""

VOICE_STYLE_VOICE = ("The user is talking by voice and your answer will be read aloud by a TTS engine: answer in short, natural spoken "
                     "sentences, no markdown, no lists, no tables, no code blocks, no emojis, no URLs unless asked. Spell numbers naturally.")
VOICE_STYLE_TEXT = ("The user typed the message: answer concisely; light markdown (short lists, `code`) is fine when it helps.")


class AgentSession:
    def __init__(self, services: Services, send: SendFn, client_call: ClientCallFn, client_info: Optional[dict] = None):
        self.id = uuid.uuid4().hex[:12]
        self.services = services
        self.send = send
        self.client_call = client_call
        self.client_info: dict = client_info or {}
        self.messages: list[dict] = []
        self.turns = 0
        self.deepseek_session_id: Optional[str] = None
        self.deepseek_parent_id: Optional[int] = None
        self.last_attachment_ids: list[str] = []
        self.session_attachment_ids: list[str] = []
        self._task: Optional[asyncio.Task] = None
        self._archived_note_added = False
        self.created_at = time.time()

    # ------------------------------------------------------------- prompt
    def _env_description(self) -> str:
        ci = self.client_info
        parts = []
        if ci.get("os"):
            parts.append(f"user's OS: {ci['os']}")
        if ci.get("hostname"):
            parts.append(f"host: {ci['hostname']}")
        if ci.get("user"):
            parts.append(f"user: {ci['user']}")
        if ci.get("shell"):
            parts.append(f"default shell: {ci['shell']}")
        if ci.get("screen"):
            parts.append(f"screen: {ci['screen']}")
        if ci.get("timezone") or ci.get("utc_offset"):
            parts.append(f"timezone: {ci.get('timezone', '')} UTC{ci.get('utc_offset', '')}")
        if ci.get("python"):
            parts.append(f"client python: {ci['python']}")
        if ci.get("home"):
            parts.append(f"home dir: {ci['home']}")
        if not ci.get("tools_enabled", True):
            parts.append("computer-control tools are DISABLED by the user")
        parts.append(f"server: {platform.system()}")
        return "; ".join(parts) if parts else "unknown"

    def _now_line(self) -> str:
        """Current date/time on the user's machine — injected every turn so the model never guesses."""
        from .tools import _client_tz
        tz, name = _client_tz(self.client_info)
        now = dt.datetime.now(tz)
        return f"Current date and time on the user's machine: {now.strftime('%A, %d %B %Y, %H:%M')} ({name})."

    def _system_message(self, source: str) -> dict:
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            env=self._env_description() + "\n" + self._now_line(),
            voice_style=VOICE_STYLE_VOICE if source == "voice" else VOICE_STYLE_TEXT)}

    def note_screenshot(self, attachment_id: str) -> None:
        self.session_attachment_ids.append(attachment_id)

    # ------------------------------------------------------------- context
    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        n = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                n += len(c) / 3.2
            if m.get("tool_calls"):
                n += len(json.dumps(m["tool_calls"], ensure_ascii=False)) / 3.2
        return int(n)

    def _trim_context(self) -> None:
        """Drop the oldest turns (they are already in memory) when the live context grows too big."""
        budget = settings.CONTEXT_BUDGET_TOKENS
        if self._estimate_tokens(self.messages) <= budget:
            return
        # Messages are grouped into "turns" starting at each user message; drop whole turns from
        # the front (they are already indexed in memory) but always keep the last few.
        while self._estimate_tokens(self.messages) > budget * 0.7:
            starts = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
            if len(starts) < 2 or len(starts) <= settings.CONTEXT_KEEP_TURNS:
                break
            del self.messages[:starts[1]]
        note = {"role": "system", "content": "Earlier parts of this conversation were archived to long-term memory. "
                                              "Use search_memory if you need details from them."}
        self.messages = [m for m in self.messages
                         if not (m.get("role") == "system" and str(m.get("content", "")).startswith("Earlier parts"))]
        self.messages.insert(0, note)
        log.info("session %s: context trimmed to ~%d tokens, %d messages", self.id, self._estimate_tokens(self.messages), len(self.messages))

    # ------------------------------------------------------------- lifecycle
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def interrupt(self) -> bool:
        if self.busy():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            return True
        return False

    def reset(self) -> None:
        self.messages.clear()
        self.turns = 0
        self.deepseek_session_id = None
        self.deepseek_parent_id = None
        self.last_attachment_ids = []
        self.session_attachment_ids = []
        self.id = uuid.uuid4().hex[:12]
        self.created_at = time.time()

    # ------------------------------------------------------------- main turn
    async def handle_user_message(self, text: str, attachment_ids: list[str], source: str = "text") -> None:
        if self.busy():
            await self.interrupt()
        self._task = asyncio.create_task(self._run_turn(text, attachment_ids, source))
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run_turn(self, text: str, attachment_ids: list[str], source: str) -> None:
        t_start = time.time()
        text = (text or "").strip()
        atts = [self.services.attachments.get(a) for a in attachment_ids or []]
        atts = [a for a in atts if a]
        if not text and not atts:
            await self.send({"type": "done", "finish_reason": "empty", "ms": 0})
            return
        self.last_attachment_ids = [a.id for a in atts]
        self.session_attachment_ids.extend(a.id for a in atts)

        user_content = text or ("(see attachments)" if atts else "")
        if atts:
            listing = "; ".join(self.services.attachments.describe(a) for a in atts)
            user_content += f"\n\n[attachments: {listing}]"

        # --- memory recall (runs before the LLM call; both embeddings in parallel)
        recalled: list[dict] = []
        if settings.MEMORY_AUTO_RECALL and text:
            try:
                recalled = await asyncio.wait_for(
                    self.services.memory.search(text, exclude_session=self.id), timeout=6.0)
            except Exception as e:  # noqa: BLE001
                log.warning("memory recall failed: %s", e)
        if recalled:
            block = self.services.memory.format_for_prompt(recalled)
            self.messages.append({"role": "system", "content": (
                "Relevant memories from earlier conversations. They are PAST exchanges, not current facts: "
                "if the user asks you to check, run, look, measure or verify something, do it with tools now "
                "instead of repeating an old answer.\n" + block)})
            await self.send({"type": "memory", "items": [{"kind": r["kind"], "score": round(r["score"], 2),
                                                          "text": r["text"][:300], "ts": r["ts"]} for r in recalled]})

        self.messages.append({"role": "user", "content": user_content})
        self.turns += 1
        self._trim_context()

        tools_enabled = bool(self.client_info.get("tools_enabled", True))
        schemas = all_schemas(tools_enabled, self.services.vision.enabled)
        ctx = ToolContext(self, self.client_call)
        assistant_text = ""
        first_token_ms: Optional[int] = None
        try:
            for round_no in range(1, settings.MAX_TOOL_ROUNDS + 1):
                await self.send({"type": "round", "round": round_no})
                messages = [self._system_message(source)] + self.messages
                round_t0 = time.time()

                async def on_event(kind: str, data: dict) -> None:
                    nonlocal first_token_ms
                    if kind == "delta":
                        if first_token_ms is None:
                            first_token_ms = int((time.time() - t_start) * 1000)
                        await self.send({"type": "delta", "content": data["content"]})
                    elif kind == "reasoning":
                        await self.send({"type": "reasoning", "content": data["content"]})
                    else:
                        await self.send({"type": "wait", **data})

                acc: Completion = await self.services.nim.chat_stream(messages, schemas, on_event=on_event)
                log.info("session %s round %d: %d chars, %d tool calls, %.1fs", self.id, round_no,
                         len(acc.content), len(acc.tool_calls), time.time() - round_t0)

                if acc.finish_reason == "length":
                    await self.send({"type": "notice", "message": "Ответ обрезан по лимиту max_tokens."})

                if not acc.tool_calls:
                    assistant_text = acc.content
                    self.messages.append({"role": "assistant", "content": acc.content})
                    break

                # --- execute tool calls (client tools may run concurrently)
                calls = []
                for i, tc in enumerate(acc.tool_calls):
                    calls.append({
                        "id": tc.get("id") or f"call_{round_no}_{i}_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"},
                    })
                self.messages.append({"role": "assistant", "content": acc.content or None, "tool_calls": calls})
                if acc.content:
                    assistant_text += acc.content + "\n"

                results = await asyncio.gather(*(self._execute_call(ctx, c) for c in calls))
                for call, result in zip(calls, results):
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "name": call["function"]["name"], "content": compact_result(result)})
            else:
                await self.send({"type": "notice", "message": f"Достигнут лимит {settings.MAX_TOOL_ROUNDS} раундов вызова инструментов."})
                self.messages.append({"role": "assistant", "content": assistant_text or "(tool round limit reached)"})

            await self.send({"type": "done", "finish_reason": "stop", "ms": int((time.time() - t_start) * 1000),
                             "first_token_ms": first_token_ms})
        except asyncio.CancelledError:
            # keep what was already said so the conversation stays coherent
            if assistant_text:
                self.messages.append({"role": "assistant", "content": assistant_text + " [interrupted by user]"})
            await self.send({"type": "done", "finish_reason": "interrupted", "ms": int((time.time() - t_start) * 1000)})
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            await self.send({"type": "error", "message": str(e)[:600]})
            await self.send({"type": "done", "finish_reason": "error", "ms": int((time.time() - t_start) * 1000)})
            return

        if assistant_text.strip():
            asyncio.create_task(self.services.memory.remember_dialog(
                self.id, text, assistant_text, {"source": source, "attachments": [a.name for a in atts]}))

    async def _execute_call(self, ctx: ToolContext, call: dict) -> Any:
        name = call["function"]["name"]
        raw = call["function"]["arguments"]
        try:
            args = json.loads(raw) if raw else {}
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            await self.send({"type": "tool_call", "id": call["id"], "name": name, "arguments": None, "raw": raw})
            return {"error": f"invalid JSON arguments: {raw[:200]}"}
        await self.send({"type": "tool_call", "id": call["id"], "name": name, "arguments": args})
        t0 = time.time()
        if name in CLIENT_TOOL_NAMES:
            if not self.client_info.get("tools_enabled", True):
                result = {"error": "computer-control tools are disabled by the user"}
            else:
                timeout = float(min(int(args.get("timeout") or 60), 600)) + 15
                try:
                    result = await self.client_call(name, args, timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    result = {"error": f"client tool failed: {e}"}
        else:
            result = await run_server_tool(name, ctx, args)
        ms = int((time.time() - t0) * 1000)
        preview = result if isinstance(result, dict) else {"result": result}
        serialized = compact_result(preview, 4000)
        payload = preview if len(serialized) < 4000 else {"preview": serialized}
        await self.send({"type": "tool_result", "id": call["id"], "name": name, "ms": ms, "result": payload})
        return result
