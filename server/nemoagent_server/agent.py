"""Agent session: conversation state, streaming tool loop, memory recall and context trimming.

Speech without function calling. When the client wants voice, the model is prompted to write the
answer itself in a TTS-ready form (rules in prompts.py, editable from the UI) and the text is streamed to the
client as speech while it is being generated (ProseSpeechRouter): the first sentence is spoken
about a second after the request, the rest follows without gaps. A line with `===` separates an
optional screen-only part (code, paths, links). Tools are used only for actions — commands, files,
screen, attachments, web search — never for talking.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .attachments import AttachmentStore
from .config import settings
from .memory import MemoryStore
from .nim import Completion, NIMClient
from .prompts import PromptStore
from .speechfmt import ProseSpeechRouter, looks_like_promise, strip_filler
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
    prompts: PromptStore


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

    def _system_message(self, tts: bool) -> dict:
        # template + style blocks live in prompts.py and can be edited from the client's prompt tab
        return {"role": "system", "content": self.services.prompts.render(
            self._env_description() + "\n" + self._now_line(), tts)}

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
    async def handle_user_message(self, text: str, attachment_ids: list[str], source: str = "text",
                                  tts: bool = False, memory: bool = False) -> None:
        if self.busy():
            await self.interrupt()
        self._task = asyncio.create_task(self._run_turn(text, attachment_ids, source, tts, memory))
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _recall(self, text: str) -> None:
        """Memory recall for the dialogue agent (it has no tools): injected as a system note."""
        try:
            recalled = await asyncio.wait_for(self.services.memory.search(text, exclude_session=self.id), timeout=6.0)
        except Exception as e:  # noqa: BLE001
            log.warning("memory recall failed: %s", e)
            return
        if not recalled:
            return
        block = self.services.memory.format_for_prompt(recalled)
        self.messages.append({"role": "system", "content": (
            "Воспоминания из прошлых разговоров (фон из ПРОШЛОГО, не текущий запрос; отвечай на последнее "
            "сообщение по существу):\n" + block)})
        await self.send({"type": "memory", "items": [{"kind": r["kind"], "score": round(r["score"], 2),
                                                      "text": r["text"][:300], "ts": r["ts"]} for r in recalled]})

    def _context_for_executor(self, limit: int = 10) -> list[dict]:
        """Recent user/assistant turns (plain strings) so the executor understands what the task is about."""
        out = []
        for m in self.messages:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
                out.append({"role": m["role"], "content": m["content"][:2000]})
        return out[-limit:]

    def _trace_params(self, tts: bool, use_memory: bool, source: str, tools: list) -> dict:
        return {"temperature": settings.LLM_TEMPERATURE, "max_tokens": settings.LLM_MAX_TOKENS,
                "thinking": settings.LLM_THINKING, "tool_choice": "auto" if tools else None,
                "tts": tts, "memory": use_memory, "source": source}

    async def _dialogue_call(self, tts: bool, use_memory: bool, source: str, t_start: float, stage: str,
                             call_no: int) -> tuple[str, Optional[str], Optional[str], Optional[int]]:
        """One streamed call of the dialogue agent (no tools).

        Returns (speech, display, task, first_token_ms). Spoken text streams to the client while it
        is generated; the `===` part goes to the screen; a `>>>` task is handed to the executor.
        """
        messages = [self._system_message(tts)] + self.messages
        router = ProseSpeechRouter()
        first_token_ms: Optional[int] = None
        await self.send({"type": "stage", "name": stage, "agent": "dialogue"})

        async def emit(events) -> None:
            for what, piece in events:
                if what == "speech":
                    await self.send({"type": "speech_delta" if tts else "delta", "content": piece})
                else:
                    await self.send({"type": "delta", "content": piece})

        async def on_event(kind: str, data: dict) -> None:
            nonlocal first_token_ms
            if kind == "delta":
                if first_token_ms is None:
                    first_token_ms = int((time.time() - t_start) * 1000)
                await emit(router.feed(data["content"]))
            elif kind == "reasoning":
                await self.send({"type": "reasoning", "content": data["content"]})
            elif kind == "wait":
                await self.send({"type": "wait", **data})

        t0 = time.time()
        await self.send({"type": "trace", "kind": "request", "agent": "dialogue", "stage": stage, "turn": self.turns,
                         "round": call_no, "model": settings.LLM_MODEL, "messages": messages, "tools": [],
                         "params": self._trace_params(tts, use_memory, source, [])})
        acc: Completion = await self.services.nim.chat_stream(messages, None, on_event=on_event)
        await emit(router.finish())
        await self.send({"type": "trace", "kind": "response", "agent": "dialogue", "stage": stage, "turn": self.turns,
                         "round": call_no, "content": acc.content, "reasoning": acc.reasoning, "tool_calls": [],
                         "finish_reason": acc.finish_reason, "usage": acc.usage, "ms": int((time.time() - t0) * 1000)})
        if acc.finish_reason == "length":
            await self.send({"type": "notice", "message": "Ответ обрезан по лимиту max_tokens."})
        speech, display, task = router.result
        log.info("session %s dialogue/%s: %d chars spoken, task=%s, %.1fs", self.id, stage, len(speech),
                 "yes" if task else "no", time.time() - t0)
        return speech, display, task, first_token_ms

    async def _run_executor(self, task: str, ctx: ToolContext, use_memory: bool, tools_enabled: bool,
                            tts: bool, source: str, call_no: int) -> tuple[str, int]:
        """The executor agent: tools loop on the task; returns (report, calls_used)."""
        await self.send({"type": "stage", "name": "executor", "agent": "executor"})
        schemas = all_schemas(tools_enabled, self.services.vision.enabled, memory_enabled=use_memory)
        system = {"role": "system", "content": self.services.prompts.render_executor(
            self._env_description() + "\n" + self._now_line())}
        work: list[dict] = self._context_for_executor() + [
            {"role": "user", "content": f"Задача от диалогового агента: {task}"}]
        last_signature: Optional[str] = None
        narration = ""
        tools_used = 0
        forced_retry = False
        for round_no in range(1, settings.MAX_TOOL_ROUNDS + 1):
            messages = [system] + work
            t0 = time.time()
            # The executor's job is to act, so the first round must call a tool (tool_choice=required);
            # later rounds are free to finish with the report. Its output is never spoken, so the
            # streaming quirks of `required` do not matter here.
            choice = "required" if (tools_used == 0 and schemas and not forced_retry) else "auto"
            await self.send({"type": "trace", "kind": "request", "agent": "executor", "stage": "executor", "turn": self.turns,
                             "round": call_no + round_no - 1, "model": settings.LLM_MODEL, "messages": messages,
                             "tools": [s["function"]["name"] for s in schemas],
                             "params": dict(self._trace_params(tts, use_memory, source, schemas), tool_choice=choice)})

            async def on_event(kind: str, data: dict) -> None:
                if kind == "delta":
                    await self.send({"type": "executor_delta", "content": data["content"]})
                elif kind == "reasoning":
                    await self.send({"type": "reasoning", "content": data["content"]})
                elif kind == "wait":
                    await self.send({"type": "wait", **data})

            acc: Completion = await self.services.nim.chat_stream(messages, schemas, tool_choice=choice, on_event=on_event)
            await self.send({"type": "trace", "kind": "response", "agent": "executor", "stage": "executor", "turn": self.turns,
                             "round": call_no + round_no - 1, "content": acc.content, "reasoning": acc.reasoning,
                             "tool_calls": acc.tool_calls, "finish_reason": acc.finish_reason, "usage": acc.usage,
                             "ms": int((time.time() - t0) * 1000)})
            log.info("session %s executor round %d (%s): %d chars, %d tool calls, %.1fs", self.id, round_no, choice,
                     len(acc.content), len(acc.tool_calls), time.time() - t0)
            if not acc.tool_calls:
                leaked = acc.content.lstrip().startswith(("[", "{")) and '"name"' in acc.content
                if choice == "required" and not forced_retry and (leaked or not acc.content.strip()):
                    # `required` sometimes comes back as raw JSON text: one more try without forcing
                    forced_retry = True
                    log.warning("session %s: executor returned a tool call as text, retrying with auto", self.id)
                    continue
                report = acc.content.strip() or narration.strip() or "(исполнитель не вернул отчёт)"
                if tools_used == 0:
                    report = "(ВНИМАНИЕ: исполнитель не вызвал ни одного инструмента, результат не проверен)\n" + report
                return report, round_no
            tools_used += len(acc.tool_calls)
            calls = []
            for i, tc in enumerate(acc.tool_calls):
                calls.append({"id": tc.get("id") or f"call_{round_no}_{i}_{uuid.uuid4().hex[:6]}", "type": "function",
                              "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"}})
            signature = json.dumps([(c["function"]["name"], c["function"]["arguments"]) for c in calls], ensure_ascii=False)
            if signature == last_signature:
                log.warning("session %s: executor repeats tool calls, stopping", self.id)
                return (narration.strip() + "\n(исполнитель зациклился на одинаковых вызовах и был остановлен)").strip(), round_no
            last_signature = signature
            if acc.content and acc.content.strip():
                narration = acc.content.strip()
            work.append({"role": "assistant", "content": acc.content.strip() or None, "tool_calls": calls})
            results = await asyncio.gather(*(self._execute_call(ctx, c) for c in calls))
            for call, result in zip(calls, results):
                work.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                             "content": compact_result(result)})
        return (narration.strip() + "\n(достигнут лимит раундов инструментов, задача могла остаться незавершённой)").strip(), settings.MAX_TOOL_ROUNDS

    async def _run_turn(self, text: str, attachment_ids: list[str], source: str, tts: bool, memory: bool = False) -> None:
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
        self.messages.append({"role": "user", "content": user_content})
        use_memory = memory or settings.MEMORY_AUTO_RECALL
        if use_memory and text:
            await self._recall(text)
        self.turns += 1
        self._trim_context()

        tools_enabled = bool(self.client_info.get("tools_enabled", True))
        ctx = ToolContext(self, self.client_call)
        assistant_text = ""
        reports: list[str] = []
        first_token_ms: Optional[int] = None
        call_no = 1
        try:
            # ---- 1. dialogue agent answers (and may hand a task to the executor)
            speech, display, task, first_token_ms = await self._dialogue_call(tts, use_memory, source, t_start, "answer", call_no)
            call_no += 1
            said = speech + (("\n" + display) if display else "")
            if said.strip():
                self.messages.append({"role": "assistant", "content": said})
                assistant_text = said
            if not task and looks_like_promise(speech):
                # "Сейчас проверю." with nothing behind it: turn the user's request itself into the task
                task = "Выполни то, о чём попросил пользователь: " + user_content
                log.info("session %s: promise without a task -> auto task", self.id)
                await self.send({"type": "notice", "message": "Голосовой агент пообещал действие без задачи — задача поставлена автоматически."})

            # ---- 2. executor does the work, 3. dialogue agent reports (at most two hops per turn)
            hops = 0
            while task and hops < 2:
                hops += 1
                if tts:
                    await self.send({"type": "speech_done", "display": display, "final": False})
                await self.send({"type": "task", "task": task})
                report, used = await self._run_executor(task, ctx, use_memory, tools_enabled, tts, source, call_no)
                call_no += used
                reports.append(report)
                await self.send({"type": "report", "task": task, "report": report})
                self.messages.append({"role": "system", "content": f"Результат исполнителя по задаче «{task}»:\n{report}"})
                speech, display, task, _ = await self._dialogue_call(tts, use_memory, source, t_start, "report", call_no)
                call_no += 1
                said = speech + (("\n" + display) if display else "")
                if said.strip():
                    self.messages.append({"role": "assistant", "content": said})
                    assistant_text = (assistant_text + "\n" + said).strip()
            if tts:
                await self.send({"type": "speech_done", "display": display, "final": True})
            await self.send({"type": "done", "finish_reason": "speak" if tts else "stop",
                             "ms": int((time.time() - t_start) * 1000), "first_token_ms": first_token_ms})
        except asyncio.CancelledError:
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
            memo = assistant_text
            if reports:
                memo += "\n[исполнитель] " + " | ".join(r[:600] for r in reports)
            asyncio.create_task(self.services.memory.remember_dialog(
                self.id, text, memo, {"source": source, "attachments": [a.name for a in atts]}))

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
