"""Agent session: conversation state, the two-agent turn, memory recall and context trimming.

One omni model, two prompts. The user's message goes to the model together with its attachments as
media parts (images, audio, video; documents as text — see media.py), so the DIALOGUE agent answers
about them directly, by voice, without any tool. Speech without function calling: when the client
wants voice, the dialogue agent is prompted to write the answer itself in a TTS-ready form (rules in
prompts.py, editable from the UI) and the text is streamed to the client as speech while it is
generated (ProseSpeechRouter): the first sentence is spoken about a second after the request, the
rest follows without gaps. A line with `===` separates an optional screen-only part (code, paths,
links); a `>>>` line hands a task to the EXECUTOR, which runs the tools (commands, files, screen,
web, memory — the screen and the attachments come back to it as images) and returns a report that
the dialogue agent then tells the user.
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

from . import media
from .attachments import AttachmentStore
from .config import settings
from .memory import MemoryStore
from .nim import Completion, NIMClient
from .prompts import PromptStore
from .speechfmt import ProseSpeechRouter, looks_like_promise, strip_filler
from .tools import CLIENT_TOOL_NAMES, ToolContext, all_schemas, compact_result, run_server_tool

log = logging.getLogger("agent")

SendFn = Callable[[dict], Awaitable[None]]
ClientCallFn = Callable[[str, dict, float], Awaitable[dict]]


def _norm_args(raw: str) -> str:
    """Tool arguments with sorted keys, so {"a":1,"b":2} and {"b":2,"a":1} count as the same call."""
    try:
        return json.dumps(json.loads(raw or "{}"), sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return raw or ""


@dataclass
class Services:
    nim: NIMClient
    memory: MemoryStore
    attachments: AttachmentStore
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
        env = "; ".join(parts) if parts else "unknown"
        # the voice that reads the answers decides the grammatical gender the assistant uses for itself
        gender = ci.get("persona_gender")
        if gender == "male":
            env += ("\nYour voice is MALE: speak about yourself in the masculine gender in Russian and other gendered "
                    "languages (я готов, я проверил, я понял, я рад).")
        elif gender == "female":
            env += ("\nYour voice is FEMALE: speak about yourself in the feminine gender in Russian and other gendered "
                    "languages (я готова, я проверила, я поняла, я рада).")
        return env

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
            n += media.estimate_tokens(m.get("content"))
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

    def _for_request(self) -> list[dict]:
        """History as sent to the model: media only in the last MEDIA_KEEP_TURNS user messages that carry it
        (older images/audio/video become a text stub — the answer about them is already in the history)."""
        return media.prune_old_media(self.messages, settings.MEDIA_KEEP_TURNS)

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
        """Recent user/assistant turns as plain text so the executor understands what the task is about."""
        out = []
        for m in self.messages:
            if m.get("role") in ("user", "assistant"):
                text = media.text_of(m.get("content"))
                if text.strip():
                    out.append({"role": m["role"], "content": text[:2000]})
        return out[-limit:]

    def _trace_params(self, tts: bool, use_memory: bool, source: str, tools: list, max_tokens: Optional[int] = None) -> dict:
        return {"temperature": settings.LLM_TEMPERATURE, "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                "thinking": settings.LLM_THINKING, "tool_choice": "auto" if tools else None,
                "tts": tts, "memory": use_memory, "source": source}

    @property
    def text_model(self) -> str:
        """The client may pick the text model in its settings (Lightning vs Super); otherwise the server default."""
        chosen = str(self.client_info.get("text_model") or "").strip()
        return chosen or settings.LLM_MODEL

    def _model_for(self, messages: list[dict]) -> str:
        """The text model unless the request carries images / audio / video — then the omni model."""
        if any(media.has_media(m.get("content")) for m in messages):
            return settings.LLM_MEDIA_MODEL
        return self.text_model

    async def _dialogue_call(self, tts: bool, use_memory: bool, source: str, t_start: float, stage: str,
                             call_no: int) -> tuple[str, Optional[str], Optional[str], Optional[int]]:
        """One streamed call of the dialogue agent (no tools).

        Returns (speech, display, task, first_token_ms). Spoken text streams to the client while it
        is generated; the `===` part goes to the screen; a `>>>` task is handed to the executor.
        """
        messages = [self._system_message(tts)] + self._for_request()
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
        model = self._model_for(messages)
        for attempt in range(2):
            router = ProseSpeechRouter()
            await self.send({"type": "trace", "kind": "request", "agent": "dialogue", "stage": stage, "turn": self.turns,
                             "round": call_no, "model": model, "messages": media.redact(messages), "tools": [],
                             "params": self._trace_params(tts, use_memory, source, [], settings.DIALOGUE_MAX_TOKENS)})
            acc: Completion = await self.services.nim.chat_stream(messages, None, model=model,
                                                                  max_tokens=settings.DIALOGUE_MAX_TOKENS, on_event=on_event)
            await emit(router.finish())
            await self.send({"type": "trace", "kind": "response", "agent": "dialogue", "stage": stage, "turn": self.turns,
                             "round": call_no, "content": acc.content, "reasoning": acc.reasoning, "tool_calls": [],
                             "finish_reason": acc.finish_reason, "usage": acc.usage, "ms": int((time.time() - t0) * 1000)})
            if acc.finish_reason == "degenerate":
                if attempt == 0 and not router.speech.strip():
                    # the loop started before anything was said: one more try
                    log.warning("session %s: dialogue output degenerated before any speech, retrying", self.id)
                    continue
                await self.send({"type": "notice", "message": "Модель зациклилась, ответ обрезан."})
            break
        if acc.finish_reason == "length":
            await self.send({"type": "notice", "message": "Ответ обрезан по лимиту max_tokens."})
        speech, display, task = router.result
        log.info("session %s dialogue/%s: %d chars spoken, task=%s, %.1fs", self.id, stage, len(speech),
                 "yes" if task else "no", time.time() - t0)
        return speech, display, task, first_token_ms

    async def _run_executor(self, task: str, ctx: ToolContext, use_memory: bool, tools_enabled: bool,
                            tts: bool, source: str, call_no: int, task_media: Optional[list[dict]] = None) -> tuple[str, int]:
        """The executor agent: tools loop on the task; returns (report, calls_used).

        `task_media` — the attachments of the current user message, so a task about them ("extract the
        table from the photo and save it") starts with the model seeing them.
        """
        await self.send({"type": "stage", "name": "executor", "agent": "executor"})
        schemas = all_schemas(tools_enabled, memory_enabled=use_memory)
        system = {"role": "system", "content": self.services.prompts.render_executor(
            self._env_description() + "\n" + self._now_line())}
        task_text = f"Задача от диалогового агента: {task}"
        task_msg: dict = {"role": "user", "content": task_text}
        if task_media:
            task_msg = {"role": "user", "content": [{"type": "text", "text": task_text + "\n(вложения пользователя приложены ниже)"},
                                                    *task_media]}
        work: list[dict] = self._context_for_executor() + [task_msg]
        seen_calls: set[str] = set()
        recent_results: list[str] = []   # last tool outputs, so a report exists even when the round limit hits
        narration = ""
        tools_used = 0
        force_tools = False
        degenerate_retry = False
        for round_no in range(1, settings.MAX_TOOL_ROUNDS + 1):
            messages = [system] + work
            t0 = time.time()
            # The executor's job is to act. The first round runs with tool_choice=auto (the guided decoding
            # behind `required` stalled for 90 s twice on the free pool); if it answers without calling any
            # tool, the round is repeated once with `required`, so the result cannot simply be made up.
            choice = "required" if (tools_used == 0 and schemas and force_tools) else "auto"
            model = self._model_for(messages)   # switches to the omni model once a screenshot / attachment is in the loop
            await self.send({"type": "trace", "kind": "request", "agent": "executor", "stage": "executor", "turn": self.turns,
                             "round": call_no + round_no - 1, "model": model, "messages": media.redact(messages),
                             "tools": [s["function"]["name"] for s in schemas],
                             "params": dict(self._trace_params(tts, use_memory, source, schemas), tool_choice=choice)})

            async def on_event(kind: str, data: dict) -> None:
                if kind == "delta":
                    await self.send({"type": "executor_delta", "content": data["content"]})
                elif kind == "reasoning":
                    await self.send({"type": "reasoning", "content": data["content"]})
                elif kind == "wait":
                    await self.send({"type": "wait", **data})

            acc: Completion = await self.services.nim.chat_stream(messages, schemas, model=model, tool_choice=choice, on_event=on_event)
            await self.send({"type": "trace", "kind": "response", "agent": "executor", "stage": "executor", "turn": self.turns,
                             "round": call_no + round_no - 1, "content": acc.content, "reasoning": acc.reasoning,
                             "tool_calls": acc.tool_calls, "finish_reason": acc.finish_reason, "usage": acc.usage,
                             "ms": int((time.time() - t0) * 1000)})
            log.info("session %s executor round %d (%s): %d chars, %d tool calls, %.1fs", self.id, round_no, choice,
                     len(acc.content), len(acc.tool_calls), time.time() - t0)
            if acc.finish_reason == "degenerate" and not acc.tool_calls and not degenerate_retry:
                # a repetition loop instead of a tool call or a report: regenerate the round once
                degenerate_retry = True
                log.warning("session %s: executor output degenerated, retrying the round", self.id)
                continue
            if not acc.tool_calls:
                if tools_used == 0 and schemas and not force_tools:
                    force_tools = True
                    log.warning("session %s: executor answered without a tool call, retrying with tool_choice=required", self.id)
                    continue
                leaked = acc.content.lstrip().startswith(("[", "{")) and '"name"' in acc.content
                report = ("" if leaked else acc.content.strip()) or narration.strip() or "(исполнитель не вернул отчёт)"
                if tools_used == 0:
                    report = "(ВНИМАНИЕ: исполнитель не вызвал ни одного инструмента, результат не проверен)\n" + report
                return report, round_no
            tools_used += len(acc.tool_calls)
            calls = []
            for i, tc in enumerate(acc.tool_calls):
                calls.append({"id": tc.get("id") or f"call_{round_no}_{i}_{uuid.uuid4().hex[:6]}", "type": "function",
                              "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"}})
            signature = json.dumps([(c["function"]["name"], _norm_args(c["function"]["arguments"])) for c in calls], ensure_ascii=False)
            if signature in seen_calls:
                log.warning("session %s: executor repeats tool calls, stopping", self.id)
                return (narration.strip() + "\n(исполнитель повторил уже сделанный вызов инструмента и был остановлен; "
                        "результаты выше — всё, что удалось получить)").strip(), round_no
            seen_calls.add(signature)
            if acc.content and acc.content.strip():
                narration = acc.content.strip()
            work.append({"role": "assistant", "content": acc.content.strip() or None, "tool_calls": calls})
            results = await asyncio.gather(*(self._execute_call(ctx, c) for c in calls))
            shown: list[tuple[str, list[dict]]] = []
            for call, result in zip(calls, results):
                if isinstance(result, dict) and result.get("_media"):
                    # a screenshot or an attachment: the tool result stays text, the media follows as a user message
                    shown.append((call["function"]["name"], result.pop("_media")))
                work.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                             "content": compact_result(result)})
                recent_results.append(f"{call['function']['name']}({_norm_args(call['function']['arguments'])[:160]}) -> "
                                      f"{compact_result(result, 500)}")
            recent_results = recent_results[-3:]
            for name, parts in shown:
                work.append({"role": "user", "content": [{"type": "text", "text": f"[содержимое от инструмента {name}]"}, *parts]})
        tail = "\n(достигнут лимит раундов инструментов, задача могла остаться незавершённой; последние результаты инструментов:)\n" + "\n".join(recent_results)
        return (narration.strip() + tail).strip(), settings.MAX_TOOL_ROUNDS

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

        # attachments go straight into the message: images/audio/video as media parts, documents as text
        parts, notes, media_tokens = await media.build_parts(atts)
        user_text = text or ("(see attachments)" if atts else "")
        if atts:
            user_text += "\n\n[attachments: " + "; ".join(notes) + "]"
            log.info("session %s: %d attachment(s) -> %d parts, ~%d tokens", self.id, len(atts), len(parts), media_tokens)
        content: Any = [{"type": "text", "text": user_text}, *parts] if parts else user_text
        self.messages.append({"role": "user", "content": content})
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
                task = "Выполни то, о чём попросил пользователь: " + user_text
                log.info("session %s: promise without a task -> auto task", self.id)
                await self.send({"type": "notice", "message": "Голосовой агент пообещал действие без задачи — задача поставлена автоматически."})

            # ---- 2. executor does the work, 3. dialogue agent reports (at most two hops per turn)
            hops = 0
            while task and hops < 2:
                hops += 1
                if tts:
                    await self.send({"type": "speech_done", "display": display, "final": False})
                await self.send({"type": "task", "task": task})
                report, used = await self._run_executor(task, ctx, use_memory, tools_enabled, tts, source, call_no,
                                                        task_media=[p for p in parts if p.get("type") != "text"] or None)
                call_no += used
                reports.append(report)
                await self.send({"type": "report", "task": task, "report": report})
                # the report comes in as a labelled user message: the model follows that far better than a
                # system note (with a system note it just repeated its "Сейчас проверю")
                self.messages.append({"role": "user", "content": (
                    f"[Результат исполнителя по задаче «{task}»]\n{report}\n\n"
                    "Сообщи пользователю итог своими словами: конкретные факты и цифры из результата; если что-то "
                    "не удалось — что именно. Не повторяй обещание и не ставь новую задачу, если она не нужна.")})
                speech, display, task, _ = await self._dialogue_call(tts, use_memory, source, t_start, "report", call_no)
                call_no += 1
                if not task and looks_like_promise(speech):
                    # still a promise instead of the outcome: one more try, then give up
                    log.warning("session %s: report stage answered with a promise, retrying", self.id)
                    self.messages.append({"role": "assistant", "content": speech})
                    self.messages.append({"role": "user", "content": "Ты снова пообещал проверить, но проверка уже выполнена. "
                                                                     "Скажи результат из сообщения выше."})
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
            import httpx
            detail = f"{type(e).__name__}: {e}".strip(": ")
            if isinstance(e, httpx.TransportError):
                # a VPN switch or a dropped network kills the pooled connection to NVIDIA mid-request
                message = "Связь с NVIDIA оборвалась (сеть или VPN переключились) — повторите сообщение."
            elif not str(e).strip():
                message = f"Ошибка на сервере: {type(e).__name__} (подробности в окне сервера)."
            else:
                message = str(e)[:600]
            await self.send({"type": "error", "message": message, "detail": detail[:600]})
            await self.send({"type": "done", "finish_reason": "error", "ms": int((time.time() - t_start) * 1000)})
            return

        if assistant_text.strip():
            memo = assistant_text
            if reports:
                memo += "\n[исполнитель] " + " | ".join(r[:600] for r in reports)
            meta = {"source": source, "attachments": [a.name for a in atts]}
            images = [a for a in atts if a.is_image]
            if images:
                # turns with pictures go to the VL collection: reachable later by text or by a similar image
                uris = [u for u in (self.services.attachments.image_data_uri(a) for a in images) if u]
                asyncio.create_task(self.services.memory.remember_media(
                    self.id, text or "(вложения)", memo, [a.public() for a in atts], uris))
            else:
                asyncio.create_task(self.services.memory.remember_dialog(self.id, text, memo, meta))

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
        preview = {k: v for k, v in result.items() if k != "_media"} if isinstance(result, dict) else {"result": result}
        serialized = compact_result(preview, 4000)
        payload = preview if len(serialized) < 4000 else {"preview": serialized}
        await self.send({"type": "tool_result", "id": call["id"], "name": name, "ms": ms, "result": payload})
        return result
