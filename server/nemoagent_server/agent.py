"""Agent session: conversation state, streaming tool loop, memory recall and context trimming.

Speech without function calling. When the client wants voice, the model is prompted to write the
answer itself in a TTS-ready form (rules in VOICE_STYLE_PROSE) and the text is streamed to the
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
from .speechfmt import ProseSpeechRouter, strip_filler
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
- When you use tools, keep the final answer focused on the outcome, not the mechanics.
- Every answer must respond to the LATEST user message. Never repeat your previous answer verbatim.
- Be a good conversation partner, not a vending machine. Small talk ("как дела", "чем хочешь заняться", jokes, opinions) gets a real, friendly answer of one to three sentences — say how you are, suggest something, ask back. Never answer a question with a bare "Хорошо" or "Привет".
- If a message is garbled, cut off (speech recognition drops words) or clearly not addressed to you, say briefly that you did not catch it and ask to repeat ("Не расслышала, повторите?") — do not greet or acknowledge as if it made sense.
- Use web_search only for facts that may have changed recently or that you do not know (news, prices, today's events); general knowledge, stories and explanations come from you directly. At most one web_search per answer.
- Do not end answers with "чем могу помочь" or similar filler; just answer."""

VOICE_STYLE_TEXT = ("The user typed the message and the answer is shown as text only: answer concisely; light markdown "
                    "(short lists, `code`) is fine when it helps.")

# Rules derived from the TeraTTSv2 model card + its character table (unicode_indexer.json):
# vocabulary = letters, space, . , ! ? : ; - ( ) « » " ' ; digits expanded only in the nominative;
# % ° № — … / \ _ * # @ & = + < > [ ] { } are dropped; abbreviations are read letter by letter.
SPEECH_RULES = """  1. Words only. Allowed characters: letters, spaces and the punctuation . , ! ? : ; - ( ) « » " '. No digits, no symbols (% ° № $ € / \\ _ * # @ & = + < > [ ] ~ |), no emoji, no markdown.
  1a. NEVER put code, shell commands, file paths, URLs, e-mails or identifiers into the spoken text — the engine cannot pronounce them. Describe them in words ("команда из трёх частей: получить процессы, отсортировать по памяти, взять первые пять").
  2. Write every number in words, in the grammatically correct form: "двадцать четыре целых девять десятых гигабайта", "пятнадцать ноль две", "минус три градуса", "восемьдесят процентов", "в две тысячи двадцать шестом году".
  3. Expand abbreviations and units into full words ("гигабайт", "операционная система", "компьютер", "километров в час"); if an abbreviation is pronounced letter by letter, write the letter names ("эс-ша-а", "ю-эс-би").
  4. In Russian speech write foreign names, brands and products in Cyrillic transliteration ("Виндоус", "Гитхаб", "Пайтон", "Ютуб", "Визуал Студио Код"). Do not mix Latin and Cyrillic inside one sentence. If the whole answer is in English, write it in English.
  5. Use the letter ё where it belongs (всё, ещё, идёт). Stress is placed automatically; only for an ambiguous homograph put + right before the stressed vowel (з+амок on a door, зам+ок on a hill).
  6. Speak like a person: natural sentences (up to about twenty words each), no lists, no headings, no tables. Put pauses with commas and full stops.
  7. Answer fully but without padding: a factual question gets the fact, a story or explanation gets a few lively sentences, small talk gets a warm reply. No closing offers or questions: never "Чем могу помочь?", "Если нужно что-то ещё, скажите", "Обращайтесь", "How can I help", "Let me know" — the user will ask if they want more."""

VOICE_STYLE_PROSE = """The user is talking by voice and your reply is read aloud by a text-to-speech engine with a tiny vocabulary. Write the reply itself as spoken text, following these rules strictly:
""" + SPEECH_RULES + """
If the answer needs code, a shell command, a file path, a link, exact figures or a table, first say it in words, then add a line containing only === and put the exact text below it (it is shown on screen, not spoken; markdown is fine there). Example:
  На диске Це свободно двадцать четыре целых девять десятых гигабайта. Команда на экране.
  ===
  Диск C: свободно 24,9 ГБ из 1765,3 ГБ
  ```powershell
  Get-PSDrive C
  ```
Before a long tool action you may write one short sentence about what you are doing ("Сейчас проверю."), then call the tools."""


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
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            env=self._env_description() + "\n" + self._now_line(),
            voice_style=VOICE_STYLE_PROSE if tts else VOICE_STYLE_TEXT)}

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
        """Automatic memory recall, presented as a search_memory tool round (an observation, not a transcript)."""
        try:
            recalled = await asyncio.wait_for(self.services.memory.search(text, exclude_session=self.id), timeout=6.0)
        except Exception as e:  # noqa: BLE001
            log.warning("memory recall failed: %s", e)
            return
        if not recalled:
            return
        call_id = f"mem_{uuid.uuid4().hex[:8]}"
        results = [{"when": dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M"), "kind": r["kind"],
                    "score": round(r["score"], 3), "text": r["text"][:2500]} for r in recalled]
        self.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "search_memory", "arguments": json.dumps({"query": text[:200]}, ensure_ascii=False)}}]})
        self.messages.append({"role": "tool", "tool_call_id": call_id, "name": "search_memory", "content": json.dumps({
            "note": ("PAST conversations (earlier sessions) — background context only, not the current request; "
                     "answer the latest message on its own merits."),
            "count": len(results), "results": results}, ensure_ascii=False)})
        await self.send({"type": "memory", "items": [{"kind": r["kind"], "score": round(r["score"], 2),
                                                      "text": r["text"][:300], "ts": r["ts"]} for r in recalled]})

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
        schemas = all_schemas(tools_enabled, self.services.vision.enabled, memory_enabled=use_memory)
        ctx = ToolContext(self, self.client_call)
        assistant_text = ""       # what goes to memory
        first_token_ms: Optional[int] = None
        finish = "stop"
        last_call_signature: Optional[str] = None
        try:
            for round_no in range(1, settings.MAX_TOOL_ROUNDS + 1):
                await self.send({"type": "round", "round": round_no})
                messages = [self._system_message(tts)] + self.messages
                round_t0 = time.time()
                prose = ProseSpeechRouter() if tts else None
                spoke_this_round = False

                async def on_event(kind: str, data: dict) -> None:
                    nonlocal first_token_ms, spoke_this_round
                    if kind == "delta":
                        if first_token_ms is None:
                            first_token_ms = int((time.time() - t_start) * 1000)
                        if prose is None:
                            await self.send({"type": "delta", "content": data["content"]})
                            return
                        for what, piece in prose.feed(data["content"]):
                            if what == "speech":
                                spoke_this_round = True
                                await self.send({"type": "speech_delta", "content": piece})
                            else:
                                await self.send({"type": "delta", "content": piece})
                    elif kind == "reasoning":
                        await self.send({"type": "reasoning", "content": data["content"]})
                    elif kind == "wait":
                        await self.send({"type": "wait", **data})

                if log.isEnabledFor(logging.DEBUG):
                    log.debug("session %s round %d messages:\n%s", self.id, round_no,
                              json.dumps(messages[1:], ensure_ascii=False, indent=1)[-6000:])
                # full trace for the client's log tab: exactly what goes to the model
                await self.send({"type": "trace", "kind": "request", "turn": self.turns, "round": round_no,
                                 "model": settings.LLM_MODEL, "messages": messages,
                                 "tools": [s["function"]["name"] for s in schemas],
                                 "params": {"temperature": settings.LLM_TEMPERATURE, "max_tokens": settings.LLM_MAX_TOKENS,
                                            "thinking": settings.LLM_THINKING, "tool_choice": "auto" if schemas else None,
                                            "tts": tts, "memory": use_memory, "source": source}})
                acc: Completion = await self.services.nim.chat_stream(messages, schemas, on_event=on_event)
                log.info("session %s round %d: %d chars, %d tool calls, %.1fs", self.id, round_no,
                         len(acc.content), len(acc.tool_calls), time.time() - round_t0)
                await self.send({"type": "trace", "kind": "response", "turn": self.turns, "round": round_no,
                                 "content": acc.content, "reasoning": acc.reasoning, "tool_calls": acc.tool_calls,
                                 "finish_reason": acc.finish_reason, "usage": acc.usage,
                                 "ms": int((time.time() - round_t0) * 1000)})
                if acc.finish_reason == "length":
                    await self.send({"type": "notice", "message": "Ответ обрезан по лимиту max_tokens."})

                if prose is not None:
                    for what, piece in prose.finish():
                        spoke_this_round = spoke_this_round or what == "speech"
                        await self.send({"type": "speech_delta" if what == "speech" else "delta", "content": piece})

                # ---------------- final answer (no tool calls)
                if not acc.tool_calls:
                    if prose is not None:
                        speech, display = prose.result
                        spoken_raw = acc.content.split("\n===")[0].strip()
                        if spoken_raw != speech.strip():
                            log.info("session %s: filler stripped: %r -> %r", self.id, spoken_raw[-120:], speech[-120:])
                        await self.send({"type": "speech_done", "display": display, "final": True})
                        assistant_text = speech + (("\n" + display) if display else "")
                        finish = "speak"
                    else:
                        assistant_text = acc.content.strip()
                    self.messages.append({"role": "assistant", "content": assistant_text})
                    break

                # ---------------- tool calls
                calls = []
                for i, tc in enumerate(acc.tool_calls):
                    calls.append({
                        "id": tc.get("id") or f"call_{round_no}_{i}_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"},
                    })
                signature = json.dumps([(c["function"]["name"], c["function"]["arguments"]) for c in calls], ensure_ascii=False)
                if signature == last_call_signature:
                    log.warning("session %s: repeated tool calls, breaking the loop", self.id)
                    await self.send({"type": "notice", "message": "Модель повторяет одни и те же вызовы, останавливаю."})
                    self.messages.append({"role": "assistant", "content": assistant_text.strip() or "(stopped: repeated tool calls)"})
                    finish = "loop"
                    break
                last_call_signature = signature

                narration = acc.content.strip() if acc.content else ""
                if prose is not None:
                    narration = strip_filler(narration) if narration else ""
                    if spoke_this_round:   # "Сейчас проверю." was spoken; let the player finish this bit
                        await self.send({"type": "speech_done", "display": None, "final": False})
                self.messages.append({"role": "assistant", "content": narration or None, "tool_calls": calls})
                if narration:
                    assistant_text += narration + "\n"

                results = await asyncio.gather(*(self._execute_call(ctx, c) for c in calls))
                for call, result in zip(calls, results):
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "name": call["function"]["name"], "content": compact_result(result)})
            else:
                await self.send({"type": "notice", "message": f"Достигнут лимит {settings.MAX_TOOL_ROUNDS} раундов вызова инструментов."})
                self.messages.append({"role": "assistant", "content": assistant_text or "(tool round limit reached)"})

            await self.send({"type": "done", "finish_reason": finish, "ms": int((time.time() - t_start) * 1000),
                             "first_token_ms": first_token_ms})
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
