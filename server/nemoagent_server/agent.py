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
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from . import media, tokencount
from .attachments import AttachmentStore
from .config import settings
from .nim import Completion, NIMClient, UpstreamError
from .tokencount import COUNTER, Count
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
    attachments: AttachmentStore      # uploads in RAM; the server keeps nothing on disk


@dataclass
class Trim:
    """The history after the automatic trim, and the exact size of the dialogue request made from it."""
    messages: list
    turns: int = 0                    # oldest user turns dropped
    tokens: int = 0                   # by how much the request shrank
    count: Optional[Count] = None     # None: the model has no exact counter (then sizes are estimates)


ARCHIVE_NOTE = ("Earlier parts of this conversation were archived to long-term memory. "
                "Use search_memory if you need details from them.")
_VIDEO_PROBES: dict[str, asyncio.Future] = {}     # clips being measured right now (one request per clip)


class AgentSession:
    def __init__(self, services: Services, send: SendFn, client_call: ClientCallFn, client_info: Optional[dict] = None):
        self.id = uuid.uuid4().hex[:12]
        # prompt overrides come from the client (hello.client.prompts) and live with this session only
        self.prompts = PromptStore((client_info or {}).get("prompts"))
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
        return {"role": "system", "content": self.prompts.render(
            self._env_description() + "\n" + self._now_line(), tts)}

    def note_screenshot(self, attachment_id: str) -> None:
        self.session_attachment_ids.append(attachment_id)

    # ------------------------------------------------------------- context
    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough size, only for a model without an exact counter (see tokencount.supports)."""
        n = 0
        for m in messages:
            n += media.estimate_tokens(m.get("content"))
            if m.get("tool_calls"):
                n += len(json.dumps(m["tool_calls"], ensure_ascii=False)) / 3.2
        return int(n)

    def _count(self, model: str, messages: list[dict], tools: Optional[list] = None,
               thinking: Optional[bool] = None) -> Optional[Count]:
        """Exact prompt tokens of a request to `model`, or None when there is no counter for it."""
        if not tokencount.supports(model):
            return None
        try:
            return COUNTER.count(messages, tools, settings.LLM_THINKING if thinking is None else thinking)
        except Exception as e:  # noqa: BLE001
            log.warning("session %s: token count failed: %s", self.id, e)
            return None

    def _dialogue_request(self, messages: list[dict], tts: bool) -> tuple[str, list[dict]]:
        """The dialogue agent's model and request for this history, exactly as _dialogue_call sends it: media only in
        the last MEDIA_KEEP_TURNS user messages (older images/audio/video become a text stub, the answer about them
        is already in the history)."""
        return self._for_model([self._system_message(tts)] + media.prune_old_media(messages, settings.MEDIA_KEEP_TURNS))

    def _trimmed(self, messages: list[dict], tts: bool) -> Trim:
        """The history after the automatic trim (nothing is changed here): once the dialogue request is bigger than
        CONTEXT_BUDGET_TOKENS, the oldest turns go (they are already in long-term memory) until it is down to 70 % of
        it, keeping at least CONTEXT_KEEP_TURNS user turns. Sizes are exact for a model with a counter."""
        budget = settings.CONTEXT_BUDGET_TOKENS
        model, req = self._dialogue_request(messages, tts)
        c = self._count(model, req)
        size = c.total if c else self._estimate_tokens(req)
        if size <= budget:
            return Trim(messages, count=c)
        per = c.per_message[1:] if c else [self._estimate_tokens([m]) for m in messages]   # [i] -> messages[i]
        starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        k, dropped = 0, 0
        while size - dropped > budget * 0.7:
            left = len(starts) - k
            if left < 2 or left <= settings.CONTEXT_KEEP_TURNS:
                break
            k += 1
            dropped = sum(per[:starts[k]])
        out = [m for m in messages[starts[k] if k else 0:]
               if not (m.get("role") == "system" and str(m.get("content", "")).startswith("Earlier parts"))]
        out.insert(0, {"role": "system", "content": ARCHIVE_NOTE})
        model, req = self._dialogue_request(out, tts)
        c2 = self._count(model, req)
        return Trim(out, turns=k, tokens=size - (c2.total if c2 else self._estimate_tokens(req)), count=c2)

    async def _trim_context(self, tts: bool = False) -> None:
        """Apply the automatic trim to the live context (see _trimmed)."""
        trim = await asyncio.to_thread(self._trimmed, self.messages, tts)
        if trim.messages is self.messages:
            return
        self.messages = trim.messages
        log.info("session %s: context trimmed: %d oldest turns dropped, request now %s tokens (-%d)", self.id, trim.turns,
                 trim.count.total if trim.count else "~?", trim.tokens)

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

    async def restore(self, history: list) -> None:
        """Rebuild the live context from the client's chat record: a chat reopened from the history, or a message
        edited or deleted in it. Every turn comes back as text; the attachments of the last MEDIA_KEEP_TURNS user
        turns come back as media while this server still holds them."""
        turns = []
        for m in history or []:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                continue
            content = str(m.get("content") or "")
            ids = [str(i) for i in (m.get("attachment_ids") or [])] if m["role"] == "user" else []
            if content.strip() or ids:
                turns.append((m["role"], content, ids))
        turns = turns[-200:]
        users = [i for i, t in enumerate(turns) if t[0] == "user"]
        keep = set(users[-settings.MEDIA_KEEP_TURNS:]) if settings.MEDIA_KEEP_TURNS > 0 else set()
        messages: list[dict] = []
        held: list[str] = []
        last_ids: list[str] = []
        for i, (role, content, ids) in enumerate(turns):
            atts = [a for a in (self.services.attachments.get(x) for x in ids) if a]
            held.extend(a.id for a in atts)
            if role == "user":
                last_ids = [a.id for a in atts]
            if atts and i in keep:
                parts, notes, _ = await media.build_parts(atts)
                text = (content or "(see attachments)") + "\n\n[attachments: " + "; ".join(notes) + "]"
                messages.append({"role": role, "content": [{"type": "text", "text": text}, *parts] if parts else text})
            elif ids:
                messages.append({"role": role, "content": content + f"\n[медиа этого сообщения ({len(ids)} шт.) уже были "
                                                                    "показаны и убраны из контекста]"})
            else:
                messages.append({"role": role, "content": content})
        self.messages = messages
        self.turns = len(users)
        self.session_attachment_ids = held
        self.last_attachment_ids = last_ids
        await self._trim_context()

    # ------------------------------------------------------------- main turn
    async def handle_user_message(self, text: str, attachment_ids: list[str], source: str = "text",
                                  tts: bool = False, memory: bool = False, memory_context: Optional[list] = None) -> None:
        if self.busy():
            await self.interrupt()
        self._task = asyncio.create_task(self._run_turn(text, attachment_ids, source, tts, memory, memory_context or []))
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _user_message(text: str, atts: list, parts: list[dict], notes: list[str]) -> tuple[str, dict]:
        """The user's message as the model gets it: the text, a note on the attachments, their media parts."""
        user_text = text or ("(see attachments)" if atts else "")
        if atts:
            user_text += "\n\n[attachments: " + "; ".join(notes) + "]"
        return user_text, {"role": "user", "content": [{"type": "text", "text": user_text}, *parts] if parts else user_text}

    # ------------------------------------------------------------- size of the next request
    async def count_request(self, text: str, attachment_ids: list[str], tts: bool,
                            still_wanted: Callable[[], bool] = lambda: True) -> dict:
        """Exact size of the request the next message would make: the dialogue agent's request with the system
        prompt, the history after the automatic trim, and this message with its attachments. Recalled memories are
        not in it (the client picks them when the message is sent). A video clip not measured yet is measured with
        one short request (NIMClient.prompt_tokens), then everything is counted again."""
        text = (text or "").strip()
        atts = [a for a in (self.services.attachments.get(x) for x in attachment_ids or []) if a]
        parts, notes = [], []
        if atts:
            parts, notes, _ = await media.build_parts(atts)
        make = (lambda t: self._user_message(t, atts, parts, notes)[1]) if (text or atts) else None
        history = list(self.messages)
        res, unknown = await asyncio.to_thread(self._preview, history, make, text, tts)
        if unknown and still_wanted():
            await self.send({"type": "token_count", **res})
            for part in unknown:
                await self._measure_video(part)
            if still_wanted():
                res, _ = await asyncio.to_thread(self._preview, history, make, text, tts)
        return res

    def _preview(self, history: list[dict], make: Optional[Callable[[str], dict]], text: str,
                 tts: bool) -> tuple[dict, list[dict]]:
        t0 = time.perf_counter()
        draft = make(text) if make else None
        trim = self._trimmed(history + ([draft] if draft else []), tts)
        model, req = self._dialogue_request(trim.messages, tts)
        res: dict = {"model": model, "limit": tokencount.CONTEXT_LIMIT, "max_prompt": tokencount.MAX_PROMPT,
                     "budget": settings.CONTEXT_BUDGET_TOKENS, "keep_turns": settings.CONTEXT_KEEP_TURNS}
        c = trim.count
        if c is None:
            return dict(res, ok=False, reason=(f"точный подсчёт есть только для {tokencount.MODEL.split('/')[-1]}"
                                               if not tokencount.supports(model) else COUNTER.error or "счётчик недоступен")), []
        tail = c.total - sum(c.per_message)                 # the generation prompt after the last message
        system = c.per_message[0]
        message = (c.per_message[-1] + tail) if draft else 0
        hist = c.total - system - message
        last = len(req) - 1
        over = max(0, c.total - tokencount.MAX_PROMPT)
        res.update(ok=True, exact=c.exact, total=c.total, fits=over == 0, over=over,
                   room=max(0, tokencount.CONTEXT_LIMIT - c.total), system=system, history=hist, message=message,
                   history_messages=sum(1 for m in req[1:last if draft else None] if m.get("role") in ("user", "assistant")),
                   attachments=[{k: it[k] for k in ("kind", "tokens", "detail", "exact")} for it in c.items
                                if draft and it["message"] == last],
                   measuring=[it["detail"] for it in c.items if not it["exact"]],
                   trimmed={"turns": trim.turns, "tokens": trim.tokens} if trim.turns else None)
        if over:
            res["fix"] = self._fit_hints(c, req, bool(draft), over, history, make, text, tts)
        res["ms"] = round((time.perf_counter() - t0) * 1000)
        seen, unknown = set(), []
        for m in req:
            for p in (m.get("content") if isinstance(m.get("content"), list) else []):
                if p.get("type") == "video_url":
                    key = tokencount.media_key(p)
                    if key not in tokencount.MEASURED_VIDEO and key not in seen:
                        seen.add(key)
                        unknown.append(p)
        return res, unknown

    def _fit_hints(self, c: Count, req: list[dict], has_draft: bool, over: int, history: list[dict],
                   make: Optional[Callable[[str], dict]], text: str, tts: bool) -> dict:
        """What would make an oversized request fit: the message cut to N characters (checked by counting the cut
        request again), or the oldest K history messages removed."""
        fix: dict = {"tokens": over}
        if has_draft and text and make:
            keep = COUNTER.count_text(text) - over
            for _ in range(4):
                if keep <= 0:
                    break
                chars = COUNTER.prefix_chars(text, keep)
                total = self._trimmed(history + [make(text[:chars])], tts).count.total
                if total <= tokencount.MAX_PROMPT:
                    fix["message_chars"] = chars
                    fix["message_chars_now"] = len(text)
                    break
                keep -= total - tokencount.MAX_PROMPT      # a token merged at the cut: cut that much more
        end = len(req) - 1 if has_draft else len(req)
        freed, count = 0, 0
        for i in range(1, end):
            freed += c.per_message[i]
            count += req[i].get("role") in ("user", "assistant")
            if freed >= over:
                fix["drop_oldest"] = {"messages": count, "tokens": freed}
                break
        return fix

    async def _measure_video(self, part: dict) -> None:
        """One short request per clip (see NIMClient.prompt_tokens); a clip already being measured is waited for."""
        key = tokencount.media_key(part)
        if key in tokencount.MEASURED_VIDEO:
            return
        job = _VIDEO_PROBES.get(key)
        if job is None:
            async def probe() -> None:
                t0 = time.time()
                total = await self.services.nim.prompt_tokens(tokencount.video_probe_messages(part), tokencount.MODEL)
                n = await asyncio.to_thread(tokencount.remember_video, part, total)
                log.info("session %s: video measured: %d tokens (one request, %.1fs)", self.id, n, time.time() - t0)
            job = _VIDEO_PROBES[key] = asyncio.ensure_future(probe())
            job.add_done_callback(lambda _f: _VIDEO_PROBES.pop(key, None))
        try:
            await asyncio.shield(job)
        except Exception as e:  # noqa: BLE001
            log.warning("session %s: video measurement failed: %s", self.id, e)

    def _inject_memories(self, items: list, max_chars: int = 6000) -> None:
        """Memories the client recalled for this message (it owns the long-term memory): a system note for the
        dialogue agent, which has no tools."""
        lines, used = [], 0
        for it in items:
            if not isinstance(it, dict):
                continue
            body = str(it.get("text") or "").strip()
            if not body:
                continue
            if len(body) > 1500:
                body = body[:1500] + " …"
            try:
                when = dt.datetime.fromtimestamp(float(it.get("ts") or 0)).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError):
                when = "?"
            score = it.get("score")
            entry = f"[{when} · {it.get('kind') or 'dialog'}" + (f" · relevance {float(score):.2f}" if isinstance(score, (int, float)) else "") + f"]\n{body}"
            if used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry)
        if not lines:
            return
        self.messages.append({"role": "system", "content": (
            "Воспоминания из прошлых разговоров (фон из ПРОШЛОГО, не текущий запрос; отвечай на последнее "
            "сообщение по существу):\n" + "\n\n".join(lines))})

    def _context_for_executor(self, limit: int = 10) -> list[dict]:
        """Recent user/assistant turns as plain text so the executor understands what the task is about."""
        out = []
        for m in self.messages:
            if m.get("role") in ("user", "assistant"):
                text = media.text_of(m.get("content"))
                if text.strip():
                    out.append({"role": m["role"], "content": text[:2000]})
        return out[-limit:]

    def _check_count(self, agent: str, job: Optional[asyncio.Future], acc: Completion) -> Optional[dict]:
        """The local count of a request next to NVIDIA's prompt_tokens for it. A request whose only unknown was a video
        clip teaches the clip's size: every later request with it is counted exactly."""
        predicted: Optional[Count] = None
        if job is not None:
            try:
                predicted = job.result() if job.done() else None
            except Exception:  # noqa: BLE001
                predicted = None
        actual = (acc.usage or {}).get("prompt_tokens")
        if predicted is None or not actual:
            return None
        actual = int(actual)
        unknown = [it for it in predicted.items if not it["exact"]]
        if len(unknown) == 1 and unknown[0]["kind"] == "video" and unknown[0].get("key"):
            n = tokencount.set_video(unknown[0]["key"], actual - predicted.total)
            log.info("session %s %s: video measured from the answer: %d tokens", self.id, agent, n)
            return {"predicted": predicted.total, "actual": actual, "match": None, "video": n}
        match = predicted.total == actual
        if match:
            log.info("session %s %s: prompt %d tokens, as counted", self.id, agent, actual)
        else:
            log.warning("session %s %s: prompt %d tokens, counted %d (%+d)%s", self.id, agent, actual, predicted.total,
                        actual - predicted.total, "" if predicted.exact else ", count not exact")
        return {"predicted": predicted.total, "actual": actual, "match": match}

    def _count_job(self, model: str, messages: list[dict], tools: Optional[list] = None,
                   thinking: Optional[bool] = None) -> Optional[asyncio.Future]:
        """Count a request in a worker thread while it is on its way (the count never delays the call)."""
        if not tokencount.supports(model):
            return None
        return asyncio.ensure_future(asyncio.to_thread(self._count, model, messages, tools, thinking))

    async def _counted(self, job: Optional[asyncio.Future]) -> None:
        if job is not None:
            try:
                await job
            except Exception:  # noqa: BLE001
                pass

    def _trace_params(self, tts: bool, use_memory: bool, source: str, tools: list, max_tokens: Optional[int] = None) -> dict:
        return {"temperature": settings.LLM_TEMPERATURE, "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                "thinking": settings.LLM_THINKING, "tool_choice": "auto" if tools else None,
                "tts": tts, "memory": use_memory, "source": source}

    ROLES = ("dialogue", "executor", "router")

    def model_for_role(self, role: str) -> str:
        """The client picks a model per role in its settings (client_info["models"]); otherwise the server defaults."""
        models = self.client_info.get("models") or {}
        chosen = str(models.get(role) or "").strip()
        if not chosen and role in ("dialogue", "executor"):
            chosen = str(self.client_info.get("text_model") or "").strip()   # older clients: one text model
        if chosen and chosen not in settings.RETIRED_MODELS:
            return chosen
        if role == "router":
            return settings.ROUTER_MODEL
        return settings.LLM_MODEL

    @property
    def models(self) -> dict:
        return {role: self.model_for_role(role) for role in self.ROLES}

    @property
    def text_model(self) -> str:
        return self.model_for_role("dialogue")

    def _for_model(self, messages: list[dict], role: str = "dialogue") -> tuple[str, list[dict]]:
        """The role's model and the messages in a form it takes. Every role is multimodal by default and gets images,
        audio and video as they are; a text-only model (TEXT_ONLY_MODELS) gets a short text note in their place."""
        model = self.model_for_role(role)
        if model in settings.TEXT_ONLY_MODELS and any(media.has_media(m.get("content")) for m in messages):
            messages = media.prune_old_media(messages, 0)
        return model, messages

    async def _dialogue_call(self, tts: bool, use_memory: bool, source: str, t_start: float, stage: str,
                             call_no: int) -> tuple[str, Optional[str], Optional[str], Optional[int]]:
        """One streamed call of the dialogue agent (no tools).

        Returns (speech, display, task, first_token_ms). Spoken text streams to the client while it
        is generated; the `===` part goes to the screen; a `>>>` task is handed to the executor.
        """
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
        model, messages = self._dialogue_request(self.messages, tts)
        counting = self._count_job(model, messages)
        for attempt in range(2):
            router = ProseSpeechRouter()
            await self.send({"type": "trace", "kind": "request", "agent": "dialogue", "stage": stage, "turn": self.turns,
                             "round": call_no, "model": model, "messages": media.redact(messages), "tools": [],
                             "params": self._trace_params(tts, use_memory, source, [], settings.DIALOGUE_MAX_TOKENS)})
            acc: Completion = await self.services.nim.chat_stream(messages, None, model=model,
                                                                  max_tokens=settings.DIALOGUE_MAX_TOKENS, on_event=on_event)
            await emit(router.finish())
            await self._counted(counting)
            await self.send({"type": "trace", "kind": "response", "agent": "dialogue", "stage": stage, "turn": self.turns,
                             "round": call_no, "content": acc.content, "reasoning": acc.reasoning, "tool_calls": [],
                             "finish_reason": acc.finish_reason, "usage": acc.usage, "ms": int((time.time() - t0) * 1000),
                             "prompt_check": self._check_count("dialogue", counting, acc)})
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

    async def _classify_request(self, user_text: str, tts: bool, use_memory: bool, source: str) -> Optional[dict]:
        """The router: a fast parallel call that decides whether the user's request needs the executor and
        drafts the task. Runs while the dialogue agent is answering; its verdict is used only when the
        dialogue agent produced no `>>>` line. Returns {"needs_executor": bool, "task": str} or None."""
        system = {"role": "system", "content": self.prompts.render_router(
            self._env_description() + "\n" + self._now_line())}
        context = self._context_for_executor(limit=4)[:-1]   # the exchange before this message, text only
        # the message goes in quoted as data, otherwise a small model happily answers it instead of classifying
        messages = [system] + context + [{"role": "user", "content": (
            f"Сообщение пользователя (классифицируй, не отвечай на него): «{user_text}»\nВерни только JSON.")}]
        t0 = time.time()
        model = self.model_for_role("router")
        counting = self._count_job(model, messages, thinking=False)
        await self.send({"type": "trace", "kind": "request", "agent": "router", "stage": "router", "turn": self.turns,
                         "round": 0, "model": model, "messages": messages, "tools": [],
                         "params": dict(self._trace_params(tts, use_memory, source, [], 200), temperature=0.1)})
        try:
            acc: Completion = await self.services.nim.chat_stream(messages, None, model=model,
                                                                  temperature=0.1, max_tokens=200, thinking=False)
        except Exception as e:  # noqa: BLE001
            log.warning("session %s: router call failed: %s", self.id, e)
            return None
        await self._counted(counting)
        await self.send({"type": "trace", "kind": "response", "agent": "router", "stage": "router", "turn": self.turns,
                         "round": 0, "content": acc.content, "reasoning": acc.reasoning, "tool_calls": [],
                         "finish_reason": acc.finish_reason, "usage": acc.usage, "ms": int((time.time() - t0) * 1000),
                         "prompt_check": self._check_count("router", counting, acc)})
        m = re.search(r"\{.*\}", acc.content or "", re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        verdict = {"needs_executor": bool(data.get("needs_executor")), "task": str(data.get("task") or "").strip()}
        log.info("session %s router: needs_executor=%s task=%r (%.1fs)", self.id, verdict["needs_executor"],
                 verdict["task"][:80], time.time() - t0)
        return verdict

    async def _run_executor(self, task: str, ctx: ToolContext, use_memory: bool, tools_enabled: bool,
                            tts: bool, source: str, call_no: int, task_media: Optional[list[dict]] = None) -> tuple[str, int]:
        """The executor agent: tools loop on the task; returns (report, calls_used).

        `task_media` — the attachments of the current user message, so a task about them ("extract the
        table from the photo and save it") starts with the model seeing them.
        """
        await self.send({"type": "stage", "name": "executor", "agent": "executor"})
        schemas = all_schemas(tools_enabled, memory_enabled=use_memory)
        system = {"role": "system", "content": self.prompts.render_executor(
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
            model, messages = self._for_model(messages, "executor")
            counting = self._count_job(model, messages, schemas)
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
            await self._counted(counting)
            await self.send({"type": "trace", "kind": "response", "agent": "executor", "stage": "executor", "turn": self.turns,
                             "round": call_no + round_no - 1, "content": acc.content, "reasoning": acc.reasoning,
                             "tool_calls": acc.tool_calls, "finish_reason": acc.finish_reason, "usage": acc.usage,
                             "ms": int((time.time() - t0) * 1000), "prompt_check": self._check_count("executor", counting, acc)})
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

    async def _run_turn(self, text: str, attachment_ids: list[str], source: str, tts: bool, memory: bool = False,
                        memory_context: Optional[list] = None) -> None:
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
        parts, notes, _ = await media.build_parts(atts)
        user_text, message = self._user_message(text, atts, parts, notes)
        if atts:
            log.info("session %s: %d attachment(s) -> %d parts", self.id, len(atts), len(parts))
        self.messages.append(message)
        use_memory = bool(memory)
        if memory_context:
            self._inject_memories(memory_context)
        self.turns += 1
        await self._trim_context(tts)

        tools_enabled = bool(self.client_info.get("tools_enabled", True))
        ctx = ToolContext(self, self.client_call)
        assistant_text = ""
        reports: list[str] = []
        first_token_ms: Optional[int] = None
        call_no = 1
        # the router classifies the request in parallel with the answer (see _classify_request)
        router_job: Optional[asyncio.Task] = None
        if settings.ROUTER_ENABLED and text:
            router_job = asyncio.create_task(self._classify_request(user_text, tts, use_memory, source))
        try:
            # ---- 1. dialogue agent answers (and may hand a task to the executor)
            speech, display, task, first_token_ms = await self._dialogue_call(tts, use_memory, source, t_start, "answer", call_no)
            call_no += 1
            said = speech + (("\n" + display) if display else "")
            if said.strip():
                self.messages.append({"role": "assistant", "content": said})
                assistant_text = said
            if (not task and router_job is not None and atts and (speech or display or "").strip()
                    and not looks_like_promise(speech)):
                # the dialogue agent saw the attachments itself and answered; the router only read the text of the
                # request, so its verdict does not send the executor to look at the same files again
                router_job.cancel()
                log.info("session %s: attachments answered by the dialogue agent, router verdict not used", self.id)
            elif not task and router_job is not None:
                # no `>>>` from the dialogue agent: the router's verdict on the request itself decides
                try:
                    verdict = await asyncio.wait_for(router_job, timeout=settings.ROUTER_TIMEOUT)
                except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
                    verdict = None
                    log.warning("session %s: router verdict unavailable: %s", self.id, e)
                if verdict and verdict["needs_executor"]:
                    task = verdict["task"] or ("Выполни то, о чём попросил пользователь: " + user_text)
                    log.info("session %s: router says the request needs the executor -> task", self.id)
                    await self.send({"type": "notice", "message": "Запрос требует действия, а голосовой агент задачу не поставил — задача сформулирована маршрутизатором."})
            elif router_job is not None:
                router_job.cancel()
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
            # what the client may keep in its long-term memory for this turn (the server remembers nothing)
            memo = {"user": text, "assistant": assistant_text, "reports": [r[:600] for r in reports], "source": source,
                    "attachments": [a.public() for a in atts]} if assistant_text.strip() else None
            await self.send({"type": "done", "finish_reason": "speak" if tts else "stop",
                             "ms": int((time.time() - t_start) * 1000), "first_token_ms": first_token_ms, "memo": memo})
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
            elif isinstance(e, UpstreamError) and e.status == 451:
                message = ("NVIDIA отклоняет запросы из текущей сети (HTTP 451): похоже, выключен VPN. "
                           "Включите VPN и повторите сообщение.")
            elif not str(e).strip():
                message = f"Ошибка на сервере: {type(e).__name__} (подробности в окне сервера)."
            else:
                message = str(e)[:600]
            await self.send({"type": "error", "message": message, "detail": detail[:600]})
            await self.send({"type": "done", "finish_reason": "error", "ms": int((time.time() - t_start) * 1000)})
            return

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
