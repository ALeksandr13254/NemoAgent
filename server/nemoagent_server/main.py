"""NemoAgent server: FastAPI + one WebSocket per client. Stateless between connections.

The server owns nothing persistent: it hides the NIM API and runs the two-agent loop for each connected
client, keeping only the live conversation of a session in RAM. Everything that must survive — chat
history, attachments, long-term memory, prompt overrides — lives on the client, which sends what the
current turn needs (recalled memories, prompt overrides, uploads that expire after UPLOAD_TTL_S).

Protocol (JSON text frames):
  client -> server
    {"type":"hello", "token": "...", "client": {os, hostname, user, shell, screen, timezone, tools_enabled,
                                                 persona_gender, models: {dialogue, executor, router},
                                                 prompts: {system, voice_prose, voice_text, executor, router}}}
    {"type":"user_message", "text": "...", "attachments": ["id", ...], "source": "voice"|"text", "tts": bool,
                            "memory": bool, "memory_context": [{kind, ts, score, text}, ...]}
        attachments: ids from POST /upload (kept in RAM for a while); tts=true: the answer is written in
        TTS form and streamed as speech_delta; memory=true: the executor gets the search_memory tool
        (which asks the client back); memory_context: memories the client recalled for this message
    {"type":"tool_result", "call_id": "...", "result": {...}}     # answer to client_tool
    {"type":"interrupt"} · {"type":"new_session"} · {"type":"load_session", "messages":[{role, content, attachment_ids?}...]}
    {"type":"sync_history", "messages":[...]}   # a message of the open chat was edited/deleted: same session, new context
    {"type":"client_info", "client": {...}}     # update capabilities / models / prompts / gender
    {"type":"get_prompts"} / {"type":"set_prompts","values":{...}} / {"type":"reset_prompts","keys":[...]}
        -> {"type":"prompts","current":{...},"defaults":{...},"overrides":{...},"overridden":[...]}
    {"type":"count_tokens", "id", "text", "attachments": [...], "tts": bool}
        -> {"type":"token_count", "id", "ok", "total", "limit", "max_prompt", "over", "room", "system", "history",
            "message", "attachments", "trimmed", "fix", "measuring", ...}: the exact size of the request this draft
            would make (AgentSession.count_request); sent twice when a video clip has to be measured first
    {"type":"count_text", "id", "texts": {key: text}} -> {"type":"text_tokens", "id", "counts": {key: tokens}}
    {"type":"ping"}
  server -> client
    {"type":"ready", "session_id", "model", "models", "vision", "default_model"}
    {"type":"stage", "name": "answer"|"executor"|"report", "agent": "dialogue"|"executor"}
    {"type":"delta", "content"} · {"type":"reasoning", "content"} · {"type":"speech_delta", "content"}
    {"type":"speech_done", "display": str|null, "final": bool}
    {"type":"task", "task"} · {"type":"executor_delta", "content"} · {"type":"report", "task", "report"}
    {"type":"tool_call", "id","name","arguments"} · {"type":"tool_result", "id","name","ms","result"}
    {"type":"client_tool", "call_id","name","arguments"}      # execute on the client (also __screenshot, __search_memory)
    {"type":"trace", "kind":"request"|"response", ...}         # full prompts for the log tab (media as sizes)
    {"type":"wait", ...} · {"type":"notice", "message"} · {"type":"error", "message", "detail"}
    {"type":"done", "finish_reason", "ms", "first_token_ms", "memo": {user, assistant, reports, attachments, source}}
        memo = what the client may store in its long-term memory for this turn
HTTP: GET /health · POST /upload (multipart 'file') · POST /embed {"kind":"text"|"vl","input":[...],"input_type"}
All HTTP calls carry `Authorization: Bearer <AGENT_TOKEN>`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .agent import AgentSession, Services
from .attachments import AttachmentStore
from .config import settings
from .nim import NIMClient
from .tokencount import COUNTER

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("server")

services: Optional[Services] = None


async def _sweeper(attachments: AttachmentStore) -> None:
    while True:
        await asyncio.sleep(60)
        try:
            n = attachments.sweep()
            if n:
                log.info("uploads expired: %d (kept %d, %.1f MB)", n, attachments.count(), attachments.total_bytes() / 1048576)
        except Exception:  # noqa: BLE001
            log.exception("upload sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global services
    for p in settings.validate():
        log.warning(p)
    nim = NIMClient()
    attachments = AttachmentStore()
    services = Services(nim=nim, attachments=attachments)
    ffmpeg = shutil.which(settings.FFMPEG)
    log.info("NemoAgent server ready on %s:%s | model %s | router %s | ffmpeg %s | no disk state",
             settings.HOST, settings.PORT, settings.LLM_MODEL, settings.ROUTER_MODEL,
             ffmpeg or "not found (only wav/mp3/mp4 attachments pass as they are)")
    sweeper = asyncio.create_task(_sweeper(attachments))
    # the token counter's files (fetched once, about 17 MB) load in the background: the server answers meanwhile
    counter = asyncio.create_task(asyncio.to_thread(lambda: COUNTER.ready))
    try:
        yield
    finally:
        sweeper.cancel()
        counter.cancel()
        await nim.aclose()


app = FastAPI(title="NemoAgent server", lifespan=lifespan)


def _check_token(authorization: Optional[str]) -> None:
    if not settings.AGENT_TOKEN:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != settings.AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _ready(session: AgentSession) -> dict:
    vision = session.text_model not in settings.TEXT_ONLY_MODELS
    return {"type": "ready", "session_id": session.id, "vision": vision,
            "modalities": ["text", "image", "audio", "video"] if vision else ["text"],
            "model": session.text_model, "models": session.models, "default_model": settings.LLM_MODEL}


@app.get("/health")
async def health():
    return {"ok": True, "model": settings.LLM_MODEL, "router_model": settings.ROUTER_MODEL,
            "vision": settings.LLM_MODEL not in settings.TEXT_ONLY_MODELS, "ffmpeg": bool(shutil.which(settings.FFMPEG)),
            "token_counter": COUNTER._tok is not None or (COUNTER.error or "loading"),
            "uploads_in_ram": services.attachments.count() if services else 0, "time": time.time()}


@app.post("/upload")
async def upload(file: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    """Keep an attachment in RAM for the next turns; the client keeps the original."""
    _check_token(authorization)
    data = await file.read()
    try:
        att = services.attachments.add(file.filename or "file", data, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    return att.public()


@app.post("/embed")
async def embed(body: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    """Embeddings for the client's long-term memory: 'text' (nemotron-3-embed-1b) or 'vl' (llama-nemotron-embed-vl-1b-v2,
    texts and data:image/... URIs in one space). The server only relays; nothing is kept."""
    _check_token(authorization)
    kind = str(body.get("kind") or "text")
    model = settings.EMBED_VL_MODEL if kind == "vl" else settings.EMBED_TEXT_MODEL
    inputs = [str(x) for x in (body.get("input") or [])][:64]
    if not inputs:
        return {"embeddings": [], "model": model}
    try:
        vecs = await services.nim.embed(model, inputs, str(body.get("input_type") or "passage"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"embedding failed: {str(e)[:200]}")
    return {"embeddings": vecs, "model": model}


class ClientLink:
    """One connected client: sends events, proxies client-side tool calls, owns one AgentSession."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self.session: Optional[AgentSession] = None
        self.client_info: dict = {}
        self._count_next: Optional[dict] = None
        self._count_task: Optional[asyncio.Task] = None

    def request_count(self, msg: dict) -> None:
        """Count requests come with every pause in typing: one runs at a time and only the latest waiting one follows
        (a count of a huge prompt takes a second or two, and a stale one is of no use)."""
        self._count_next = msg
        if self._count_task is None or self._count_task.done():
            self._count_task = asyncio.create_task(self._count_loop())

    async def _count_loop(self) -> None:
        while self._count_next is not None and self.session is not None:
            msg, self._count_next = self._count_next, None
            try:
                res = await self.session.count_request(msg.get("text") or "", msg.get("attachments") or [],
                                                       bool(msg.get("tts")), still_wanted=lambda: self._count_next is None)
            except Exception as e:  # noqa: BLE001
                log.warning("token count failed: %s", e)
                res = {"ok": False, "reason": f"подсчёт не удался: {str(e)[:160]}"}
            await self.send({"type": "token_count", "id": msg.get("id"), **res})

    async def send(self, msg: dict) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass

    async def call_client(self, name: str, args: dict, timeout: float) -> dict:
        call_id = uuid.uuid4().hex[:10]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = fut
        await self.send({"type": "client_tool", "call_id": call_id, "name": name, "arguments": args})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return {"error": f"client did not answer within {int(timeout)}s"}
        finally:
            self._pending.pop(call_id, None)

    def resolve(self, call_id: str, result: dict) -> None:
        fut = self._pending.get(call_id)
        if fut and not fut.done():
            fut.set_result(result if isinstance(result, dict) else {"result": result})

    def cancel_pending(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result({"error": "cancelled"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    link = ClientLink(ws)
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=15)
        hello = json.loads(raw)
    except Exception:
        await ws.close(code=4000)
        return
    if hello.get("type") != "hello" or (settings.AGENT_TOKEN and hello.get("token") != settings.AGENT_TOKEN):
        await link.send({"type": "error", "message": "unauthorized"})
        await ws.close(code=4001)
        return
    link.client_info = hello.get("client") or {}
    link.session = AgentSession(services, link.send, link.call_client, link.client_info)
    log.info("client connected: %s", {k: link.client_info.get(k) for k in ("os", "hostname", "user")})
    await link.send(_ready(link.session))

    turn_task: Optional[asyncio.Task] = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "user_message":
                if turn_task and not turn_task.done():
                    await link.session.interrupt()
                    link.cancel_pending()
                turn_task = asyncio.create_task(link.session.handle_user_message(
                    msg.get("text") or "", msg.get("attachments") or [], msg.get("source") or "text",
                    tts=bool(msg.get("tts")), memory=bool(msg.get("memory")), memory_context=msg.get("memory_context") or []))
            elif t == "tool_result":
                link.resolve(msg.get("call_id", ""), msg.get("result") or {})
            elif t == "interrupt":
                if await link.session.interrupt():
                    link.cancel_pending()
                    await link.send({"type": "notice", "message": "interrupted"})
            elif t == "new_session":
                await link.session.interrupt()
                link.cancel_pending()
                link.session.reset()
                await link.send(_ready(link.session))
            elif t in ("load_session", "sync_history"):
                # load_session: the client reopens a chat from its history (a fresh session for it);
                # sync_history: a message of the open chat was edited or deleted (same session, context rebuilt)
                await link.session.interrupt()
                link.cancel_pending()
                if t == "load_session":
                    link.session.reset()
                await link.session.restore(msg.get("messages") or [])
                log.info("session %s: %s, %d messages", link.session.id,
                         "chat restored" if t == "load_session" else "history synced after an edit", len(link.session.messages))
                if t == "load_session":
                    await link.send(_ready(link.session))
            elif t == "client_info":
                incoming = msg.get("client") or {}
                old_gender = link.client_info.get("persona_gender")
                link.client_info.update(incoming)
                link.session.client_info = link.client_info
                if "prompts" in incoming:
                    link.session.prompts.reset()
                    link.session.prompts.set(incoming.get("prompts") or {})
                if "text_model" in incoming or "models" in incoming:
                    log.info("session %s: models -> %s", link.session.id, {k: v.split("/")[-1] for k, v in link.session.models.items()})
                    await link.send({"type": "model", "model": link.session.text_model, "models": link.session.models,
                                     "vision": link.session.text_model not in settings.TEXT_ONLY_MODELS})
                new_gender = incoming.get("persona_gender")
                if new_gender and old_gender and new_gender != old_gender and link.session.messages:
                    # the voice changed mid-conversation: the history is full of the old gender, so say it out loud
                    link.session.messages.append({"role": "system", "content": (
                        "Голос ассистента переключён на " + ("мужской" if new_gender == "male" else "женский") +
                        ": с этого момента говори о себе в " + ("мужском" if new_gender == "male" else "женском") +
                        " роде, даже если раньше в разговоре было иначе.")})
            elif t == "get_prompts":
                await link.send({"type": "prompts", **link.session.prompts.snapshot()})
            elif t == "set_prompts":
                await link.send({"type": "prompts", "saved": True, **link.session.prompts.set(msg.get("values") or {})})
            elif t == "reset_prompts":
                await link.send({"type": "prompts", "saved": True, **link.session.prompts.reset(msg.get("keys"))})
            elif t == "count_tokens":
                link.request_count(msg)
            elif t == "count_text":
                texts = {str(k): str(v or "") for k, v in (msg.get("texts") or {}).items()}
                try:
                    counts = await asyncio.to_thread(lambda: {k: COUNTER.count_text(v) for k, v in texts.items()})
                    await link.send({"type": "text_tokens", "id": msg.get("id"), "counts": counts})
                except Exception as e:  # noqa: BLE001
                    await link.send({"type": "text_tokens", "id": msg.get("id"), "error": str(e)[:200]})
            elif t == "ping":
                await link.send({"type": "pong", "t": msg.get("t")})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("ws error: %s", e)
    finally:
        if link.session:
            await link.session.interrupt()
        link.cancel_pending()
        log.info("client disconnected")


def run() -> None:
    import uvicorn
    uvicorn.run("nemoagent_server.main:app", host=settings.HOST, port=settings.PORT, log_level="warning",
                ws_max_size=64 * 1024 * 1024, timeout_keep_alive=75)


if __name__ == "__main__":
    run()
