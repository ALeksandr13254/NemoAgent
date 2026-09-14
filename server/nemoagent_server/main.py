"""NemoAgent server: FastAPI + one WebSocket per client.

Protocol (JSON text frames):
  client -> server
    {"type":"hello", "token": "...", "client": {os, hostname, user, shell, screen, timezone, tools_enabled}}
    {"type":"user_message", "text": "...", "attachments": ["id", ...], "source": "voice"|"text", "tts": bool, "memory": bool}
        attachments (images, audio, video, documents) are sent to the omni model inside the message
        tts=true: the answer is written in TTS form and streamed as speech_delta events
        memory=true: long-term memory is recalled for this message and the search_memory tool is offered
    {"type":"tool_result", "call_id": "...", "result": {...}}
    {"type":"interrupt"}          # stop the current answer (barge-in)
    {"type":"new_session"}
    {"type":"client_info", "client": {...}}     # update capabilities/toggles
    {"type":"get_prompts"} / {"type":"set_prompts","values":{system,voice_prose,voice_text,executor}} / {"type":"reset_prompts","keys":[...]}
        -> {"type":"prompts","current":{...},"defaults":{...},"overridden":[...]}   (editable system prompt parts)
    {"type":"ping"}
  server -> client
    {"type":"ready", "session_id": "...", "vision": true, "memory": {...}, "model": "..."}
    {"type":"stage", "name": "answer"|"executor"|"report", "agent": "dialogue"|"executor"}
    {"type":"delta", "content": "..."}         {"type":"reasoning", "content": "..."}
    {"type":"task", "task": "..."}             {"type":"executor_delta", "content": "..."}   {"type":"report", "task", "report"}
    {"type":"tool_call", "id","name","arguments"}            # informational (server tools)
    {"type":"client_tool", "call_id","name","arguments"}     # execute on the client, reply with tool_result
    {"type":"tool_result", "id","name","ms","result"}
    {"type":"speech_delta", "content": "..."}  # TTS-ready text (stream it to the TTS)
    {"type":"speech_done", "display": str|null, "final": bool}   # display = screen-only part after ===, if any
    {"type":"trace", "kind":"request", "messages":[...full prompt, media as sizes...], "tools":[...], "params":{...}}
    {"type":"trace", "kind":"response", "content", "reasoning", "tool_calls", "usage", "ms"}
    {"type":"memory", "items":[...]}            {"type":"wait", ...}   {"type":"notice", "message"}
    {"type":"done", "finish_reason", "ms", "first_token_ms"}    {"type":"error", "message"}
Uploads: POST /upload (multipart 'file', header Authorization: Bearer <AGENT_TOKEN>) -> attachment json.
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

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .agent import AgentSession, Services
from .attachments import AttachmentStore
from .config import settings
from .memory import MemoryStore
from .nim import NIMClient
from .prompts import PromptStore

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("server")

services: Optional[Services] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global services
    for p in settings.validate():
        log.warning(p)
    nim = NIMClient()
    memory = MemoryStore(nim)
    attachments = AttachmentStore()
    prompts = PromptStore()
    services = Services(nim=nim, memory=memory, attachments=attachments, prompts=prompts)
    ffmpeg = shutil.which(settings.FFMPEG)
    log.info("NemoAgent server ready on %s:%s | model %s | ffmpeg %s | memory %s",
             settings.HOST, settings.PORT, settings.LLM_MODEL, ffmpeg or "not found (only wav/mp3/mp4 attachments pass as they are)",
             memory.count())
    try:
        yield
    finally:
        await nim.aclose()


app = FastAPI(title="NemoAgent server", lifespan=lifespan)


def _check_token(authorization: Optional[str]) -> None:
    if not settings.AGENT_TOKEN:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != settings.AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _ready(session: AgentSession) -> dict:
    return {"type": "ready", "session_id": session.id, "vision": True, "modalities": ["text", "image", "audio", "video"],
            "memory": services.memory.count(), "model": settings.LLM_MODEL}


@app.get("/health")
async def health():
    return {"ok": True, "model": settings.LLM_MODEL, "vision": True, "ffmpeg": bool(shutil.which(settings.FFMPEG)),
            "memory": services.memory.count() if services else {}, "time": time.time()}


@app.post("/upload")
async def upload(file: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    _check_token(authorization)
    data = await file.read()
    try:
        att = services.attachments.add(file.filename or "file", data, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    return att.public()


@app.get("/memory/recent")
async def memory_recent(limit: int = 20, authorization: Optional[str] = Header(default=None)):
    _check_token(authorization)
    return services.memory.recent(limit)


@app.post("/memory/prune")
async def memory_prune(authorization: Optional[str] = Header(default=None)):
    """Drop trivial/duplicate dialog memories (test chatter like 'Проверка.')."""
    _check_token(authorization)
    return {"removed": services.memory.prune(), "left": services.memory.count()}


@app.post("/memory/clear")
async def memory_clear(authorization: Optional[str] = Header(default=None)):
    """Forget everything (irreversible)."""
    _check_token(authorization)
    return {"removed": services.memory.clear(), "left": services.memory.count()}


class ClientLink:
    """One connected client: sends events, proxies client-side tool calls, owns one AgentSession."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self.session: Optional[AgentSession] = None
        self.client_info: dict = {}

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
                    tts=bool(msg.get("tts")), memory=bool(msg.get("memory"))))
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
            elif t == "client_info":
                link.client_info.update(msg.get("client") or {})
                link.session.client_info = link.client_info
            elif t == "get_prompts":
                await link.send({"type": "prompts", **services.prompts.snapshot()})
            elif t == "set_prompts":
                await link.send({"type": "prompts", "saved": True, **services.prompts.set(msg.get("values") or {})})
            elif t == "reset_prompts":
                await link.send({"type": "prompts", "saved": True, **services.prompts.reset(msg.get("keys"))})
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
