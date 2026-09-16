"""Client orchestrator: server link, local UI hub, microphone -> STT -> server, server -> TTS/tools."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Optional

import httpx
import numpy as np
import websockets

from . import executor
from pathlib import Path

from .config import settings

log = logging.getLogger("core")

SAFE_TOOLS = {"__screenshot", "system_info", "list_windows", "read_file", "list_directory"}


class ClientCore:
    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ui_clients: set = set()
        self.server_ws = None
        self.connected = False
        self.session_id: Optional[str] = None
        self.server_info: dict = {}
        self.stt = None
        self.tts = None
        self.player = None
        self.speaker = None
        self.mic = None
        self.status = {"stt": "off" if not settings.STT_ENABLED else "loading",
                       "tts": "off" if not settings.TTS_ENABLED else "loading", "mic": "loading"}
        self.state = {
            "tts_mode": "always" if settings.TTS_ENABLED else "off",   # always | voice | off
            "auto_listen": settings.AUTO_LISTEN,
            "barge_in": settings.BARGE_IN,
            "tools_enabled": settings.TOOLS_ENABLED,
            "confirm": settings.TOOL_CONFIRM,
            "tts_language": settings.TTS_LANGUAGE,   # auto | ru | en: voice for Latin words and numbers
            "text_model": settings.TEXT_MODEL,       # text model asked from the server ("" = server default)
            "voice_ru": settings.TTS_VOICE_RU,
            "voice_en": settings.TTS_VOICE_EN,
            "tts_speed": settings.TTS_SPEED,
            "stt_language": settings.STT_LANGUAGE,
            "speaker_device": settings.SPEAKER_DEVICE or "",
            "mic_device": settings.MIC_DEVICE or "",
            "memory_recall": False,     # 🗂 button: recall long-term memory for the messages while it is on
        }
        # what the user picks in the settings panel survives a restart (client/data/settings.json)
        self._settings_path = Path(__file__).resolve().parent.parent / "data" / "settings.json"
        self._load_state()
        self._sync_runtime_settings()
        self.current_source = "text"
        self.turn_t0 = 0.0
        self.assistant_buffer = ""
        self.turn_active = False
        self.speech_mode = False        # the model answered through the `speak` tool this turn
        self.dictation = False          # read-aloud tab: recognised speech goes into its text, not to the agent
        self._dictate_once = False      # one push-to-talk phrase for the read-aloud tab
        self._dictation_forced_listen = False
        self._reader: Optional[dict] = None   # active read-aloud job: {"gen", "index", "total"}
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._last_level_sent = 0.0
        self._http = httpx.AsyncClient(timeout=120)

    # ============================================================== settings persistence
    PERSIST_KEYS = ("tts_mode", "auto_listen", "barge_in", "tools_enabled", "confirm", "voice_ru", "voice_en", "tts_speed",
                    "stt_language", "tts_language", "text_model", "speaker_device", "mic_device")

    def _load_state(self) -> None:
        try:
            if self._settings_path.exists():
                saved = json.loads(self._settings_path.read_text("utf-8"))
                for k in self.PERSIST_KEYS:
                    if k in saved:
                        self.state[k] = saved[k]
                log.info("settings restored from %s", self._settings_path)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot read %s: %s", self._settings_path, e)

    def _save_state(self) -> None:
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(json.dumps({k: self.state[k] for k in self.PERSIST_KEYS if k in self.state},
                                                      ensure_ascii=False, indent=1), "utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("cannot save %s: %s", self._settings_path, e)

    def _sync_runtime_settings(self) -> None:
        """Push the settings state into the module-level `settings` the audio code reads."""
        settings.TTS_VOICE_RU = self.state["voice_ru"]
        settings.TTS_VOICE_EN = self.state["voice_en"]
        try:
            settings.TTS_SPEED = float(self.state["tts_speed"])
        except (TypeError, ValueError):
            pass
        settings.STT_LANGUAGE = self.state["stt_language"] or "auto"
        settings.TTS_LANGUAGE = self.state.get("tts_language") if self.state.get("tts_language") in ("ru", "en") else "auto"
        settings.BARGE_IN = bool(self.state["barge_in"])
        settings.SPEAKER_DEVICE = str(self.state.get("speaker_device") or "") or None
        settings.MIC_DEVICE = str(self.state.get("mic_device") or "") or None

    # ============================================================== lifecycle
    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        from .local_ui import make_app
        import uvicorn
        config = uvicorn.Config(make_app(self), host=settings.UI_HOST, port=settings.UI_PORT, log_level="warning",
                                ws_max_size=64 * 1024 * 1024)
        server = uvicorn.Server(config)
        ui_task = asyncio.create_task(server.serve())
        log.info("UI: http://%s:%s", settings.UI_HOST, settings.UI_PORT)
        if settings.OPEN_BROWSER:
            import webbrowser
            self.loop.call_later(1.0, lambda: webbrowser.open(f"http://{settings.UI_HOST}:{settings.UI_PORT}"))
        threading.Thread(target=self._load_tts, daemon=True, name="load-tts").start()
        threading.Thread(target=self._load_stt, daemon=True, name="load-stt").start()
        try:
            await self._server_loop()
        finally:
            ui_task.cancel()

    # ------------------------------------------------------------ model loading
    def _post(self, coro_fn, *args) -> None:
        """Schedule a coroutine on the loop from a worker thread."""
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro_fn(*args)))

    def _load_tts(self) -> None:
        if not settings.TTS_ENABLED:
            return
        try:
            from .player import StreamPlayer
            from .speech import Speaker
            from .tts import TeraTTS
            self.player = StreamPlayer(44100, device=settings.SPEAKER_DEVICE or None)
            self.tts = TeraTTS()
            self.speaker = Speaker(self.tts, self.player, on_state=lambda d: self._post(self._on_speaker_state, d))
            self.status["tts"] = f"ready ({'GPU' if 'CUDA' in self.tts.provider else 'CPU'})"
        except Exception as e:  # noqa: BLE001
            log.exception("TTS load failed")
            self.status["tts"] = f"error: {e}"[:120]
            self.state["tts_mode"] = "off"
        self._post(self.broadcast_status)

    def _load_stt(self) -> None:
        if settings.STT_ENABLED:
            try:
                from .stt import SpeechToText
                self.stt = SpeechToText()
                self.status["stt"] = f"ready ({'GPU' if self.stt.device == 'cuda' else 'CPU'})"
            except Exception as e:  # noqa: BLE001
                log.exception("STT load failed")
                self.status["stt"] = f"error: {e}"[:120]
        self._post(self.broadcast_status)
        try:
            from .audio_in import Microphone
            self.mic = Microphone(on_utterance=self._on_utterance, on_speech_start=self._on_speech_start,
                                  is_agent_speaking=lambda: bool(self.speaker and self.speaker.is_speaking),
                                  on_level=self._on_level, device=settings.MIC_DEVICE, on_health=self._on_mic_health)
            self.mic.set_enabled(self.state["auto_listen"] and self.stt is not None)
            self.status["mic"] = "ready"
        except Exception as e:  # noqa: BLE001
            log.exception("microphone failed")
            self.status["mic"] = f"error: {e}"[:120]
        self._post(self.broadcast_status)

    # ============================================================== server link
    async def _server_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(settings.ws_url(), max_size=64 * 1024 * 1024, ping_interval=20,
                                              ping_timeout=20, open_timeout=10) as ws:
                    self.server_ws = ws
                    await ws.send(json.dumps({"type": "hello", "token": settings.AGENT_TOKEN,
                                              "client": self._client_info()}, ensure_ascii=False))
                    self.connected = True
                    backoff = 1.0
                    await self.broadcast_status()
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        await self._on_server_message(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("server connection: %s", str(e)[:200])
            self.connected = False
            self.server_ws = None
            if self.turn_active:
                self.turn_active = False
                await self.broadcast({"type": "done", "finish_reason": "disconnected"})
            await self.broadcast_status()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.7, 15.0)

    def _client_info(self) -> dict:
        info = executor.client_description()
        info["tools_enabled"] = self.state["tools_enabled"]
        if self.state.get("text_model"):
            info["text_model"] = self.state["text_model"]
        return info

    async def send_server(self, msg: dict) -> bool:
        ws = self.server_ws
        if not ws:
            await self.broadcast({"type": "error", "message": "нет соединения с сервером"})
            return False
        try:
            await ws.send(json.dumps(msg, ensure_ascii=False))
            return True
        except Exception as e:  # noqa: BLE001
            await self.broadcast({"type": "error", "message": f"send failed: {e}"})
            return False

    async def _on_server_message(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "ready":
            self.session_id = msg.get("session_id")
            self.server_info = {k: msg.get(k) for k in ("vision", "vision_error", "memory", "model", "media_model", "default_model")}
            await self.broadcast_status()
            return
        if t == "model":   # the server confirmed the text model chosen in the settings
            self.server_info["model"] = msg.get("model")
            await self.broadcast_status()
            return
        if t == "delta":
            content = msg.get("content") or ""
            if not self.assistant_buffer:
                content = content.lstrip()      # models like to start with blank lines
                if not content:
                    return
            self.assistant_buffer += content
            # Plain content is display-only: speech arrives separately as speech_delta (from the
            # `speak` tool or the server-side rewrite). Only if the server sends none at all do we
            # fall back to speaking the raw text at `done`.
            await self.broadcast(msg)
            return
        if t == "speech_delta":
            piece = msg.get("content") or ""
            if not self.speech_mode:
                self.speech_mode = True
                if self.speaker:
                    self.speaker.begin()        # drops any fallback speech started from plain content
            if self.speaker and self.state["tts_mode"] != "off":
                self.speaker.feed(piece, prepared=True)
            await self.broadcast(msg)
            return
        if t == "speech_done":
            if self.speaker:
                self.speaker.end()
            await self.broadcast(msg)
            return
        if t == "done":
            self.turn_active = False
            if self.speaker and self._speak_enabled() and not self.speech_mode and self.assistant_buffer.strip() \
                    and msg.get("finish_reason") in ("stop", None):
                self.speaker.say(self.assistant_buffer)     # last-resort fallback: raw text, cleaned locally
            msg = dict(msg, total_ms=int((time.time() - self.turn_t0) * 1000) if self.turn_t0 else None)
            await self.broadcast(msg)
            return
        if t == "client_tool":
            asyncio.ensure_future(self._handle_client_tool(msg.get("call_id"), msg.get("name"), msg.get("arguments") or {}))
            return
        await self.broadcast(msg)

    def _speak_enabled(self) -> bool:
        mode = self.state["tts_mode"]
        return bool(self.speaker) and (mode == "always" or (mode == "voice" and self.current_source == "voice"))

    # ============================================================== tools
    async def _handle_client_tool(self, call_id: str, name: str, args: dict) -> None:
        t0 = time.time()
        if name == "__screenshot":
            result = await asyncio.to_thread(executor.execute, name, args)
        elif not self.state["tools_enabled"]:
            result = {"error": "computer-control tools are disabled in the client settings"}
        else:
            mode = self.state["confirm"]
            need = (mode == "always" and name not in SAFE_TOOLS) or (mode == "dangerous" and executor.is_dangerous(name, args))
            approved = True
            if need:
                approved = await self._ask_confirm(name, args)
            if not approved:
                result = {"error": "the user rejected this action"}
            else:
                await self.broadcast({"type": "client_tool_start", "name": name, "summary": executor.summarize(name, args)})
                result = await asyncio.to_thread(executor.execute, name, args)
        if name != "__screenshot":
            await self.broadcast({"type": "client_tool_done", "name": name, "ms": int((time.time() - t0) * 1000),
                                  "ok": "error" not in result})
        await self.send_server({"type": "tool_result", "call_id": call_id, "result": result})

    async def _ask_confirm(self, name: str, args: dict) -> bool:
        cid = uuid.uuid4().hex[:8]
        fut: asyncio.Future = self.loop.create_future()
        self._pending_confirms[cid] = fut
        await self.broadcast({"type": "confirm", "id": cid, "name": name, "summary": executor.summarize(name, args),
                              "timeout": settings.TOOL_CONFIRM_TIMEOUT})
        if self.speaker and self.state["tts_mode"] != "off":
            self.speaker.say("Нужно подтверждение действия." if self.current_source == "voice" else "")
        try:
            return bool(await asyncio.wait_for(fut, timeout=settings.TOOL_CONFIRM_TIMEOUT))
        except asyncio.TimeoutError:
            await self.broadcast({"type": "confirm_expired", "id": cid})
            return False
        finally:
            self._pending_confirms.pop(cid, None)

    # ============================================================== user input
    async def send_user_message(self, text: str, attachments: list[str], source: str) -> None:
        text = (text or "").strip()
        if not text and not attachments:
            return
        self.current_source = source
        self.turn_t0 = time.time()
        self.assistant_buffer = ""
        self.turn_active = True
        self.speech_mode = False
        if self.speaker:
            self.speaker.cancel()
        tts = self._speak_enabled()
        memory = bool(self.state.get("memory_recall"))
        await self.broadcast({"type": "user_message", "text": text, "attachments": attachments, "source": source, "memory": memory})
        ok = await self.send_server({"type": "user_message", "text": text, "attachments": attachments,
                                     "source": source, "tts": tts, "memory": memory})
        if not ok:
            self.turn_active = False
            await self.broadcast({"type": "done", "finish_reason": "error"})

    async def interrupt(self, reason: str = "user") -> None:
        if self.speaker:
            self.speaker.cancel()
        if self.turn_active:
            await self.send_server({"type": "interrupt"})
        await self.broadcast({"type": "interrupted", "reason": reason})

    async def upload_attachment(self, filename: str, data: bytes, mime: Optional[str]) -> dict:
        url = settings.http_url() + "/upload"
        try:
            r = await self._http.post(url, files={"file": (filename, data, mime or "application/octet-stream")},
                                      headers={"Authorization": f"Bearer {settings.AGENT_TOKEN}"})
            if r.status_code != 200:
                return {"error": f"upload failed: {r.status_code} {r.text[:200]}"}
            return r.json()
        except Exception as e:  # noqa: BLE001
            return {"error": f"upload failed: {e}"}

    async def take_screenshot_attachment(self, monitor: Optional[int] = None) -> dict:
        shot = await asyncio.to_thread(executor.execute, "__screenshot", {"monitor": monitor})
        if shot.get("error"):
            return shot
        import base64
        data = base64.b64decode(shot["png_base64"])
        return await self.upload_attachment(f"screenshot_{time.strftime('%H%M%S')}.png", data, "image/png")

    # ------------------------------------------------------------ microphone callbacks (threads)
    def _on_level(self, level: float, speech: bool) -> None:
        now = time.time()
        if now - self._last_level_sent >= 0.1:
            self._last_level_sent = now
            self._post(self.broadcast, {"type": "mic", "level": round(min(1.0, level * 8), 3), "speech": speech})

    def _on_mic_health(self, text: str) -> None:
        # from the VAD thread: "ready" or a problem description, shown on the STT pill / settings
        self.status["mic"] = text
        self._post(self.broadcast_status)

    def _on_speech_start(self) -> None:
        # the VAD heard speech while the agent talks: stop talking right away (barge-in)
        if self.state["barge_in"]:
            self._post(self._barge_in)

    async def _barge_in(self) -> None:
        if self.speaker and self.speaker.is_speaking:
            log.info("barge-in: user started speaking")
            await self.interrupt("barge-in")

    def _on_utterance(self, audio: np.ndarray, duration: float, during_speech: bool = False) -> None:
        self._post(self._process_utterance, audio, duration, during_speech)

    async def _process_utterance(self, audio: np.ndarray, duration: float, during_speech: bool = False) -> None:
        if not self.stt:
            return
        if during_speech and not self.state["barge_in"]:
            return
        await self.broadcast({"type": "stt", "state": "transcribing", "duration": round(duration, 2)})
        t0 = time.time()
        try:
            text = await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as e:  # noqa: BLE001
            await self.broadcast({"type": "stt", "state": "error", "message": str(e)[:200]})
            return
        ms = int((time.time() - t0) * 1000)
        if not text:
            await self.broadcast({"type": "stt", "state": "empty", "ms": ms})
            return
        await self.broadcast({"type": "stt", "state": "done", "text": text, "ms": ms})
        if self.dictation or self._dictate_once:
            # read-aloud tab is dictating: the phrase goes into its text box instead of the conversation
            self._dictate_once = False
            await self.broadcast({"type": "dictation", "text": text, "ms": ms})
            return
        await self.send_user_message(text, [], "voice")

    # ============================================================== read-aloud tab
    async def _on_speaker_state(self, d: dict) -> None:
        """Events from the TTS worker: metrics go to the UI as they are; sentence progress drives the read-aloud tab."""
        kind = d.get("type")
        job = self._reader
        if kind == "speaking":
            if job and d.get("gen") == job["gen"]:
                job["index"] += 1
                await self.broadcast({"type": "read", "state": "sentence", "index": job["index"], "total": max(job["total"], job["index"]),
                                      "text": d.get("text", "")})
            return
        if kind == "speech_drained":
            if job and d.get("gen") == job["gen"]:
                # everything is synthesised; wait for the player to finish the tail before saying "done"
                for _ in range(300):
                    if not (self.player and self.player.is_playing) or self._reader is not job:
                        break
                    await asyncio.sleep(0.2)
                if self._reader is job:
                    self._reader = None
                    await self.broadcast({"type": "read", "state": "done"})
            return
        if kind == "speech_cancelled":
            if job and d.get("gen") == job["gen"]:
                self._reader = None
                await self.broadcast({"type": "read", "state": "stopped"})
            return
        await self.broadcast(d)

    async def _set_dictation(self, on: bool) -> None:
        on = bool(on) and self.stt is not None and self.mic is not None
        if on and self.speaker and self.speaker.is_speaking:
            self.speaker.cancel()               # dictating into a text that is being read makes no sense
        if on and not self.dictation:
            self._dictation_forced_listen = not bool(self.state["auto_listen"])
            self.mic.set_enabled(True)
        elif not on and self.dictation and self._dictation_forced_listen and self.mic:
            self.mic.set_enabled(bool(self.state["auto_listen"]) and self.stt is not None)
            self._dictation_forced_listen = False
        self.dictation = on
        await self.broadcast({"type": "read", "state": "dictation", "dictation": on,
                              **({} if on or self.stt else {"message": "распознавание речи не загружено"})})
        await self.broadcast_status()

    async def _start_reading(self, text: str) -> None:
        text = (text or "").strip()
        if not self.speaker:
            await self.broadcast({"type": "read", "state": "error", "message": "синтез речи не загружен"})
            return
        if not text:
            return
        if self.dictation:
            await self._set_dictation(False)
        from .tts import SentenceSplitter
        sp = SentenceSplitter()
        total = len(list(sp.feed(text)) + list(sp.finish()))
        gen = self.speaker.begin()
        self._reader = {"gen": gen, "index": 0, "total": total}
        await self.broadcast({"type": "read", "state": "start", "total": total, "chars": len(text)})
        log.info("read aloud: %d chars, %d sentences", len(text), total)
        self.speaker.feed(text)
        self.speaker.end()

    async def _stop_reading(self) -> None:
        if self.speaker:
            self.speaker.cancel()               # emits speech_cancelled -> "stopped" for the UI
        if self._reader:
            self._reader = None
            await self.broadcast({"type": "read", "state": "stopped"})

    # ============================================================== UI hub
    async def broadcast(self, msg: dict) -> None:
        if not self.ui_clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        dead = []
        for ws in list(self.ui_clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ui_clients.discard(ws)

    def status_payload(self) -> dict:
        devices: list[dict] = []
        inputs: list[dict] = []
        try:
            from .player import list_input_devices, list_output_devices
            devices = list_output_devices()
            inputs = list_input_devices()
        except Exception:  # noqa: BLE001
            pass
        return {"type": "status", "server": self.connected, "session_id": self.session_id, "server_info": self.server_info,
                "stt": self.status["stt"], "tts": self.status["tts"], "mic": self.status["mic"],
                "speaker": getattr(self.player, "device_name", None), "output_devices": devices,
                "microphone": getattr(self.mic, "device_name", None), "input_devices": inputs,
                "listening": bool(self.mic and self.mic.enabled), "dictation": self.dictation, "settings": self.state,
                "voices": {"ru": ["ru_f1", "ru_m5", "ru_f2", "ru_m1"],
                           "en": ["eng_f3", "eng_f5", "eng_m3", "eng_m4", "eng_f4_whisper", "eng_m2_whisper"]}}

    async def broadcast_status(self) -> None:
        await self.broadcast(self.status_payload())

    async def on_ui_message(self, ws, msg: dict) -> None:
        t = msg.get("type")
        if t == "send":
            await self.send_user_message(msg.get("text") or "", msg.get("attachments") or [], "text")
        elif t == "interrupt":
            await self.interrupt()
        elif t == "new_session":
            if self.speaker:
                self.speaker.cancel()
            self.turn_active = False
            await self.send_server({"type": "new_session"})
            await self.broadcast({"type": "cleared"})
        elif t == "screenshot":
            res = await self.take_screenshot_attachment(msg.get("monitor"))
            await ws.send_text(json.dumps({"type": "attachment", "request_id": msg.get("request_id"), **res}, ensure_ascii=False))
        elif t == "settings":
            await self._apply_settings(msg.get("settings") or {})
        elif t == "ptt":
            if self.mic:
                if msg.get("state") == "down" and msg.get("dictate"):
                    self._dictate_once = True
                self.mic.set_ptt(msg.get("state") == "down")
        elif t == "dictate":
            await self._set_dictation(bool(msg.get("on")))
        elif t == "read":
            await self._start_reading(str(msg.get("text") or ""))
        elif t == "read_stop":
            await self._stop_reading()
        elif t == "confirm_reply":
            fut = self._pending_confirms.get(msg.get("id", ""))
            if fut and not fut.done():
                fut.set_result(bool(msg.get("approved")))
        elif t == "say":
            if self.speaker:
                self.speaker.say(str(msg.get("text") or ""))
        elif t == "get_state":
            await ws.send_text(json.dumps(self.status_payload(), ensure_ascii=False))
        elif t in ("get_prompts", "set_prompts", "reset_prompts"):
            await self.send_server(msg)      # the server answers with a "prompts" message, relayed to the UI
        elif t == "ping":
            await ws.send_text(json.dumps({"type": "pong", "t": msg.get("t")}))

    async def _apply_settings(self, s: dict) -> None:
        for key in ("tts_mode", "auto_listen", "barge_in", "tools_enabled", "confirm", "voice_ru", "voice_en", "tts_speed",
                    "stt_language", "tts_language", "text_model", "memory_recall"):
            if key in s:
                self.state[key] = s[key]
        if "text_model" in s and self.server_ws:
            await self.send_server({"type": "client_info", "client": {"text_model": str(s["text_model"] or "")}})
        for key in ("speaker_device", "mic_device"):   # unchanged: don't reopen the stream
            if key in s and self.state.get(key, "") == str(s[key] or ""):
                s = {k: v for k, v in s.items() if k != key}
        self._sync_runtime_settings()
        if self.mic:
            self.mic.set_enabled(bool(self.state["auto_listen"]) and self.stt is not None)
        if self.state["tts_mode"] == "off" and self.speaker:
            self.speaker.cancel()
        if "tools_enabled" in s and self.server_ws:
            await self.send_server({"type": "client_info", "client": {"tools_enabled": bool(self.state["tools_enabled"])}})
        if "speaker_device" in s:
            self.state["speaker_device"] = str(s["speaker_device"] or "")
            await asyncio.to_thread(self._switch_output_device, self.state["speaker_device"])
        if "mic_device" in s:
            self.state["mic_device"] = str(s["mic_device"] or "")
            await asyncio.to_thread(self._switch_input_device, self.state["mic_device"])
        self._save_state()
        await self.broadcast_status()

    def _switch_input_device(self, spec: str) -> None:
        if not self.mic:
            return
        try:
            self.mic.reopen(spec or None)
            settings.MIC_DEVICE = spec or None
            self.status["mic"] = "ready"
            log.info("microphone switched to %s", self.mic.device_name)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot open microphone %r: %s", spec, e)
            self.status["mic"] = f"error: {e}"[:120]

    def _switch_output_device(self, spec: str) -> None:
        """Re-open the output stream on another device (from the settings dropdown)."""
        if not self.speaker:
            return
        from .player import StreamPlayer
        try:
            new_player = StreamPlayer(44100, device=spec or None)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot open output device %r: %s", spec, e)
            self.status["tts"] = f"ready, device error: {e}"[:120]
            return
        old = self.player
        self.speaker.cancel()
        self.speaker.player = new_player
        self.player = new_player
        settings.SPEAKER_DEVICE = spec or None
        try:
            old.close()
        except Exception:
            pass
        self.status["tts"] = f"ready ({'GPU' if self.tts and 'CUDA' in self.tts.provider else 'CPU'})"
        log.info("output device switched to %s", new_player.device_name)
