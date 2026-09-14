"""Local web UI served by the client (http://127.0.0.1:8765) + WebSocket bridge to ClientCore."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).parent / "ui"


def make_app(core) -> FastAPI:
    app = FastAPI(title="NemoAgent client UI")
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.post("/ui/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        return await core.upload_attachment(file.filename or "file", data, file.content_type)

    @app.websocket("/ui")
    async def ui_ws(ws: WebSocket):
        await ws.accept()
        core.ui_clients.add(ws)
        try:
            await ws.send_text(json.dumps(core.status_payload(), ensure_ascii=False))
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await core.on_ui_message(ws, msg)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            core.ui_clients.discard(ws)

    return app
