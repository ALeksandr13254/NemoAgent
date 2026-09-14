"""Local web UI served by the client (http://127.0.0.1:8765) + WebSocket bridge to ClientCore."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).parent / "ui"


class NoCacheStatic(StaticFiles):
    """Browsers happily reuse a cached app.js after an update, leaving new buttons dead."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp


def _asset_version() -> str:
    h = hashlib.md5()
    for name in ("app.js", "style.css"):
        try:
            h.update((UI_DIR / name).read_bytes())
        except FileNotFoundError:
            pass
    return h.hexdigest()[:10]


def make_app(core) -> FastAPI:
    app = FastAPI(title="NemoAgent client UI")
    app.mount("/static", NoCacheStatic(directory=str(UI_DIR)), name="static")

    @app.get("/")
    async def index():
        # version query on the assets = a fresh script after every update, whatever the cache did
        v = _asset_version()
        html = (UI_DIR / "index.html").read_text("utf-8")
        html = html.replace('href="/static/style.css"', f'href="/static/style.css?v={v}"')
        html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={v}"')
        return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})

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
