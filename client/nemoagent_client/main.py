"""NemoAgent client entry point."""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# quiet the ML libraries before they are imported
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import settings  # noqa: E402


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "websockets", "uvicorn", "faster_whisper", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not settings.AGENT_TOKEN:
        logging.warning("AGENT_TOKEN is empty in client/.env — the server will reject the connection unless it also has no token")
    logging.info("NemoAgent client → server %s | UI http://%s:%s", settings.SERVER_URL, settings.UI_HOST, settings.UI_PORT)
    from .core import ClientCore
    try:
        asyncio.run(ClientCore().run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
