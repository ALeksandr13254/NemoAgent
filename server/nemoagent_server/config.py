"""Server configuration: everything comes from environment / server/.env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_DIR / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().strip('"').strip("'")


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = _env(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    v = _env(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


class Settings:
    # --- network ---
    HOST = _env("HOST", "0.0.0.0")
    PORT = _int("PORT", 8700)
    AGENT_TOKEN = _env("AGENT_TOKEN", "")          # shared secret between client and server

    # --- NVIDIA NIM ---
    NVIDIA_API_KEY = _env("NVIDIA_API_KEY", "")
    NIM_BASE_URL = _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1/")
    # Two models: the fast text model answers every request that carries no media (first token ~1 s,
    # no queue), the omni model takes over automatically when a request contains images / audio / video
    # (attachments, screenshots) — its free pool is small and often overloaded (503 "16/16").
    LLM_MODEL = _env("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    LLM_MEDIA_MODEL = _env("LLM_MEDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    LLM_THINKING = _bool("LLM_THINKING", False)     # the model reasons by default; off saves seconds per answer
    LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.3)  # model card: 0.2 without reasoning, 0.6 with it
    LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 4096)            # executor rounds (write_file content can be long)
    DIALOGUE_MAX_TOKENS = _int("DIALOGUE_MAX_TOKENS", 1200)   # spoken answer + screen part; bounds a runaway to ~30 s
    LLM_TOP_P = _env("LLM_TOP_P")                   # unset = model default
    UPSTREAM_TIMEOUT = _int("UPSTREAM_TIMEOUT", 600)  # seconds of NIM silence we tolerate
    MAX_TOOL_ROUNDS = _int("MAX_TOOL_ROUNDS", 12)

    EMBED_TEXT_MODEL = _env("EMBED_TEXT_MODEL", "nvidia/nemotron-3-embed-1b")
    EMBED_VL_MODEL = _env("EMBED_VL_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")

    # --- media: attachments are sent to the model as image_url / audio_url / video_url parts ---
    MEDIA_IMAGE_MAX_SIDE = _int("MEDIA_IMAGE_MAX_SIDE", 1600)   # ~1600 prompt tokens per image at this size
    MEDIA_AUDIO_MAX_S = _int("MEDIA_AUDIO_MAX_S", 1200)         # longer recordings are cut (model limit: 1 hour)
    MEDIA_VIDEO_MAX_S = _int("MEDIA_VIDEO_MAX_S", 120)          # model limit: 2 minutes
    MEDIA_VIDEO_MAX_HEIGHT = _int("MEDIA_VIDEO_MAX_HEIGHT", 480)
    MEDIA_PDF_MAX_PAGES = _int("MEDIA_PDF_MAX_PAGES", 20)
    MEDIA_TEXT_MAX_CHARS = _int("MEDIA_TEXT_MAX_CHARS", 60000)  # per document, after which it is cut
    MEDIA_MAX_INLINE_MB = _int("MEDIA_MAX_INLINE_MB", 24)       # biggest file we inline as base64 (a 19 MB request went through)
    MEDIA_KEEP_TURNS = _int("MEDIA_KEEP_TURNS", 2)              # media of older user messages is replaced by a text stub
    FFMPEG = _env("FFMPEG", "ffmpeg")                           # needed for anything but wav/mp3/mp4 and for long/large clips

    # --- memory (RAG over past dialogs) ---
    MEMORY_DB = Path(_env("MEMORY_DB", str(SERVER_DIR / "data" / "memory.sqlite3")))
    # Memory (RAG over past dialogs) is used only when the client asks for it (the 🗂 button); set
    # true to recall automatically on every message instead.
    MEMORY_AUTO_RECALL = _bool("MEMORY_AUTO_RECALL", False)
    MEMORY_TOP_K = _int("MEMORY_TOP_K", 4)
    MEMORY_MIN_SCORE = _float("MEMORY_MIN_SCORE", 0.5)
    CONTEXT_BUDGET_TOKENS = _int("CONTEXT_BUDGET_TOKENS", 60000)
    CONTEXT_KEEP_TURNS = _int("CONTEXT_KEEP_TURNS", 6)

    # --- web (web_search / fetch_page tools) ---
    # Optional search API keys (first configured one is used; without any, Bing's RSS feed is the fallback):
    TAVILY_API_KEY = _env("TAVILY_API_KEY", "")           # tavily.com — free tier, made for agents
    BRAVE_SEARCH_API_KEY = _env("BRAVE_SEARCH_API_KEY", "")  # brave.com/search/api — free tier
    SERPER_API_KEY = _env("SERPER_API_KEY", "")           # serper.dev — Google results
    WEB_PROXY = _env("WEB_PROXY")                    # optional proxy for the search engine and page fetches
    WEB_USER_AGENT = _env("WEB_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

    # --- uploads ---
    UPLOAD_DIR = Path(_env("UPLOAD_DIR", str(SERVER_DIR / "data" / "uploads")))
    UPLOAD_MAX_MB = _int("UPLOAD_MAX_MB", 100)

    LOG_LEVEL = _env("LOG_LEVEL", "info")

    @classmethod
    def validate(cls) -> list[str]:
        problems = []
        if not cls.NVIDIA_API_KEY:
            problems.append("NVIDIA_API_KEY is not set (server/.env)")
        if not cls.AGENT_TOKEN:
            problems.append("AGENT_TOKEN is not set (server/.env) — clients cannot authenticate")
        return problems


settings = Settings()
