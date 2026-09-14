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
    LLM_MODEL = _env("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    LLM_THINKING = _bool("LLM_THINKING", False)     # reasoning adds seconds of latency; off for voice
    LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.6)
    LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 4096)
    LLM_TOP_P = _env("LLM_TOP_P")                   # unset = model default
    UPSTREAM_TIMEOUT = _int("UPSTREAM_TIMEOUT", 600)  # seconds of NIM silence we tolerate
    MAX_TOOL_ROUNDS = _int("MAX_TOOL_ROUNDS", 12)

    EMBED_TEXT_MODEL = _env("EMBED_TEXT_MODEL", "nvidia/nemotron-3-embed-1b")
    EMBED_VL_MODEL = _env("EMBED_VL_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")

    # --- memory (RAG over past dialogs) ---
    MEMORY_DB = Path(_env("MEMORY_DB", str(SERVER_DIR / "data" / "memory.sqlite3")))
    MEMORY_AUTO_RECALL = _bool("MEMORY_AUTO_RECALL", True)
    MEMORY_TOP_K = _int("MEMORY_TOP_K", 4)
    MEMORY_MIN_SCORE = _float("MEMORY_MIN_SCORE", 0.42)
    CONTEXT_BUDGET_TOKENS = _int("CONTEXT_BUDGET_TOKENS", 60000)
    CONTEXT_KEEP_TURNS = _int("CONTEXT_KEEP_TURNS", 6)

    # --- DeepSeek reverse API (vision / documents / web search) ---
    DEEPSEEK_AUTH_TOKEN = _env("DEEPSEEK_AUTH_TOKEN", "")
    DEEPSEEK_COOKIES_FILE = Path(_env("DEEPSEEK_COOKIES_FILE", str(SERVER_DIR / "deepseek_cookies.json")))
    DEEPSEEK_TIMEZONE_OFFSET = _int("DEEPSEEK_TIMEZONE_OFFSET", 10800)
    DEEPSEEK_PROXY = _env("DEEPSEEK_PROXY")          # e.g. http://127.0.0.1:2080
    DEEPSEEK_THINKING = _bool("DEEPSEEK_THINKING", False)
    DEEPSEEK_MAX_FILES = _int("DEEPSEEK_MAX_FILES", 50)
    DEEPSEEK_MAX_FILE_MB = _int("DEEPSEEK_MAX_FILE_MB", 100)

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
        if not cls.DEEPSEEK_AUTH_TOKEN:
            problems.append("DEEPSEEK_AUTH_TOKEN is not set — attachments / screen / web search will be unavailable")
        return problems


settings = Settings()
