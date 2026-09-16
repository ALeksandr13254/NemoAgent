"""Client configuration (client/.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

CLIENT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(CLIENT_DIR / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().strip('"').strip("'")


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


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
    # --- connection to the agent server ---
    SERVER_URL = _env("SERVER_URL", "http://127.0.0.1:8700")
    AGENT_TOKEN = _env("AGENT_TOKEN", "")

    # --- local UI (browser page) ---
    UI_HOST = _env("UI_HOST", "127.0.0.1")
    UI_PORT = _int("UI_PORT", 8765)
    OPEN_BROWSER = _bool("OPEN_BROWSER", True)

    # --- speech to text (faster-whisper) ---
    STT_ENABLED = _bool("STT_ENABLED", True)
    STT_MODEL = _env("STT_MODEL", "large-v3-turbo")
    STT_DEVICE = _env("STT_DEVICE", "auto")            # auto | cuda | cpu
    STT_COMPUTE = _env("STT_COMPUTE", "auto")          # auto | float16 | int8_float16 | int8
    STT_LANGUAGE = _env("STT_LANGUAGE", "ru")          # ru | en | auto (auto can drift to other languages on background audio)
    STT_BEAM = _int("STT_BEAM", 3)                     # beam 3 costs nothing on GPU and drops fewer words

    # --- microphone / VAD ---
    MIC_DEVICE = _env("MIC_DEVICE")                    # name or index; empty = default
    AUTO_LISTEN = _bool("AUTO_LISTEN", True)           # hands-free: VAD decides when you speak
    VAD_THRESHOLD = _float("VAD_THRESHOLD", 0.55)
    VAD_SILENCE_MS = _int("VAD_SILENCE_MS", 750)       # silence that ends an utterance (a mid-sentence pause is ~0.5 s)
    VAD_MIN_SPEECH_MS = _int("VAD_MIN_SPEECH_MS", 250)
    VAD_MAX_UTTERANCE_S = _int("VAD_MAX_UTTERANCE_S", 45)
    VAD_PRE_ROLL_MS = _int("VAD_PRE_ROLL_MS", 300)
    BARGE_IN = _bool("BARGE_IN", True)                 # speaking while the agent talks interrupts it
    BARGE_IN_MIN_SPEECH_MS = _int("BARGE_IN_MIN_SPEECH_MS", 450)

    # --- text to speech (TeraTTSv2) ---
    TTS_ENABLED = _bool("TTS_ENABLED", True)
    TTS_MODEL_DIR = Path(_env("TTS_MODEL_DIR", str(CLIENT_DIR / "models" / "TeraTTSv2")))
    TTS_PROVIDER = _env("TTS_PROVIDER", "auto")        # auto | cuda | cpu
    TTS_THREADS = _int("TTS_THREADS", 0)               # 0 = automatic
    TTS_LANGUAGE = _env("TTS_LANGUAGE", "auto")        # auto (by script) | ru | en — which voice reads Latin words and numbers
    TEXT_MODEL = _env("TEXT_MODEL", "")                # text model asked from the server (empty = server default); switchable in the UI
    TTS_VOICE_RU = _env("TTS_VOICE_RU", "ru_f1")
    TTS_VOICE_EN = _env("TTS_VOICE_EN", "eng_f3")
    TTS_SPEED = _float("TTS_SPEED", 1.0)               # duration_scale: <1 faster, >1 slower
    TTS_FIRST_CHUNK_CHARS = _int("TTS_FIRST_CHUNK_CHARS", 70)
    TTS_MAX_CHARS = _int("TTS_MAX_CHARS", 180)
    TTS_STREAM_FRAMES = _int("TTS_STREAM_FRAMES", 8)   # vocoder frames per streamed audio chunk
    SPEAKER_DEVICE = _env("SPEAKER_DEVICE")

    # --- computer control ---
    TOOLS_ENABLED = _bool("TOOLS_ENABLED", True)
    TOOL_CONFIRM = _env("TOOL_CONFIRM", "dangerous")   # never | dangerous | always
    TOOL_CONFIRM_TIMEOUT = _int("TOOL_CONFIRM_TIMEOUT", 90)
    SCREENSHOT_MAX_SIDE = _int("SCREENSHOT_MAX_SIDE", 1600)

    LOG_LEVEL = _env("LOG_LEVEL", "info")

    @classmethod
    def ws_url(cls) -> str:
        u = cls.SERVER_URL.rstrip("/")
        if u.startswith("https://"):
            return "wss://" + u[len("https://"):] + "/ws"
        if u.startswith("http://"):
            return "ws://" + u[len("http://"):] + "/ws"
        if u.startswith("ws"):
            return u + "/ws"
        return "ws://" + u + "/ws"

    @classmethod
    def http_url(cls) -> str:
        u = cls.SERVER_URL.rstrip("/")
        if u.startswith("ws://"):
            return "http://" + u[5:]
        if u.startswith("wss://"):
            return "https://" + u[6:]
        if not u.startswith("http"):
            return "http://" + u
        return u


settings = Settings()
