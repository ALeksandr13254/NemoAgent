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


def _allowed(value: str | None, options: tuple) -> str:
    return value if value in options else options[0]


# Reasoning effort for the dialogue agent and the executor, asked from the server (shown in the UI settings).
# Nemotron models only switch thinking on/off (any level but off enables it); GLM budgets it with
# reasoning_effort (off means the minimum, low), DeepSeek with a numeric reasoning_effort 1-100
# (off/low/high/max = 1/33/66/100); StepFun has no documented NIM switch and follows the on/off path.
# The router always classifies without reasoning.
REASONING_MODES = ("off", "low", "high", "max")


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
    TTS_LANGUAGE = _env("TTS_LANGUAGE", "ru")        # auto (by script) | ru | en — which voice reads Latin words and numbers
    # Model per role, asked from the server (shown in the UI settings). Every role runs on the multimodal Nano Omni
    # by default and sees attachments and screenshots itself; Nemotron 3.5 Super VL 120B, GLM 5.3 Flash,
    # Step 5 Preview and DeepSeek V4.1 Flash are also multimodal (text and images, no audio/video),
    # Nemotron 3.5 Lightning and GLM 5.3 take text only
    # (their calls get a text note instead of media). A value outside this list (for example the retired Nemotron 3
    # Super 120B left in an old .env) falls back to the first entry.
    TEXT_MODELS = ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nvidia/nemotron-3.5-super-vl-120b-a12b",
                   "nvidia/nemotron-3.5-lightning-30b-a3b", "z-ai/glm-5.3-flash", "z-ai/glm-5.3",
                   "stepfun-ai/step-5-preview", "deepseek-ai/deepseek-v4.1-flash")
    MODEL_DIALOGUE = _allowed(_env("MODEL_DIALOGUE"), TEXT_MODELS)
    MODEL_EXECUTOR = _allowed(_env("MODEL_EXECUTOR"), TEXT_MODELS)
    MODEL_ROUTER = _allowed(_env("MODEL_ROUTER"), TEXT_MODELS)
    REASONING = _allowed(_env("REASONING"), REASONING_MODES)
    TTS_VOICE_RU = _env("TTS_VOICE_RU", "ru_f1")
    TTS_VOICE_EN = _env("TTS_VOICE_EN", "eng_f5")
    TTS_SPEED = _float("TTS_SPEED", 1.0)               # duration_scale: <1 faster, >1 slower
    TTS_FIRST_CHUNK_CHARS = _int("TTS_FIRST_CHUNK_CHARS", 70)
    TTS_MAX_CHARS = _int("TTS_MAX_CHARS", 180)
    TTS_STREAM_FRAMES = _int("TTS_STREAM_FRAMES", 8)   # vocoder frames per streamed audio chunk
    SPEAKER_DEVICE = _env("SPEAKER_DEVICE")

    # --- everything persistent (chats, attachments, memory, settings, prompt overrides) lives under DATA_DIR ---
    DATA_DIR = Path(_env("CLIENT_DATA_DIR", str(CLIENT_DIR / "data")))

    # --- long-term memory (RAG over past dialogs), stored here on the client ---
    MEMORY_TOP_K = _int("MEMORY_TOP_K", 4)             # memories mixed into a message when 🗂 is on
    # Cosine threshold for that. Query -> passage cosines of these models are low: a relevant memory scores
    # 0.27-0.7 (nemotron-3-embed-1b) or 0.26-0.55 (llama-nemotron-embed-vl-1b-v2), an unrelated one rarely
    # above 0.24. The old 0.5 dropped 85-90% of the relevant memories; 0.27 keeps about 90% of them.
    MEMORY_MIN_SCORE = _float("MEMORY_MIN_SCORE", 0.27)
    ATTACHMENT_KEEP_HOURS = _int("ATTACHMENT_KEEP_HOURS", 1)   # uploads that never made it into a chat are dropped after this

    # --- computer control ---
    TOOLS_ENABLED = _bool("TOOLS_ENABLED", True)
    TOOL_CONFIRM = _env("TOOL_CONFIRM", "dangerous")   # never | dangerous | always
    TOOL_CONFIRM_TIMEOUT = _int("TOOL_CONFIRM_TIMEOUT", 90)
    SCREENSHOT_MAX_SIDE = _int("SCREENSHOT_MAX_SIDE", 1600)
    # monitors the screenshot button / look_at_screen capture (1-based numbers, one image per monitor); empty = all
    SCREENSHOT_MONITORS = [int(x) for x in (_env("SCREENSHOT_MONITORS", "") or "").replace(";", ",").split(",") if x.strip().isdigit()]

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
