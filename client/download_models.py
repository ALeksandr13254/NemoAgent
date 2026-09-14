"""Download the models the client needs: TeraTTSv2 (~1.2 GB) and faster-whisper large-v3-turbo (~1.6 GB).

    .venv\\Scripts\\python.exe download_models.py            # both
    .venv\\Scripts\\python.exe download_models.py --tts     # only TeraTTS
    .venv\\Scripts\\python.exe download_models.py --stt     # only Whisper
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def download_tts(target: Path) -> None:
    from huggingface_hub import snapshot_download
    target.mkdir(parents=True, exist_ok=True)
    print(f"[tts] downloading TeraSpace/TeraTTSv2 -> {target}")
    snapshot_download(repo_id="TeraSpace/TeraTTSv2", local_dir=str(target))
    print("[tts] done")


def download_stt(model: str) -> None:
    from nemoagent_client.cuda import setup_cuda_paths
    setup_cuda_paths()
    from faster_whisper import WhisperModel
    print(f"[stt] downloading faster-whisper {model} (HF cache)")
    WhisperModel(model, device="cpu", compute_type="int8")
    print("[stt] done")


def main() -> None:
    from nemoagent_client.config import settings
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--stt", action="store_true")
    a = ap.parse_args()
    both = not (a.tts or a.stt)
    if a.tts or both:
        if (settings.TTS_MODEL_DIR / "config.json").exists():
            print(f"[tts] already present: {settings.TTS_MODEL_DIR}")
        else:
            download_tts(settings.TTS_MODEL_DIR)
    if a.stt or both:
        download_stt(settings.STT_MODEL)


if __name__ == "__main__":
    main()
