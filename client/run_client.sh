#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating client virtual environment..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install --no-deps silero-vad
fi
if [ ! -f .env ]; then
  echo "[!] client/.env not found. Copy .env.example to .env and set SERVER_URL / AGENT_TOKEN."
  exit 1
fi
export PYTHONUTF8=1
exec .venv/bin/python -m nemoagent_client.main
