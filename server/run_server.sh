#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating server virtual environment..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
if [ ! -f .env ]; then
  echo "[!] server/.env not found. Copy .env.example to .env and fill in the keys."
  exit 1
fi
exec .venv/bin/python -m nemoagent_server.main
