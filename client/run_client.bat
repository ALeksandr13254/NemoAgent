@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating client virtual environment...
    py -3.11 -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    ".venv\Scripts\python.exe" -m pip install --no-deps silero-vad
)
if not exist ".env" (
    echo [!] client\.env not found. Copy .env.example to .env and set SERVER_URL / AGENT_TOKEN.
    pause
    exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m nemoagent_client.main
pause
