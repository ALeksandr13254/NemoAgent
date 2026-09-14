@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating server virtual environment...
    py -3.11 -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if not exist ".env" (
    echo [!] server\.env not found. Copy .env.example to .env and fill in the keys.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m nemoagent_server.main
pause
