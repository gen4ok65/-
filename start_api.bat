@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo [music-lyrics-ai] Starting REST API on http://127.0.0.1:8000 ...
%PY% -m music_lyrics_ai.cli serve-api --preset melody --host 127.0.0.1 --port 8000
pause
