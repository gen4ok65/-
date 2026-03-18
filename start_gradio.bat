@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo [music-lyrics-ai] Starting local Gradio studio...
echo [music-lyrics-ai] If this is the first run, wait for the model download and do not press Ctrl+C.
%PY% -m music_lyrics_ai.cli launch --preset melody
pause
