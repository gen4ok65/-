@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo [music-lyrics-ai] Preloading MusicGen model before opening the app...
echo [music-lyrics-ai] Keep this window open until you see 'Model ready'.
%PY% -m music_lyrics_ai.cli preload-model --preset melody
pause
