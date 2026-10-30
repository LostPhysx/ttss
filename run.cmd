@echo off
REM Start the TTSS server at http://127.0.0.1:7860 (everything stays inside .tools\)
setlocal
cd /d "%~dp0"
if not exist ".tools\venv\Scripts\python.exe" (
  echo Environment missing - running setup first...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" || exit /b 1
)
set "HF_HOME=%~dp0.tools\hf-home"
set "TORCH_HOME=%~dp0.tools\torch-home"
if "%PORT%"=="" set "PORT=7860"
REM 0.0.0.0 = reachable from other devices on the local network (phone). Set HOST=127.0.0.1 for this PC only.
if "%HOST%"=="" set "HOST=0.0.0.0"
start "" "http://127.0.0.1:%PORT%"
".tools\venv\Scripts\python.exe" -m uvicorn app.server:app --host %HOST% --port %PORT%
