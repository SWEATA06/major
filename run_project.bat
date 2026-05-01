@echo off
REM run_project.bat - Start backend and frontend in separate PowerShell windows

REM Start backend (run from repository root so Python imports resolve)
start "Backend" powershell -NoExit -Command "Set-Location -LiteralPath '%~dp0'; $env:PYTHONPATH=(Get-Location).Path; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

REM Start frontend (installs dependencies first time)
start "Frontend" powershell -NoExit -Command "Set-Location -LiteralPath '%~dp0frontend'; if (-not (Test-Path 'node_modules')) { npm install }; npm run dev"

echo Launched Backend and Frontend windows. Close this window when finished.
pause >nul
