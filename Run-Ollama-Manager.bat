@echo off
REM Ollama Docker Manager - Windows Batch Launcher
cd /d "%~dp0"
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "run_manager.ps1"
pause
