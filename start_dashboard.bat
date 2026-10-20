@echo off
rem Double-click to open the Duds Studio dashboard (http://127.0.0.1:5000)
cd /d "%~dp0"
"venv\Scripts\python.exe" src\dashboard.py
pause
