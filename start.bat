@echo off
cd /d %~dp0
.venv\Scripts\python.exe live_sub.py %*
pause
