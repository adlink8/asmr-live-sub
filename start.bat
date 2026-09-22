@echo off
cd /d %~dp0
.venv\Scripts\python.exe live_sub.py --model anime --mt sakura --mt-ngl 99 --strategy adaptive --max-s 5.0 --hang-s 2.0 %*
pause
