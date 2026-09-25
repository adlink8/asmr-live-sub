@echo off
cd /d D:\ADLINK\asmr-live-sub
if not exist logs mkdir logs
echo ====== %date% %time% nightly start ====== >> logs\nightly_run.log
.venv\Scripts\python.exe tools\nightly_collect.py >> logs\nightly_run.log 2>&1
echo ====== %date% %time% nightly exit %errorlevel% ====== >> logs\nightly_run.log
