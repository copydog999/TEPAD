@echo off
REM =====================================================================
REM TEPAD data bridge: serves tick,G,P frames to sim_ui.exe over TCP.
REM Started by scripts\run_sim.bat, but you can also run it alone.
REM Keep this file ASCII-only (cmd.exe reads .bat with the ANSI code page).
REM   options: %1 = extra args are appended, e.g.  --port COM4 --baud 115200
REM =====================================================================
setlocal
cd /d "%~dp0.."

if defined TEPAD_PY set "PY=%TEPAD_PY%"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" set "PY=C:\ProgramData\miniconda3\python.exe"
if not defined PY set "PY=python"

echo TEPAD data bridge - interpreter: %PY%
echo Serves tick,G,P on 127.0.0.1:18801, loops one simulated day (720 ticks).
echo Default pace is 100 ms/tick; the UI "Playback" dropdown changes it live
echo (it sends "SPEED <sec>" back to this window), no restart needed.
echo Data is CLEAN on purpose (no python-side fault injection): the only anomalies
echo are the ones you fire from the UI buttons, so any alarm you see is yours.
echo Keep this window open. Close it to stop the stream.
echo.

"%PY%" scripts\serial_bridge.py --tcp --interval 0.1 --loop %*

echo.
echo Bridge exited (port busy, python error, or you stopped it).
endlocal
