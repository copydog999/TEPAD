@echo off
REM =====================================================================
REM TEPAD simulation UI launcher (Windows, one click)
REM
REM   scripts\run_sim.bat        prepare calibration -> build -> start bridge -> start UI
REM   scripts\run_sim.bat check   environment check only, no windows opened
REM
REM Keep this file ASCII-only: cmd.exe reads .bat with the system ANSI
REM code page, so UTF-8 Chinese here breaks parsing (GBK console, no BOM).
REM =====================================================================
setlocal
cd /d "%~dp0.."

set "EXE=sim_ui\build\x64\Release\sim_ui.exe"
set "CAL=sim_ui\calibration\dual_window_calibration.json"
set "MSB=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"

REM Pick a Python 3.8+ interpreter. Bare "python" on this machine is an old
REM Anaconda 3.7, which cannot run src\simulation. Override with TEPAD_PY.
if not defined PY (
  if defined TEPAD_PY set "PY=%TEPAD_PY%"
  if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
  if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
  if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" set "PY=C:\ProgramData\miniconda3\python.exe"
  if not defined PY set "PY=python"
)
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [x] "%PY%" is older than Python 3.8 ^(setup.py requires 3.8+^).
  echo     Fix: set TEPAD_PY=C:\path\to\python.exe  then run this script again.
  exit /b 1
)

if /i "%~1"=="check" goto :check

echo === TEPAD dual-window simulation UI ===

REM [1/4] calibration json: export it if missing
if exist "%CAL%" (
  echo [1/4] calibration file ok
) else (
  echo [1/4] exporting calibration, about 1 min...
  "%PY%" scripts\export_simulation_calibration.py --golden sim_ui\selftest\golden\golden_frames.csv
  if errorlevel 1 (
    echo [x] calibration export failed, check your python env
    exit /b 1
  )
)

REM [2/4] MFC host app: build it if missing
if exist "%EXE%" (
  echo [2/4] sim_ui.exe ok
) else (
  if not exist "%MSB%" (
    echo [x] MSBuild not found. Install VS 2022 "Desktop C++" + "C++ MFC" component.
    exit /b 1
  )
  echo [2/4] building sim_ui, about 1 min...
  "%MSB%" sim_ui\sim_ui.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
  if errorlevel 1 (
    echo [x] build failed
    exit /b 1
  )
)

REM [3/4] data bridge = TCP server, loops one simulated day with injected faults.
REM It runs in its own window via run_bridge.bat (nesting "cmd /k" with quoted
REM redirection here silently swallows the script argument).
echo [3/4] starting data bridge on 127.0.0.1:18801
start "TEPAD data bridge - keep it open" "%~dp0run_bridge.bat"
REM The bridge first simulates a day (720 ticks); give it a few seconds to bind the port.
ping -n 6 127.0.0.1 >nul

REM [4/4] host app (--auto = connect to the local bridge without extra clicks)
echo [4/4] starting sim_ui.exe --auto
start "" "%EXE%" --auto

echo.
echo Done. Curves should start moving on their own.
echo Calibration loads at startup; "eta=..." in the info line means OK.
echo Use the two "inject" buttons on row 5 to fire a spike / shading and read the alarm delay.
echo If the link line says connect failed, wait a second and click the connect button.
echo Closing the black "TEPAD data bridge" window stops the stream.
goto :eof

:check
echo === environment check ===
if exist "%CAL%" (echo [ok] calibration  %CAL%) else (echo [-- ] missing: run scripts\run_sim.bat to export it)
if exist "%EXE%" (echo [ok] host app     %EXE%) else (echo [-- ] not built: run scripts\run_sim.bat to build it)
if exist "%MSB%" (echo [ok] msbuild      %MSB%) else (echo [x ] missing MSBuild: VS 2022 Desktop C++ + C++ MFC component required)
echo [ok] python       %PY%
"%PY%" --version
endlocal
