@echo off
rem run_simulink.bat - double-click wrapper for run_simulink.ps1 (args are passed through)
rem   run_simulink.bat            full pipeline: selftest -> build -> verify -> injection scan -> plots
rem   run_simulink.bat -Quick     stop after the model checks (no scan, no figures)
rem Comments stay ASCII: cmd reads .bat in the OEM code page, UTF-8 Chinese turns into garbage.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_simulink.ps1" %*
endlocal
