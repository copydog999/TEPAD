@echo off
REM TEPAD one-click run (Windows)
setlocal
cd /d "%~dp0.."

echo === TEPAD pipeline ===
if not exist "datasets\ding\raw\train" (
  echo [!] datasets are not ready; run first: python scripts\download_datasets.py --source ^<dir^>
)

python src\pipeline\run_all.py --datasets %1 --output results\ %2
if errorlevel 1 (
  echo [x] run failed
  exit /b 1
)
echo [ok] results written to results\
REM Note: append --baselines to also run the Ding cross-model comparison (table 6 / figure 7, about 6-8 minutes)
endlocal
