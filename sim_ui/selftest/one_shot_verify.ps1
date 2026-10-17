# one_shot_verify.ps1 -- start bridge + UI, click both inject buttons, print the
# on-screen alarm delay for each. Everything lives inside one process tree, because
# child processes started from a CI/agent shell are reaped when the command ends.
#
#   powershell -ExecutionPolicy Bypass -File sim_ui\selftest\one_shot_verify.ps1
param([int]$WarmupSeconds = 12)

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$exe = Join-Path $root 'sim_ui\build\x64\Release\sim_ui.exe'
$smoke = Join-Path $PSScriptRoot 'ui_smoke.ps1'

# Only one sim_ui may exist, otherwise UI Automation can click a stale instance.
Get-Process sim_ui -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*serial_bridge*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

$bridge = Start-Process -FilePath (Join-Path $root 'scripts\run_bridge.bat') -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 3
$ui = Start-Process -FilePath $exe -ArgumentList '--auto' -PassThru
Write-Host ('bridge=' + $bridge.Id + ' ui=' + $ui.Id)
Start-Sleep -Seconds $WarmupSeconds

function Get-UiLines {
    & powershell -ExecutionPolicy Bypass -File $smoke 'status' |
        Where-Object { $_ -like '*UI:*' }
}

function Show-Status([string]$tag) {
    Write-Host ('----- ' + $tag + ' -----')
    Get-UiLines | ForEach-Object { Write-Host $_ }
}

# Keep this file ASCII-only: Windows PowerShell 5.1 reads -File scripts with the ANSI
# code page, so literal Chinese here would never match what the UI shows.
$star = [string][char]0x2605

# The 60-tick window keeps alarming for dozens of frames after a fault, and one tick is
# 0.1s of wall clock now, so a fixed 2s sleep leaves the detector still hot: the next
# injection would then "alarm after 1 frame" and the weak-sigma case would look broken.
# Wait for the per-frame alarm mark to disappear, so every case starts from normal data.
function Wait-DetectorQuiet([int]$LimitSeconds = 120) {
    $frameWord = -join (@(0x5E27) | ForEach-Object { [char]$_ })   # "zhen" = frame
    $statsPattern = '^UI: (' + [regex]::Escape($frameWord) + '|frames) '
    $deadline = (Get-Date).AddSeconds($LimitSeconds)
    while ((Get-Date) -lt $deadline) {
        $lines = Get-UiLines
        # Only trust a read that actually returned the live statistics row; an empty
        # result (window gone / UI busy) must never be mistaken for "quiet".
        $stats = @($lines | Where-Object { $_ -match $statsPattern })
        if ($stats.Count -eq 0) {
            Start-Sleep -Seconds 2
            continue
        }
        if (-not ($stats[0] -like ('*' + $star + '*'))) {
            Start-Sleep -Seconds 1
            Write-Host ('detector quiet, ' + $stats[0])
            return $true
        }
        Start-Sleep -Seconds 2
    }
    Write-Host 'WARN: detector still alarming when the wait limit was reached'
    return $false
}

Show-Status 'baseline (no manual injection yet)'

foreach ($case in @(
    @{ action = 'spike'; value = '';  expect = 'alarm'; label = 'default 8-sigma spike' },
    @{ action = 'shade'; value = '';  expect = 'alarm'; label = 'default 60-tick shading' },
    @{ action = 'spike'; value = '2'; expect = 'quiet'; label = 'weak 2-sigma spike' },
    @{ action = 'spike'; value = '3'; expect = 'any';   label = 'marginal 3-sigma spike' },
    @{ action = 'spike'; value = '8'; expect = 'alarm'; label = 'explicit 8-sigma spike' })) {
    Write-Host ('===== manual ' + $case.action + ': ' + $case.label + ' =====')
    if ($case.value -eq '') {
        & powershell -ExecutionPolicy Bypass -File $smoke $case.action 40
    } else {
        & powershell -ExecutionPolicy Bypass -File $smoke $case.action 40 $case.value
    }
    # exit 0 = the screen says "alarmed after N frames", 2 = "not alarmed" (verdict reached).
    $code = $LASTEXITCODE
    $alarmed = ($code -eq 0)
    $verdict = if ($case.expect -eq 'any') { 'INFO' }
               elseif ($alarmed -eq ($case.expect -eq 'alarm')) { 'PASS' }
               else { 'FAIL' }
    Write-Host ('EXPECT=' + $case.expect + ' EXIT=' + $code + ' VERDICT=' + $verdict)
    Wait-DetectorQuiet | Out-Null
}

# Bilingual check: the language button must rebuild every caption, and the English
# verdict line must still be parseable by ui_smoke.ps1.
Write-Host '===== switch the UI to English, then inject the default spike ====='
& powershell -ExecutionPolicy Bypass -File $smoke 'language' | ForEach-Object { Write-Host $_ }
Wait-DetectorQuiet | Out-Null
& powershell -ExecutionPolicy Bypass -File $smoke spike 40 8
$code = $LASTEXITCODE
$verdict = if ($code -eq 0) { 'PASS' } else { 'FAIL' }
Write-Host ('EXPECT=alarm EXIT=' + $code + ' VERDICT=' + $verdict)

Write-Host '--- done ---'

Stop-Process -Id $ui.Id -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -gt $bridge.StartTime } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Stop-Process -Id $bridge.Id -Force -ErrorAction SilentlyContinue
Write-Host 'cleanup done'
