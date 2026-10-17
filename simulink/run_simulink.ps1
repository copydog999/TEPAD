<#
.SYNOPSIS
    One-command run of the TEPAD Simulink simulation (headless MATLAB -batch, exits automatically when done).

.USAGE
    .\run_simulink.ps1                          # full pipeline: self-test → build → model checks → injection scan → plots
    .\run_simulink.ps1 -Quick                   # stop after the model checks (no figures, fastest)
    .\run_simulink.ps1 -NoRebuild               # reuse the existing .slx, do not rebuild
    .\run_simulink.ps1 -Target tepadsim_injection_scan -NoScan    # run a single step
    .\run_simulink.ps1 -MatlabRoot 'D:\MATLAB\R2024b'

.NOTES
    Why -batch instead of -r: -batch brings back the exit code when done, so CI/scripts can judge;
    MATLAB started with -r is an interactive window and stays around.
    Note this uses bin\win64\MATLAB.exe — bin\matlab.exe is blocked on some machines by the
    RUNASADMIN compatibility layer of AppCompatLayers and will not start in a non-admin session.
    MATLAB stdout is GBK on a Chinese Windows, so logs are read with -Encoding Default.
#>
[CmdletBinding()]
param(
    [string] $Target = 'tepadsim_run_all',
    [string] $MatlabRoot = 'C:\Program Files\MATLAB\R2024b',
    [switch] $Quick,        # = -NoScan
    [switch] $NoScan,
    [switch] $NoRebuild,
    [int]    $TimeoutSec = 1800
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$src = Join-Path $root 'src'
$exe = Join-Path $MatlabRoot 'bin\win64\MATLAB.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "MATLAB executable not found: $exe (use -MatlabRoot to point at the install root)"
}
if (-not (Get-Command (Join-Path $src ($Target + '.m')) -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath (Join-Path $src ($Target + '.m')))) {
        throw "entry function $Target.m not found under simulink/src"
    }
}

# ---- compose the -batch expression: add path → call entry → a false return exits non-zero ----
$opts = @()
if ($Quick -or $NoScan) { $opts += "'skipScan', true" }
if ($NoRebuild)         { $opts += "'rebuild', false" }
$optText = ''
if ($opts.Count) { $optText = $opts -join ', ' }   # used as a raw argument list, it must not start with a comma

$srcFwd = $src -replace '\\', '/'
# Keep the expression pure ASCII: the -batch string passes through cmd, so non-ASCII always arrives mangled.
# Only entries returning a logical get an output; void functions such as tepadsim_open_model are called directly.
$logicalTargets = 'tepadsim_run_all', 'tepadsim_selftest', 'tepadsim_verify_model'
if ($Target -in $logicalTargets) {
    $call = "r = $Target($optText); " +
            "if islogical(r) && ~r, error('tepadsim:failed', '%s returned false', '$Target'); end"
} else {
    $call = "$Target($optText)"
}
$expr = "addpath('$srcFwd'); $call"

# ---- start MATLAB and write the log under simulink/logs --------------------
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outFile = Join-Path $logDir ("run-{0}.out.txt" -f $stamp)
$errFile = Join-Path $logDir ("run-{0}.err.txt" -f $stamp)

Write-Host "[run_simulink] MATLAB = $exe"
Write-Host "[run_simulink] batch  = $expr"
Write-Host "[run_simulink] log    = $outFile"

$argList = '-batch "' + $expr + '"'
# Working directory is simulink/, not src/: Simulink caches slprj/ and *.slxc in the cwd,
# and putting them in src/ would mix them with the sources (both are already in .gitignore).
$proc = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $root `
    -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
# Event notification must be enabled, otherwise $proc.ExitCode is empty after WaitForExit(ms) polling
# (only the synchronous -Wait form obtains the exit code).
$proc.EnableRaisingEvents = $true
if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    try { $proc.Kill() } catch { }
    throw "MATLAB did not finish within $TimeoutSec s and was terminated; partial log: $outFile"
}
Start-Sleep -Seconds 1     # let the handles flush, otherwise the file may still read empty below

if (Test-Path -LiteralPath $outFile) {
    Get-Content -LiteralPath $outFile -Encoding Default | Write-Host
}
$code = $proc.ExitCode
if ($code -ne 0 -and (Test-Path -LiteralPath $errFile)) {
    Write-Host '---- stderr ----'
    Get-Content -LiteralPath $errFile -Encoding Default | Write-Host
}
Write-Host "[run_simulink] exit code = $code"
exit $code
