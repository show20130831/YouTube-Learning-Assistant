<#
.SYNOPSIS
    Run the local worker once and append its output to logs\worker.log.

.DESCRIPTION
    Called by the scheduled task created with register-task.ps1. Uses the project's virtual
    environment (created by `uv sync`), so uv does not need to be on PATH for the task.
    Exit codes: 0 ok, 1 error, 2 YouTube blocked caption requests from this network.
#>
$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "worker.log"
if ((Test-Path $log) -and (Get-Item $log).Length -gt 1MB) {
    Move-Item -Force $log "$log.1"  # keep one previous log
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$yla = Join-Path $root ".venv\Scripts\yla.exe"

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -FilePath $log -Append -Encoding utf8
if (-not (Test-Path $yla)) {
    "yla not found at $yla; run 'uv sync' in $root" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
& $yla worker 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
$code = $LASTEXITCODE
"exit code: $code" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
