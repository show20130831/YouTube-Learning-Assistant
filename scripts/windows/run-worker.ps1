<#
.SYNOPSIS
    Run a yla command once (the local worker by default) and append its output to a log in logs\.

.DESCRIPTION
    Called by the scheduled tasks created with register-task.ps1 (worker) and
    register-drift-task.ps1 (weekly drift check, through run-drift.ps1). Uses the project's virtual
    environment (created by `uv sync`), so uv does not need to be on PATH for the task.
    Worker exit codes: 0 ok, 1 error, 2 YouTube blocked caption requests from this network.
#>
param(
    [string[]]$YlaArgs = @("worker"),
    [string]$LogName = "worker.log"
)
$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir $LogName
if ((Test-Path $log) -and (Get-Item $log).Length -gt 1MB) {
    Move-Item -Force $log "$log.1"  # keep one previous log
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# Python writes UTF-8; without this PowerShell decodes it with the console code page (cp950 on
# Traditional Chinese Windows) and Chinese channel names end up garbled in the log.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$yla = Join-Path $root ".venv\Scripts\yla.exe"

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -FilePath $log -Append -Encoding utf8
if (-not (Test-Path $yla)) {
    "yla not found at $yla; run 'uv sync' in $root" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
& $yla @YlaArgs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
$code = $LASTEXITCODE
"exit code: $code" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
