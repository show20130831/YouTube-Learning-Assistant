<#
.SYNOPSIS
    Run the weekly drift check (yla eval drift) and append its output to logs\drift.log.
#>
& (Join-Path $PSScriptRoot "run-worker.ps1") -YlaArgs @("eval", "drift") -LogName "drift.log"
exit $LASTEXITCODE
