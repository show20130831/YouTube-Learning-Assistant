<#
.SYNOPSIS
    Register (or update) the weekly scheduled task that runs the drift check.

.DESCRIPTION
    - Runs every -Day at -Time (default Sunday 14:00: after the 12:00 digest, so the daily pipeline
      has already used its share of the free LLM quota).
    - Re-runs the primary model on the evaluation videos (transcripts are cached on this computer)
      and records the result; the next digest warns if the model drifted.
    - Same conditions as the worker task: wakes the computer, runs late if missed, current user only.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\register-drift-task.ps1
#>
param(
    [string]$Day = "Sunday",
    [string]$Time = "14:00",
    [string]$TaskName = "YouTube Learning Assistant - drift check"
)
$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "run-drift.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $Time
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force `
    -Description "Weekly check that the LLM behind the configured model ID has not changed (YouTube Learning Assistant)." |
    Out-Null

$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Output "Registered '$TaskName': next run $($info.NextRunTime)"
