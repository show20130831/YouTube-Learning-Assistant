<#
.SYNOPSIS
    Register (or update) the daily scheduled task that runs the local worker.

.DESCRIPTION
    - Runs daily at -Time (default 10:30, before the 11:00 cloud pipeline).
    - Wakes the computer from sleep (also needs "Allow wake timers" enabled in Power Options).
    - Runs as soon as possible after a missed start, e.g. when the computer was off at 10:30.
    - Runs as the current user only while signed in, so no administrator rights or stored
      password are needed. No automatic restart: retrying while YouTube blocks requests only
      makes it worse.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\register-task.ps1
    powershell -ExecutionPolicy Bypass -File scripts\windows\register-task.ps1 -Time 09:45
#>
param(
    [string]$Time = "10:30",
    [string]$TaskName = "YouTube Learning Assistant - local worker"
)
$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "run-worker.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force `
    -Description "Discover new videos and fetch captions from the home network (YouTube Learning Assistant)." |
    Out-Null

$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Output "Registered '$($task.TaskName)': state $($task.State), next run $($info.NextRunTime)"
Write-Output "Run it now with: Start-ScheduledTask -TaskName '$TaskName'"
