$ErrorActionPreference = 'Stop'
$r49Root = 'E:\GAN\research\sim2real_two_papers_20261008_0305'
$r49Work = Join-Path $r49Root 'work\R49_LOCAL_HEUN128_RECOVERY_001'
$r49Active = Get-Content -LiteralPath (Join-Path $r49Work 'ACTIVE_JOB.json') -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
$r49Process = Get-Process -Id $r49Active.actual_worker_PID -ErrorAction SilentlyContinue
$r49Status = [ordered]@{
    observed_UTC = [DateTime]::UtcNow.ToString('o')
    entry = $r49Active.entry
    worker_PID = $r49Active.actual_worker_PID
    saved_native_exit = $r49Active.actual_worker_native_exit
    process_found_now = [bool]$r49Process
    process_start_time = if ($r49Process) { $r49Process.StartTime.ToUniversalTime().ToString('o') } else { $null }
    cumulative_CPU_s = if ($r49Process) { $r49Process.CPU } else { $null }
    current_RSS_bytes = if ($r49Process) { $r49Process.WorkingSet64 } else { $null }
    process_peak_RSS_bytes = if ($r49Process) { $r49Process.PeakWorkingSet64 } else { $null }
    remote_previous_replay_terminal = 'unknown'
}
$r49ErrorFile = Join-Path $r49Work ($r49Active.entry + '.stderr.log')
if (Test-Path -LiteralPath $r49ErrorFile) { $r49Status['stderr_bytes'] = (Get-Item -LiteralPath $r49ErrorFile).Length }
foreach ($r49ProgressName in @('GENERATION_PROGRESS.json', 'EVALUATION_PROGRESS.json', 'REPLAY_PROGRESS.json', 'LOCAL_PIPELINE_PROGRESS.json', 'LOCAL_PIPELINE_TERMINAL.json', 'LOCAL_RESOURCE_WAIT.json')) {
    $r49ProgressFile = Join-Path $r49Work $r49ProgressName
    if (Test-Path -LiteralPath $r49ProgressFile) {
        $r49Status[$r49ProgressName] = Get-Content -LiteralPath $r49ProgressFile -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
    }
}
$r49Status | ConvertTo-Json -Depth 7
