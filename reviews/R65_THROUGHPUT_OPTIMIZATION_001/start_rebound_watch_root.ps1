$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\GAN\research\sim2real_two_papers_20261008_0305'
$stageRoot = Join-Path $projectRoot 'work\R65_FULL_EMPIRICAL_D_RESPONSE_001'
$controlRoot = Join-Path $stageRoot 'controller_runtime'
$packageRoot = Join-Path $projectRoot 'reviews\R65_THROUGHPUT_OPTIMIZATION_001\new_controller_package'
$closure = Get-Content -LiteralPath (Join-Path $projectRoot 'reviews\R65_THROUGHPUT_OPTIMIZATION_001\ROOT_CONTROLLER_REBIND_REVIEW.json') -Raw | ConvertFrom-Json
if ($closure.status -ne 'ROOT_CONTROLLER_REBIND_REVIEW_COMPLETE') { throw 'Root preparation review is incomplete.' }
foreach ($name in @('ROOT_WATCHER_LAUNCH.json', 'PIPELINE_WATCH_STARTED.json', 'PIPELINE_STOP_REQUEST.json')) {
    if (Test-Path -LiteralPath (Join-Path $controlRoot $name)) { throw "Existing controller record requires inspection: $name" }
}
$entry = Join-Path $packageRoot 'watch_R65_pipeline_root.py'
$entryHash = (Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant()
if ($entryHash -ne '45895134eb3b6cf34a1604e9dc8ac5e84ded8c1f699b90ae12929ffec7820469') { throw 'Reviewed watcher source has changed.' }
$runtimeTemp = Join-Path $controlRoot 'launcher_tmp'
New-Item -ItemType Directory -Force -Path $runtimeTemp | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp
$env:PYTHONDONTWRITEBYTECODE = '1'
$stdoutPath = Join-Path $controlRoot 'ROOT_WATCHER.stdout.log'
$stderrPath = Join-Path $controlRoot 'ROOT_WATCHER.stderr.log'
if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath)) { throw 'Prior watcher logs must be preserved and inspected.' }
$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
$watchProcess = Start-Process -FilePath $pythonExecutable -ArgumentList @('-B', $entry) -WorkingDirectory $packageRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Start-Sleep -Seconds 1
$watchProcess.Refresh()
$record = [ordered]@{
    UTC = [DateTimeOffset]::UtcNow.ToString('o')
    PID = $watchProcess.Id
    process_running_at_receipt = -not $watchProcess.HasExited
    entry = $entry
    source_SHA256 = $entryHash
    executable = $pythonExecutable
    stdout = $stdoutPath
    stderr = $stderrPath
    intended_CPU_count = 1
    intended_process_commit_and_RSS_limit_bytes = 1073741824
    resource_enforcement = 'controller ResourceOwner; actual live affinity is checked separately'
    agents_idle_before_launch = $true
    scientific_job_submitted_by_this_launcher = $false
    training_or_recovery_launch = $false
    automatic_failed_attempt_retry = $false
    final_model_complete = $false
}
if ($watchProcess.HasExited) { $record['native_exit'] = $watchProcess.ExitCode }
$destination = Join-Path $controlRoot 'ROOT_WATCHER_LAUNCH.json'
[IO.File]::WriteAllText($destination, ($record | ConvertTo-Json -Depth 5) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
$record | ConvertTo-Json -Depth 5
if ($watchProcess.HasExited) { Get-Content -LiteralPath $stderrPath -Raw; throw 'Watcher exited; inspect retained evidence before any retry.' }
