param(
    [string]$Python = 'E:\GAN\runtime_envs\fresh_noise_cpu_20261007_103015\venv\Scripts\python.exe',
    [int]$TimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$prepDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($prepDir.Substring(0, 2).ToUpperInvariant() -notin @('E:', 'D:')) {
    throw 'All preparation files must stay on E:/D:.'
}
Set-Location -LiteralPath $prepDir
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Inspected CPU Python is unavailable.' }
$threadNames = @(
    'OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS','GOTO_NUM_THREADS','TBB_NUM_THREADS','RAYON_NUM_THREADS',
    'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'
)
foreach ($key in $threadNames) { [Environment]::SetEnvironmentVariable($key, '1', 'Process') }
$cacheNames = @(
    'TEMP','TMP','TMPDIR','XDG_CACHE_HOME','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR',
    'TRITON_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX'
)
$cacheRoot = Join-Path $prepDir 'runtime_cache'
[System.IO.Directory]::CreateDirectory($cacheRoot) | Out-Null
foreach ($key in $cacheNames) {
    $path = Join-Path $cacheRoot $key.ToLowerInvariant()
    [System.IO.Directory]::CreateDirectory($path) | Out-Null
    [Environment]::SetEnvironmentVariable($key, $path, 'Process')
}
foreach ($pair in @(
    @('PYTHONDONTWRITEBYTECODE','1'), @('PYTHONNOUSERSITE','1'),
    @('CUDA_VISIBLE_DEVICES','-1'), @('OMP_DYNAMIC','FALSE'),
    @('MKL_DYNAMIC','FALSE'), @('TORCHDYNAMO_DISABLE','1'),
    @('PYTHONPATH',$prepDir)
)) { [Environment]::SetEnvironmentVariable($pair[0], $pair[1], 'Process') }

$fixturePath = Join-Path $prepDir 'artificial_fixture.json'
$fixture = Get-Content -LiteralPath $fixturePath -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($binding in $fixture.immutable_text_inputs) {
    $actual = (Get-FileHash -LiteralPath $binding.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $binding.sha256) { throw "Immutable report mismatch before execution: $($binding.path)" }
}
$attemptIndex = 1
do {
    $attemptDir = Join-Path $prepDir ('attempt_{0:D3}' -f $attemptIndex)
    $attemptIndex++
} while (Test-Path -LiteralPath $attemptDir)
[System.IO.Directory]::CreateDirectory($attemptDir) | Out-Null
$stdoutPath = Join-Path $attemptDir 'stdout.txt'
$stderrPath = Join-Path $attemptDir 'stderr.txt'
$proofPath = Join-Path $attemptDir 'PROOF_ATTEMPT_RECEIPT.json'
$receiptPath = Join-Path $prepDir 'CODE_PREPARATION_RECEIPT.json'
$previousAttempts = @()
if (Test-Path -LiteralPath $receiptPath) {
    $previous = Get-Content -LiteralPath $receiptPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($null -ne $previous.attempts) { $previousAttempts = @($previous.attempts) }
}
$start = [DateTimeOffset]::UtcNow
$timer = [Diagnostics.Stopwatch]::StartNew()
$info = [Diagnostics.ProcessStartInfo]::new()
$info.FileName = $Python
$info.WorkingDirectory = $prepDir
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
$info.ArgumentList.Add('-B')
$info.ArgumentList.Add((Join-Path $prepDir 'artificial_vjp_proof.py'))
$info.ArgumentList.Add('--output-dir')
$info.ArgumentList.Add($attemptDir)
$process = [Diagnostics.Process]::new()
$process.StartInfo = $info
if (-not $process.Start()) { throw 'Python did not start.' }
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$observedPeak = [int64]0
$terminationReason = $null
while (-not $process.WaitForExit(250)) {
    $process.Refresh()
    $observedPeak = [Math]::Max($observedPeak, [int64]$process.PeakWorkingSet64)
    if ($observedPeak -ge 1073741824) { $terminationReason = 'parent_observer_peak_RSS_limit' }
    if ($timer.Elapsed.TotalSeconds -gt $TimeoutSeconds) { $terminationReason = 'bounded_artificial_proof_timeout' }
    if ($null -ne $terminationReason) {
        $process.Kill($true)
        $process.WaitForExit()
        break
    }
}
$exitCode = $process.ExitCode
try {
    $process.Refresh()
    $observedPeak = [Math]::Max($observedPeak, [int64]$process.PeakWorkingSet64)
} catch {}
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
$timer.Stop()
[IO.File]::WriteAllText($stdoutPath, $stdout, [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($stderrPath, $stderr, [Text.UTF8Encoding]::new($false))
$end = [DateTimeOffset]::UtcNow
function Bind-File([string]$Path) {
    [pscustomobject]@{
        path = [IO.Path]::GetFullPath($Path)
        bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$reportsUnchanged = $true
foreach ($binding in $fixture.immutable_text_inputs) {
    $actual = (Get-FileHash -LiteralPath $binding.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $binding.sha256) { $reportsUnchanged = $false }
}
if (Test-Path -LiteralPath $proofPath) {
    $receipt = Get-Content -LiteralPath $proofPath -Raw -Encoding utf8 | ConvertFrom-Json
} else {
    $receipt = [pscustomobject]@{
        schema_version=1;status='failed';artificial_only_code_preparation=$true
        proof_receipt_missing=$true;immutable_reports_byte_identical=$reportsUnchanged
        source_and_fixture_bindings=@(
            'weighted_empirical_w1.py','artificial_vjp_proof.py','resource_guard.py',
            'run_artificial.ps1','artificial_fixture.json'
        ) | ForEach-Object { Bind-File (Join-Path $prepDir $_) }
        output_bindings=@()
        limitations=@('The proof did not reach a validated terminal receipt; no passing claim is made.')
    }
}
$finalConsole = $null
try { $finalConsole = $stdout.Trim() | ConvertFrom-Json } catch {}
$finalChildPeak = if ($null -ne $finalConsole -and $null -ne $finalConsole.peak_RSS_bytes) {
    [int64]$finalConsole.peak_RSS_bytes
} else { [int64]0 }
$boundPeak = [Math]::Max($observedPeak, $finalChildPeak)
$success = $exitCode -eq 0 -and $receipt.status -eq 'passed' -and $reportsUnchanged -and
    $boundPeak -gt 0 -and $boundPeak -lt 1073741824 -and $null -eq $terminationReason
$receipt.status = if ($success) { 'passed' } else { 'failed' }
$launcher = [pscustomobject]@{
    start_utc=$start.ToString('o');end_utc=$end.ToString('o')
    elapsed_seconds=$timer.Elapsed.TotalSeconds
    python_executable=$Python;python_flag='-B'
    script=(Join-Path $prepDir 'artificial_vjp_proof.py')
    working_directory=$prepDir;output_directory=$attemptDir
    exit_code=$exitCode;timeout_seconds=$TimeoutSeconds;termination_reason=$terminationReason
    observer_peak_working_set_bytes=$observedPeak;final_child_peak_RSS_bytes=$finalChildPeak
    max_bound_peak_RSS_bytes=$boundPeak;peak_RSS_below_1_GiB=($boundPeak -gt 0 -and $boundPeak -lt 1073741824)
    original_reports_hashes_unchanged=$reportsUnchanged
    stdout=(Bind-File $stdoutPath);stderr=(Bind-File $stderrPath)
}
$receipt | Add-Member -NotePropertyName launcher -NotePropertyValue $launcher -Force
$attempt = [pscustomobject]@{
    directory=$attemptDir;status=$receipt.status;exit_code=$exitCode
    proof_receipt= $(if (Test-Path -LiteralPath $proofPath) { Bind-File $proofPath } else { $null })
    stdout=(Bind-File $stdoutPath);stderr=(Bind-File $stderrPath)
    resource_peak_bytes=$boundPeak;elapsed_seconds=$timer.Elapsed.TotalSeconds
}
$receipt | Add-Member -NotePropertyName attempts -NotePropertyValue @($previousAttempts + $attempt) -Force
$receipt | Add-Member -NotePropertyName receipt_scope -NotePropertyValue 'Artificial code preparation only; parent scientific protocol and R63 remain unchanged.' -Force
$receipt.output_bindings = @($receipt.output_bindings) + @((Bind-File $stdoutPath),(Bind-File $stderrPath))
if (Test-Path -LiteralPath $proofPath) { $receipt.output_bindings += (Bind-File $proofPath) }
[IO.File]::WriteAllText(
    $receiptPath, (($receipt | ConvertTo-Json -Depth 60) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
[pscustomobject]@{
    status=$receipt.status;receipt=$receiptPath;exit_code=$exitCode
    passed_groups=$receipt.passed_groups;test_groups=$receipt.test_groups
    peak_RSS_bytes=$boundPeak;elapsed_seconds=$timer.Elapsed.TotalSeconds
    reports_unchanged=$reportsUnchanged
    receipt_sha256=(Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json
if (-not $success) { exit 1 }
