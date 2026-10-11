$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Qualifier $root) -notin @('E:','D:')) { throw 'E/D output root required' }
$python='E:\GAN\runtime_envs\fresh_noise_cpu_20261007_103015\venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
$env:TEMP=Join-Path $root 'cache\tmp'
$env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $root 'cache\torch'
$env:MPLCONFIGDIR=Join-Path $root 'cache\matplotlib'
$env:XDG_CACHE_HOME=Join-Path $root 'cache\xdg'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$binding=Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root 'FROZEN_TOLERANCES.json')
$started=[DateTimeOffset]::UtcNow.ToString('o')
$process=Start-Process -FilePath $python -ArgumentList @('-B',(Join-Path $root 'artificial_fixture.py')) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root 'fixture.stdout.log') -RedirectStandardError (Join-Path $root 'fixture.stderr.log') -Wait -PassThru
$receipt=@{start_UTC=$started;end_UTC=[DateTimeOffset]::UtcNow.ToString('o');PID=$process.Id;native_exit_code=$process.ExitCode;python=$python;arguments=@('-B','artificial_fixture.py');tolerance_sha256_before_launch=$binding.Hash.ToLower();stdout='fixture.stdout.log';stderr='fixture.stderr.log';science_registered=$false;science_launched=$false}
$receipt|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $root 'NATIVE_EXIT_RECEIPT.json') -Encoding UTF8
$receipt|ConvertTo-Json -Depth 5
exit $process.ExitCode
