"""Pause only the identified R65 worker, run one artificial check, always resume.

This entry changes no frozen file, model, optimizer, random stream or checkpoint.
It never restarts a training job. The benchmark is killed on time/resource excess.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC')
OUT = Path(__file__).resolve().parent
MAX_SECONDS = 180
os.sched_setaffinity(0, {11})

def utc():
    return datetime.now(timezone.utc).isoformat()

def write(name, value):
    temp = OUT/(name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(OUT/name)

def read(name):
    return json.loads((ROOT/name).read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def identity(pid):
    p = Path('/proc')/str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return {'PID':pid, 'start_ticks':int(fields[19]),
            'cwd':str((p/'cwd').resolve()), 'command':(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
            'state':fields[0]}

def same_identity(before, after):
    return all(before[k] == after[k] for k in ['PID', 'start_ticks', 'cwd', 'command'])

def gpu_rows():
    run = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                          '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)
    assert run.returncode == 0, run.stderr
    return [{'PID':int(v.split(',')[0]), 'bytes':int(v.split(',')[1])*1024**2}
            for v in run.stdout.splitlines() if v.strip()]

def process_rss(pid):
    try:
        return next(int(s.split()[1])*1024 for s in (Path('/proc')/str(pid)/'status').read_text().splitlines()
                    if s.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        return 0

def interrupted(signum, frame):
    raise RuntimeError('Guard interrupted by signal '+str(signum))

assert not (OUT/'GUARD_RECEIPT.json').exists(), 'No duplicate attempt'
manifest = json.loads((OUT/'BENCHMARK_REGISTRATION.json').read_text())
assert manifest['scientific_data_reads'] == 0 and manifest['scientific_updates'] == 0
assert manifest['maximum_benchmark_seconds'] == MAX_SECONDS
for row in manifest['files']:
    assert digest(OUT/row['name']) == row['sha256'], row['name']
review = read('ROOT_IMPLEMENTATION_REVIEW.json')
for row in review['files']:
    assert digest(ROOT/row['name']) == row['sha256'], row['name']
active = read('REMOTE_ACTIVE_JOB.json')
assert active['entry'] == 'train_final.py' and active.get('actual_worker_native_exit') is None
handshake = read('train_final.py.SCIENCE_PID.json')
pid = active['actual_worker_PID']
assert pid == handshake['PID'] == 291399
assert handshake['nonce'] == '37c43e76bb064c44937ebe888d58d0da'
before = identity(pid)
assert before['cwd'] == str(ROOT) and 'linux_owned_worker.py' in before['command']
assert before['state'] not in ['T', 't', 'Z', 'X']
gpus = gpu_rows()
assert {row['PID'] for row in gpus} == {pid}, gpus
for signum in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]:
    signal.signal(signum, interrupted)
record = {'started_UTC':utc(), 'training_identity_before':before, 'original_frozen_file_count':len(review['files']),
          'scientific_data_reads':0, 'scientific_updates':0, 'training_restarted':False,
          'model_or_optimizer_changed':False, 'GPU_before':gpus}
child = None
paused = False
try:
    os.kill(pid, signal.SIGSTOP)
    paused = True
    deadline = time.monotonic()+5
    while identity(pid)['state'] not in ['T', 't']:
        assert time.monotonic() < deadline, 'Could not confirm stopped worker'
        time.sleep(.05)
    record['training_paused_UTC'] = utc()
    write('GUARD_STARTED.json', record)
    time.sleep(.5)
    config = read('config.json')
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '12'
    for name in ['tmp','cuda','torch','matplotlib']:
        (OUT/'cache'/name).mkdir(parents=True, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               TMPDIR=str(OUT/'cache/tmp'), TMP=str(OUT/'cache/tmp'), TEMP=str(OUT/'cache/tmp'),
               CUDA_CACHE_PATH=str(OUT/'cache/cuda'), TORCH_HOME=str(OUT/'cache/torch'),
               MPLCONFIGDIR=str(OUT/'cache/matplotlib'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [config['server_python'], '-B', str(OUT/manifest['entry']), *manifest['arguments']]
    record['benchmark_command'] = command
    started = time.monotonic()
    with (OUT/'BENCHMARK.stdout.log').open('wb') as out, (OUT/'BENCHMARK.stderr.log').open('wb') as err:
        child = subprocess.Popen(command, cwd=OUT, env=env, stdout=out, stderr=err,
                                 start_new_session=True, preexec_fn=lambda:os.sched_setaffinity(0, set(range(12))))
        record['benchmark_PID'] = child.pid
        record['combined_GPU_peak_bytes'] = 0
        record['combined_RSS_peak_bytes'] = 0
        while child.poll() is None:
            assert identity(pid)['state'] in ['T','t'], 'Training unexpectedly resumed during benchmark'
            assert time.monotonic()-started <= MAX_SECONDS, 'Artificial benchmark time limit'
            rows = gpu_rows()
            assert {r['PID'] for r in rows}.issubset({pid,child.pid}), 'Unexpected GPU process'
            used = sum(r['bytes'] for r in rows)
            rss = process_rss(pid)+process_rss(child.pid)
            record['combined_GPU_peak_bytes'] = max(record['combined_GPU_peak_bytes'], used)
            record['combined_RSS_peak_bytes'] = max(record['combined_RSS_peak_bytes'], rss)
            assert used <= 10*1024**3 and rss <= 48*1024**3, 'Combined resource limit'
            time.sleep(.5)
        record['benchmark_native_exit'] = child.returncode
        record['benchmark_elapsed_s'] = time.monotonic()-started
    assert child.returncode == 0, 'Artificial benchmark failed; retain original implementation'
    record['benchmark_status'] = 'completed'
except BaseException as exc:
    record['benchmark_status'] = 'failed'
    record['error'] = repr(exc)
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        record['benchmark_native_exit'] = child.returncode
    if paused:
        current = identity(pid)
        assert same_identity(before,current), 'Worker identity changed; do not signal a different process'
        os.kill(pid, signal.SIGCONT)
        deadline = time.monotonic()+5
        while identity(pid)['state'] in ['T','t']:
            assert time.monotonic() < deadline, 'Training resume confirmation timed out'
            time.sleep(.05)
        record['training_resumed_UTC'] = utc()
        record['training_identity_after'] = identity(pid)
        record['training_resumed'] = True
    for row in review['files']:
        assert digest(ROOT/row['name']) == row['sha256'], row['name']
    record['original_frozen_files_unchanged'] = True
    record['finished_UTC'] = utc()
    write('GUARD_RECEIPT.json', record)
    print(json.dumps(record), flush=True)
raise SystemExit(0 if record.get('benchmark_status') == 'completed' and record.get('training_resumed') else 1)
