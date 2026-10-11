"""Save an immutable, read-only OS-level R65 runtime observation.

No scientific imports, arrays, model forwards, process signals, or job launch.
"""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
OUT = ROOT / 'reviews/R65_RUNTIME_OBSERVATIONS_001'
CFG = json.loads((STAGE / 'config.json').read_text('utf8'))
K = ctypes.WinDLL('kernel32', use_last_error=True)
K.GetCurrentProcess.restype = ctypes.c_void_p
K.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
K.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
HANDLE = K.GetCurrentProcess()
assert K.SetProcessAffinityMask(HANDLE, 8)
assert K.SetProcessWorkingSetSizeEx(HANDLE, 16 * 1024**2, 1024**3, 6)

CODE = 'from pathlib import Path\nROOT=Path(' + repr(CFG['remote_root']) + ')\n' + r'''
from datetime import datetime,timezone
import json,os,subprocess,time
def read_json(name):
    p=ROOT/name
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except json.JSONDecodeError: return {'concurrent_write_observed':True}
active=read_json('REMOTE_ACTIVE_JOB.json')
pid=active['actual_worker_PID']
proc=Path('/proc',str(pid))
assert proc.exists(),'Owned worker no longer present'
assert proc.joinpath('cwd').resolve()==ROOT.resolve(),'Owned worker cwd mismatch'
cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace')
assert 'train_final.py' in cmd,'Owned worker entry mismatch'
def sample():
    stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    status=proc.joinpath('status').read_text().splitlines()
    return {
      'UTC':datetime.now(timezone.utc).isoformat(),'monotonic_s':time.monotonic(),
      'utime_ticks':int(stat[11]),'stime_ticks':int(stat[12]),
      'process_start_ticks':int(stat[19]),'num_threads':int(stat[17]),
      'state':stat[0],
      'memory_bytes':{l.split(':')[0]:int(l.split()[1])*1024 for l in status if l.startswith(('VmRSS:','VmHWM:'))},
      'IO':{l.split(':')[0]:int(l.split(':')[1]) for l in proc.joinpath('io').read_text().splitlines()},
      'wchan':proc.joinpath('wchan').read_text(),
      'physical_progress':read_json('PHYSICAL_PROGRESS.json')}
a=sample()
time.sleep(2)
b=sample()
elapsed=b['monotonic_s']-a['monotonic_s']
cpu_seconds=((b['utime_ticks']+b['stime_ticks'])-(a['utime_ticks']+a['stime_ticks']))/os.sysconf('SC_CLK_TCK')
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
stdout=ROOT/'train_final.py.supervisor.stdout.log'
tail=''
if stdout.exists():
    with stdout.open('rb') as stream:
        stream.seek(0,2); size=stream.tell(); stream.seek(max(0,size-12000));tail=stream.read().decode('utf8','replace')
result={'read_only':True,'science_imports':False,'owned_PID':pid,'owned_cwd':str(ROOT),
 'owned_command':cmd,'CPU_affinity':sorted(os.sched_getaffinity(pid)),
 'clock_ticks_per_s':os.sysconf('SC_CLK_TCK'),'samples':[a,b],
 'sample_wall_s':elapsed,'process_CPU_s':cpu_seconds,'equivalent_busy_CPU_cores':cpu_seconds/elapsed,
 'loadavg':os.getloadavg(),'GPU_query':{'native_exit':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr},
 'supervisor_tail':tail,'training_failure':read_json('TRAINING_FAILURE.json'),
 'worker_native_receipt':read_json('train_final.py.NATIVE_RECEIPT.json')}
print(json.dumps(result,allow_nan=False))
'''
P = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes',
                    '-o', 'ConnectTimeout=8', 'rylan@100.82.231.102', '/usr/bin/python3 -B -'],
                   input=CODE, text=True, capture_output=True, timeout=30)
assert P.returncode == 0, P.stderr
VALUE = json.loads(P.stdout)
NOW = datetime.now(timezone.utc)
VALUE.update(observed_UTC=NOW.isoformat(), controller_PID=os.getpid(),
             controller_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             frozen_config_SHA256=hashlib.sha256((STAGE/'config.json').read_bytes()).hexdigest(),
             local_CPU_affinity_mask=8, local_working_set_cap_bytes=1024**3)
OUT.mkdir(parents=True, exist_ok=True)
DEST = OUT/(NOW.strftime('%Y%m%d_%H%M%S_%f')+'.json')
with DEST.open('xb') as stream:
    stream.write((json.dumps(VALUE, indent=2, allow_nan=False)+'\n').encode())
READBACK = json.loads(DEST.read_text('utf8'))
assert READBACK == VALUE
LAST = VALUE['samples'][-1]['physical_progress']
print(json.dumps({'file':str(DEST),'UTC':VALUE['observed_UTC'],
 'current_update':LAST.get('current_update'),'last_completed_update':LAST.get('last_completed_update'),
 'phase':LAST.get('phase'),'progress_updated_UTC':LAST.get('UTC'),
 'busy_CPU_cores_over_2s':VALUE['equivalent_busy_CPU_cores'],
 'memory_bytes':VALUE['samples'][-1]['memory_bytes'],'GPU_query':VALUE['GPU_query'],
 'training_failure':VALUE['training_failure'],'worker_native_receipt':VALUE['worker_native_receipt']},indent=2))
