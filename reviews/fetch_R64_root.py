"""Normal private artifact transfer after actual native completion, no migration test."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import json
import os
import subprocess

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
C = json.loads((S/'config.json').read_text('utf8'))
remote, host = C['remote_root'], 'rylan@100.82.231.102'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
mask, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system))
allowed = [i for i in range(64) if mask.value & (1 << i)]
chosen = allowed[min(3, len(allowed)-1)]
assert kernel.SetProcessAffinityMask(handle, 1 << chosen)
events, copied = [], []


def call(argv, code=None, timeout=None):
    p = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=timeout)
    events.append(dict(argv=argv, native_exit=p.returncode, stderr=p.stderr))
    assert p.returncode == 0, events[-1]
    return p.stdout


code = 'from pathlib import Path\nimport json\np=Path('+repr(remote)+')\n'+'''
a=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert a['entry']=='diagnose_train.py'and a.get('actual_worker_native_exit')==0 and a['termination_reason']is None
assert not Path('/proc',str(a['actual_worker_PID'])).exists()
n=json.loads((p/'diagnose_train.py.NATIVE_RECEIPT.json').read_text())
assert n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0
assert not(p/'DIAGNOSTIC_FAILURE.json').exists()
progress=json.loads((p/'DIAGNOSTIC_PROGRESS.json').read_text())
assert progress['phase']=='registered_diagnostic_endpoint_complete'
result=json.loads((p/'RESULT.json').read_text())
assert len(result['figures'])==4
records=[]
for f in sorted(p.rglob('*')):
    rel=f.relative_to(p)
    if f.is_file()and'cache'not in rel.parts and'__pycache__'not in rel.parts and f.suffix!='.tar':
        records.append({'name':str(rel),'bytes':f.stat().st_size})
print(json.dumps({'files':records,'actual_PID_gone':True,'native_exit':0,'final_phase':progress['phase']}))
'''
remote_state = json.loads(call(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                              host, '/usr/bin/python3 -B -'], code, timeout=45))
bound = json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
protected = {row['name'] for row in bound['files']} | {'ROOT_IMPLEMENTATION_REVIEW.json'}
selected = [row['name'] for row in remote_state['files'] if row['name'] not in protected or not (S/row['name']).exists()]
parents = sorted(set(str(Path(name).parent) for name in selected))
for parent in parents:
    dest = S/parent
    dest.mkdir(parents=True, exist_ok=True)
    names = [name for name in selected if str(Path(name).parent) == parent]
    assert all('..' not in Path(name).parts and not Path(name).is_absolute() for name in names)
    print(json.dumps(dict(copying_parent=parent, files=len(names))), flush=True)
    call(['scp', '-q', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
          *[host+':'+remote+'/'+name for name in names], str(dest)+'/'])
    copied.extend(names)
    progress = dict(UTC=datetime.now(timezone.utc).isoformat(), control_PID=os.getpid(), CPU_affinity=[chosen],
                    copied=copied, expected_files=len(selected), events=events, normal_transfer=True, migration_consistency_test=False)
    (S/'FETCH_PROGRESS.json').write_bytes((json.dumps(progress, indent=2)+'\n').encode())
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), control_PID=os.getpid(), CPU_affinity=[chosen],
               remote_completion=remote_state, copied=copied, events=events, normal_transfer=True,
               migration_consistency_test=False, byte_hash_copy_comparison_performed=False,
               local_copied_bytes=sum((S/name).stat().st_size for name in copied))
(S/'FETCH_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2)+'\n').encode())
print(json.dumps(dict(files_copied=len(copied), bytes=receipt['local_copied_bytes'], native_exit=0)))
