"""Root-owned transport and invocation of an explicitly reviewed artificial benchmark."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
STAGE = PROJECT/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
cfg = json.loads((STAGE/'config.json').read_text('utf8'))
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', required=True)
args = parser.parse_args()
assert args.attempt.startswith('attempt_') and args.attempt[8:].isdigit()
out = ROOT/args.attempt
manifest_path = out/'BENCHMARK_REGISTRATION.json'
manifest = json.loads(manifest_path.read_text('utf8'))
assert manifest['root_reviewed'] is True
assert not (out/'LOCAL_INVOCATION_RECEIPT.json').exists()
remote = cfg['remote_root']+'/engineering_benchmarks/'+args.attempt
assert manifest['remote_directory'] == remote
names = [row['name'] for row in manifest['files']]+['BENCHMARK_REGISTRATION.json']
bundle = out/'BENCHMARK_BUNDLE.tar'
with tarfile.open(bundle, 'x') as tar:
    for name in names:
        p = out/name
        assert p.parent == out and p.is_file(), name
        if name != 'BENCHMARK_REGISTRATION.json':
            want = next(row['sha256'] for row in manifest['files'] if row['name'] == name)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == want, name
        tar.add(p, arcname=name)
ssh = ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
       'rylan@100.82.231.102', '/usr/bin/python3 -B -']

def call(command, code=None, timeout=30):
    run = subprocess.run(command, input=code, text=True, capture_output=True, timeout=timeout)
    if run.returncode:
        raise RuntimeError({'native_exit':run.returncode, 'stderr':run.stderr[-4000:], 'stdout':run.stdout[-4000:]})
    return run.stdout

create = 'from pathlib import Path\np=Path('+repr(remote)+')\np.mkdir(parents=True,exist_ok=False)\n'
call(ssh, create)
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(bundle),
      'rylan@100.82.231.102:'+remote+'/BENCHMARK_BUNDLE.tar'])
extract = 'from pathlib import Path\nimport tarfile\np=Path('+repr(remote)+')\n' + '''
with tarfile.open(p/'BENCHMARK_BUNDLE.tar') as t:
    assert all(Path(m.name).name==m.name and m.isfile() for m in t.getmembers())
    t.extractall(p,filter='data')
'''
call(ssh, extract)
start = datetime.now(timezone.utc).isoformat()
invoke = ['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
          'rylan@100.82.231.102','/usr/bin/python3 -B '+shlex.quote(remote+'/remote_artificial_benchmark_guard.py')]
run = subprocess.run(invoke, text=True, capture_output=True, timeout=220)
(out/'REMOTE_GUARD.stdout.log').write_text(run.stdout, encoding='utf8')
(out/'REMOTE_GUARD.stderr.log').write_text(run.stderr, encoding='utf8')
receipt = {'started_UTC':start, 'finished_UTC':datetime.now(timezone.utc).isoformat(),
           'native_exit':run.returncode, 'remote_directory':remote,
           'science_inputs_decoded':False, 'training_restart':False}
(out/'LOCAL_INVOCATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
fetch = 'from pathlib import Path\nimport json\np=Path('+repr(remote)+')\nnames='+repr(manifest['result_files'])+'\n' + '''
result={}
for name in names:
    v=p/name
    if v.is_file():
        assert v.stat().st_size<2*1024**2
        result[name]=v.read_text()
print(json.dumps(result))
'''
results = json.loads(call(ssh, fetch))
for name, value in results.items():
    assert Path(name).name == name
    (out/name).write_text(value, encoding='utf8')
print(json.dumps(receipt))
print(run.stdout[-9000:])
if run.stderr:
    print(run.stderr[-4000:])
raise SystemExit(run.returncode)
