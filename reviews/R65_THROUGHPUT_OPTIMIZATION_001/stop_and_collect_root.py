"""Invoke the reviewed original-attempt archive and collect exact JSON evidence."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import ctypes
import hashlib
import json
import shlex
import subprocess

P = Path(__file__).resolve().parent
k = ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
receipt_path = P/'REMOTE_CONTROL_LOCATION.json'
assert not receipt_path.exists(), 'Inspect prior invocation; never repeat the stop blindly'
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
remote = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ENGINEERING_UPGRADE_'+stamp
entry = P/'remote_stop_and_archive_original.py'
location = {'remote_directory':remote,'original_attempt':'37c43e76bb064c44937ebe888d58d0da',
            'collector_SHA256':hashlib.sha256(entry.read_bytes()).hexdigest(),'created_UTC':stamp}
receipt_path.write_text(json.dumps(location,indent=2)+'\n',encoding='utf8')
ssh = ['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
       'rylan@100.82.231.102','/usr/bin/python3 -B -']
create = subprocess.run(ssh,input='from pathlib import Path\nPath('+repr(remote)+').mkdir(exist_ok=False)\n',
                        text=True,capture_output=True,timeout=20)
assert create.returncode == 0,create.stderr
upload = subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(entry),
                         'rylan@100.82.231.102:'+remote+'/'+entry.name],text=True,capture_output=True,timeout=30)
assert upload.returncode == 0,upload.stderr
invoke = ssh[:-1]+['/usr/bin/python3 -B '+shlex.quote(remote+'/'+entry.name)]
run = subprocess.run(invoke,text=True,capture_output=True,timeout=150)
(P/'STOP_ARCHIVE_NATIVE.stdout.log').write_text(run.stdout,encoding='utf8')
(P/'STOP_ARCHIVE_NATIVE.stderr.log').write_text(run.stderr,encoding='utf8')
(P/'STOP_ARCHIVE_NATIVE_RECEIPT.json').write_text(json.dumps({'native_exit':run.returncode,
    'UTC':datetime.now(timezone.utc).isoformat(),'remote_directory':remote},indent=2)+'\n',encoding='utf8')
code = 'from pathlib import Path\nimport json,base64\np=Path('+repr(remote)+')\n' + '''
items={}
for q in list((p/'saved_evidence').rglob('*.json'))+[p/'STOP_AND_ARCHIVE_RECEIPT.json',p/'STOP_INTENT.json']:
    if q.is_file():
        assert q.stat().st_size<=16*1024**2
        items[q.relative_to(p).as_posix()]=base64.b64encode(q.read_bytes()).decode()
print(json.dumps(items))
'''
fetch = subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=30)
assert fetch.returncode == 0,fetch.stderr
data = json.loads(fetch.stdout)
for name,encoded in data.items():
    dest=P/'recovery_actual'/name
    assert dest.resolve().is_relative_to((P/'recovery_actual').resolve())
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(base64.b64decode(encoded))
print(json.dumps({'native_exit':run.returncode,'JSON_evidence_files_collected':len(data),
                  'remote_directory':remote}))
print(run.stdout[-4000:])
if run.stderr:
    print(run.stderr[-4000:])
raise SystemExit(run.returncode)
