"""Collect a terminal R78 run, preserving all scientific outputs and failures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import tarfile

P = Path(__file__).resolve().parent
R = P.parents[1]
S = R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P/'collection_runtime')
C = json.loads((S/'config.json').read_text('utf8'))
remote = C['remote_root']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


assert not (P/'COLLECTION_RECEIPT.json').exists()
code = 'from pathlib import Path\nimport os,json,hashlib,tarfile\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
completion=json.loads((p/'PIPELINE_COMPLETION.json').read_text())
assert not Path('/proc/'+str(completion['shell_PID'])).exists(),'pipeline shell still present'
for entry,row in completion['scientific_native_receipts'].items():
 assert row['actual_worker_native_exit']is not None
 assert not Path('/proc/'+str(row['actual_worker_PID'])).exists(),entry
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
files=[]
for path in sorted(p.rglob('*')):
 relative=path.relative_to(p)
 if 'cache'in relative.parts or '__pycache__'in relative.parts:continue
 assert not path.is_symlink(),str(relative)
 if path.is_file()and path.suffix not in{'.tar','.tmp'}and relative.as_posix()!='OUTPUT_MANIFEST.json':files.append(path)
manifest=[dict(name=path.relative_to(p).as_posix(),bytes=path.stat().st_size,sha256=sha(path))for path in files]
(p/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=p/'R78_TERMINAL_OUTPUT.tar'
assert not archive.exists(),'preserve existing archive; recover collection instead of recreating'
with tarfile.open(archive,'w')as tar:
 for path in files+[p/'OUTPUT_MANIFEST.json']:tar.add(path,arcname=path.relative_to(p).as_posix(),recursive=False)
print(json.dumps(dict(remote_archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
 files=len(files),completion_status=completion['status'],
 scientific_native_exits={k:v['actual_worker_native_exit']for k,v in completion['scientific_native_receipts'].items()},
 all_owned_worker_and_shell_processes_absent=True,pipeline_shell_native_exit='unknown')))
'''
proc = subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,
    capture_output=True,encoding='utf8',timeout=60)
for kind,value in [('stdout',proc.stdout),('stderr',proc.stderr)]:
    (P/('COLLECT_REMOTE.'+kind+'.log')).write_text(value,'utf8')
assert proc.returncode == 0,proc.stderr
receipt = json.loads(proc.stdout)
package = P/'R78_TERMINAL_OUTPUT.tar'
assert not package.exists()
fetch = subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'rylan@100.82.231.102:'+receipt['remote_archive'],str(package)],
    text=True,capture_output=True,timeout=300)
(P/'FETCH.stderr.log').write_text(fetch.stderr,'utf8')
assert fetch.returncode == 0,fetch.stderr
assert package.stat().st_size == receipt['bytes'] and sha(package) == receipt['sha256']
with tarfile.open(package)as tar:
    for item in tar.getmembers():
        relative = Path(item.name)
        assert item.isfile() and not relative.is_absolute() and '..'not in relative.parts and '\\'not in item.name
        target = S/relative
        assert target.resolve().is_relative_to(S.resolve())
        if target.exists():
            h = hashlib.sha256()
            with tar.extractfile(item)as f:
                for block in iter(lambda:f.read(1048576),b''):
                    h.update(block)
            assert h.hexdigest() == sha(target),item.name
    tar.extractall(S,filter='data')
manifest = json.loads((S/'OUTPUT_MANIFEST.json').read_text('utf8'))
for row in manifest:
    assert (S/row['name']).stat().st_size == row['bytes'] and sha(S/row['name']) == row['sha256'],row['name']
receipt.update(UTC=datetime.now(timezone.utc).isoformat(),local_archive=str(package),local_root=str(S),
    all_file_SHA_verified=True,guard={**guard,**memory()})
(P/'COLLECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
