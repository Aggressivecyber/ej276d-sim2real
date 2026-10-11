"""Archive only this terminal owned run and verify all byte hashes on E."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,tarfile,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'collection_runtime');cfg=json.loads((S/'config.json').read_text('utf8'));remote=cfg['remote_root']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
code='from pathlib import Path\nimport os,json,hashlib,tarfile\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
native=json.loads((p/'train_full.py.NATIVE_RECEIPT.json').read_text())
assert native['actual_worker_native_exit']is not None
assert not Path('/proc/'+str(native['actual_worker_PID'])).exists()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb')as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
files=[f for f in sorted(p.iterdir())if f.is_file()and f.suffix not in {'.tar','.tmp'}and f.name!='OUTPUT_MANIFEST.json']
manifest=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f))for f in files]
(p/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=p/'R72_TERMINAL_OUTPUT.tar'
assert not archive.exists()
with tarfile.open(archive,'w')as tar:
 for f in files+[p/'OUTPUT_MANIFEST.json']:tar.add(f,arcname=f.name,recursive=False)
print(json.dumps(dict(remote_archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),native_exit=native['actual_worker_native_exit'],files=len(files))))
'''
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=60)
(P/'COLLECT_REMOTE.stdout.log').write_text(run.stdout,'utf8');(P/'COLLECT_REMOTE.stderr.log').write_text(run.stderr,'utf8')
assert run.returncode==0,run.stderr
receipt=json.loads(run.stdout);package=P/'R72_TERMINAL_OUTPUT.tar';assert not package.exists()
fetch=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+receipt['remote_archive'],str(package)],text=True,capture_output=True,timeout=300)
(P/'FETCH.stderr.log').write_text(fetch.stderr,'utf8');assert fetch.returncode==0,fetch.stderr
assert sha(package)==receipt['sha256']
with tarfile.open(package)as tar:
    for item in tar.getmembers():
        assert item.isfile()and'/'not in item.name and'\\'not in item.name
        target=S/item.name
        if target.exists():
            with tar.extractfile(item)as stream:assert hashlib.sha256(stream.read()).hexdigest()==sha(target),item.name
    tar.extractall(S,filter='data')
manifest=json.loads((S/'OUTPUT_MANIFEST.json').read_text('utf8'))
for item in manifest:assert sha(S/item['name'])==item['sha256'],item['name']
receipt.update(UTC=datetime.now(timezone.utc).isoformat(),local_archive=str(package),local_root=str(S),all_file_SHA_verified=True,guard={**guard,**memory()})
(P/'COLLECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
