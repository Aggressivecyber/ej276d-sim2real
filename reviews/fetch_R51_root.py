from pathlib import Path
from datetime import datetime,timezone
import subprocess,json
R=Path(__file__).resolve().parents[1];S=R/'work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001'
C=json.loads((S/'config.json').read_text());host='rylan@100.82.231.102';remote=C['remote_root'];events=[]
def call(args,code=None):
 p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=60)
 events.append({'argv':args,'native_exit':p.returncode,'stderr':p.stderr})
 if p.returncode:raise RuntimeError(events[-1])
 return p.stdout
code='from pathlib import Path\nimport json\np=Path('+repr(remote)+')\nn=json.loads((p/"numeric_worker.py.NATIVE_RECEIPT.json").read_text());assert n["actual_worker_native_exit"]==0 and n["termination_reason"] is None\nprint(json.dumps([str(f.relative_to(p)) for f in sorted(p.rglob("*")) if f.is_file() and "cache" not in f.relative_to(p).parts]))\n'
files=json.loads(call(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],code))
selected=[n for n in files if not(S/n).exists()]
for parent in sorted(set(str(Path(n).parent) for n in selected)):
 dest=S/parent;dest.mkdir(parents=True,exist_ok=True)
 names=[n for n in selected if str(Path(n).parent)==parent]
 assert all('..' not in Path(n).parts and not Path(n).is_absolute() for n in names)
 call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[host+':'+remote+'/'+n for n in names],str(dest)+'/' ])
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'events':events,'copied':selected,'normal_transfer':True,'migration_consistency_test':False}
(S/'FETCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'files':selected,'bytes':sum((S/n).stat().st_size for n in selected)},indent=2))
