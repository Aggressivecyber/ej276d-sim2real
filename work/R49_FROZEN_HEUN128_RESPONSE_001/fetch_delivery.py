from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,base64,time,sys
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text())
SSH=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code="from pathlib import Path;import json;r=Path("+repr(C['remote_root'])+");print(json.dumps([str(p.relative_to(r))for p in r.rglob('*')if p.is_file()and 'cache'not in p.relative_to(r).parts]))"
encoded=base64.b64encode(code.encode()).decode()
q=subprocess.run(SSH+[C['server_python']+' -B -c '+"\"import base64;exec(base64.b64decode('"+encoded+"'))\""],capture_output=True,text=True)
if q.returncode:raise RuntimeError(('ordinary file listing failed',q.stderr))
names=json.loads(q.stdout)
skip={p.name for p in R.glob('*.py')}|{'config.json','config_cpu.json','PROTOCOL.md','SOURCE_BINDINGS.json','NOISE_CALIBRATION.json','NOISE_CALIBRATION.npz'}
chosen=[]
for name in names:
    rel=Path(name)
    if rel.is_absolute()or'..'in rel.parts:raise ValueError('unsafe returned relative path')
    if name in skip:continue
    if (R/rel).exists():continue
    (R/rel).parent.mkdir(exist_ok=True,parents=True);chosen.append(name)
lines=['get '+json.dumps(C['remote_root']+'/'+n)+' '+json.dumps(str(R/n).replace('\\','/'))for n in chosen]
batch=R/'DELIVERY_TRANSFER_BATCH.txt';batch.write_bytes(('\n'.join(lines)+'\n').encode())
start=time.monotonic()
with(R/'delivery_sftp.stdout.log').open('wb')as out,(R/'delivery_sftp.stderr.log').open('wb')as err:
    q=subprocess.run(['sftp','-b',str(batch),'-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102'],stdout=out,stderr=err,cwd=R)
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'native_exit':q.returncode,
     'files_requested':len(chosen),'requested_paths':chosen,'elapsed_s':time.monotonic()-start,
     'normal_transfer_no_equivalence_test':True,'historical_arrays_models_copied':False,
     'new_files_returned_bytes':sum((R/n).stat().st_size for n in chosen if(R/n).exists())}
(R/'DELIVERY_TRANSFER_RECEIPT.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in rec.items()if k!='requested_paths'}),flush=True)
raise SystemExit(q.returncode)
