from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,os
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
if len(sys.argv)>1:
    names=sys.argv[1:]
else:
    names=['NOISE_CALIBRATION.json','NOISE_CALIBRATION.npz','FORMULA_CHECK.json','calibrate.py.NATIVE_RECEIPT.json',
           'calibrate.py.WORKER_RECEIPT.json','calibrate.py.SCIENCE_PID.json','calibrate.py.RSS_ACK.json',
           'calibrate.py.stdout.log','calibrate.py.stderr.log','REMOTE_RUN_HISTORY.json']
ops=[]
for n in names:
    if '..' in Path(n).parts:raise ValueError(n)
    dest=R/n;dest.parent.mkdir(exist_ok=True,parents=True)
    q=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+C['remote_root']+'/'+n,str(dest)],capture_output=True,text=True,timeout=60)
    ops.append({'file':n,'exit':q.returncode,'stderr':q.stderr,'bytes':dest.stat().st_size if dest.exists()else None})
    if q.returncode:break
h=R/'TRANSFER_HISTORY.json';history=json.loads(h.read_text('utf8'))if h.exists()else[]
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'ordinary_transfer_no_hash_test':True,'files':ops}
history.append(rec);h.write_bytes((json.dumps(history,indent=2)+'\n').encode())
print(json.dumps(rec),flush=True)
raise SystemExit(0 if all(v['exit']==0 for v in ops)else 1)
