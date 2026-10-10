from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,ast,os
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text())
rows=[]
for n in ['linux_supervise.py','prepare_stage.py']:
    d=(R/n).read_bytes();ast.parse(d.decode('utf8'));rows.append({'name':n,'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()})
q=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=8',str(R/'linux_supervise.py'),
                  'rylan@100.82.231.102:'+C['remote_root']+'/linux_supervise.py'],capture_output=True,text=True,timeout=30)
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'native_exit':q.returncode,'source_bindings':rows,
     'changes':'R48 startup56GiB RAM admission is strict; old lower-quota branch removed',
     'science_before_correction':0,'original_PRETRAIN_FREEZE_retained':True,'migration_test':False,'stderr':q.stderr}
(R/'PRETRAIN_RESOURCE_CORRECTION.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps(rec),flush=True)
raise SystemExit(q.returncode)
