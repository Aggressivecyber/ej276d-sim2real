from pathlib import Path
import ast,json,hashlib,os,datetime
ROOT=Path(__file__).resolve().parent
names=['config.json','PROTOCOL.md','model.py','legacy_model.py','online_training.py','train_gpu.py','reorder_core.py','fine_core.py','prior.py','supervise.py','owned_worker.py']
bindings={}
for n in names:
    p=ROOT/n;b=p.read_bytes()
    if p.suffix=='.py':ast.parse(b.decode('utf8'))
    bindings[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(ROOT/'PRETRAIN_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'training_updates_before_freeze':0,'old_response_training':0},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST_pass':True,'bindings':len(bindings)}))
