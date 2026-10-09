from pathlib import Path
import ast,hashlib,json,os,datetime
ROOT=Path(__file__).resolve().parent
names=['config.json','PROTOCOL.md','train_gpu.py','online_training.py','model.py','legacy_model.py','fine_core.py','prior.py','supervise.py','owned_worker.py','prepare_extra_D.py']
rows=[]
for n in names:
    b=(ROOT/n).read_bytes()
    if n.endswith('.py'):ast.parse(b.decode('utf8'))
    rows.append({'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
record={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_first_training_update':True,'arm':'D5','source_bindings':rows,'old_R23_unchanged':True,'migration_consistency_test':False,'actual_execution_from_config':json.loads((ROOT/'config.json').read_text('utf8'))['execution']}
(ROOT/'PRETRAIN_FROZEN_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':record['UTC'],'training_sources_AST_pass':True,'current_bytes':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file()),'scientific_started':False}))
