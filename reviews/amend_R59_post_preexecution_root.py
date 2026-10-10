from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';C=json.loads((S/'config.json').read_text('utf8'))
record=R/'reviews/R59_POST_PREEXECUTION_001';record.mkdir(exist_ok=False)
old=json.loads((S/'POSTPROCESS_FILE_BINDING.json').read_text('utf8'))
names=['flow_diagnostics.py','numerical_check.py','POSTPROCESS_ROOT_REVIEW.json']
new={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in names}
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'];events=[]
prefix='from pathlib import Path\nimport json,hashlib,ast,shutil\np=Path('+repr(C['remote_root'])+')\nnames='+repr(names)+'\nold='+repr(old['bound_SHA256'])+'\nnew='+repr(new)+'\n'
def call(argv,text=None):
    p=subprocess.run(argv,input=text,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0,events[-1]
    return p.stdout
before=json.loads(call(ssh,prefix+'''
a=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text());assert a['entry']=='train_final.py'
assert not any((p/n).exists() for n in ['numerical_check.py.used','generate_all.py.used','evaluate_saved.py.used','replay_all.py.used','summarize_and_plot.py.used'])
assert not (p/'GENERATION_RECEIPT.json').exists()
for n,h in old.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
b=p/'postprocess_preexecution_v1';b.mkdir(exist_ok=False)
for n in names+['POSTPROCESS_FILE_BINDING.json']:shutil.copy2(p/n,b/n)
print(json.dumps({'prior_files':{n:(p/n).read_text() for n in names},'training_PID':a['actual_worker_PID'],'post_science_executed':False}))
'''))
for n,text in before['prior_files'].items():(record/n).write_text(text,'utf8')
(record/'PRIOR_BINDING.json').write_text(json.dumps(old,indent=2)+'\n','utf8')
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n) for n in names],'rylan@100.82.231.102:'+C['remote_root']+'/'])
updated=json.loads(call(ssh,prefix+'''
v=json.loads((p/'POSTPROCESS_FILE_BINDING.json').read_text())
for n,h in new.items():
 assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
 if n.endswith('.py'):ast.parse((p/n).read_text(),filename=n)
 v['bound_SHA256'][n]=h
v['preexecution_amendment']='Explicit covariance component label includes model projection; save existing TRAIN solver projection diagnostics. No numeric acceptance changes.'
(p/'POSTPROCESS_FILE_BINDING.json').write_text(json.dumps(v,indent=2)+'\\n')
print(json.dumps(v))
'''))
(S/'POSTPROCESS_FILE_BINDING.json').write_text(json.dumps(updated,indent=2)+'\n','utf8')
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','prior_hashes':{n:old['bound_SHA256'][n] for n in names},'new_hashes':new,
    'training_PID':before['training_PID'],'training_or_protocol_or_acceptance_changed':False,'new_scientific_execution':0,
    'reason':updated['preexecution_amendment'],'events':events}
(record/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'amended_before_any_post_execution':names,'training_PID':before['training_PID'],'acceptance_changed':False}))
