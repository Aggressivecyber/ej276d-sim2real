"""Preserve the failed attempt and deploy a reviewed serialization-only repair."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';A=S/'repairs/AUDIT_JSON_001'
C=json.loads((S/'config.json').read_text('utf8'));review=json.loads((A/'REPAIR_REVIEW.json').read_text('utf8'))
assert review['pass'] and review['all_audit_values_equal_before_after'] and review['all_28_checks_pass']
host='rylan@100.82.231.102';remote=C['remote_root'];events=[]
def call(args,code=None):
    p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=120)
    events.append({'argv':args,'native_exit':p.returncode,'stderr':p.stderr});assert p.returncode==0,events[-1]
    return p.stdout
def ssh(code):return call(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],code)
prefix='from pathlib import Path\nimport json,hashlib,shutil,ast\np=Path('+repr(remote)+')\na=p/"repairs/AUDIT_JSON_001"\nreview='+repr(review)+'\n'
observed=json.loads(ssh(prefix+'''
failed=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert failed['entry']=='numerical_check.py' and failed['actual_worker_native_exit']==1
assert 'bool_' in failed['worker_self_receipt']['error'] and failed['GPU_process_memory_peak_bytes']==0
for pid in [failed['actual_worker_PID'],failed['supervisor_PID']]:assert not Path('/proc',str(pid)).exists()
assert not (p/'TRAIN_SOLVER_CHECK.json').exists() and not (p/'GENERATION_RECEIPT.json').exists()
assert hashlib.sha256((p/'post_audit.py').read_bytes()).hexdigest()==review['old_post_audit_SHA256']
assert hashlib.sha256((p/'PRECISION_RESOLVED_RESPONSE/FINAL_FLOW.pt').read_bytes()).hexdigest()==review['frozen_weight_SHA256']
binding=json.loads((p/'POSTPROCESS_FILE_BINDING.json').read_text())['bound_SHA256']
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in binding.items())
a.mkdir(parents=True,exist_ok=False)
names=['post_audit.py','POSTPROCESS_FILE_BINDING.json','REMOTE_ACTIVE_JOB.json','REMOTE_RUN_HISTORY.json']+[f.name for f in p.glob('numerical_check.py.*')if f.is_file()]
for n in sorted(set(names)):shutil.copy2(p/n,a/n)
marker=p/'LAUNCH_numerical_check.py.json';marker.rename(a/marker.name)
print(json.dumps({'failed_native':failed,'original_binding_archived':True,'failed_processes_absent':True,'no_numeric_forward_or_DEV_output':True}))
'''))
for source,dest in [(S/'post_audit.py',remote+'/post_audit.py'),(S/'POSTPROCESS_FILE_BINDING.json',remote+'/POSTPROCESS_FILE_BINDING.json'),(A/'REPAIR_REVIEW.json',remote+'/repairs/AUDIT_JSON_001/REPAIR_REVIEW.json')]:
    call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(source),host+':'+dest])
verified=json.loads(ssh(prefix+'''
binding=json.loads((p/'POSTPROCESS_FILE_BINDING.json').read_text())
old=json.loads((a/'POSTPROCESS_FILE_BINDING.json').read_text())
changes=[n for n,h in binding['bound_SHA256'].items()if old['bound_SHA256'][n]!=h]
assert changes==['post_audit.py']
for n,h in binding['bound_SHA256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
ast.parse((p/'post_audit.py').read_text())
assert hashlib.sha256((p/'PRECISION_RESOLVED_RESPONSE/FINAL_FLOW.pt').read_bytes()).hexdigest()==review['frozen_weight_SHA256']
print(json.dumps({'changed_bound_files':changes,'all_31_file_hashes_verified':True,'frozen_weight_unchanged':True,'AST_pass':True}))
'''))
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'observed':observed,'verified':verified,'events':events,'new_scientific_execution':0,'next_action':'one reviewed retry of the not-yet-executed numerical calculation, then the original post sequence'}
(A/'DEPLOY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'UTC':receipt['UTC'],**verified},indent=2))
