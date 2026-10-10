from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os,ast
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';A=S/'repairs/ADC_CHECK_001'
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
C=read(S/'config.json');old=read(A/'ROOT_IMPLEMENTATION_REVIEW.json');failed=read(A/'diagnose_train.py.NATIVE_RECEIPT.json')
assert failed['actual_worker_native_exit']==failed['worker_self_receipt']['native_exit']==1
failure=read(A/'DIAGNOSTIC_FAILURE.json');assert failure['counter']['neural_forward_rows_completed']==0
assert "canonical_REAL_torch(tx).cpu().numpy()"in failure['traceback']
assert not(S/'DATA_ACCESS_RECEIPT.json').exists()and not(S/'FORWARD_RECEIPT.json').exists()
changed=[row['name']for row in old['files']if sha(S/row['name'])!=row['sha256']]
assert set(changed)=={'artificial_check.py','ARTIFICIAL_CHECK_LOCAL.json'},changed
assert read(S/'ARTIFICIAL_CHECK_LOCAL.json')['pass']
for record in read(S/'INPUT_BINDING.json')['original_acceptance']:assert sha(Path(record['path']))==record['sha256']
ast.parse((S/'artificial_check.py').read_text('utf8'))
review={'UTC':datetime.now(timezone.utc).isoformat(),'scope':'Cross-backend ADC artificial assertion only','previous_failure_native_exit':1,
 'previous_failure_before_TRAIN_decode':True,'previous_project_neural_forward_rows':0,'changed_files':changed,
 'repair':'Require identical integer ADC codes and float64 voltage difference <= machine epsilon across NumPy/torch backends. Preserve strict bitwise original-vs-traced solver output on each backend.',
 'original_ADC_functions_and_frozen_solve_unchanged':True,'scientific_model_numeric_thresholds_unchanged':True,
 'all_frozen_input_and_helper_bindings_unchanged':True,'old_artificial_SHA256':sha(A/'artificial_check.py'),'new_artificial_SHA256':sha(S/'artificial_check.py'),
 'local_CPU_recheck_pass':True,'server_CPU_CUDA_recheck_pending':True,'failed_receipt_elapsed_s':failed['elapsed_s']}
write(A/'REPAIR_REVIEW.json',review)
updated=dict(old,binding_revision=2,previous_binding='repairs/ADC_CHECK_001/ROOT_IMPLEMENTATION_REVIEW.json',artificial_assertion_repair='repairs/ADC_CHECK_001/REPAIR_REVIEW.json')
updated['UTC']=datetime.now(timezone.utc).isoformat()
updated['files']=[{'name':v['name'],'bytes':(S/v['name']).stat().st_size,'sha256':sha(S/v['name'])}for v in old['files']]
write(S/'ROOT_IMPLEMENTATION_REVIEW.json',updated)
host='rylan@100.82.231.102';remote=C['remote_root'];events=[]
def invoke(args,code=None):
    if code is not None:compile(code,'remote_repair.py','exec')
    p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(A/'DEPLOYMENT_RECEIPT.json',{'UTC':datetime.now(timezone.utc).isoformat(),'events':events})
    assert p.returncode==0,events[-1]
    return p.stdout
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
prefix='from pathlib import Path\nimport json,hashlib,shutil,os,subprocess,time\np=Path('+repr(remote)+')\na=p/"repairs/ADC_CHECK_001"\n'
code=prefix+'''
native=json.loads((p/'diagnose_train.py.NATIVE_RECEIPT.json').read_text());assert native['actual_worker_native_exit']==1
assert not Path('/proc',str(native['actual_worker_PID'])).exists()
assert not Path('/proc',str(native['supervisor_PID'])).exists()
failure=json.loads((p/'DIAGNOSTIC_FAILURE.json').read_text());assert failure['counter']['neural_forward_rows_completed']==0
assert not(p/'DATA_ACCESS_RECEIPT.json').exists()and not(p/'FORWARD_RECEIPT.json').exists()
old=json.loads((p/'ROOT_IMPLEMENTATION_REVIEW.json').read_text())
for v in old['files']:assert hashlib.sha256((p/v['name']).read_bytes()).hexdigest()==v['sha256'],v['name']
a.mkdir(parents=True,exist_ok=False)
for f in p.iterdir():
    if f.is_file():shutil.copy2(f,a/f.name)
(p/'diagnose_train.py.used').rename(a/'original_submission_guard.used')
(p/'DIAGNOSTIC_FAILURE.json').rename(a/'original_failure.json')
(p/'LAUNCH_ONCE.json').rename(a/'original_launch_marker.json')
print(json.dumps({'archived_before_edit':True,'old_native_exit':1,'old_PID_absent':True,'TRAIN_decode':0,'new_project_forward_rows':0}))
'''
invoke(ssh,code)
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(S/'artificial_check.py'),str(S/'ARTIFICIAL_CHECK_LOCAL.json'),str(S/'ROOT_IMPLEMENTATION_REVIEW.json'),host+':'+remote+'/'])
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(A/'REPAIR_REVIEW.json'),host+':'+remote+'/repairs/ADC_CHECK_001/'])
code=prefix+'python='+repr(C['server_python'])+'\n'+r'''
review=json.loads((p/'ROOT_IMPLEMENTATION_REVIEW.json').read_text())
for v in review['files']:assert hashlib.sha256((p/v['name']).read_bytes()).hexdigest()==v['sha256']
marker=p/'ADC_CHECK_RETRY_LAUNCH_ONCE.json'
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'diagnose_train.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p),'prior_failed_attempt_preserved':True}
marker.write_text(json.dumps(record,indent=2)+'\n');time.sleep(1.5)
record['immediate_exit']=child.poll();print(json.dumps(record,indent=2))
'''
print(invoke(ssh,code))
