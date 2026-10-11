"""Launch one reviewed resume after the archived pre-training directory collision."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import json
import subprocess

P=Path(__file__).resolve().parent
S=P.parents[1]/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
cfg=json.loads((S/'config.json').read_text('utf8'))
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
receipt=P/'RESUME_002_LAUNCH_RECEIPT.json'
assert not receipt.exists()
prior_stderr=P/'RESUME_002_LAUNCH_NATIVE.stderr.log'
if prior_stderr.exists():
    prior_stdout=P/'RESUME_002_LAUNCH_NATIVE.stdout.log'
    assert not prior_stdout.read_text('utf8') and 'SyntaxError: unterminated string literal' in prior_stderr.read_text('utf8')
    for path in [prior_stdout,prior_stderr]:
        saved=path.with_name(path.name.replace('.log','.pre_execution_syntax.log'))
        assert not saved.exists()
        path.rename(saved)
code='from pathlib import Path\nimport os,json,hashlib,subprocess,time\np=Path('+repr(cfg['remote_root'])+')\npython='+repr(cfg['server_python'])+'\n'+r'''
os.sched_setaffinity(0,{11})
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert not any((Path('/proc')/str(pid)).exists() for pid in [291399,291394,328790,328795])
assert sha(p/'ROOT_IMPLEMENTATION_REVIEW.json')=='5d26170c74bfd1294511d2684455942bd29581fca7ac6f4414e95aaa6134339c'
manifest=json.loads((p/'ROOT_IMPLEMENTATION_REVIEW.json').read_text())
assert len(manifest['files'])==115
for row in manifest['files']:assert sha(p/row['name'])==row['sha256'],row['name']
assert sha(p/'RECOVERY_REQUEST.json')=='8293af6a09b19e6404212b84c8b091dee94cd2a8d60a605b4cb7bad87916588e'
request=json.loads((p/'RECOVERY_REQUEST.json').read_text())
assert request['previous_native_attempt_id']=='c9306986025c4474b87e5e800b300a43'
assert request['committed_update']==500
assert sha(p/'cache/RECOVERY_SAME_RUN.pt')==request['checkpoint_SHA256']
assert sha(p/request['checkpoint_publication_receipt_path'])==request['checkpoint_publication_receipt_SHA256']
assert not any((p/name).exists() for name in ['train_final.py.used','artificial_results','train_final.py.SCIENCE_PID.json','train_final.py.NATIVE_RECEIPT.json','REMOTE_ACTIVE_JOB.json'])
marker=p/'LAUNCH_train_final.py_resume_002.json'
with marker.open('x') as f:json.dump({'launch_intent_unix':time.time(),'root_authorized_same_checkpoint_resume':True},f)
out=(p/'train_final.py.resume_002.supervisor.stdout.log').open('xb')
err=(p/'train_final.py.resume_002.supervisor.stderr.log').open('xb')
cpus=list(range(12))
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'train_final.py'],cwd=p,
 stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
 preexec_fn=lambda:os.sched_setaffinity(0,cpus))
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'remote_root':str(p),
 'same_logical_run':True,'resume_checkpoint':500,'entry':'train_final.py','CPU_affinity':cpus,
 'new_review_SHA256':sha(p/'ROOT_IMPLEMENTATION_REVIEW.json'),'recovery_request_SHA256':sha(p/'RECOVERY_REQUEST.json')}
time.sleep(2)
record['immediate_supervisor_exit']=child.poll()
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text()) if (p/'REMOTE_ACTIVE_JOB.json').exists() else None
marker.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
'''
# This is literal Python passed on stdin, never interpolated into a shell command.
run=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
 'rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=25)
(P/'RESUME_002_LAUNCH_NATIVE.stdout.log').write_text(run.stdout,encoding='utf8')
(P/'RESUME_002_LAUNCH_NATIVE.stderr.log').write_text(run.stderr,encoding='utf8')
assert run.returncode==0,(run.stdout,run.stderr)
record=json.loads(run.stdout)
record['local_observed_UTC']=datetime.now(timezone.utc).isoformat()
receipt.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
assert record['immediate_supervisor_exit'] is None and record['actual_job'] is not None,record
print(json.dumps({'same_run_resume_launched':True,'supervisor_PID':record['supervisor_PID'],
 'actual_worker_PID':record['actual_job']['actual_worker_PID'],'native_exit':record['actual_job'].get('actual_worker_native_exit'),
 'checkpoint':500,'new_review_SHA256':record['new_review_SHA256']}))
