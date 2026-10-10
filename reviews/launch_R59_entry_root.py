from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os,sys
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
C=json.loads((S/'config.json').read_text());entry=sys.argv[1]
previous={'numerical_check.py':'train_final.py','generate_all.py':'numerical_check.py',
    'evaluate_saved.py':'generate_all.py','replay_all.py':'evaluate_saved.py','summarize_and_plot.py':'replay_all.py'}
assert entry in previous
prefix='from pathlib import Path\nimport json,subprocess,time,os,hashlib\np=Path('+repr(C['remote_root'])+')\nentry='+repr(entry)+'\npython='+repr(C['server_python'])+'\nprevious='+repr(previous[entry])+'\narm='+repr(C['arm'])+'\n'
code=prefix+'''
assert (p/entry).exists()
active=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert active['entry']==previous and active.get('actual_worker_native_exit')==0 and active['termination_reason']is None,active
assert not Path('/proc',str(active['actual_worker_PID'])).exists(),active
binding=json.loads((p/'POSTPROCESS_FILE_BINDING.json').read_text())['bound_SHA256']
for n,h in binding.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
train=json.loads((p/arm/'TRAINING_RECEIPT.json').read_text())
assert train['updates']==50000 and train['selection']=='sole final50000'
assert train['actual_unique_SOURCE_REAL_reference_REAL_identity']==[6593,21047,21047]
assert train['teacher_waveforms_constructed']==0 and all(train[k]==0 for k in ['DEV_read','EVAL_read','CAL_decode'])
if entry!='numerical_check.py':assert json.loads((p/'TRAIN_SOLVER_CHECK.json').read_text())['both_pass']
if entry in ['evaluate_saved.py','replay_all.py']:assert json.loads((p/'GENERATION_NUMERIC_RECEIPT.json').read_text())['status']=='complete'
if entry=='summarize_and_plot.py':
 assert json.loads((p/'EVALUATION_NUMERIC_RECEIPT.json').read_text())['status']=='complete'
 assert json.loads((p/'REPLAY_RECEIPT.json').read_text())['rows']==4000
marker=p/('LAUNCH_'+entry+'.json')
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/(entry+'.supervisor.stdout.log')).open('wb');err=(p/(entry+'.supervisor.stderr.log')).open('wb')
supervisor='linux_supervise_reporting.py'if entry=='summarize_and_plot.py'else'linux_supervise.py'
cpus=sorted(os.sched_getaffinity(0))[:12]
child=subprocess.Popen([python,'-B',str(p/supervisor),entry],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
record={'entry':entry,'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p),'inherited_CPU_affinity':cpus}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text());record['immediate_exit']=child.poll()
pid=record['actual_job'].get('actual_worker_PID')
if pid and Path('/proc',str(pid)).exists():
 record['actual_CPU_affinity']=sorted(os.sched_getaffinity(pid));assert record['actual_CPU_affinity']==cpus
print(json.dumps(record,indent=2))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
record={'UTC':datetime.now(timezone.utc).isoformat(),'entry':entry,'control_PID':os.getpid(),'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(S/('LOCAL_LAUNCH_'+entry+'.json')).write_text(json.dumps(record,indent=2)+'\n','utf8')
assert p.returncode==0,record
print(p.stdout)
