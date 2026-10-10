"""Release only the archived completed failure's once-per-entry marker."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,shutil
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';A=S/'repairs/AUDIT_JSON_001'
C=json.loads((S/'config.json').read_text('utf8'))
for n in ['LOCAL_LAUNCH_numerical_check.py.json','LOCAL_PIPELINE_CONTROL.json','LOCAL_PIPELINE_CONTROL_AUDIT_JSON_REPAIR.started.json']:
    dest=A/('ADMISSION_FAILURE_'+n);assert not dest.exists();shutil.copy2(S/n,dest)
code='from pathlib import Path\nimport json,hashlib,shutil\np=Path('+repr(C['remote_root'])+')\na=p/"repairs/AUDIT_JSON_001"\n'+'''
active=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert active['actual_worker_PID']==187137 and active['actual_worker_native_exit']==1 and active['entry']=='numerical_check.py'
for pid in [187137,187132,191357]:assert not Path('/proc',str(pid)).exists()
error=(p/'numerical_check.py.supervisor.stderr.log').read_text()
assert 'already submitted; resolve existing identity and native receipt' in error
assert not (p/'TRAIN_SOLVER_CHECK.json').exists() and not (p/'GENERATION_RECEIPT.json').exists()
for n in ['numerical_check.py.supervisor.stderr.log','numerical_check.py.supervisor.stdout.log','LAUNCH_numerical_check.py.json']:
    dst=a/('ADMISSION_FAILURE_'+n);assert not dst.exists();shutil.copy2(p/n,dst)
marker=p/'numerical_check.py.used'
assert marker.read_bytes()==(a/marker.name).read_bytes()
marker.rename(a/'RELEASED_numerical_check.py.used')
(p/'LAUNCH_numerical_check.py.json').rename(a/'RELEASED_RETRY_LAUNCH_numerical_check.py.json')
print(json.dumps({'released_archived_failed_entry_marker':True,'admission_supervisor_PID':191357,'admission_native_exit':1,'admission_new_worker_started':False,'admission_error':error,'prior_native_failure_preserved':True,'all_known_processes_absent':True}))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,p.stderr
rec={'UTC':datetime.now(timezone.utc).isoformat(),**json.loads(p.stdout)}
(A/'ADMISSION_MARKER_RESOLUTION.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
watch=(R/'reviews/watch_R61_repaired_pipeline_root.py').read_text('utf8').replace('LOCAL_PIPELINE_CONTROL_AUDIT_JSON_REPAIR.started.json','LOCAL_PIPELINE_CONTROL_AUDIT_JSON_RESOLVED.started.json')
with (R/'reviews/watch_R61_resolved_repair_pipeline_root.py').open('x',encoding='utf8',newline='\n')as f:f.write(watch)
print(json.dumps(rec,indent=2))
