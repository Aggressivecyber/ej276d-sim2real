"""One native local execution with a60-second hard timeout and no restart."""
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys, time
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'supervisor_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert read(P/'REGISTRATION.json')['status']=='REGISTERED_BEFORE_SCALAR_DECODE'
with (P/'RUN_USED.json').open('x',encoding='utf8') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat()),f)
start=time.monotonic()
proc=subprocess.Popen([sys.executable,'-B',str(S/'charge_certificate.py')],cwd=S,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8')
jobs=read(R/'jobs.json');j=next(x for x in jobs['jobs']if x['stage']==S.name)
j.update(status='running',active_scientific_jobs=1,worker_PID=proc.pid)
jobs.update(active_scientific_jobs=1,active_local_scientific_jobs=1);write(R/'jobs.json',jobs)
reason=None
try:stdout,stderr=proc.communicate(timeout=60)
except subprocess.TimeoutExpired:
 reason='hard_60_second_local_cap';proc.kill();stdout,stderr=proc.communicate(timeout=10)
(P/'NATIVE.stdout.log').write_text(stdout,encoding='utf8');(P/'NATIVE.stderr.log').write_text(stderr,encoding='utf8')
native=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_worker_PID=proc.pid,actual_worker_native_exit=proc.returncode,
            elapsed_s=time.monotonic()-start,termination_reason=reason,worker_cap_s=60,resource=dict(guard,**memory()))
write(P/'NATIVE_RECEIPT.json',native)
status='completed_pending_root_review' if proc.returncode==0 else 'closed_invalid_execution'
jobs=read(R/'jobs.json');j=next(x for x in jobs['jobs']if x['stage']==S.name)
j.update(status=status,active_scientific_jobs=0,native=native)
jobs.update(active_scientific_jobs=0,active_local_scientific_jobs=0);write(R/'jobs.json',jobs)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=native['UTC'],active_scientific_jobs=0,current_job=dict(stage=S.name,status='terminal',**native),
                       current_stage_progress=status,current_activity='root_charge_certificate_review',
                       last_native_observation='reviews/'+S.name+'/NATIVE_RECEIPT.json')
 write(R/name,d)
print(json.dumps(dict(native=native,stdout=stdout,stderr=stderr),ensure_ascii=False),flush=True)
sys.exit(0 if proc.returncode==0 else 1)
