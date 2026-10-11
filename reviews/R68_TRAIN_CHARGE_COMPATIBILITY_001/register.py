"""Read-only native reconciliation, then freeze the one local scalar diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess, sys

P=Path(__file__).resolve().parent; R=P.parents[1]
S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard=apply(P)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (P/'REGISTRATION.json').exists()
roots={str(k):read(R/('work/'+stage+'/config.json'))['remote_root'] for k,stage in
       [(65,'R65_FULL_EMPIRICAL_D_RESPONSE_001'),(66,'R66_EQUAL_COST_PILOT_001'),(67,'R67_EQUAL_COST_REPAIR_001')]}
code='import os,json,subprocess\nfrom pathlib import Path\nfrom datetime import datetime,timezone\nroots='+repr(roots)+'\n'+r'''
os.sched_setaffinity(0,{11})
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cwd=str((p/'cwd').resolve());cmd=(p/'cmdline').read_bytes().split(b'\x00')
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if cwd.startswith(base) and cmd and b'python' in cmd[0] and stat[0]!='Z':
   active.append(dict(PID=int(p.name),cwd=cwd,state=stat[0],executable=Path(os.fsdecode(cmd[0])).name))
 except (OSError,ValueError):pass
native={}
for stage,root in roots.items():
 f=Path(root)/('train_final.py.NATIVE_RECEIPT.json' if stage=='65' else 'pilot.py.NATIVE_RECEIPT.json')
 n=json.loads(f.read_text());native[stage]={k:n.get(k) for k in
 ['actual_worker_PID','supervisor_PID','actual_worker_native_exit','elapsed_s','stop_UTC','termination_reason']}
assert not active,active
assert [native[k]['actual_worker_native_exit'] for k in ['65','66','67']]==[-15,1,0]
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),active_project_python_processes=active,native=native,GPU_processes=gpu)))
'''
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],
                   input=code,text=True,capture_output=True,timeout=30)
(P/'PREFLIGHT.stdout.log').write_text(run.stdout,encoding='utf8')
(P/'PREFLIGHT.stderr.log').write_text(run.stderr,encoding='utf8')
assert run.returncode==0,run.stderr
native=json.loads(run.stdout);write(P/'NATIVE_PREFLIGHT.json',native)
before=P/'before_registration_metadata';before.mkdir()
for name in ['jobs.json','PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md']:
 shutil.copy2(R/name,before/name)
now=datetime.now(timezone.utc).isoformat()
jobs=read(R/'jobs.json')
old=next(j for j in jobs['jobs'] if j['stage']=='R65_FULL_EMPIRICAL_D_RESPONSE_001')
write(P/'STALE_R65_JOB_ENTRY.json',old)
old.update(status='retired_incomplete_cost_hold',active_scientific_jobs=0,updated_UTC=now,
           current_job=dict(stage=old['stage'],status='terminal',**native['native']['65']),
           actual_native_exit=-15,last_completed_update=613,durable_checkpoint_update=500,
           in_memory_updates_501_to_613_retained=False,automatic_resume=False,
           terminal_evidence='reviews/R66_EQUAL_COST_PILOT_001/R65_RETIREMENT_RECEIPT.json',
           reconciliation='reviews/R68_TRAIN_CHARGE_COMPATIBILITY_001/NATIVE_PREFLIGHT.json')
for k,stage,status in [(66,'R66_EQUAL_COST_PILOT_001','closed_invalid_execution'),(67,'R67_EQUAL_COST_REPAIR_001','closed_negative_local_support')]:
 assert not any(j['stage']==stage for j in jobs['jobs'])
 jobs['jobs'].append(dict(stage=stage,status=status,owner='/root',active_scientific_jobs=0,
                         protocol='tasks/'+stage+'.md',output_root='work/'+stage,remote_root=roots[str(k)],
                         native=native['native'][str(k)],model_goal_complete=False,
                         delivery_receipt='R67_DELIVERY_RECEIPT.json',reconciled_UTC=now))
stage='R68_TRAIN_CHARGE_COMPATIBILITY_001'
jobs['jobs'].append(dict(stage=stage,status='registered_before_scalar_decode',owner='/root',
                        kind='fixed_finite_TRAIN_scalar_arithmetic',active_scientific_jobs=0,
                        protocol='tasks/'+stage+'.md',output_root='work/'+stage,registered_UTC=now,
                        model_goal_complete=False))
jobs.update(active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,updated_UTC=now)
write(R/'jobs.json',jobs)
write(P/'LEDGER_RECONCILIATION.json',dict(UTC=now,reason='jobs.json retained the original running R65 entry and omitted terminal R66/R67.',
      native_evidence='NATIVE_PREFLIGHT.json',original_snapshot='before_registration_metadata/jobs.json',
      original_snapshot_SHA256=sha(before/'jobs.json'),new_job_ledger_SHA256=sha(R/'jobs.json'),
      active_remote_scientific_jobs=0,R65_original_50000_incomplete=True,resource=dict(guard,**memory())))
archive=R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/ENDPOINT_GEOMETRY_PRIVATE.npz'
config=dict(stage=stage,registered_UTC=now,input_archive=archive.relative_to(R).as_posix(),input_SHA256=sha(archive),
            SOURCE_N=6593,REAL_N=21047,CPU_threads=1,RSS_limit_bytes=1024**3,worker_hard_cap_s=60,
            relative_tolerance='1/100000',ADC_charge_radius='600/16383',float64_slack='1/10000000',
            SOURCE_field='SOURCE_original_q',REAL_field='REAL_canonical_q',original_A_F_complete=False)
write(S/'config.json',config)
shutil.copy2(R/('tasks/'+stage+'.md'),S/'PROTOCOL.md')
compile((S/'charge_certificate.py').read_text('utf8'),str(S/'charge_certificate.py'),'exec')
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=now,status='R68_registered_scalar_compatibility',project_status='R68_registered_scalar_compatibility',
      current_stage=stage,current_task='tasks/'+stage+'.md',active_scientific_jobs=0,current_job=None,
      current_activity='preregistered_TRAIN_charge_envelope_analysis',current_stage_progress='registered_not_started',
      current_stage_registration='reviews/'+stage+'/REGISTRATION.json',
      next_action='One local exact scalar CDF-envelope analysis; no network, no new pilot, no automatic R65 resume.',
      latest_diagnostic_conclusion='R67 full-D failed joint equal-cost benefit; original A-F incomplete.')
 write(R/name,d)
header='# Current: R68 registered TRAIN charge compatibility analysis\n\nR67 did not establish joint equal-cost benefit; R65 remains retired and on cost hold. R68 performs one local exact CDF-envelope calculation from saved TRAIN charge scalars, with all original parents, a conservative charge/ADC interval and a60-second cap. No model inference, training, new pilot or DEV/EVAL access. A positive finite TRAIN bound cannot establish failure of the original fixed-HGB held-out gate. Original A-F remains unchanged and incomplete. Native process state was refreshed and the stale jobs ledger reconciled; no remote scientific worker is active. See tasks/R68_TRAIN_CHARGE_COMPATIBILITY_001.md.\n\n## Earlier records\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md']:
 (R/name).write_text(header+(R/name).read_text('utf8'),encoding='utf8')
bound=[S/'charge_certificate.py',S/'config.json',S/'PROTOCOL.md',R/'PROJECT_ACCEPTANCE.json',R/'PROJECT_ACCEPTANCE.md',
       R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/source_flow_core.py',
       R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/endpoint_geometry.py']
write(P/'REGISTRATION.json',dict(UTC=now,status='REGISTERED_BEFORE_SCALAR_DECODE',input_archive=config['input_archive'],
      input_SHA256=config['input_SHA256'],files=[dict(path=f.relative_to(R).as_posix(),sha256=sha(f)) for f in bound],
      scope='one fixed local TRAIN-scalar certificate; no retraining or pilot',resource=guard))
print(json.dumps(dict(status='REGISTERED_BEFORE_SCALAR_DECODE',input_SHA256=config['input_SHA256'],native=native,
                     ledger_reconciled=True,resource=dict(guard,**memory())),ensure_ascii=False),flush=True)
