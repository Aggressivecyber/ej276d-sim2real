"""Read resume receipts and compare retained progress records; no model execution."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import hashlib
import json
import subprocess

P=Path(__file__).resolve().parent
S=P.parents[1]/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
cfg=json.loads((S/'config.json').read_text('utf8'))
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
code='from pathlib import Path\nimport json,os,hashlib\np=Path('+repr(cfg['remote_root'])+')\n'+r'''
os.sched_setaffinity(0,{11})
out={}
for name in ['SAME_RUN_RESTORE_RECEIPT.json','RECOVERY_REQUEST.json','ARTIFICIAL_SUITE_RECEIPT.json',
             'PHYSICAL_PROGRESS.json','TRAIN_PROGRESS.json','REMOTE_ACTIVE_JOB.json']:
    f=p/name
    if f.exists():out[name]=json.loads(f.read_text())
restore=out['SAME_RUN_RESTORE_RECEIPT.json']
assert restore['PID']==330411 and restore['committed_update_restored']==500
assert restore['all_eight_random_states_and_original_optimizer_restored']
assert restore['model_state_SHA256_after_restore']==restore['model_state_SHA256_in_checkpoint']
assert restore['checkpoint_SHA256']=='63d958bfffbb75b57362005361f4adf3ef828de201bd6b2ebd5933df34fa3b5e'
assert out['ARTIFICIAL_SUITE_RECEIPT.json']['pass'] is True
assert (Path('/proc')/str(restore['PID'])).exists()
out['worker_process_state']=(Path('/proc')/str(restore['PID'])/'stat').read_text().rsplit(')',1)[1].split()[0]
old=p/'native_attempts/37c43e76bb064c44937ebe888d58d0da/stage/FULL_EMPIRICAL_RESPONSE/TRAINING_TRACE.json'
new=p/'FULL_EMPIRICAL_RESPONSE/TRAINING_TRACE.json'
out['old_trace']=json.loads(old.read_text())
out['current_trace']=json.loads(new.read_text())
out['old_trace_SHA256']=hashlib.sha256(old.read_bytes()).hexdigest()
print(json.dumps(out),flush=True)
'''
run=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',
 '/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=20)
assert run.returncode==0,run.stderr
data=json.loads(run.stdout)
now=datetime.now(timezone.utc)
out=P/'resume_observations'
out.mkdir(exist_ok=True)
target=out/(now.strftime('%Y%m%dT%H%M%S_%f')+'.json')
target.write_text(json.dumps({'observed_UTC':now.isoformat(),**data},indent=2)+'\n',encoding='utf8')
fields=['last_update','last_up_to100_mean','precision_order_last_up_to100_mean',
 'max_output_relative_q_error_f64','max_output_relative_q_error_f32',
 'unique_SOURCE_REAL_reference_REAL_identity','effective_counter']
def exact(x):
    if isinstance(x,float):return ('float64hex',x.hex())
    if isinstance(x,list):return [exact(v) for v in x]
    if isinstance(x,dict):return {k:exact(v) for k,v in x.items()}
    return x
old={r['global_step']:r for r in data['old_trace']}
restore=data['SAME_RUN_RESTORE_RECEIPT.json']
physical=data['PHYSICAL_PROGRESS.json']
# The on-disk trace is replaced only at the first new 100-update publication.
# Retained records from the archived attempt are not evidence of the new run.
new={r['global_step']:r for r in data['current_trace']
     if r['native_attempt_id']==restore['native_attempt_id'] and r['PID']==restore['PID']
     and 500<r['global_step']<=physical['last_completed_update']}
comparisons=[]
for step in sorted(set(old)&set(new)):
    if step<=500:continue
    checks={field:exact(old[step][field])==exact(new[step][field]) for field in fields}
    comparisons.append({'global_step':step,'record_fields_equal':checks,'all_compared_fields_equal':all(checks.values()),
      'scope':'Recorded endpoint and 100-update aggregate metrics, not a full model-state comparison.'})
start=datetime.fromisoformat(restore['UTC'])
finish=datetime.fromisoformat(physical['UTC'])
completed=physical['last_completed_update']-500
old_seconds=(datetime.fromisoformat(old[600]['UTC'])-datetime.fromisoformat(old[500]['UTC'])).total_seconds()/100
speed={'original_step_500_to_600_seconds_per_update':old_seconds,
 'resume_observed_completed_updates':completed,'resume_restore_to_observation_elapsed_seconds':(finish-start).total_seconds(),
 'measurement_contains_partial_current_update':True}
if completed:
    seconds=(finish-start).total_seconds()/completed
    speed.update(resume_approx_seconds_per_update=seconds,approx_step_time_reduction_fraction=1-seconds/old_seconds,
      approximate_remaining_training_hours=(50000-physical['last_completed_update'])*seconds/3600)
if 600 in new:
    seconds=(datetime.fromisoformat(new[600]['UTC'])-start).total_seconds()/100
    assert seconds>0 and physical['last_completed_update']>=600
    speed.update(resume_first_100_full_updates_seconds_per_update=seconds,
      first_100_step_time_reduction_fraction=1-seconds/old_seconds)
result={'UTC':now.isoformat(),'status':('PAUSED_NATIVE_STATE_VERIFIED' if data['worker_process_state']=='T' else 'RESTORE_AND_NATIVE_TRAIN_PROGRESS_VERIFIED'),
 'worker_process_state':data['worker_process_state'],'remaining_hours_estimate_applies_only_if_training_resumes':data['worker_process_state']=='T',
 'actual_worker_PID':restore['PID'],'committed_update_restored':500,'checkpoint_SHA256':restore['checkpoint_SHA256'],
 'restored_model_state_SHA256':restore['model_state_SHA256_after_restore'],
 'original_optimizer_and_eight_random_states_restored':True,'artificial_suite_pass':True,
 'last_completed_update':physical['last_completed_update'],'speed':speed,'progress_comparisons':comparisons,
 'comparison_lineage_filter':'Only the resumed native_attempt_id, PID and physically completed updates; retained old on-disk records excluded.',
 'raw_observation_path':str(target),'raw_observation_SHA256':hashlib.sha256(target.read_bytes()).hexdigest()}
(P/'LATEST_RESUME_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
