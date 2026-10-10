from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';D=R/'reviews/R59_TRANSFER_RECOVERY_001'
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
old=read(D/'LOCAL_PIPELINE_CONTROL.failed.json')
assert old['status']=='control_failed_no_remote_restart' and 'fetch_R59_root.py' in old['error'] and 'TimeoutExpired' in old['error']
fetch=read(S/'FETCH_summarize_and_plot.py.json')
assert fetch['required_terminal_entry']=='summarize_and_plot.py' and fetch['normal_transfer']
assert all(e['native_exit']==0 for e in fetch['events'])
assert datetime.fromisoformat(fetch['UTC'])>datetime.fromisoformat(old['UTC'])
live=read(S/'LIVE_OBSERVATION.json')
assert not live['actual_process']['exists']
active=read(S/'REMOTE_ACTIVE_JOB.json')
assert active['entry']=='summarize_and_plot.py' and active['actual_worker_native_exit']==0 and active['termination_reason']is None
entries=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py']
for n in entries:
    a=read(S/(n+'.NATIVE_RECEIPT.json'))
    assert a['actual_worker_native_exit']==0 and a['worker_self_receipt']['native_exit']==0 and a['termination_reason']is None
for n in fetch['copied']:assert (S/n).is_file(),n
for n in ['watch_R59_pipeline_root.py','deliver_R59_root.py','recover_R59_transfer_root.py']:
    ast.parse((R/'reviews'/n).read_text('utf8'))
now=datetime.now(timezone.utc).isoformat()
receipt={'UTC':now,'PID':os.getpid(),'recovered':True,'cause':'Outer controller 60-second aggregate transfer timeout; all six scientific entries already completed with native exit 0.',
    'failed_control_receipt':'reviews/R59_TRANSFER_RECOVERY_001/LOCAL_PIPELINE_CONTROL.failed.json',
    'original_controller_SHA256':hashlib.sha256((D/'watch_R59_pipeline_root.executed.py').read_bytes()).hexdigest(),
    'normal_refetch_receipt':'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001/FETCH_summarize_and_plot.py.json',
    'all_transfer_native_exits':0,'files_copied':len(fetch['copied']),'copied_bytes':sum((S/n).stat().st_size for n in fetch['copied']),
    'all_scientific_native_exits':0,'scientific_entries_restarted':0,'new_training_or_inference':0,
    'pipeline_controller_restarted':False,'controller_fix':'Future aggregate fetch call allows 600 seconds; each SSH/SCP operation retains 60 seconds.',
    'model_goal_complete':False}
write(D/'RECEIPT.json',receipt)
state={**old,'UTC':now,'status':'registered_pipeline_complete_pending_root_QA','recovery_PID':os.getpid(),
    'original_control_native_exit':1,'recovery_receipt':'reviews/R59_TRANSFER_RECOVERY_001/RECEIPT.json',
    'remote_actual_process':live['actual_process'],'fetch':{'files_copied':receipt['files_copied'],'bytes':receipt['copied_bytes']},'model_goal_complete':False}
state.pop('error');state.pop('traceback');state.pop('remote_job_not_terminated')
write(S/'LOCAL_PIPELINE_CONTROL.json',state)
print(json.dumps(receipt,indent=2))
