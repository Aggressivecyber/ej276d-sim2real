"""Control only: await the live registered run and execute its frozen post sequence."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
assert R.drive.upper()in ['E:','D:']
PID=os.getpid();START=time.monotonic();STATE=S/'LOCAL_PIPELINE_CONTROL.json'
sequence=['train_final.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py']
events=[{'kind':'root_reviewed_serialization_repair','record':'repairs/AUDIT_JSON_001/REPAIR_REVIEW.json','failed_attempt_retained':True}];current='numerical_check.py';last=None
def utc():return datetime.now(timezone.utc).isoformat()
def write(status,**extra):
    v={'UTC':utc(),'control_PID':PID,'current_entry':current,'control_only_no_scientific_imports':True,
       'status':status,'elapsed_s':time.monotonic()-START,'events':events,**extra}
    tmp=STATE.with_suffix('.tmp.json');tmp.write_text(json.dumps(v,indent=2)+'\n','utf8');os.replace(tmp,STATE)
    return v
def emit(v):print(json.dumps(v),flush=True)
def call(script,*args):
    timeout=600 if script=='fetch_R61_root.py' else 60
    p=subprocess.run([sys.executable,'-B',str(R/'reviews'/script),*args],cwd=R,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError({'script':script,'args':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    return json.loads(p.stdout)
with (S/'LOCAL_PIPELINE_CONTROL_AUDIT_JSON_RESOLVED.started.json').open('x',encoding='utf8')as f:
    json.dump({'UTC':utc(),'control_PID':PID,'sequence':sequence,'authorization':'existing registered R61 protocol only'},f,indent=2)
try:
    while True:
        if (S/'LOCAL_PIPELINE_CONTROL.stop_requested').exists():
            emit(write('control_stopped_at_explicit_marker',remote_job_not_terminated=True));break
        try:v=call('poll_R61_root.py')
        except BaseException as e:
            event={'UTC':utc(),'kind':'observation_failed_no_relaunch','error':repr(e)};events.append(event)
            emit(write('observation_retry_same_job',last_error=event));time.sleep(30);continue
        live=json.loads((S/'REMOTE_ACTIVE_JOB.json').read_text('utf8'));actual=v['actual_process']
        if live['entry']!=current:raise RuntimeError({'unexpected_entry':live['entry'],'expected':current,'actual':actual})
        if actual['exists']:
            if not actual['identity_matches_registered_stage'] or current not in actual['argv']:
                raise RuntimeError({'unverified_live_identity':actual,'entry':current})
            g=v.get('training_progress')or{};key=(current,g.get('global_step'))
            record=write('verified_wait',remote_actual_process=actual,training_step=g.get('global_step'),scientific_native_exit=None)
            if key!=last:emit({k:record[k]for k in ['UTC','status','current_entry','control_PID','training_step','remote_actual_process']});last=key
            time.sleep(30);continue
        if live.get('actual_worker_native_exit')is None:
            events.append({'UTC':utc(),'kind':'PID_absent_without_native_receipt_no_relaunch','actual':actual})
            emit(write('missing_terminal_receipt_manual_review',remote_actual_process=actual));break
        if live['actual_worker_native_exit']!=0 or live['termination_reason']is not None:
            emit(write('scientific_entry_failed_no_retry',terminal=live,failures=v['failure']));break
        events.append({'UTC':utc(),'kind':'verified_native_completion','entry':current,'PID':live['actual_worker_PID'],'native_exit':0})
        if current=='numerical_check.py'and not json.loads((S/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']:
            emit(write('original_TRAIN_numerical_gate_failed_no_DEV'));break
        if current==sequence[-1]:
            transfer=call('fetch_R61_root.py',current)
            emit(write('registered_pipeline_complete_pending_root_QA',fetch=transfer,model_goal_complete=False));break
        next_entry=sequence[sequence.index(current)+1]
        emit(write('launching_next_registered_entry',next_entry=next_entry))
        result=call('launch_R61_entry_root.py',next_entry)
        if result.get('immediate_exit')not in [None,0]or result['actual_job']['entry']!=next_entry:
            raise RuntimeError({'next_entry_admission_requires_review':result,'no_automatic_retry':True})
        events.append({'UTC':utc(),'kind':'launched_once','entry':next_entry,'supervisor_PID':result['supervisor_PID'],'actual_worker_PID':result['actual_job'].get('actual_worker_PID')})
        current=next_entry;last=None
        time.sleep(2)
except BaseException as e:
    emit(write('control_failed_no_remote_restart',error=repr(e),traceback=traceback.format_exc(),remote_job_not_terminated=True))
    raise
