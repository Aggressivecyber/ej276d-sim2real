from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
R=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
assert json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text())['both_pass']
if (R/'PIPELINE_STARTED.json').exists():raise RuntimeError('resolve actual existing pipeline state; do not duplicate')
record={'PID':os.getpid(),'start_UTC':utc(),'new_training_updates':0,'compute_budget_enforced':False,'completed':[]}
write('PIPELINE_STARTED.json',record)
start=time.monotonic()
extras={
 'generate_all.py':['GENERATION_RECEIPT.json','GENERATION_NUMERIC_RECEIPT.json','INITIAL_NOISE_DESCRIPTIONS.json','REAL_IDENTITY_DESCRIPTIONS.json'],
 'evaluate_saved.py':['metrics.json','EVALUATION_NUMERIC_RECEIPT.json','SAME_TREATED_REAL_CONTROL_PAIRED.json'],
 'replay_all.py':['REPLAY_RECEIPT.json'],
 'summarize_and_plot.py':['GATE_METRICS.json','ACCEPTANCE_A_C.json','METRICS_COMPACT.json','SUMMARY_PLOT_NUMERIC_RECEIPT.json'],
 'render_train.py':['TRAIN_PLOT_NUMERIC_RECEIPT.json']}
for entry in extras:
 record['current_entry']=entry;record['last_update_UTC']=utc();write('PIPELINE_PROGRESS.json',record)
 print(json.dumps({'pipeline_entry':entry,'UTC':utc()}),flush=True)
 result=subprocess.run([sys.executable,'-B',str(R/'launch_post.py'),entry],cwd=R)
 common=[entry+'.NATIVE_RECEIPT.json',entry+'.WORKER_RECEIPT.json',entry+'.SCIENCE_PID.json',
         entry+'.RSS_ACK.json',entry+'.stdout.log',entry+'.stderr.log','REMOTE_RUN_HISTORY.json']
 fetch=subprocess.run([sys.executable,'-B',str(R/'fetch_small.py'),*common],cwd=R)
 row={'entry':entry,'launch_native_exit':result.returncode,'native_return_exit':fetch.returncode,'stop_UTC':utc()}
 record['completed'].append(row);write('PIPELINE_PROGRESS.json',record)
 if result.returncode or fetch.returncode:
  record.update(status='engineering_failure_preserved',stop_UTC=utc());write('PIPELINE_TERMINAL.json',record)
  raise SystemExit(result.returncode or fetch.returncode)
 fetch=subprocess.run([sys.executable,'-B',str(R/'fetch_small.py'),*extras[entry]],cwd=R)
 row['summary_return_exit']=fetch.returncode;write('PIPELINE_PROGRESS.json',record)
 if fetch.returncode:
  record.update(status='result_return_failure_preserved',stop_UTC=utc());write('PIPELINE_TERMINAL.json',record)
  raise SystemExit(fetch.returncode)
 print(json.dumps({'completed_entry':entry,'UTC':utc(),'native_exit':result.returncode}),flush=True)
record.update(status='all_registered_science_complete',stop_UTC=utc(),transport_control_wall_s=time.monotonic()-start)
write('PIPELINE_TERMINAL.json',record);print(json.dumps(record),flush=True)
