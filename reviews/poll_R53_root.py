from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001';C=json.loads((S/'config.json').read_text());host='rylan@100.82.231.102'
code='from pathlib import Path\nimport json,os\np=Path('+repr(C['remote_root'])+')\n'+'''
names=['REMOTE_ACTIVE_JOB.json','TRAIN_PROGRESS.json','TEACHER_VALIDATION.json','TRAINING_FAILURE.json','TRAIN_NOT_STARTED.json','TRAIN_STARTED.json','JOINT_PULSE_RESPONSE/INITIALIZATION.json','JOINT_PULSE_RESPONSE/TRAINING_RECEIPT.json','TRAIN_SOLVER_CHECK.json','GENERATION_PROGRESS.json','GENERATION_NUMERIC_RECEIPT.json','EVALUATION_PROGRESS.json','EVALUATION_NUMERIC_RECEIPT.json','REPLAY_PROGRESS.json','REPLAY_RECEIPT.json','GENERATION_FAILURE.json','EVALUATION_FAILURE.json','REPLAY_FAILURE.json']
names += [str(q.relative_to(p)) for q in sorted(p.glob('*.NATIVE_RECEIPT.json'))]
out={'files':{n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}}
a=out['files'].get('REMOTE_ACTIVE_JOB.json',{});pid=a.get('actual_worker_PID')
if pid:
 d=Path('/proc')/str(pid)
 out['actual_process']={'PID':pid,'exists':d.exists()}
 if d.exists():
  out['actual_process'].update(cwd=str((d/'cwd').resolve()),argv=(d/'cmdline').read_bytes().replace(b'\\0',b' ').decode(),status=[l for l in (d/'status').read_text().splitlines() if l.startswith(('State:','VmRSS:','VmHWM:'))])
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,{'native_exit':p.returncode,'stderr':p.stderr}
x=json.loads(p.stdout);now=datetime.now(timezone.utc).isoformat()
for name,value in x['files'].items():
 dest=S/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2)+'\n','utf8')
receipt={'UTC':now,'control_PID':os.getpid(),'SSH_native_exit':0,**x};(S/'LIVE_OBSERVATION.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
a=x['files'].get('REMOTE_ACTIVE_JOB.json',{});alive=x.get('actual_process',{}).get('exists',False);terminal='actual_worker_native_exit' in a
phase=a.get('entry','unknown').removesuffix('.py')
status=phase+'_running' if alive and not terminal else (phase+'_native_complete' if terminal and a['actual_worker_native_exit']==0 else 'inspect_native_state')
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 path=R/fn;v=json.loads(path.read_text('utf8'));v.update(current_stage_progress=status,active_scientific_jobs=int(alive and not terminal),active_remote_scientific_jobs=int(alive and not terminal),active_local_scientific_jobs=0)
 v['status']='R53_'+status+'_long_term_goal_active'
 if fn=='PROJECT_PROGRESS.json':v['project_status']=v['status']
 v['execution_location']['current_stage_GPU_used']='TRAIN_STARTED.json' in x['files']
 v['execution_location']['preflight_required']=False
 v['current_job']={'stage':C['stage'],'status':status,'actual_worker_PID':a.get('actual_worker_PID'),'supervisor_PID':a.get('supervisor_PID'),'actual_native_exit':a.get('actual_worker_native_exit'),'last_observed_UTC':now,'actual_process_exists':alive}
 v['current_activity']='root_R53_'+phase if alive and not terminal else 'root_R53_terminal_review'
 path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')

started=int('TRAIN_STARTED.json' in x['files'])
updates=x['files'].get('TRAIN_PROGRESS.json',{}).get('global_step',0)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
 path=R/name;v=json.loads(path.read_text('utf8'));rows=[q for q in v[key]if q.get('stage',q.get('id'))==C['stage']];assert len(rows)==1
 rows[0].update(status=status,new_training_runs_started=started,actual_worker_PID=a.get('actual_worker_PID'),supervisor_PID=a.get('supervisor_PID'),last_observed_UTC=now,actual_native_exit=a.get('actual_worker_native_exit'),actual_updates_observed=updates,active_scientific_jobs=int(alive and not terminal))
 path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
path=R/'RESOURCE_BUDGET.json';v=json.loads(path.read_text('utf8'));b=v['improvement_block']
rows=[q for q in b['registered_stages']if q['stage']==C['stage']];assert len(rows)==1
rows[0].update(status=status,response_training_runs_started=started,actual_updates_observed=updates,last_observed_UTC=now,active_scientific_jobs=int(alive and not terminal))
b['response_training_runs_registered']=sum(q.get('response_runs_reserved',q.get('planned_new_training_runs',0))for q in b['registered_stages'])
b['response_training_runs_started']=sum(q.get('response_training_runs_started',0)for q in b['actual_stage_receipts']if q['stage']!=C['stage'])+started
v['last_observed_UTC']=now;path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
t=x['files'].get('TEACHER_VALIDATION.json',{})
progress={k:x['files'][k]for k in ['GENERATION_PROGRESS.json','EVALUATION_PROGRESS.json']if k in x['files']}
if 'TRAIN_SOLVER_CHECK.json' in x['files']:
 n=x['files']['TRAIN_SOLVER_CHECK.json'];progress['numerical']={'pass':n['both_pass'],'models':{k:{m:v[m]for m in ['RMS_median_LSB','RMS_p95_LSB','preADC_relative_q_max','pass']}for k,v in n['models'].items()}}
if 'REPLAY_PROGRESS.json' in x['files']:
 rp=x['files']['REPLAY_PROGRESS.json'];progress['replay']={'rows':rp['rows'],'elapsed_s':rp['elapsed_s'],'domains':{k:v['bitwise_pass']for k,v in rp['domains'].items()}}
print(json.dumps({'UTC':now,'status':status,'actual_process':x.get('actual_process'),'training_progress':x['files'].get('TRAIN_PROGRESS.json') if phase=='train_final' else None,'teacher':{k:t.get(k) for k in ['passed','max_relative_q_error','max_PSD_change']},'post_progress':progress,'failure':{k:v for k,v in x['files'].items()if 'FAILURE' in k},'native':a.get('actual_worker_native_exit')},indent=2))
