from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';C=json.loads((S/'config.json').read_text('utf8'))
code='from pathlib import Path\nimport json,os,time\np=Path('+repr(C['remote_root'])+')\n'+'''
names=['REMOTE_ACTIVE_JOB.json','TRAIN_PROGRESS.json','TRAINING_FAILURE.json','TRAIN_STARTED.json','ARTIFICIAL_CHECK_SERVER.json','ARTIFICIAL_CHECK_CUDA.json','TRAIN_POPULATION_AND_COORDINATES.json',
 'DIRECT_DISTRIBUTION_RESPONSE/INITIALIZATION.json','DIRECT_DISTRIBUTION_RESPONSE/TRAINING_RECEIPT.json','TRAIN_SOLVER_CHECK.json','GENERATION_PROGRESS.json','GENERATION_NUMERIC_RECEIPT.json',
 'EVALUATION_PROGRESS.json','EVALUATION_NUMERIC_RECEIPT.json','REPLAY_PROGRESS.json','REPLAY_RECEIPT.json','GENERATION_FAILURE.json','EVALUATION_FAILURE.json','REPLAY_FAILURE.json']
names += [str(q.relative_to(p)) for q in sorted(p.glob('*.NATIVE_RECEIPT.json'))]
out={'files':{},'transient_read_failures':[]}
for n in names:
 if not (p/n).is_file():continue
 for trial in range(3):
  try:out['files'][n]=json.loads((p/n).read_text());break
  except json.JSONDecodeError as e:
   if trial==2:out['transient_read_failures'].append({'path':n,'error':str(e)})
   else:time.sleep(.05)
a=out['files'].get('REMOTE_ACTIVE_JOB.json',{});pid=a.get('actual_worker_PID')
actual={'PID':pid,'exists':False,'identity_matches_registered_stage':False}
if pid:
 q=Path('/proc')/str(pid);actual['exists']=q.exists()
 if q.exists():
  try:
   cwd=str((q/'cwd').resolve());argv=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');stat=(q/'stat').read_text().rsplit(')',1)[1].split();ticks=os.sysconf('SC_CLK_TCK')
   actual.update(cwd=cwd,argv=argv,identity_matches_registered_stage=cwd==str(p) and 'linux_owned_worker.py' in argv,
    status=[s for s in(q/'status').read_text().splitlines() if s.startswith(('State:','VmRSS:','VmHWM:'))],process_CPU_s=(int(stat[11])+int(stat[12]))/ticks,
    process_threads=int(stat[17]),process_elapsed_s=float(Path('/proc/uptime').read_text().split()[0])-int(stat[19])/ticks)
  except OSError as e:actual['observation_race']=str(e)
out['actual_process']=actual
if not a and (p/'supervisor.stderr.log').is_file():out['supervisor_error_tail']=(p/'supervisor.stderr.log').read_text(errors='replace')[-8000:]
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,{'native_exit':p.returncode,'stderr':p.stderr}
x=json.loads(p.stdout);NOW=datetime.now(timezone.utc).isoformat()
def write(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path==R/'RESOURCE_BUDGET.json':text=text.replace('\n','\r\n')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode('utf8'))
for name,value in x['files'].items():write(S/name,value)
write(S/'LIVE_OBSERVATION.json',{'UTC':NOW,'control_PID':os.getpid(),'SSH_native_exit':0,**x})
a=x['files'].get('REMOTE_ACTIVE_JOB.json',{});actual=x['actual_process'];alive=actual['exists'] and actual['identity_matches_registered_stage']
terminal='actual_worker_native_exit' in a and not alive;phase=a.get('entry','unknown').removesuffix('.py')
status=phase+'_running' if alive else phase+'_native_complete' if terminal and a['actual_worker_native_exit']==0 else phase+'_native_failed' if terminal else 'awaiting_verified_worker_identity'
active=int(alive) if alive or terminal else None
started=int('TRAIN_STARTED.json' in x['files']);updates=x['files'].get('TRAIN_PROGRESS.json',{}).get('global_step',0)
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    path=R/name;v=json.loads(path.read_text('utf8'))
    v.update(status='R59_'+status+'_long_term_goal_active',current_stage_progress=status,active_scientific_jobs=active,active_remote_scientific_jobs=active,active_local_scientific_jobs=0,
        model_goal_complete=False,application_goal_status='active')
    if name=='PROGRAM.json':v['updated_at_utc']=NOW
    else:v.update(project_status=v['status'],updated_UTC=NOW)
    v['execution_location']['current_stage_GPU_used']=bool(a.get('GPU_admission_applied',False));v['execution_location']['preflight_required']=False
    v['current_job']={'stage':C['stage'],'status':status,'actual_worker_PID':a.get('actual_worker_PID'),'supervisor_PID':a.get('supervisor_PID'),
        'actual_native_exit':a.get('actual_worker_native_exit'),'last_observed_UTC':NOW,'actual_process':actual,'active_scientific_jobs':active}
    v['current_activity']='root_R59_'+phase if alive else 'root_R59_terminal_or_identity_review';write(path,v)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    path=R/name;v=json.loads(path.read_text('utf8'));rows=[r for r in v[key] if r.get('stage',r.get('id'))==C['stage']];assert len(rows)==1
    rows[0].update(status=status,new_training_runs_started=started,actual_updates_observed=updates,actual_worker_PID=a.get('actual_worker_PID'),supervisor_PID=a.get('supervisor_PID'),
        last_observed_UTC=NOW,actual_native_exit=a.get('actual_worker_native_exit'),active_scientific_jobs=active);write(path,v)
path=R/'RESOURCE_BUDGET.json';v=json.loads(path.read_text('utf8'));block=v['improvement_block'];rows=[r for r in block['registered_stages'] if r['stage']==C['stage']];assert len(rows)==1
rows[0].update(status=status,response_training_runs_started=started,actual_updates_observed=updates,last_observed_UTC=NOW,active_scientific_jobs=active)
block['response_training_runs_registered']=sum(r.get('response_runs_reserved',r.get('planned_new_training_runs',0)) for r in block['registered_stages'])
block['response_training_runs_started']=sum(r.get('response_training_runs_started',0) for r in block['actual_stage_receipts'] if r['stage']!=C['stage'])+started
v['last_observed_UTC']=NOW;write(path,v)
print(json.dumps({'UTC':NOW,'status':status,'actual_process':actual,'training_progress':x['files'].get('TRAIN_PROGRESS.json') if phase=='train_final' else None,
    'artificial_CPU_pass':'ARTIFICIAL_CHECK_SERVER.json' in x['files'],'artificial_CUDA_pass':'ARTIFICIAL_CHECK_CUDA.json' in x['files'],
    'failure':{k:v for k,v in x['files'].items() if 'FAILURE' in k},'native':a.get('actual_worker_native_exit'),'transient_read_failures':x['transient_read_failures'],
    'supervisor_error_tail':x.get('supervisor_error_tail')},indent=2))
