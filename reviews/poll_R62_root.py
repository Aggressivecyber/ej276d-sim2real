from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];ID='R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';S=R/'work'/ID;C=json.loads((S/'config.json').read_text('utf8'))
code='from pathlib import Path\nimport json,os,time\np=Path('+repr(C['remote_root'])+')\n'+'''
names=['REMOTE_ACTIVE_JOB.json','DIAGNOSTIC_PROGRESS.json','DIAGNOSTIC_FAILURE.json','ARTIFICIAL_CHECK_SERVER.json','ARTIFICIAL_CHECK_CUDA.json','ENVIRONMENT.json',
 'diagnose_train.py.NATIVE_RECEIPT.json','diagnose_train.py.WORKER_RECEIPT.json','PATH_SUMMARY.json','BASELINE_JOINT_SUMMARY.json','LOSS_COVERAGE_SUMMARY.json','PSD_SUMMARY.json','PROJECTION_SUMMARY.json','RESULT.json','RANDOM_STREAM_RECEIPT.json','FIGURE_MANIFEST.json']
out={'files':{},'transient_read_failures':[]}
for n in names:
 if not(p/n).is_file():continue
 for trial in range(3):
  try:out['files'][n]=json.loads((p/n).read_text());break
  except json.JSONDecodeError as e:
   if trial==2:out['transient_read_failures'].append({'path':n,'error':str(e)})
   else:time.sleep(.05)
a=out['files'].get('REMOTE_ACTIVE_JOB.json',{});pid=a.get('actual_worker_PID');actual={'PID':pid,'exists':False,'identity_matches_registered_stage':False}
if pid:
 q=Path('/proc')/str(pid);actual['exists']=q.exists()
 if q.exists():
  try:
   cwd=str((q/'cwd').resolve());argv=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');stat=(q/'stat').read_text().rsplit(')',1)[1].split();ticks=os.sysconf('SC_CLK_TCK')
   actual.update(cwd=cwd,argv=argv,identity_matches_registered_stage=cwd==str(p)and'linux_owned_worker.py'in argv,
    status=[s for s in(q/'status').read_text().splitlines()if s.startswith(('State:','VmRSS:','VmHWM:'))],process_CPU_s=(int(stat[11])+int(stat[12]))/ticks,
    process_elapsed_s=float(Path('/proc/uptime').read_text().split()[0])-int(stat[19])/ticks)
  except OSError as e:actual['observation_race']=str(e)
out['actual_process']=actual
if(out['files'].get('DIAGNOSTIC_FAILURE.json')or a.get('actual_worker_native_exit')not in[None,0])and(p/'diagnose_train.py.stderr.log').exists():out['stderr_tail']=(p/'diagnose_train.py.stderr.log').read_text(errors='replace')[-10000:]
if not a and(p/'supervisor.stderr.log').is_file():out['stderr_tail']=(p/'supervisor.stderr.log').read_text(errors='replace')[-10000:]
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,{'native_exit':p.returncode,'stderr':p.stderr}
x=json.loads(p.stdout);NOW=datetime.now(timezone.utc).isoformat()
def write(path,v):
 value=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if path==R/'RESOURCE_BUDGET.json':value=value.replace('\n','\r\n')
 path.write_bytes(value.encode('utf8'))
for n,v in x['files'].items():write(S/n,v)
write(S/'LIVE_OBSERVATION.json',{'UTC':NOW,'control_PID':os.getpid(),'SSH_native_exit':0,**x})
a=x['files'].get('REMOTE_ACTIVE_JOB.json',{});actual=x['actual_process'];alive=actual['exists']and actual['identity_matches_registered_stage']
terminal=a.get('actual_worker_native_exit')is not None and not actual['exists']
status='diagnostic_running'if alive else'diagnostic_native_complete'if terminal and a['actual_worker_native_exit']==0 else'diagnostic_native_failed'if terminal else'awaiting_verified_identity'
active=int(alive)if alive or terminal else None;progress=x['files'].get('DIAGNOSTIC_PROGRESS.json',{});counter=progress.get('counter',{})
for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 v=json.loads((R/name).read_text('utf8'));v.update(status='R62_'+status+'_long_term_goal_active',current_stage_progress=status,
   active_scientific_jobs=active,active_remote_scientific_jobs=active,active_local_scientific_jobs=0,model_goal_complete=False,application_goal_status='active')
 v['current_job']={'stage':ID,'status':status,'actual_worker_PID':a.get('actual_worker_PID'),'supervisor_PID':a.get('supervisor_PID'),'actual_native_exit':a.get('actual_worker_native_exit'),
   'last_observed_UTC':NOW,'actual_process':actual,'active_scientific_jobs':active,'phase':progress.get('phase'),'counter':counter}
 if name=='PROGRAM.json':v.update(updated_at_utc=NOW)
 else:v.update(project_status=v['status'],updated_UTC=NOW,current_activity='root_R62_'+status)
 write(R/name,v)
for name,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 v=json.loads((R/name).read_text('utf8'));row=next(q for q in v[key]if q[ident]==ID)
 row.update(status=status,actual_worker_PID=a.get('actual_worker_PID'),supervisor_PID=a.get('supervisor_PID'),last_observed_UTC=NOW,
  actual_native_exit=a.get('actual_worker_native_exit'),active_scientific_jobs=active,counter=counter);write(R/name,v)
v=json.loads((R/'RESOURCE_BUDGET.json').read_text('utf8'));row=next(q for q in v['improvement_block']['registered_stages']if q['stage']==ID)
row.update(status=status,active_scientific_jobs=active,last_observed_UTC=NOW,neural_forward_rows_observed=counter.get('neural_forward_rows_completed',0),response_training_runs_started=0)
v['last_observed_UTC']=NOW;write(R/'RESOURCE_BUDGET.json',v)
print(json.dumps({'UTC':NOW,'status':status,'actual_process':actual,'phase':progress.get('phase'),'counter':counter,'native':a.get('actual_worker_native_exit'),
 'failure':x['files'].get('DIAGNOSTIC_FAILURE.json'),'stderr_tail':x.get('stderr_tail'),'transient_read_failures':x['transient_read_failures']},indent=2))
