"""Normal SSH/transfer controls; no migration consistency tests or scientific imports."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,tarfile,time
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
REMOTE=C['remote_root'];PY=C['server_python'];MODE=sys.argv[1];START=time.monotonic();commands=[]
SSH=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
SCP=['scp','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8']
def run(kind,args,body=None):
 p=subprocess.run(args,input=body,text=True,capture_output=True)
 (R/(MODE+'_'+kind+'.stdout.log')).write_bytes(p.stdout.encode())
 (R/(MODE+'_'+kind+'.stderr.log')).write_bytes(p.stderr.encode())
 commands.append({'kind':kind,'native_exit':p.returncode,'argv':args})
 if p.returncode:raise RuntimeError((kind,p.returncode,p.stderr[-1000:]))
 return p.stdout
try:
 if MODE=='preflight':
  code=f'''from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,shutil,subprocess
project='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
leaf=Path({REMOTE!r});jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python' in argv and (project in argv or cwd.startswith(project)):jobs.append({{'PID':int(p.name),'argv':argv,'cwd':cwd}})
 except(OSError,PermissionError):pass
mem={{l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}}
def output(args):return subprocess.check_output(args,text=True).strip()
paths={C['endpoint_weight_paths']!r}
rec={{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'cwd':os.getcwd(),'argv':sys.argv,
 'CPU_logical':os.cpu_count(),'MemAvailable_bytes':mem['MemAvailable'],'own_project_jobs':jobs,
 'disk_free_bytes':shutil.disk_usage(project).free,'remote_leaf':str(leaf),'remote_leaf_exists':leaf.exists(),
 'GPU_jobs_authorized':0,'GPU_snapshot':output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
 'frozen_models_exist':{{name:Path(path).is_file() for name,path in paths.items()}}}}
print(json.dumps(rec))
'''
  v=json.loads(run('query',SSH+[PY+' -B -'],code))
  (R/'SERVER_PREFLIGHT.json').write_bytes((json.dumps(v,indent=2)+'\n').encode())
  assert not v['own_project_jobs'] and not v['remote_leaf_exists']
  assert v['disk_free_bytes']>=8*1024**3+1024**3 and all(v['frozen_models_exist'].values())
  if v['MemAvailable_bytes']<56*1024**3:
   C['RSS_limit_bytes']=min(C['RSS_limit_bytes'],max(0,v['MemAvailable_bytes']-8*1024**3))
   assert C['RSS_limit_bytes']>=4*1024**3
   for name in ['config.json','config_remote.json']:
    d=json.loads((R/name).read_text('utf8'));d['RSS_limit_bytes']=C['RSS_limit_bytes'];d['lower_RAM_quota_reason']='actual admission MemAvailable below56GiB'
    (R/name).write_bytes((json.dumps(d,indent=2)+'\n').encode())
  print(json.dumps(v),flush=True)
 elif MODE=='progress':
  code=f"from pathlib import Path\nimport json\nr=Path({REMOTE!r})\nfor n in ['GENERATION_PROGRESS.json','EVALUATION_PROGRESS.json','REPLAY_PROGRESS.json']:\n p=r/n\n if p.exists():print(json.dumps({{'file':n,'content':json.loads(p.read_text())}}))\n"
  print(run('read_saved_progress',SSH+[PY+' -B -'],code),flush=True)
 elif MODE=='deploy_check':
  assert json.loads((R/'SERVER_PREFLIGHT.json').read_text())['own_project_jobs']==[]
  archive=R/'UPLOAD_CHECK.tar'
  names=['network.py','source_flow_core.py','check_models.py','linux_supervise.py','linux_owned_worker.py',
         'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']
  with tarfile.open(archive,'w') as t:
   for name in names:t.add(R/name,arcname=name,recursive=False)
   t.add(R/'config_remote.json',arcname='config.json',recursive=False)
  run('create',SSH+[PY+' -B -'],f"from pathlib import Path\np=Path({REMOTE!r})\nassert not p.exists() and not p.is_symlink()\np.mkdir(parents=True,exist_ok=False)\nprint(p)\n")
  run('upload',SCP+[str(archive),'rylan@100.82.231.102:'+REMOTE+'/'+archive.name])
  run('extract',SSH+['cd '+REMOTE+' && tar -xf '+archive.name+' --no-same-owner --no-same-permissions'])
 elif MODE=='deploy_evaluation':
  assert json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']
  archive=R/'UPLOAD_CPU_EVALUATION.tar'
  run('upload',SCP+[str(archive),'rylan@100.82.231.102:'+REMOTE+'/'+archive.name])
  run('extract',SSH+['cd '+REMOTE+' && tar -xf '+archive.name+' --no-same-owner --no-same-permissions'])
 elif MODE=='deploy_correction':
  archive=R/'UPLOAD_EVALUATION_CORRECTION.tar'
  with tarfile.open(archive,'w')as t:
   for name in ['evaluate_saved.py','budget_comparison.py','FIXED_SUPPORT_CORRECTION_RECEIPT.json']:t.add(R/name,arcname=name,recursive=False)
  run('upload',SCP+[str(archive),'rylan@100.82.231.102:'+REMOTE+'/'+archive.name])
  run('extract',SSH+['cd '+REMOTE+' && tar -xf '+archive.name+' --no-same-owner --no-same-permissions'])
 elif MODE.startswith('fetch_'):
  phase=MODE[6:];assert phase in ['check','evaluation','replay']
  names={'check':['SHORT_2950_HEUN64','LONG_50000_HEUN64','TRAIN_SOLVER_CHECK.json','LOADED_MODEL_BINDINGS.json','SERVER_ENVIRONMENT.json'],
   'evaluation':['SHORT_2950_HEUN64/generated.npz','LONG_50000_HEUN64/generated.npz','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL','results','metrics.json',
    'GENERATION_RECEIPT.json','GENERATION_NUMERIC_RECEIPT.json','SOURCE_PROBES_SUMMARY.json','JOINT8_SUMMARY.json',
    'GUIDE_PERTURBATION_SUMMARY.json','GUIDE_PERTURBATION_COMPLETE_SUMMARY.json','RAW_RR_REUSE_BINDING.json',
    'SAME_TREATED_REAL_CONTROL_PAIRED.json','EVALUATION_NUMERIC_RECEIPT.json','C2ST_RUNNING.json','BOUNDARY_METRICS.json','MEAN_SD_METRICS.json','DEV_COVARIANCE_DESCRIPTION.json',
    'REAL_CONTROL_INPUT_BINDING.json','GENERATION_PROGRESS.json','EVALUATION_PROGRESS.json','GENERATION_FAILURE.json','EVALUATION_FAILURE.json'],
   'replay':['REPLAY_RECEIPT.json','REPLAY_PROGRESS.json','REPLAY_FAILURE.json']}[phase]
  entries={'check':['check_models.py'],'evaluation':['generate_all.py','evaluate_saved.py'],'replay':['replay_all.py']}[phase]
  names+=['REMOTE_RUN_HISTORY.json','REMOTE_ACTIVE_JOB.json','SERVER_START_RESOURCE.json']
  for entry in entries:names += [entry+s for s in ['.NATIVE_RECEIPT.json','.SCIENCE_PID.json','.RSS_ACK.json','.WORKER_RECEIPT.json','.stdout.log','.stderr.log']]
  archive='RETURN_'+phase.upper()+'.tar'
  code=f'''from pathlib import Path
import tarfile,json,os
R=Path({REMOTE!r})
with tarfile.open(R/{archive!r},'w') as t:
 for name in {names!r}:
  p=R/name
  if p.exists():t.add(p,arcname=name)
print(json.dumps({{'archive':str(R/{archive!r}),'bytes':(R/{archive!r}).stat().st_size,'PID':os.getpid()}}))
'''
  run('archive',SSH+[PY+' -B -'],code)
  run('download',SCP+['rylan@100.82.231.102:'+REMOTE+'/'+archive,str(R/archive)])
  with tarfile.open(R/archive,'r') as t:t.extractall(R,filter='data')
 else:raise ValueError(MODE)
finally:
 rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'mode':MODE,'commands':commands,
  'elapsed_s':time.monotonic()-START,'scientific_compute_s':0,'normal_transfer_only':True,'migration_content_consistency_test':False}
 (R/(MODE.upper()+'_NATIVE_RECEIPT.json')).write_bytes((json.dumps(rec,indent=2)+'\n').encode())
 print(json.dumps(rec),flush=True)
