from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R45_REMOTE_20261010_010004_UTC'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
SCRIPT = r'''
from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,subprocess,shutil
project='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
old=Path(project)/'R43_REMOTE_20261009_233131_UTC'
leaf=Path(project)/'R45_REMOTE_20261010_010004_UTC'
jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python' in argv and (project in argv or cwd.startswith(project)):jobs.append({'PID':int(p.name),'argv':argv,'cwd':cwd})
 except(OSError,PermissionError):pass
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
def output(args):return subprocess.check_output(args,text=True).strip()
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'cwd':os.getcwd(),'argv':sys.argv,
 'CPU_logical':os.cpu_count(),'MemAvailable_bytes':mem['MemAvailable'],
 'GPU':output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
 'GPU_compute_jobs':output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
 'own_project_jobs':jobs,'disk_free_bytes':shutil.disk_usage(project).free,'remote_leaf':str(leaf),'remote_leaf_exists':leaf.exists(),
 'R43_config':json.loads((old/'config.json').read_text()),
 'inherited_files_exist':{n:(old/n).is_file() for n in ['SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt','SCALAR_TIME_FLOW_FINAL/FINAL_OPTIMIZER.pt','SCALAR_TIME_FLOW_FINAL/TRAINING_RECEIPT.json','SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz','UNIT_CHARGE_CONSTANTS.npz']}}
print(json.dumps(r))
'''
started = datetime.now(timezone.utc).isoformat()
args = ['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',PYTHON+' -B -']
p = subprocess.run(args,input=SCRIPT,text=True,capture_output=True)
(ROOT/'SERVER_PREFLIGHT.stdout.log').write_text(p.stdout,encoding='utf8')
(ROOT/'SERVER_PREFLIGHT.stderr.log').write_text(p.stderr,encoding='utf8')
receipt = {'local_PID':os.getpid(),'start_UTC':started,'stop_UTC':datetime.now(timezone.utc).isoformat(),'native_exit':p.returncode,'argv':args}
(ROOT/'RESOURCE_CHECK_NATIVE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
if p.returncode:raise SystemExit(p.returncode)
r = json.loads(p.stdout)
(ROOT/'SERVER_PREFLIGHT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
assert not r['own_project_jobs'] and not r['GPU_compute_jobs'] and not r['remote_leaf_exists']
assert r['MemAvailable_bytes']>=56*1024**3
assert int(r['GPU'].split(',')[2].strip())*1024**2>=11*1024**3
assert r['disk_free_bytes']>=12*1024**3 and all(r['inherited_files_exist'].values())
print(json.dumps({'local_PID':os.getpid(),'server_PID':r['PID'],'UTC':r['UTC'],'GPU':r['GPU'],'MemAvailable_bytes':r['MemAvailable_bytes'],'own_project_jobs':r['own_project_jobs'],'remote_leaf':REMOTE,'native_exit':p.returncode}),flush=True)
