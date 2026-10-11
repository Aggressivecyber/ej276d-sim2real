"""Read-only live observation. Never starts, stops or alters a scientific job."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
C=json.loads((S/'config.json').read_text('utf8'))
code='from pathlib import Path\nimport json,os,time,subprocess\np=Path('+repr(C['remote_root'])+')\n'+'''
def read(path):
 for i in range(3):
  try:return json.loads(path.read_text())
  except json.JSONDecodeError:
   if i==2:raise
   time.sleep(.05)
a=read(p/'REMOTE_ACTIVE_JOB.json');pid=a.get('actual_worker_PID');q=Path('/proc')/str(pid)
v={'worker_PID':pid,'entry':a['entry'],'exists':q.exists(),'native_exit':a.get('actual_worker_native_exit'),'supervisor_PID':a['supervisor_PID']}
if q.exists():
 v['cwd']=str((q/'cwd').resolve());v['argv']=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace')
 v['identity_matches']=v['cwd']==str(p)and 'linux_owned_worker.py' in v['argv']and a['entry']in v['argv']
 v['CPU_affinity']=sorted(os.sched_getaffinity(pid))
 v['memory_status']=[x for x in(q/'status').read_text().splitlines()if x.startswith(('VmRSS:','VmHWM:'))]
 v['actual_state']=next(x for x in(q/'status').read_text().splitlines()if x.startswith('State:'))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
v['current_worker_GPU_process_bytes']=next((int(x.split(',')[1].strip())*1048576 for x in gpu.splitlines()if x.strip()and int(x.split(',')[0])==pid),None)
if (p/'TRAIN_PROGRESS.json').exists():
 t=read(p/'TRAIN_PROGRESS.json');v.update(step=t['global_step'],progress_UTC=t['UTC'],training_elapsed_s=t['elapsed_s'],
  max_relative_q_f32=t['max_output_q_residual_f32'],GPU_reserved_peak_bytes=t['GPU_reserved_peak_bytes'],DEV_read=t['counter']['DEV_read'])
v['failure_files']=[q.name for q in p.glob('*FAILURE*.json')]
print(json.dumps(v))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,{'native_exit':p.returncode,'stderr':p.stderr}
v=json.loads(p.stdout);v.update(observed_UTC=datetime.now(timezone.utc).isoformat(),root_observer_PID=os.getpid(),read_only=True)
(S/'ROOT_CONTINUATION_OBSERVATION.json').write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps(v,indent=2))
