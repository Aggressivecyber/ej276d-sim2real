from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, os

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'work/R54_CONTINUOUS_CHARGE_FLOW_001'
cfg=json.loads((STAGE/'config.json').read_text('utf8'))
remote=cfg['remote_root'];host='rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R54_ROOT_')
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
events=[]

def invoke(argv,code=None):
    p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    (STAGE/'DEPLOYMENT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),'events':events},indent=2)+'\n','utf8')
    if p.returncode:raise RuntimeError(events[-1])
    return p.stdout

prefix='from pathlib import Path\nimport json,subprocess,time,os\np=Path('+repr(remote)+')\n'
checks=prefix+'\n'.join('assert Path('+repr(cfg[k])+').exists(), '+repr(k) for k in ['server_python','TRAIN_path','original_stream','constants_path','train_solver_input','full_coupling_path','partial_coupling_path','noise_calibration_path','original_initial_state_path','original_training_receipt','historical_R53_training_receipt'])
checks+='\nassert p.parent.is_dir()\np.mkdir(exist_ok=False)\nprint(str(p))\n'
invoke(ssh,checks)
files=['config.json','INPUT_BINDING.json','teacher_core.py','train_final.py','network.py','source_flow_core.py','noise_core.py','linux_supervise.py','linux_owned_worker.py','PROTOCOL.md','Q_DEFINITION.json','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json']
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(STAGE/n) for n in files],host+':'+remote+'/'])
code=prefix+'python='+repr(cfg['server_python'])+'\n'+'''
marker=p/'LAUNCH_train_final.py.json'
with marker.open('x') as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'train_final.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
active=p/'REMOTE_ACTIVE_JOB.json'
record['actual_job']=json.loads(active.read_text()) if active.exists() else None
record['immediate_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
print(invoke(ssh,code))
