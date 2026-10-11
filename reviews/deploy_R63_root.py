from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';C=json.loads((S/'config.json').read_text('utf8'))
remote=C['remote_root'];host='rylan@100.82.231.102';events=[]
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R63_ROOT_')
review=json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'));assert review['ready_for_single_registered_training'] and not review['scientific_execution_started']
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
def invoke(argv,code=None):
    p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    (S/'DEPLOYMENT_RECEIPT.json').write_bytes((json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),'events':events},indent=2)+'\n').encode())
    if p.returncode:raise RuntimeError(events[-1])
    return p.stdout
prefix='from pathlib import Path\nimport json,subprocess,time,os\np=Path('+repr(remote)+')\n'
checks=prefix+'\n'.join('assert Path('+repr(p)+').is_file()' for p in [C['server_python'],*C['input_paths'].values()])
checks+='\nassert p.parent.is_dir()\np.mkdir(exist_ok=False)\nprint(str(p))\n'
invoke(ssh,checks)
names=[record['name']for record in review['files']]+['ROOT_IMPLEMENTATION_REVIEW.json']
assert all((S/n).is_file() for n in names)
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n) for n in names],host+':'+remote+'/'])
code=prefix+'python='+repr(C['server_python'])+'\n'+'''
marker=p/'LAUNCH_train_final.py.json'
with marker.open('x') as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'train_final.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
active=p/'REMOTE_ACTIVE_JOB.json';record['actual_job']=json.loads(active.read_text()) if active.is_file() else None
record['immediate_supervisor_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
print(invoke(ssh,code))
