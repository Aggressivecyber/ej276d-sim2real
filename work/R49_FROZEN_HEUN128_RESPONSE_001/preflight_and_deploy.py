from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,base64,ast,sys,time
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
SSH=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
def put(n,x):(R/n).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
start=time.monotonic()
code=base64.b64encode((R/'server_preflight.py').read_bytes()).decode()
q=subprocess.run(SSH+[C['server_python']+' -B -c '+"\"import base64;exec(base64.b64decode('"+code+"'))\""],capture_output=True,text=True)
put('RESOURCE_CHECK_NATIVE.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'argv':q.args,'native_exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'science_started':False})
if q.returncode:raise RuntimeError('resource preflight transport failed; no science submitted')
pre=json.loads(q.stdout);put('SERVER_PREFLIGHT.json',pre)
if pre['own_project_jobs']:raise RuntimeError(('owned science status must be resolved',pre['own_project_jobs']))
if pre['remote_leaf_exists']:raise RuntimeError('existing leaf requires status resolution, no automatic resubmit')
if pre['MemAvailable_bytes']<C['MemAvailable_min_bytes']:raise RuntimeError('machine memory unavailable')
if pre['disk_free_bytes']<60*1024**2:raise RuntimeError('physical capacity below necessary core output; not a cumulative byte quota')
q=subprocess.run(SSH+['mkdir -p '+C['remote_root']],capture_output=True,text=True)
ops=[{'operation':'new independent remote directory','native_exit':q.returncode,'stderr':q.stderr}]
if q.returncode:raise RuntimeError(ops[-1])
names=['config.json','config_cpu.json','NOISE_CALIBRATION.npz','NOISE_CALIBRATION.json',
       'linux_supervise_post.py','linux_owned_worker.py','PROTOCOL.md','SOURCE_BINDINGS.json']
names+= [p.name for p in R.glob('*.py')if p.name not in ['prepare_stage.py','preflight_and_deploy.py','server_preflight.py','linux_supervise_post.py','linux_owned_worker.py','launch_post.py','fetch_small.py']]
for n in names:
    if n.endswith('.py'):ast.parse((R/n).read_text('utf8'),filename=n)
args=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8']+[str(R/n)for n in names]+['rylan@100.82.231.102:'+C['remote_root']+'/']
q=subprocess.run(args,capture_output=True,text=True)
ops.append({'operation':'normal selected source/control upload','files':names,'native_exit':q.returncode,'stderr':q.stderr})
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'operations':ops,
     'elapsed_s':time.monotonic()-start,'migration_equivalence_test':False,'new_scientific_jobs':0}
put('DEPLOY_RECEIPT.json',rec)
if q.returncode:raise RuntimeError('normal upload failed; no science submitted')
q=subprocess.run(SSH+['mkdir -p '+ ' '.join(C['remote_root']+'/'+a for a in C['arms'])],capture_output=True,text=True)
rec['operations'].append({'operation':'result subdirectories','native_exit':q.returncode,'stderr':q.stderr});put('DEPLOY_RECEIPT.json',rec)
if q.returncode:raise RuntimeError('output path creation failed')
print(json.dumps({'deployed':rec,'server_preflight':pre}),flush=True)
