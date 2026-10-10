from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,tarfile,ast,hashlib
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));REMOTE=C['remote_root']
def write(n,v):(R/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
names=['config.json','PROTOCOL.json','network.py','noise_core.py','source_flow_core.py','calibrate.py','training.py',
       'train_control.py','train_completion.py','check_models.py','linux_owned_worker.py','linux_supervise.py']
rows=[]
for n in names:
    data=(R/n).read_bytes()
    if n.endswith('.py'):ast.parse(data.decode('utf8'))
    rows.append({'name':n,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
write('PRETRAIN_FREEZE.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'sources':rows,
     'config_final_before_science':True,'migration_consistency_test':False,'task_SHA':'901517e92b400f65a7206ad01e2fff8e7fbb6625ae2b0f995c1af91bd874bb8f'})
archive=R/'UPLOAD_SOURCE.tar'
with tarfile.open(archive,'w',format=tarfile.USTAR_FORMAT) as t:
    for n in names+['PRETRAIN_FREEZE.json']:t.add(R/n,arcname=n,recursive=False)
common=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
ops=[]
q=subprocess.run(common+["mkdir -- '"+REMOTE+"'"],capture_output=True,text=True,timeout=30)
ops.append({'kind':'mkdir_new_exclusive','exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
if q.returncode:write('DEPLOYMENT_RECEIPT.json',{'operations':ops});raise RuntimeError('remote leaf creation failed; inspect, no blind resubmit')
q=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=8',str(archive),'rylan@100.82.231.102:'+REMOTE+'/UPLOAD_SOURCE.tar'],capture_output=True,text=True,timeout=60)
ops.append({'kind':'source_upload','exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'archive_bytes':archive.stat().st_size})
if q.returncode:write('DEPLOYMENT_RECEIPT.json',{'operations':ops});raise RuntimeError('upload failed')
q=subprocess.run(common+["cd '"+REMOTE+"' && tar -xf UPLOAD_SOURCE.tar"],capture_output=True,text=True,timeout=30)
ops.append({'kind':'source_unpack','exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
write('DEPLOYMENT_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'operations':ops,'remote_root':REMOTE,
       'ordinary_transfer_only_no_migration_hash_test':True,'historical_inputs_copied':0})
if q.returncode:raise RuntimeError('unpack failed')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote':REMOTE,'source_deploy_native0':True,'bytes':archive.stat().st_size}),flush=True)
