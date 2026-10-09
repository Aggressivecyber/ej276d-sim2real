from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
code='''from pathlib import Path
import json
p=Path(__REMOTE__)
names=['TRAINING_FAILURE.json','JOINT_KERNEL_FINAL/FAILED_TRAINING.json','KERNEL_FORMULA_CHECK.json','JOINT_KERNEL_FINAL/INITIAL_GRADIENT_SCALE.json','train_server.py.stderr.log']
out={}
for n in names:
 q=p/n
 if q.exists():out[n]=q.read_text()
print(json.dumps(out))
'''.replace('__REMOTE__',repr(cfg['remote_root']))
cmd=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',cfg['server_python']+' -B -']
r=subprocess.run(cmd,input=code,capture_output=True,text=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert r.returncode==0,r.stderr
folder=ROOT/'failure001';folder.mkdir(exist_ok=False)
data=json.loads(r.stdout)
for n,body in data.items():(folder/n.replace('/','__')).write_text(body,'utf8')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'SSH_native_exit':r.returncode,'science_restart_or_refit':0,'files':list(data)}
(folder/'INSPECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({**receipt,'failure':json.loads(data.get('TRAINING_FAILURE.json','{}')),'partial':json.loads(data.get('JOINT_KERNEL_FINAL/FAILED_TRAINING.json','{}')),'calibration_saved':'JOINT_KERNEL_FINAL/INITIAL_GRADIENT_SCALE.json'in data}),flush=True)
