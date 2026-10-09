from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,ast,hashlib
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
assert not (ROOT/'TRANSFER_RECEIPT.json').exists()
files=['train_server.py','diffusion_core.py','flow_core.py','linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']
bindings={}
for name in files:
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text('utf8'))
    bindings[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'PRETRAIN_FREEZE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':bindings,
    'existing_inputs_not_rehashed_for_migration':True,'fixed_recipe':True},indent=2)+'\n','utf8')
with tarfile.open(ROOT/'UPLOAD.tar','w',format=tarfile.USTAR_FORMAT)as tar:
    for n in files:tar.add(ROOT/n,arcname='config.json'if n=='remote_config.json'else n,recursive=False)
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code='''from pathlib import Path
import json,os
p=Path(__REMOTE__);assert not p.exists()and not p.is_symlink()
assert all(not q.is_symlink()for q in p.parents);p.mkdir()
old=Path(__OLD__);assert (old/'AFFINE_CONSTANTS.npz').is_file()and(old/'inputs/TRAIN_PRIVATE.npz').is_file()
print(json.dumps({'PID':os.getpid(),'created':str(p),'old_inputs_present':True}))
'''.replace('__REMOTE__',repr(cfg['remote_root'])).replace('__OLD__',repr(cfg['remote_R33_root']))
first=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(ROOT/'DEPLOY_CREATE.stdout.log').write_text(first.stdout,'utf8');(ROOT/'DEPLOY_CREATE.stderr.log').write_text(first.stderr,'utf8')
assert first.returncode==0,first.stderr
scp=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(ROOT/'UPLOAD.tar'),'rylan@100.82.231.102:'+cfg['remote_root']+'/UPLOAD.tar']
upload=subprocess.run(scp,capture_output=True,text=True,timeout=40,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert upload.returncode==0,upload.stderr
code='''from pathlib import Path
import os,tarfile,json
p=Path(__REMOTE__)
with tarfile.open(p/'UPLOAD.tar')as tar:tar.extractall(p,filter='data')
assert(p/'train_server.py').is_file()and(p/'config.json').is_file()
print(json.dumps({'PID':os.getpid(),'entry_exists':True,'cwd':str(p)}))
'''.replace('__REMOTE__',repr(cfg['remote_root']))
last=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert last.returncode==0,last.stderr
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'create_native_exit':first.returncode,'upload_native_exit':upload.returncode,
    'extract_native_exit':last.returncode,'remote_root':cfg['remote_root'],'upload_tar_bytes':(ROOT/'UPLOAD.tar').stat().st_size,
    'migration_content_tests':0,'create_result':json.loads(first.stdout),'extract_result':json.loads(last.stdout)}
(ROOT/'TRANSFER_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
