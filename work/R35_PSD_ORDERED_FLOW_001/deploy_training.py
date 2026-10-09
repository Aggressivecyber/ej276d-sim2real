from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,ast,hashlib
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
if (ROOT/'TRANSFER_RECEIPT.json').exists():raise RuntimeError('training sources already deployed')
files=['train_server.py','psd_order_core.py','model.py','legacy_model.py','flow_core.py','reorder_core.py',
    'linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']
bindings={}
for name in files:
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text('utf8'))
    bindings[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'PRETRAIN_FREEZE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':bindings,
    'old_inputs_not_rehashed_for_migration':True,'one_recipe':True},indent=2)+'\n','utf8')
with tarfile.open(ROOT/'UPLOAD.tar','w',format=tarfile.USTAR_FORMAT) as tar:
    for name in files:tar.add(ROOT/name,arcname='config.json' if name=='remote_config.json' else name,recursive=False)
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code='''from pathlib import Path
import json,os
p=Path(__REMOTE__)
assert not p.exists() and not p.is_symlink(),'unique leaf exists'
assert all(not parent.is_symlink() for parent in [p.parent,*p.parents])
p.mkdir()
old=Path(__OLD__)
assert (old/'INITIAL_FLOW.pt').is_file() and (old/'AFFINE_CONSTANTS.npz').is_file()
assert (old/'inputs/TRAIN_PRIVATE.npz').is_file()
print(json.dumps({'PID':os.getpid(),'created':str(p),'old_inputs_present':True}))
'''.replace('__REMOTE__',repr(cfg['remote_root'])).replace('__OLD__',repr(cfg['remote_R33_root']))
p=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(ROOT/'DEPLOY_CREATE.stdout.log').write_text(p.stdout,'utf8');(ROOT/'DEPLOY_CREATE.stderr.log').write_text(p.stderr,'utf8')
assert p.returncode==0,p.stderr
scp=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(ROOT/'UPLOAD.tar'),'rylan@100.82.231.102:'+cfg['remote_root']+'/UPLOAD.tar']
t=subprocess.run(scp,capture_output=True,text=True,timeout=40,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert t.returncode==0,t.stderr
code='''from pathlib import Path
import os,tarfile,json
p=Path(__REMOTE__)
with tarfile.open(p/'UPLOAD.tar') as tar:tar.extractall(p,filter='data')
assert (p/'train_server.py').is_file() and (p/'config.json').is_file()
print(json.dumps({'PID':os.getpid(),'entry_exists':True,'cwd':str(p),'normal_extraction':True}))
'''.replace('__REMOTE__',repr(cfg['remote_root']))
u=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert u.returncode==0,u.stderr
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'create_native_exit':p.returncode,
    'upload_native_exit':t.returncode,'extract_native_exit':u.returncode,'remote_root':cfg['remote_root'],
    'upload_tar_bytes':(ROOT/'UPLOAD.tar').stat().st_size,'only_new_source_control_transferred':True,
    'source_control_bindings_local_only':'PRETRAIN_FREEZE.json','migration_content_tests':0,
    'create_result':json.loads(p.stdout),'extract_result':json.loads(u.stdout)}
(ROOT/'TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
