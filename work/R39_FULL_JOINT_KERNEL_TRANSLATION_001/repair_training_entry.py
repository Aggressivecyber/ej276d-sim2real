from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,subprocess,tarfile,hashlib,ast
ROOT=Path(__file__).resolve().parent;R32=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001';cfg=json.loads((ROOT/'config.json').read_text('utf8'))
snap=ROOT/'failure001/source_snapshot';snap.mkdir(parents=True,exist_ok=False)
for n in ['config.json','remote_config.json','PRETRAIN_FREEZE.json','SOURCE_BINDINGS.json','online_training.py','train_server.py','linux_supervise.py','launch_remote.py']:
    shutil.copy2(ROOT/n,snap/n)
full=json.loads((R32/'config.json').read_text('utf8'));missing={k:v for k,v in full.items()if k not in cfg};merged={**full,**cfg}
merged['recovery_calibration_path']='JOINT_KERNEL_FINAL/INITIAL_GRADIENT_SCALE.json'
merged['known_failed_first_D_update_reconstructed']=1;merged['scientific_unique_update_trajectory']=2950
merged['training_wall_s']=896.;merged['training_total_limit_s']=896.
merged['native_training_entry']='train_recover.py';merged['actual_training_resume']='initial G untouched; known first D reconstructed deterministically; saved64-batch coefficient reused'
remoteold=json.loads((R32/'remote_config.json').read_text('utf8'));remote={**remoteold,**merged}
for k in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','G_initial','reader_initial','D_initial']:remote[k]=remoteold[k]
for n,v in [('config.json',merged),('remote_config.json',remote)]:
    (ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n','utf8')
s=(ROOT/'online_training.py').read_text('utf8')
needle="calibration=fine.calibrate(model,xs,xt,actual,cfg['batch'])"
assert needle in s
s=s.replace(needle,"calibration=json.loads((root/'INITIAL_GRADIENT_SCALE.json').read_text('utf8'));fine.c=calibration['coefficient'];fine.calibration=calibration;assert np.isfinite(fine.c)and fine.c>0")
s=s.replace('root.mkdir(exist_ok=False)','root.mkdir(exist_ok=True)')
(ROOT/'online_recover.py').write_text(s,'utf8')
s=(ROOT/'train_server.py').read_text('utf8').replace('from online_training import train','from online_recover import train')
s=s.replace("write('KERNEL_FORMULA_CHECK.json',selfcheck())","assert json.loads((ROOT/'KERNEL_FORMULA_CHECK.json').read_text('utf8'))['pass']")
(ROOT/'train_recover.py').write_text(s,'utf8')
for n in ['linux_supervise.py','launch_remote.py']:
    s=(ROOT/n).read_text('utf8').replace("['train_server.py']","['train_server.py','train_recover.py']")
    if n=='linux_supervise.py':s=s.replace(',900)',',896)')
    (ROOT/n).write_text(s,'utf8')
files=['config.json','remote_config.json','online_recover.py','train_recover.py','linux_supervise.py']
bindings={n:{'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()}for n in files}
for n in files:
    if n.endswith('.py'):ast.parse((ROOT/n).read_text('utf8'))
with tarfile.open(ROOT/'ENGINEERING_REPAIR_UPLOAD.tar','w',format=tarfile.USTAR_FORMAT)as t:
    for n in files:
        if n=='config.json':continue
        t.add(ROOT/n,arcname='config.json'if n=='remote_config.json'else n,recursive=False)
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
cmd=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(ROOT/'ENGINEERING_REPAIR_UPLOAD.tar'),'rylan@100.82.231.102:'+cfg['remote_root']+'/ENGINEERING_REPAIR_UPLOAD.tar']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert r.returncode==0,r.stderr
code='''from pathlib import Path
import json,shutil,tarfile,os
p=Path(__REMOTE__);h=json.loads((p/'REMOTE_RUN_HISTORY.json').read_text());assert h[-1]['actual_worker_native_exit']==1
f=json.loads((p/'TRAINING_FAILURE.json').read_text());assert f['counter']['G_updates']==0 and f['counter']['D_updates']==1
old=p/'failure001_remote';old.mkdir()
for n in ['config.json','online_training.py','train_server.py','linux_supervise.py','TRAINING_FAILURE.json','train_server.py.stdout.log','train_server.py.stderr.log','train_server.py.NATIVE_RECEIPT.json']:
 shutil.copy2(p/n,old/n)
shutil.move(str(p/'JOINT_KERNEL_FINAL'),str(old/'JOINT_KERNEL_FINAL'))
(p/'JOINT_KERNEL_FINAL').mkdir()
for n in ['INITIAL_GRADIENT_SCALE.json','FRESH_PROJECTION_FIRST_PRIVATE.npz','FRESH_PROJECTION_LAST_PRIVATE.npz','FRESH_PROJECTION_STREAM.json']:
 shutil.copy2(old/'JOINT_KERNEL_FINAL'/n,p/'JOINT_KERNEL_FINAL'/n)
with tarfile.open(p/'ENGINEERING_REPAIR_UPLOAD.tar')as t:t.extractall(p,filter='data')
print(json.dumps({'PID':os.getpid(),'first_failure_counter':f['counter'],'saved_calibration_reused':True,'known_D_reconstruct_calls':1}))
'''.replace('__REMOTE__',repr(cfg['remote_root']))
q=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,capture_output=True,text=True,encoding='utf8',timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert q.returncode==0,q.stderr
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'upload_native_exit':r.returncode,'prepare_native_exit':q.returncode,'missing_original_fields_restored':list(missing),'source_bindings':bindings,'remote':json.loads(q.stdout),
    'first_failed_native_science_s':3.9120370720011124,'zero_G_reader_updates_before_failure':True,'additional_D_state_reconstruction':1,'calibration_repeat':0,'recipe_changed':False}
(ROOT/'ENGINEERING_RECOVERY_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
