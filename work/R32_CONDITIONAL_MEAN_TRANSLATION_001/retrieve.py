from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,shutil
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
if (ROOT/'RETURN_TRANSFER_RECEIPT.json').exists():raise RuntimeError('results already returned')
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code='''from pathlib import Path
import json,tarfile
p=Path(__REMOTE_ROOT__)
h=json.loads((p/'REMOTE_RUN_HISTORY.json').read_text())
assert h[-1]['entry']=='train_server.py' and h[-1]['actual_worker_native_exit']==0
files=[]
with tarfile.open(p/'RETURN.tar','w')as tar:
    for arm in ['SERVER_FRESH_CONTROL','CONDITIONAL_MEAN']:
        for q in sorted((p/arm).iterdir()):
            if q.is_file():tar.add(q,arcname=arm+'/'+q.name,recursive=False);files.append((arm+'/'+q.name,q.stat().st_size))
    for q in sorted(p.iterdir()):
        if not q.is_file()or q.name in ['UPLOAD.tar','RETURN.tar']:continue
        if q.name in ['MEAN_TARGETS_PRIVATE.npz','MEAN_TARGETS_METADATA.json','MEAN_FORMULA_CHECK.json','PAIRED_FLOW_RECEIPT.json','REORDER_CONSTRUCTION_RECEIPT.json']:
            name=q.name
        else:name='remote_receipts/'+q.name
        tar.add(q,arcname=name,recursive=False);files.append((name,q.stat().st_size))
assert sum(n for _,n in files)<512*1024**2
print(json.dumps({'return_tar_bytes':(p/'RETURN.tar').stat().st_size,'files':files,'normal_pack_no_content_testing':True}))
'''.replace('__REMOTE_ROOT__',repr(cfg['remote_root']))
p=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if p.returncode:raise RuntimeError(p.stderr)
info=json.loads(p.stdout)
command=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+cfg['remote_root']+'/RETURN.tar',str(ROOT/'REMOTE_RETURN.tar')]
t=subprocess.run(command,capture_output=True,text=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if t.returncode:raise RuntimeError(t.stderr)
with tarfile.open(ROOT/'REMOTE_RETURN.tar')as tar:tar.extractall(ROOT,filter='data')
history=json.loads((ROOT/'remote_receipts/REMOTE_RUN_HISTORY.json').read_text('utf8'))
for row in history:row['execution_location']='registered Linux server';row['remote_root']=cfg['remote_root']
(ROOT/'RUN_HISTORY.json').write_text(json.dumps(history,indent=2)+'\n','utf8')
shutil.copy2(ROOT/'remote_receipts/TRAINING_NUMERIC_RECEIPT.json',ROOT/'TRAINING_NUMERIC_RECEIPT.json')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote_pack_native_exit':p.returncode,
    'scp_native_exit':t.returncode,'command':command,'return_tar_bytes':(ROOT/'REMOTE_RETURN.tar').stat().st_size,
    'normal_transfer_and_tar_extraction':True,'migration_content_testing':0,'remote_outputs':info,
    'remote_science_cumulative_s':sum(r['elapsed_s']for r in history)}
(ROOT/'RETURN_TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
tr=json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text())
print(json.dumps({'PID':os.getpid(),'UTC':receipt['UTC'],'native_pack':p.returncode,'native_scp':t.returncode,
    'return_bytes':receipt['return_tar_bytes'],'remote_science_s':receipt['remote_science_cumulative_s'],
    'records':{k:{'updates':r['updates'],'selected_step':r['selected_step'],'selected_J':r['selected_validation']['J'],
        'final_J':r['final_validation']['J'],'GPU_peak_allocated_bytes':r['GPU_peak_allocated_bytes'],'mean_negative_batches':r['mean_negative_batches']}for k,r in tr['records'].items()}}),flush=True)
