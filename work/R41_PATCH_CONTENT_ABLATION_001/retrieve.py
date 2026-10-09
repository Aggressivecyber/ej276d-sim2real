from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,shutil
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
assert not (ROOT/'RETURN_TRANSFER_RECEIPT.json').exists()
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code='''from pathlib import Path
import json,tarfile
p=Path(__REMOTE__);h=json.loads((p/'REMOTE_RUN_HISTORY.json').read_text());assert h[-1]['actual_worker_native_exit']is not None
files=[]
with tarfile.open(p/'RETURN.tar','w',format=tarfile.USTAR_FORMAT)as tar:
    for q in sorted((p/'NO_PATCH_NCE_FINAL').iterdir()):
        if q.is_file():n='NO_PATCH_NCE_FINAL/'+q.name;tar.add(q,arcname=n,recursive=False);files.append([n,q.stat().st_size])
    for q in sorted(p.iterdir()):
        if not q.is_file()or q.name in ['UPLOAD.tar','RETURN.tar','ENGINEERING_REPAIR_UPLOAD.tar']:continue
        n=q.name if q.name in ['MATCHED_TRAINING_FLOW_RECEIPT.json','INITIAL_EQUIVALENCE.json','TRAINING_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json']else'remote_receipts/'+q.name
        tar.add(q,arcname=n,recursive=False);files.append([n,q.stat().st_size])
assert sum(v for _,v in files)<384*1024**2
print(json.dumps({'return_tar_bytes':(p/'RETURN.tar').stat().st_size,'files':files}))
'''.replace('__REMOTE__',repr(cfg['remote_root']))
r=subprocess.run(SSH+[cfg['server_python']+' -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert r.returncode==0,r.stderr
info=json.loads(r.stdout);cmd=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+cfg['remote_root']+'/RETURN.tar',str(ROOT/'REMOTE_RETURN.tar')]
s=subprocess.run(cmd,text=True,capture_output=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert s.returncode==0,s.stderr
with tarfile.open(ROOT/'REMOTE_RETURN.tar')as tar:tar.extractall(ROOT,filter='data')
history=json.loads((ROOT/'remote_receipts/REMOTE_RUN_HISTORY.json').read_text('utf8'))
for row in history:row.update(execution_location='registered Linux server',remote_root=cfg['remote_root'])
(ROOT/'RUN_HISTORY.json').write_text(json.dumps(history,indent=2)+'\n','utf8')
shutil.copy2(ROOT/'remote_receipts/TRAINING_NUMERIC_RECEIPT.json',ROOT/'TRAINING_NUMERIC_RECEIPT.json')
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'pack_native_exit':r.returncode,'scp_native_exit':s.returncode,
    'return_tar_bytes':(ROOT/'REMOTE_RETURN.tar').stat().st_size,'migration_content_tests':0,'remote_outputs':info,
    'remote_science_s':sum(q['elapsed_s']for q in history)}
(ROOT/'RETURN_TRANSFER_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'native_pack':r.returncode,'native_scp':s.returncode,'bytes':rec['return_tar_bytes'],
    'science_s':rec['remote_science_s'],'remote_worker_exit':history[-1]['actual_worker_native_exit'],
    'training_counter':json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text('utf8'))['counter']}),flush=True)
