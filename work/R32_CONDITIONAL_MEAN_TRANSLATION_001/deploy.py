from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));REMOTE=CFG['remote_root']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8']
def run(args,stdin=None):
    p=subprocess.run(args,input=stdin,text=True,capture_output=True,encoding='utf8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:raise RuntimeError(json.dumps({'args':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
    return {'argv':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if (ROOT/'TRANSFER_RECEIPT.json').exists():raise RuntimeError('already deployed; no duplicate migration')
with tarfile.open(ROOT/'UPLOAD.tar','w')as t:
    for name,info in CFG['remote_input_sources'].items():t.add(info['path'],arcname='inputs/'+name,recursive=False)
    for name in ['online_training.py','model.py','legacy_model.py','prior.py','fine_core.py','reorder_core.py','conditional_mean.py',
                 'train_server.py','linux_owned_worker.py','linux_supervise.py','pid_probe.py','PROTOCOL.md']:
        t.add(ROOT/name,arcname=name,recursive=False)
    t.add(ROOT/'remote_config.json',arcname='config.json',recursive=False)
code="from pathlib import Path; p=Path("+repr(REMOTE)+"); assert not p.exists() and not p.is_symlink(); p.mkdir(); print(p)"
create=run(SSH+[CFG['server_python']+' -B -'],code+'\n')
transfer=run(SCP+[str(ROOT/'UPLOAD.tar'),'rylan@100.82.231.102:'+REMOTE+'/UPLOAD.tar'])
code="import tarfile; from pathlib import Path; p=Path("+repr(REMOTE)+"); t=tarfile.open(p/'UPLOAD.tar'); t.extractall(p,filter='data'); print({'entry_exists':(p/'train_server.py').is_file(),'inputs':len(list((p/'inputs').iterdir()))})"
extract=run(SSH+[CFG['server_python']+' -B -'],code+'\n')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote_root':REMOTE,
    'upload_tar_bytes':(ROOT/'UPLOAD.tar').stat().st_size,'readonly_reused_input_bytes':sum(x['bytes']for x in CFG['remote_input_sources'].values()),
    'create':create,'transfer':transfer,'extract':extract,'migration_content_tests':0,'science_started':False}
(ROOT/'TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
