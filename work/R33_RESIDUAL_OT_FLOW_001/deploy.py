from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,tarfile,ast,hashlib
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));REMOTE=CFG['remote_root']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-q','-o','BatchMode=yes','-o','ConnectTimeout=8']
def run(args,stdin=None):
    p=subprocess.run(args,input=stdin,text=True,capture_output=True,encoding='utf8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:raise RuntimeError(json.dumps({'argv':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
    return {'argv':args,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if (ROOT/'TRANSFER_RECEIPT.json').exists():raise RuntimeError('already deployed; inspect identity, no duplicate')
sources=['flow_core.py','train_server.py','model.py','legacy_model.py','reorder_core.py','linux_owned_worker.py','linux_supervise.py']
binding=[]
for name in sources:
    raw=(ROOT/name).read_bytes();ast.parse(raw.decode('utf8'));binding.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
(ROOT/'PRETRAIN_FREEZE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),
    'scientific_config':CFG,'source_bindings':binding,'no_response_updates_before_freeze':True},indent=2)+'\n','utf8')
with tarfile.open(ROOT/'UPLOAD.tar','w')as t:
    for name,info in CFG['remote_input_sources'].items():t.add(info['path'],arcname='inputs/'+name,recursive=False)
    for name in [*sources,'PROTOCOL.md','PRETRAIN_FREEZE.json']:t.add(ROOT/name,arcname=name,recursive=False)
    t.add(ROOT/'remote_config.json',arcname='config.json',recursive=False)
code="from pathlib import Path; p=Path("+repr(REMOTE)+"); assert not p.exists() and not p.is_symlink(); p.mkdir(); print(p)"
create=run(SSH+[CFG['server_python']+' -B -'],code+'\n')
transfer=run(SCP+[str(ROOT/'UPLOAD.tar'),'rylan@100.82.231.102:'+REMOTE+'/UPLOAD.tar'])
code="import tarfile; from pathlib import Path; p=Path("+repr(REMOTE)+"); t=tarfile.open(p/'UPLOAD.tar'); t.extractall(p,filter='data'); print({'entry_exists':(p/'train_server.py').is_file(),'inputs':len(list((p/'inputs').iterdir()))})"
extract=run(SSH+[CFG['server_python']+' -B -'],code+'\n')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote_root':REMOTE,
    'upload_tar_bytes':(ROOT/'UPLOAD.tar').stat().st_size,'readonly_reused_input_bytes':sum(v['bytes']for v in CFG['remote_input_sources'].values()),
    'create':create,'transfer':transfer,'extract':extract,'migration_content_tests':0,'science_started':False}
(ROOT/'TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':receipt['UTC'],'upload_tar_bytes':receipt['upload_tar_bytes'],
                  'create_native':0,'scp_native':0,'extract_native':0,'entry_exists':True}))
