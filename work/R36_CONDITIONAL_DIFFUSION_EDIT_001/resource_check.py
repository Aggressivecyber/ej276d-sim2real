from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,time
ROOT=Path(__file__).resolve().parent
REMOTE='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R35_REMOTE_20261009_190721_UTC'
PY='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
code=r'''
from pathlib import Path
import os,json,sys,datetime,subprocess,shutil,importlib.metadata,time
start=time.monotonic();prefix='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'; jobs=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in [os.getpid(),os.getppid()]: continue
    try:
        if entry.stat().st_uid!=os.getuid(): continue
        argv=(entry/'cmdline').read_bytes().replace(b'\x00',b' ').decode('utf8','replace')
        cwd=os.readlink(entry/'cwd')
        if prefix in argv or cwd.startswith(prefix):jobs.append({'PID':int(entry.name),'argv':argv,'cwd':cwd})
    except (OSError,PermissionError):pass
mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True)
import numpy as np,torch,scipy
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python_path':sys.executable,'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
    'device':'cuda:0' if torch.cuda.is_available() else 'cpu','GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'own_project_jobs':jobs,'CPU_N':os.cpu_count(),'MemAvailable':mem['MemAvailable'],
    'disk_free_bytes':shutil.disk_usage(prefix).free,'gpu_query_exit':gpu.returncode,'gpu_resource':gpu.stdout.strip(),
    'new_remote_leaf_exists':Path(__REMOTE__).exists(),'check_elapsed_s':time.monotonic()-start,'new_response_operations':0}))
'''.replace('__REMOTE__',repr(REMOTE))
start=time.monotonic()
p=subprocess.Popen(SSH+[PY+' -B -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
try:out,err=p.communicate(code,timeout=40)
except subprocess.TimeoutExpired:
    p.kill();out,err=p.communicate()
    (ROOT/'RESOURCE_CHECK_TIMEOUT.json').write_text(json.dumps({'PID':os.getpid(),'SSH_PID':p.pid,'native_exit':p.returncode,'UTC':datetime.now(timezone.utc).isoformat()}),'utf8')
    raise
(ROOT/'RESOURCE_CHECK.stdout.log').write_text(out,'utf8');(ROOT/'RESOURCE_CHECK.stderr.log').write_text(err,'utf8')
receipt={'local_PID':os.getpid(),'SSH_PID':p.pid,'native_exit':p.returncode,'elapsed_s':time.monotonic()-start,'UTC':datetime.now(timezone.utc).isoformat(),'command':SSH+[PY+' -B -']}
(ROOT/'RESOURCE_CHECK_NATIVE.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
assert p.returncode==0,err
info=json.loads(out)
old=json.loads((ROOT.parent/'R33_RESIDUAL_OT_FLOW_001/remote_receipts/SERVER_ENVIRONMENT.json').read_text('utf8'))
same={k:info[k]==old[k] for k in ['python','numpy','torch','scipy','device','GPU']}
info['R33_environment_exact_fields_match']=same
(ROOT/'SERVER_PREFLIGHT.json').write_text(json.dumps(info,indent=2)+'\n','utf8')
print(json.dumps({'control':receipt,'server':info}),flush=True)
assert all(same.values()),'R33 environment differs'
assert not info['own_project_jobs'],'unknown active project job; do not submit'
assert not info['new_remote_leaf_exists'],'new remote leaf already exists; do not submit'
