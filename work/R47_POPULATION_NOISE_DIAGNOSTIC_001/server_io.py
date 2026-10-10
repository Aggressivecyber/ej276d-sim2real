"""Normal small controls and transfer; no migration consistency testing."""
from pathlib import Path
from datetime import datetime, timezone
import json,os,sys,subprocess,time,tarfile
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text());MODE=sys.argv[1]
REMOTE=C['remote_root'];PY=C['server_python'];START=time.monotonic();calls=[]
SSH=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
SCP=['scp','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8']
def run(name,args,body=None):
    p=subprocess.run(args,input=body,text=True,capture_output=True)
    (R/(MODE+'_'+name+'.stdout.log')).write_bytes(p.stdout.encode())
    (R/(MODE+'_'+name+'.stderr.log')).write_bytes(p.stderr.encode())
    calls.append({'kind':name,'argv':args,'native_exit':p.returncode})
    if p.returncode:raise RuntimeError((name,p.returncode,p.stderr[-1000:]))
    return p.stdout
try:
    if MODE=='preflight':
        code=f'''from pathlib import Path
import os,sys,json,shutil
from datetime import datetime,timezone
project='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305';jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python' in argv and (project in argv or cwd.startswith(project)):jobs.append({{'PID':int(p.name),'argv':argv,'cwd':cwd}})
 except(OSError,PermissionError):pass
mem={{l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}}
inputs={C['inputs']!r}
print(json.dumps({{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'cwd':os.getcwd(),'own_project_jobs':jobs,
 'CPU_logical':os.cpu_count(),'MemAvailable_bytes':mem['MemAvailable'],'disk_free_bytes':shutil.disk_usage(project).free,
 'remote_leaf':{REMOTE!r},'remote_leaf_exists':Path({REMOTE!r}).exists(),'inputs_exist':{{n:Path(p).is_file() for n,p in inputs.items()}},'GPU_science_jobs':0}}))
'''
        v=json.loads(run('query',SSH+[PY+' -B -'],code));(R/'SERVER_PREFLIGHT.json').write_bytes((json.dumps(v,indent=2)+'\n').encode())
        assert not v['own_project_jobs'] and not v['remote_leaf_exists'] and all(v['inputs_exist'].values()),v
        assert v['disk_free_bytes']>=C['disk_free_reserve_bytes']+64*1024**2
        if v['MemAvailable_bytes']<56*1024**3:
            C['RSS_limit_bytes']=min(C['RSS_limit_bytes'],v['MemAvailable_bytes']-8*1024**3)
            assert C['RSS_limit_bytes']>=4*1024**3
            C['lower_RAM_quota_reason']='actual MemAvailable below56GiB'
            (R/'config.json').write_bytes((json.dumps(C,indent=2)+'\n').encode())
        print(json.dumps(v),flush=True)
    elif MODE=='deploy':
        assert json.loads((R/'SERVER_PREFLIGHT.json').read_text())['own_project_jobs']==[]
        names=['config.json','numeric_worker.py','evaluation.py','diagnostic_core.py','common_math.py','linux_supervise.py','linux_owned_worker.py','PROTOCOL.md','SOURCE_REUSE_BINDINGS.json']
        archive=R/'UPLOAD_CODE.tar'
        with tarfile.open(archive,'w') as t:
            for n in names:t.add(R/n,arcname=n,recursive=False)
        run('create',SSH+[PY+' -B -'],f"from pathlib import Path\np=Path({REMOTE!r});assert not p.exists() and not p.is_symlink();p.mkdir(parents=True,exist_ok=False);print(p)\n")
        run('upload',SCP+[str(archive),'rylan@100.82.231.102:'+REMOTE+'/'+archive.name])
        run('extract',SSH+['cd '+REMOTE+' && tar -xf '+archive.name+' --no-same-owner --no-same-permissions'])
    elif MODE=='fetch':
        archive='RETURN_DIAGNOSTIC.tar'
        code=f'''from pathlib import Path
import tarfile,json,os
R=Path({REMOTE!r});names=['results','metrics.json','METRICS_COMPACT.json','NUMERIC_RECEIPT.json','INPUT_IDENTITY_CHECK.json','DIAGNOSTIC_FAILURE.json','COUNTERS.json','DIAGNOSTIC_PROGRESS.json','REMOTE_RUN_HISTORY.json','REMOTE_ACTIVE_JOB.json','SERVER_START_RESOURCE.json']
names+=["numeric_worker.py"+s for s in ['.NATIVE_RECEIPT.json','.SCIENCE_PID.json','.RSS_ACK.json','.WORKER_RECEIPT.json','.stdout.log','.stderr.log']]
with tarfile.open(R/{archive!r},'w')as t:
 for n in names:
  p=R/n
  if p.exists():t.add(p,arcname=n)
print(json.dumps({{'PID':os.getpid(),'archive':str(R/{archive!r}),'bytes':(R/{archive!r}).stat().st_size}}))
'''
        run('archive',SSH+[PY+' -B -'],code)
        run('download',SCP+['rylan@100.82.231.102:'+REMOTE+'/'+archive,str(R/archive)])
        with tarfile.open(R/archive,'r')as t:t.extractall(R,filter='data')
    else:raise ValueError(MODE)
finally:
    rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'mode':MODE,'commands':calls,'elapsed_s':time.monotonic()-START,
         'science_s':0,'normal_transfer_only':True,'migration_content_consistency_test':False}
    (R/(MODE.upper()+'_NATIVE_RECEIPT.json')).write_bytes((json.dumps(rec,indent=2)+'\n').encode());print(json.dumps(rec),flush=True)
