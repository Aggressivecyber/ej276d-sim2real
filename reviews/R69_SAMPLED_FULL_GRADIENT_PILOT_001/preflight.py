"""Read native terminal identities and machine availability; never launch science."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
g=apply(P)
roots={str(k):json.loads((R/('work/'+s+'/config.json')).read_text('utf8'))['remote_root']for k,s in
 [(65,'R65_FULL_EMPIRICAL_D_RESPONSE_001'),(66,'R66_EQUAL_COST_PILOT_001'),(67,'R67_EQUAL_COST_REPAIR_001')]}
code='import os,json,subprocess,shutil\nfrom pathlib import Path\nfrom datetime import datetime,timezone\nroots='+repr(roots)+'\n'+r'''
os.sched_setaffinity(0,{11})
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:
  cwd=str((p/'cwd').resolve());cmd=(p/'cmdline').read_bytes().split(b'\0');stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if cwd.startswith(base)and cmd and b'python'in cmd[0]and stat[0]!='Z':active.append(dict(PID=int(p.name),cwd=cwd,state=stat[0]))
 except(OSError,ValueError):pass
native={}
for k,root in roots.items():
 n=json.loads((Path(root)/('train_final.py.NATIVE_RECEIPT.json'if k=='65'else'pilot.py.NATIVE_RECEIPT.json')).read_text())
 native[k]={a:n.get(a)for a in['actual_worker_PID','supervisor_PID','actual_worker_native_exit','elapsed_s','termination_reason']}
assert not active,active
assert [native[k]['actual_worker_native_exit']for k in['65','66','67']]==[-15,1,0]
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),active_project_python_processes=active,native=native,GPU_processes=gpu,GPU_free_MiB=free,MemAvailable_bytes=mem['MemAvailable'],server_disk_free_bytes=shutil.disk_usage(base).free)))
'''
q=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
for label,value in [('stdout',q.stdout),('stderr',q.stderr)]: (P/('PREFLIGHT.'+label+'.log')).write_text(value,encoding='utf8')
assert q.returncode==0,q.stderr
d=json.loads(q.stdout);d.update(local_resource=dict(g,**memory()),local_E_free_bytes=shutil.disk_usage('E:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free)
(P/'NATIVE_PREFLIGHT.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
print(json.dumps(d))
