"""Publish frozen R65 sources and start its one owned precheck/training entry."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
import subprocess
import tarfile

R = Path(__file__).resolve().parents[1]
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
C = json.loads((S/'config.json').read_text('utf8'))
remote = C['remote_root']
host = 'rylan@100.82.231.102'
k = ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h = k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
cache = S/'cache/local_deployment'
cache.mkdir(parents=True,exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):
    os.environ[key] = str(cache)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_')
review = json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
assert review['ready_for_launch'] and review['original_A_F_changed'] is False
assert not (S/'DEPLOYMENT_RECEIPT.json').exists(), 'Inspect existing deployment/native identity before any retry'
events = []
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for record in review['files']:
    assert sha(S/record['name']) == record['sha256'],record['name']
names = [record['name'] for record in review['files']]+['ROOT_IMPLEMENTATION_REVIEW.json']
assert len(names) == len(set(names))
archive = cache/'R65_FROZEN_SOURCE_BUNDLE.tar'
with tarfile.open(archive,'w') as bundle:
    for name in names:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert (S/name).is_file()
        bundle.add(S/name,arcname=name,recursive=False)
ssh = ['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']


def invoke(argv,code=None):
    p = subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    receipt = {'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),
        'CPU_affinity':[3],'native_RSS_hard_max_bytes':1024**3,'events':events,
        'normal_transfer_no_separate_migration_consistency_test':True}
    (S/'DEPLOYMENT_RECEIPT.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    assert p.returncode == 0,events[-1]
    return p.stdout


prefix = 'from pathlib import Path\nimport json,subprocess,time,os\np=Path('+repr(remote)+')\n'
checks = prefix+'\n'.join('assert Path('+repr(p)+').is_file()' for p in [C['server_python'],*C['input_paths'].values()])
checks += '\nassert p.parent.is_dir()\np.mkdir(exist_ok=False)\nprint(str(p))\n'
invoke(ssh,checks)
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(archive),host+':'+remote+'/R65_FROZEN_SOURCE_BUNDLE.tar'])
extract = prefix+'''
import tarfile
with tarfile.open(p/'R65_FROZEN_SOURCE_BUNDLE.tar','r') as bundle:
    for member in bundle.getmembers():
        q=Path(member.name)
        assert member.isfile() and not q.is_absolute() and '..' not in q.parts
        assert p.resolve() in (p/q).resolve().parents
    bundle.extractall(p,filter='data')
    member_count=len(bundle.getmembers())
print(json.dumps({'files_extracted':member_count,'scientific_arrays_decoded':False}))
'''
invoke(ssh,extract)
launch = prefix+'python='+repr(C['server_python'])+'\n'+'''
marker=p/'LAUNCH_train_final.py.json'
with marker.open('x') as stream:
    json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},stream)
out=(p/'train_final.py.supervisor.stdout.log').open('wb')
err=(p/'train_final.py.supervisor.stderr.log').open('wb')
cpus=sorted(os.sched_getaffinity(0))[:12]
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'train_final.py'],
    cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
    preexec_fn=lambda:os.sched_setaffinity(0,cpus))
record={'entry':'train_final.py','supervisor_PID':child.pid,'launch_time_unix':time.time(),
        'stage':str(p),'inherited_CPU_affinity':cpus}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
active=p/'REMOTE_ACTIVE_JOB.json'
record['actual_job']=json.loads(active.read_text()) if active.exists() else None
record['immediate_supervisor_exit']=child.poll()
pid=(record['actual_job'] or {}).get('actual_worker_PID')
if pid and Path('/proc',str(pid)).exists():
    record['actual_CPU_affinity']=sorted(os.sched_getaffinity(pid))
    assert record['actual_CPU_affinity']==cpus
print(json.dumps(record,indent=2))
'''
observed = json.loads(invoke(ssh,launch))
(S/'LAUNCH_RECEIPT.json').write_bytes((json.dumps(observed,indent=2)+'\n').encode())
assert observed['actual_job'] is not None,observed
assert observed['immediate_supervisor_exit'] is None,observed
print(json.dumps(observed,indent=2))
