"""Register-bound launch and native observation for sequential R78 evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import tarfile

P = Path(__file__).resolve().parent
R = P.parents[1]
S = R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P/'control_runtime')
C = json.loads((S/'config.json').read_text('utf8'))
remote = C['remote_root']


def read(path):
    return json.loads(path.read_text('utf8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ssh(code,label,timeout=45):
    proc = subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8',
        'rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,
        capture_output=True,encoding='utf8',timeout=timeout)
    for kind,value in [('stdout',proc.stdout),('stderr',proc.stderr)]:
        (P/(label+'.'+kind+'.log')).write_text(value,'utf8')
    assert proc.returncode == 0,proc.stderr
    return json.loads(proc.stdout)


def update(state,terminal):
    now = datetime.now(timezone.utc).isoformat()
    active = state.get('active')or{}
    progress = state.get('pipeline_progress')or{}
    completion = state.get('completion')or{}
    current = dict(stage=C['stage'],remote_root=remote,status='terminal'if terminal else'running',
        pipeline_shell_PID=(state.get('pipeline_identity')or{}).get('shell_PID',state.get('shell_PID')),
        actual_worker_PID=active.get('actual_worker_PID'),current_entry=progress.get('current_entry'),
        completion_status=completion.get('status'),pipeline_shell_native_exit='unknown',wall_cap_s=None)
    for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
        d = read(R/name)
        d.update(updated_UTC=now,active_scientific_jobs=0 if terminal else 1,current_job=current,
            current_stage_progress=completion.get('status')if terminal else progress.get('current_entry','native_startup'),
            last_native_observation='reviews/'+P.name+('/LATEST_NATIVE_OBSERVATION.json'if'active'in state else'/LAUNCH_RECEIPT.json'))
        write(R/name,d)
    for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
        d = read(R/name)
        row = next(v for v in d[key]if v.get('stage',v.get('id')) == C['stage'])
        row.update(current_job=current,status=completion.get('status','terminal')if terminal else'running',updated_UTC=now)
        d.update(updated_UTC=now,active_scientific_jobs=0 if terminal else 1,
            active_remote_scientific_jobs=0 if terminal else 1,active_local_scientific_jobs=0)
        write(R/name,d)


mode = sys.argv[1]
if mode == 'launch':
    assert not (P/'LAUNCH_RECEIPT.json').exists()
    reg = read(P/'REGISTRATION.json')
    assert reg['status'] == 'REGISTERED_BEFORE_SCIENCE'
    assert read(R/'jobs.json')['active_scientific_jobs'] == 0
    for row in reg['local_files']:
        assert sha(R/row['path']) == row['sha256'],row['path']
    package = P/'R78_FROZEN_INSTALL.tar'
    assert not package.exists()
    with tarfile.open(package,'w')as tar:
        for path in sorted(S.iterdir()):
            if path.is_file():tar.add(path,arcname=path.name,recursive=False)
    ssh('from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'
        'assert p.parent == Path("/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305")\n'
        'p.mkdir()\nprint(json.dumps({"created":str(p)}))','REMOTE_PREPARE')
    proc = subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',
        str(package),'rylan@100.82.231.102:'+remote+'/R78_FROZEN_INSTALL.tar'],
        text=True,capture_output=True,timeout=30)
    (P/'UPLOAD.stderr.log').write_text(proc.stderr,'utf8')
    assert proc.returncode == 0,proc.stderr
    code = 'from pathlib import Path\nimport os,json,hashlib,tarfile,subprocess\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
with tarfile.open(p/'R78_FROZEN_INSTALL.tar')as tar:
 for item in tar.getmembers():assert item.isfile()and'/'not in item.name and'\\'not in item.name
 tar.extractall(p,filter='data')
binding=json.loads((p/'FILE_BINDING.json').read_text())
for row in binding['files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['name']
cfg=json.loads((p/'config.json').read_text())
for path in p.glob('*.py'):compile(path.read_text(),str(path),'exec')
checked=subprocess.run(['/bin/bash','-n',str(p/'run_pipeline.sh')],capture_output=True,text=True)
assert checked.returncode==0,checked.stderr
train=json.loads(Path(cfg['training']['native_receipt']['path']).read_text())
assert train['actual_worker_native_exit']==0 and train['termination_reason']is None
assert not Path('/proc/'+str(train['actual_worker_PID'])).exists()
retired=json.loads((Path(cfg['R65_root'])/'R65_COST_HOLD_TERMINATION.json').read_text())
assert retired['native_exit']==-15
with(p/'PIPELINE.stdout.log').open('wb')as out,(p/'PIPELINE.stderr.log').open('wb')as err:
 child=subprocess.Popen(['/bin/bash',str(p/'run_pipeline.sh')],cwd=p,stdin=subprocess.DEVNULL,
  stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['exact_CPU_affinity'])))
stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(remote_root=str(p),shell_PID=child.pid,shell_start_ticks=int(stat[19]),
 installed_file_count=len(binding['files']),installed_FILE_BINDING_SHA256=hashlib.sha256((p/'FILE_BINDING.json').read_bytes()).hexdigest())),flush=True)
'''
    result = ssh(code,'INSTALL_AND_LAUNCH')
    result.update(UTC=datetime.now(timezone.utc).isoformat(),local_resource={**guard,**memory()})
    write(P/'LAUNCH_RECEIPT.json',result)
    update(result,False)
    print(json.dumps(result),flush=True)
elif mode == 'observe':
    code = 'from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
def read(name):
 path=p/name
 return json.loads(path.read_text())if path.exists()else None
def proc(pid):
 if pid is None:return None
 path=Path('/proc')/str(pid)
 if not path.exists():return None
 try:
  stat=(path/'stat').read_text().rsplit(')',1)[1].split()
  return dict(PID=pid,state=stat[0],start_ticks=int(stat[19]),cwd=str((path/'cwd').resolve()),CPU_affinity=sorted(os.sched_getaffinity(pid)))
 except FileNotFoundError:return None
active=read('REMOTE_ACTIVE_JOB.json');identity=read('PIPELINE_IDENTITY.json')
current=(read('PIPELINE_PROGRESS.json')or{}).get('current_entry')
result=dict(pipeline_identity=identity,pipeline_progress=read('PIPELINE_PROGRESS.json'),
 completion=read('PIPELINE_COMPLETION.json'),active=active,
 generation=read('GENERATION_RECEIPT.json'),replay=read('REPLAY_RECEIPT.json'),
 evaluation_progress=read('EVALUATION_PROGRESS.json'),evaluation_receipt=read('EVALUATION_NUMERIC_RECEIPT.json'),
 gate=read('GATE_METRICS.json'),summary=read('SUMMARY_PLOT_NUMERIC_RECEIPT.json'),
 failures={n:read(n)for n in ['GENERATION_FAILURE.json','REPLAY_FAILURE.json','EVALUATION_FAILURE.json','SUMMARY_FAILURE.json']})
result['pipeline_process']=proc(identity['shell_PID'])if identity else None
result['worker']=proc(active.get('actual_worker_PID')or active.get('launcher_PID'))if active else None
result['current_native']=read(current+'.NATIVE_RECEIPT.json')if current else None
result['entry_stderr_tail']=(p/(current+'.stderr.log')).read_text()[-4000:]if current and(p/(current+'.stderr.log')).exists()else None
result['pipeline_stderr_tail']=(p/'PIPELINE.stderr.log').read_text()[-4000:]if(p/'PIPELINE.stderr.log').exists()else None
print(json.dumps(result))
'''
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    result = ssh(code,'OBSERVE_'+stamp)
    result['observed_UTC'] = datetime.now(timezone.utc).isoformat()
    write(P/'LATEST_NATIVE_OBSERVATION.json',result)
    update(result,result['completion'] is not None)
    print(json.dumps(dict(UTC=result['observed_UTC'],worker=result['worker'],
        pipeline_process=result['pipeline_process'],progress=result['pipeline_progress'],
        generation_status=(result['generation']or{}).get('status'),
        replay_pass=(result['replay']or{}).get('pass'),evaluation_progress=result['evaluation_progress'],
        primary_A_C_pass=(result['gate']or{}).get('primary_A_C_pass'),completion=result['completion'],
        failures=result['failures'],stderr=result['entry_stderr_tail'],pipeline_stderr=result['pipeline_stderr_tail'])),flush=True)
else:
    raise ValueError(mode)
