"""Register-bound launch, native observation and verified local collection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,subprocess,tarfile
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'control_runtime');C=json.loads((S/'config.json').read_text('utf8'));remote=C['remote_root']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def ssh(code,label,timeout=30):
    q=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,encoding='utf8',timeout=timeout)
    for kind,value in [('stdout',q.stdout),('stderr',q.stderr)]:(P/(label+'.'+kind+'.log')).write_text(value,'utf8')
    assert q.returncode==0,q.stderr
    return json.loads(q.stdout)
def update(state,terminal):
    now=datetime.now(timezone.utc).isoformat();native=state.get('native')or{};active=state.get('active')or{}
    current=dict(stage=C['stage'],remote_root=remote,status='terminal'if terminal else'running',
        supervisor_PID=native.get('supervisor_PID',active.get('supervisor_PID',state.get('supervisor_PID'))),
        actual_worker_PID=native.get('actual_worker_PID',active.get('actual_worker_PID')),native_exit=native.get('actual_worker_native_exit'),wall_cap_s=None)
    for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
        d=read(R/name);d.update(updated_UTC=now,active_scientific_jobs=0 if terminal else 1,current_job=current,
            current_stage_progress=(state.get('result')or{}).get('status')if terminal else(state.get('progress')or{}).get('phase','native_startup'),
            last_native_observation='reviews/'+P.name+('/LATEST_NATIVE_OBSERVATION.json'if'active'in state else'/LAUNCH_RECEIPT.json'))
        write(R/name,d)
    for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
        d=read(R/name);j=next(v for v in d[key]if v.get('stage',v.get('id'))==C['stage'])
        j.update(current_job=current,status=(state.get('result')or{}).get('status','terminal')if terminal else'running',updated_UTC=now)
        d.update(updated_UTC=now,active_scientific_jobs=0 if terminal else 1,active_remote_scientific_jobs=0 if terminal else 1,active_local_scientific_jobs=0)
        write(R/name,d)
mode=sys.argv[1]
if mode=='launch':
    assert not(P/'LAUNCH_RECEIPT.json').exists()
    reg=read(P/'REGISTRATION.json');assert reg['status']=='REGISTERED_BEFORE_SCIENCE'
    for row in reg['local_files']:assert sha(R/row['path'])==row['sha256'],row['path']
    package=P/'R72_FROZEN_INSTALL.tar';assert not package.exists()
    with tarfile.open(package,'w')as t:
        for f in sorted(S.iterdir()):
            if f.is_file():t.add(f,arcname=f.name,recursive=False)
    ssh('from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\nassert p.parent==Path("/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305")\np.mkdir()\nprint(json.dumps({"created":str(p)}))','REMOTE_PREPARE')
    cp=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(package),'rylan@100.82.231.102:'+remote+'/R72_FROZEN_INSTALL.tar'],text=True,capture_output=True,timeout=30)
    (P/'UPLOAD.stderr.log').write_text(cp.stderr,'utf8');assert cp.returncode==0,cp.stderr
    code='from pathlib import Path\nimport os,json,hashlib,tarfile,subprocess\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
with tarfile.open(p/'R72_FROZEN_INSTALL.tar')as t:
 for m in t.getmembers():assert m.isfile()and'/'not in m.name and'\\'not in m.name
 t.extractall(p,filter='data')
binding=json.loads((p/'FILE_BINDING.json').read_text())
for row in binding['files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['name']
cfg=json.loads((p/'config.json').read_text())
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
retired=json.loads((Path(cfg['R65_root'])/'R65_COST_HOLD_TERMINATION.json').read_text());assert retired['native_exit']==-15
with(p/'SUPERVISOR.stdout.log').open('wb')as out,(p/'SUPERVISOR.stderr.log').open('wb')as err:
 child=subprocess.Popen([cfg['server_python'],'-B',str(p/'linux_supervise.py'),'train_full.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['exact_CPU_affinity'])))
print(json.dumps(dict(remote_root=str(p),supervisor_PID=child.pid,installed_file_count=len(binding['files']),installed_FILE_BINDING_SHA256=hashlib.sha256((p/'FILE_BINDING.json').read_bytes()).hexdigest())),flush=True)
'''
    result=ssh(code,'INSTALL_AND_LAUNCH');result.update(UTC=datetime.now(timezone.utc).isoformat(),local_guard={**guard,**memory()})
    write(P/'LAUNCH_RECEIPT.json',result);update(result,False);print(json.dumps(result))
elif mode=='observe':
    code='from pathlib import Path\nimport os,json\nos.sched_setaffinity(0,{11})\np=Path('+repr(remote)+')\n'+r'''
def read(n):
 f=p/n
 return json.loads(f.read_text())if f.exists()else None
def proc(pid):
 if pid is None:return None
 f=Path('/proc')/str(pid)
 if not f.exists():return None
 try:
  stat=(f/'stat').read_text().rsplit(')',1)[1].split()
  return dict(PID=pid,state=stat[0],start_ticks=int(stat[19]),cwd=str((f/'cwd').resolve()),CPU_affinity=sorted(os.sched_getaffinity(pid)))
 except FileNotFoundError:return None
a=read('REMOTE_ACTIVE_JOB.json')
d=dict(active=a,handshake=read('train_full.py.SCIENCE_PID.json'),native=read('train_full.py.NATIVE_RECEIPT.json'),progress=read('PROGRESS.json'),result=read('RESULT.json'),cost_admission=read('COST_ADMISSION.json'),failure=read('FAILURE.json'),artificial_checks=read('ARTIFICIAL_CHECKS.json'))
if a:d.update(worker=proc(a.get('actual_worker_PID')or a.get('launcher_PID')),supervisor=proc(a.get('supervisor_PID')))
d['stderr_tail']=(p/'train_full.py.stderr.log').read_text()[-5000:]if(p/'train_full.py.stderr.log').exists()else None
d['supervisor_stderr_tail']=(p/'SUPERVISOR.stderr.log').read_text()[-3000:]if(p/'SUPERVISOR.stderr.log').exists()else None
print(json.dumps(d))
'''
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');result=ssh(code,'OBSERVE_'+stamp)
    result['observed_UTC']=datetime.now(timezone.utc).isoformat();write(P/'LATEST_NATIVE_OBSERVATION.json',result)
    terminal=result.get('native')is not None;update(result,terminal)
    cost=result.get('cost_admission')or{}
    print(json.dumps(dict(UTC=result['observed_UTC'],worker=result.get('worker'),progress=result.get('progress'),
        native_exit=(result.get('native')or{}).get('actual_worker_native_exit'),artificial=result.get('artificial_checks'),
        cost={k:cost.get(k)for k in ['pass_gate','median_update_seconds','p90_update_seconds','estimate_full_run_seconds','restored_state_SHA256']},
        result_status=(result.get('result')or{}).get('status'),failure=result.get('failure'),stderr=result['stderr_tail'],supervisor_stderr=result['supervisor_stderr_tail'])))
else:raise ValueError(mode)
