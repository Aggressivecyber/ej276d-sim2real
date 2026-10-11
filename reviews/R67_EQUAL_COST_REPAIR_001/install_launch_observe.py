"""Install once, then observe only this R67 native process identity."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,tarfile,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R67_EQUAL_COST_REPAIR_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
guard=apply(P);C=json.loads((S/'config.json').read_text());remote=C['remote_root']
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(code,name,timeout=25):
    out=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=timeout)
    (P/(name+'.stdout.log')).write_text(out.stdout,encoding='utf8');(P/(name+'.stderr.log')).write_text(out.stderr,encoding='utf8')
    assert out.returncode==0,out.stderr
    return json.loads(out.stdout)
mode=sys.argv[1]
if mode=='launch':
    assert not (P/'LAUNCH_RECEIPT.json').exists()
    binding=json.loads((S/'FILE_BINDING.json').read_text())
    assert any(x['name']=='R63/train_final.py'for x in binding['files'])
    package=P/'R67_FROZEN_INSTALL.tar'
    with tarfile.open(package,'w')as t:
        for f in sorted(S.iterdir()):
            if f.is_file():t.add(f,arcname=f.name,recursive=False)
    prep=run('from pathlib import Path\nimport os,json\np=Path('+repr(remote)+')\nos.sched_setaffinity(0,{11})\nassert p.parent==Path("/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305")\np.mkdir()\nprint(json.dumps({"created":str(p)}))\n','REMOTE_PREPARE')
    cp=subprocess.run(['scp','-q',str(package),'rylan@100.82.231.102:'+remote+'/R67_FROZEN_INSTALL.tar'],text=True,capture_output=True,timeout=25)
    (P/'UPLOAD.stdout.log').write_text(cp.stdout,encoding='utf8');(P/'UPLOAD.stderr.log').write_text(cp.stderr,encoding='utf8');assert cp.returncode==0,cp.stderr
    code='from pathlib import Path\nimport os,json,tarfile,hashlib,subprocess\np=Path('+repr(remote)+')\nos.sched_setaffinity(0,{11})\n'+r'''
with tarfile.open(p/'R67_FROZEN_INSTALL.tar')as t:
 for m in t.getmembers():assert m.isfile()and'/'not in m.name and'\\'not in m.name
 t.extractall(p,filter='data')
binding=json.loads((p/'FILE_BINDING.json').read_text())
for r in binding['files']:
 f=Path(r['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['name']
cfg=json.loads((p/'config.json').read_text())
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
old=Path(cfg['R65_root']);retired=json.loads((old/'R65_COST_HOLD_TERMINATION.json').read_text())
prior=Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R66_ROOT_20261011_070346_UTC')
prior_native=json.loads((prior/'pilot.py.NATIVE_RECEIPT.json').read_text())
assert prior_native['actual_worker_native_exit']==1
assert prior_native['elapsed_s']+cfg['worker_cap_s']<cfg['aggregate_budget_s']
for key in ['actual_worker_PID','supervisor_PID']:
 q=Path('/proc')/str(prior_native[key])
 if q.exists():assert(q/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
assert retired['native_exit']==-15
for key in ['worker','supervisor']:
 pid=retired['before'][key]['PID'];proc=Path('/proc')/str(pid)
 if proc.exists():assert(proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
with(p/'SUPERVISOR.stdout.log').open('wb')as out,(p/'SUPERVISOR.stderr.log').open('wb')as err:
 child=subprocess.Popen([cfg['server_python'],'-B',str(p/'linux_supervise.py'),'pilot.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['exact_CPU_affinity'])))
print(json.dumps(dict(remote_root=str(p),supervisor_PID=child.pid,installed_file_count=len(binding['files']),installed_FILE_BINDING_SHA256=hashlib.sha256((p/'FILE_BINDING.json').read_bytes()).hexdigest())),flush=True)
'''
    result=run(code,'INSTALL_AND_LAUNCH');result.update(UTC=datetime.now(timezone.utc).isoformat(),local_guard=guard)
    write(P/'LAUNCH_RECEIPT.json',result)
    for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
        d=json.loads((R/name).read_text());d.update(status='R67_capped_pilot_launching',project_status='R67_capped_pilot_launching',
            active_scientific_jobs=1,current_activity='equal_cost_capped_pilot',current_stage_progress='launching_native_identity_pending',
            current_job=dict(stage=C['stage'],remote_root=remote,supervisor_PID=result['supervisor_PID'],wall_cap_s=C['worker_cap_s'],native_exit=None),
            last_native_observation='reviews/R67_EQUAL_COST_REPAIR_001/LAUNCH_RECEIPT.json')
        write(R/name,d)
    print(json.dumps(result,indent=2))
elif mode=='observe':
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    code='from pathlib import Path\nimport json,os\np=Path('+repr(remote)+')\nos.sched_setaffinity(0,{11})\n'+r'''
def read(n):
 f=p/n
 return json.loads(f.read_text())if f.exists()else None
def proc(pid):
 if pid is None:return None
 f=Path('/proc')/str(pid)
 if not f.exists():return None
 try:
  stat=(f/'stat').read_text().rsplit(')',1)[1].split()
  return dict(PID=pid,state=stat[0],start_ticks=int(stat[19]),command=(f/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=str((f/'cwd').resolve()),CPU_affinity=sorted(os.sched_getaffinity(pid)))
 except FileNotFoundError:return None
a=read('REMOTE_ACTIVE_JOB.json')
res=dict(active=a,handshake=read('pilot.py.SCIENCE_PID.json'),native=read('pilot.py.NATIVE_RECEIPT.json'),progress=read('PROGRESS.json'),
 result=read('RESULT.json'),baseline_budget=read('BASELINE_BUDGET_GATE.json'))
if a:res.update(worker=proc(a.get('actual_worker_PID')or a.get('launcher_PID')),supervisor=proc(a.get('supervisor_PID')))
res['stderr_tail']=(p/'pilot.py.stderr.log').read_text()[-6000:]if(p/'pilot.py.stderr.log').exists()else None
res['supervisor_stderr_tail']=(p/'SUPERVISOR.stderr.log').read_text()[-3000:]if(p/'SUPERVISOR.stderr.log').exists()else None
print(json.dumps(res))
'''
    result=run(code,'OBSERVE_'+stamp);result['observed_UTC']=datetime.now(timezone.utc).isoformat()
    write(P/'LATEST_NATIVE_OBSERVATION.json',result)
    for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
        d=json.loads((R/name).read_text());a=result.get('active')or{};terminal=result.get('native')is not None
        d.update(active_scientific_jobs=0 if terminal else 1,current_stage_progress=(result.get('result')or{}).get('status')if terminal else(result.get('progress')or{}).get('phase','startup'),
            last_native_observation='reviews/R67_EQUAL_COST_REPAIR_001/LATEST_NATIVE_OBSERVATION.json',
            current_job=dict(stage=C['stage'],remote_root=remote,supervisor_PID=a.get('supervisor_PID'),actual_worker_PID=a.get('actual_worker_PID'),
                native_exit=a.get('actual_worker_native_exit'),wall_cap_s=C['worker_cap_s'],status='terminal'if terminal else'running'))
        write(R/name,d)
    print(json.dumps(dict(UTC=result['observed_UTC'],worker=result.get('worker'),supervisor=result.get('supervisor'),
        progress=result.get('progress'),native_exit=(result.get('native')or{}).get('actual_worker_native_exit'),
        result_status=(result.get('result')or{}).get('status'),baseline_budget=result.get('baseline_budget'),stderr=result.get('stderr_tail'),
        supervisor_stderr=result.get('supervisor_stderr_tail')),indent=2))
else:raise ValueError(mode)
