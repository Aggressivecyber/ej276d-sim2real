"""Collect a terminal pilot without rerunning any scientific computation."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,subprocess,hashlib,tarfile,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R67_EQUAL_COST_REPAIR_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P)
C=json.loads((S/'config.json').read_text());remote=C['remote_root']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
code='from pathlib import Path\nimport os,json,hashlib,tarfile,subprocess\nfrom datetime import datetime,timezone\np=Path('+repr(remote)+')\nos.sched_setaffinity(0,{11})\n'+r'''
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
native=json.loads((p/'pilot.py.NATIVE_RECEIPT.json').read_text())
assert native['actual_worker_native_exit'] is not None
identities={}
for key in ['actual_worker_PID','supervisor_PID']:
 pid=native[key];proc=Path('/proc')/str(pid)
 state=None
 if proc.exists():
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split();state=stat[0]
  assert state=='Z',(key,state)
 identities[key]=dict(PID=pid,state=state,terminal=True)
cfg=json.loads((p/'config.json').read_text())
assert sha(Path(cfg['R65_root'])/'cache/RECOVERY_SAME_RUN.pt')==cfg['checkpoint_SHA256']
manifest=[]
for f in sorted(p.rglob('*')):
 if not f.is_file()or'cache'in f.relative_to(p).parts or f.suffix=='.tar':continue
 manifest.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
bundle=p/'R67_COMPLETED_EXPORT.tar'
assert not bundle.exists()
with tarfile.open(bundle,'w')as t:
 for row in manifest:t.add(p/row['path'],arcname=row['path'],recursive=False)
result=dict(UTC=datetime.now(timezone.utc).isoformat(),native=native,identities=identities,manifest=manifest,
 bundle_SHA256=sha(bundle),bundle_bytes=bundle.stat().st_size,old_checkpoint_unchanged=True,
 GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(result),flush=True)
'''
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
(P/'COLLECT_NATIVE.stdout.log').write_text(run.stdout,encoding='utf8');(P/'COLLECT_NATIVE.stderr.log').write_text(run.stderr,encoding='utf8')
assert run.returncode==0,run.stderr
receipt=json.loads(run.stdout);archive=P/'R67_COMPLETED_EXPORT.tar'
cp=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+remote+'/R67_COMPLETED_EXPORT.tar',str(archive)],text=True,capture_output=True,timeout=60)
assert cp.returncode==0,cp.stderr
assert sha(archive)==receipt['bundle_SHA256']
with tarfile.open(archive)as t:
    for m in t.getmembers():
        assert m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts
        old=S/m.name
        if old.exists():assert sha(old)==next(x['sha256']for x in receipt['manifest']if x['path']==m.name),m.name
    t.extractall(S,filter='data')
for row in receipt['manifest']:assert sha(S/row['path'])==row['sha256'],row['path']
receipt.update(local_guard=guard,local_memory=memory(),downloaded_and_reopened=True)
write(P/'COLLECTION_RECEIPT.json',receipt)
result=json.loads((S/'RESULT.json').read_text())
environment=json.loads((S/'ENVIRONMENT.json').read_text())
assert environment['CPU_affinity']==list(range(12))
resource_proof=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_CPU_affinity=environment['CPU_affinity'],torch_threads=12,
    prior_attempt_native_wall_s=C['prior_attempt_native_wall_s'],repair_native_wall_s=receipt['native']['elapsed_s'],
    cumulative_native_wall_s=C['prior_attempt_native_wall_s']+receipt['native']['elapsed_s'],original_cumulative_cap_s=600)
assert resource_proof['cumulative_native_wall_s']<=600
write(P/'CUMULATIVE_BUDGET_AND_CPU_RECEIPT.json',resource_proof)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=json.loads((R/name).read_text());d.update(status='R67_terminal_pending_root_review',project_status='R67_terminal_pending_root_review',
        active_scientific_jobs=0,current_stage_progress=result['status'],current_activity='root_pilot_result_review',
        last_native_observation='reviews/R67_EQUAL_COST_REPAIR_001/COLLECTION_RECEIPT.json',
        current_job=dict(stage=C['stage'],remote_root=remote,status='terminal',actual_worker_PID=receipt['native']['actual_worker_PID'],
            supervisor_PID=receipt['native']['supervisor_PID'],native_exit=receipt['native']['actual_worker_native_exit'],wall_cap_s=C['worker_cap_s']))
    write(R/name,d)
print(json.dumps(dict(status=result['status'],native_exit=receipt['native']['actual_worker_native_exit'],elapsed_s=result['elapsed_s'],
    native_elapsed_s=receipt['native']['elapsed_s'],files=len(receipt['manifest']),bytes=receipt['bundle_bytes'],GPU_processes=receipt['GPU_processes'],
    CPU_affinity=environment['CPU_affinity'],primary=result.get('primary'),failed_harm_count=len(result.get('failed_harm_checks',[]))),indent=2))
