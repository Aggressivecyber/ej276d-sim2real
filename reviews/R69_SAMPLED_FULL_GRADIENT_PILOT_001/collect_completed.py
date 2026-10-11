"""Collect the single terminal native attempt without scientific re-execution."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,subprocess,hashlib,tarfile
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'collection_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
C=read(S/'config.json');remote=C['remote_root']
assert not(P/'COLLECTION_RECEIPT.json').exists()
code='from pathlib import Path\nimport os,json,hashlib,tarfile,subprocess\nfrom datetime import datetime,timezone\np=Path('+repr(remote)+')\nos.sched_setaffinity(0,{11})\n'+r'''
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
native=json.loads((p/'pilot.py.NATIVE_RECEIPT.json').read_text())
assert native['actual_worker_native_exit'] is not None
identities={}
for key in ['actual_worker_PID','supervisor_PID']:
 pid=native[key];proc=Path('/proc')/str(pid);state=None
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
bundle=p/'R69_COMPLETED_EXPORT.tar';assert not bundle.exists()
with tarfile.open(bundle,'w')as t:
 for row in manifest:t.add(p/row['path'],arcname=row['path'],recursive=False)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),native=native,identities=identities,manifest=manifest,
 bundle_SHA256=sha(bundle),bundle_bytes=bundle.stat().st_size,old_checkpoint_unchanged=True,
 GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip())),flush=True)
'''
q=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
for k,v in [('stdout',q.stdout),('stderr',q.stderr)]:(P/('COLLECT_NATIVE.'+k+'.log')).write_text(v,encoding='utf8')
assert q.returncode==0,q.stderr
receipt=json.loads(q.stdout);archive=P/'R69_COMPLETED_EXPORT.tar'
cp=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102:'+remote+'/R69_COMPLETED_EXPORT.tar',str(archive)],text=True,capture_output=True,timeout=60)
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
env=read(S/'ENVIRONMENT.json');assert env['CPU_affinity']==list(range(12))
native=receipt['native'];result=read(S/'RESULT.json')
resource=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_CPU_affinity=env['CPU_affinity'],worker_cap_s=600,
 native_elapsed_s=native['elapsed_s'],native_exit=native['actual_worker_native_exit'],
 job_tree_RSS_peak_bytes=native['job_tree_RSS_peak_bytes'],worker_peak_RSS_bytes=native['actual_worker_RSS_peak_bytes'],
 GPU_process_memory_peak_bytes=native['GPU_process_memory_peak_bytes'],GPU_allocated_peak_bytes=result.get('GPU_allocated_peak_bytes'),
 GPU_reserved_peak_bytes=result.get('GPU_reserved_peak_bytes'),single_scientific_job=not native['start_resource']['other_own_project_jobs'])
write(P/'RESOURCE_RECEIPT.json',resource)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=datetime.now(timezone.utc).isoformat(),status='R69_terminal_pending_root_review',
  project_status='R69_terminal_pending_root_review',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,
  current_activity='R69_root_result_review',current_stage_progress=result['status'],last_native_observation='reviews/'+S.name+'/COLLECTION_RECEIPT.json',
  current_job=dict(stage=S.name,status='terminal',remote_root=remote,actual_worker_PID=native['actual_worker_PID'],supervisor_PID=native['supervisor_PID'],native_exit=native['actual_worker_native_exit'],wall_cap_s=600))
 write(R/name,d)
for name,key,field in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 d=read(R/name);e=next(e for e in d[key]if e[field]==S.name)
 e.update(status='terminal_pending_root_review',native=native,active_scientific_jobs=0,result_status=result['status'])
 d.update(active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
print(json.dumps(dict(status=result['status'],files=len(receipt['manifest']),native_exit=native['actual_worker_native_exit'],native_wall_s=native['elapsed_s'],GPU_processes=receipt['GPU_processes'],primary=result.get('primary'),failed_harm_checks=len(result.get('failed_harm_checks',[]))),ensure_ascii=False,indent=2))
