"""Save the actual cost hold, reviewed implementation and receipts to private Git/D."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys

P=Path(__file__).resolve().parent
R=P.parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
import local_resource_guard as guard
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
resource=guard.apply(P/'root_delivery')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(R).as_posix()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def call(argv,timeout=60):
    if argv[0]=='git':argv=['git','-c','core.longpaths=true',*argv[1:]]
    v=subprocess.run(argv,cwd=R,text=True,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
    assert v.returncode==0,{'argv':argv,'exit':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
    return v.stdout.strip()
final=R/'R65_OPTIMIZATION_PAUSE_DELIVERY_RECEIPT.json'
assert not final.exists()
assert read(R/'PROJECT_PROGRESS.json')['status']=='R65_paused_pending_low_cost_benefit_evidence'
review=read(S/'ROOT_IMPLEMENTATION_REVIEW.json')
assert len(review['files'])==115
for row in review['files']:assert sha(S/row['name'])==row['sha256'],row['name']
stopped=read(S/'controller_runtime/PIPELINE_WATCH_STOPPED.json')
assert stopped['state']=='LOCAL_WATCH_STOPPED'
pids=(ctypes.c_ulong*32768)();needed=ctypes.c_ulong()
assert ctypes.WinDLL('psapi',use_last_error=True).EnumProcesses(pids,ctypes.sizeof(pids),ctypes.byref(needed))
assert 56812 not in pids[:needed.value//ctypes.sizeof(ctypes.c_ulong)]
write(S/'controller_runtime/ROOT_WATCHER_ACTUAL_VERIFICATION.json',{
 'UTC':datetime.now(timezone.utc).isoformat(),'PID':56812,'actual_PID_present':False,
 'latest_state':'LOCAL_WATCH_STOPPED','stop_receipt_SHA256':sha(S/'controller_runtime/PIPELINE_WATCH_STOPPED.json'),
 'native_exit':None,'remote_training_paused_separately':True,
 'original_prior_watcher_evidence_preserved':'reviews/R65_THROUGHPUT_OPTIMIZATION_001/old_controller_runtime_20261011_045543_stopped'})
downloaded=P/'paused_native_metadata'
before=P/'before_paused_native_metadata_download'
assert not downloaded.exists() and not before.exists()
downloaded.mkdir();before.mkdir()
cfg=read(S/'config.json')
names=['ARTIFICIAL_SUITE_RECEIPT.json','ARTIFICIAL_SUITE_PROGRESS.json','SAME_RUN_RESTORE_RECEIPT.json',
       'RECOVERY_REQUEST.json','PHYSICAL_PROGRESS.json','TRAIN_PROGRESS.json','RECOVERY_CHECKPOINT.json','REMOTE_ACTIVE_JOB.json']
sources=['rylan@100.82.231.102:'+cfg['remote_root']+'/'+name for name in names]
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*sources,str(downloaded)],30)
assert read(downloaded/'SAME_RUN_RESTORE_RECEIPT.json')['PID']==330411
assert read(downloaded/'PHYSICAL_PROGRESS.json')['last_completed_update']==613
assert read(downloaded/'ARTIFICIAL_SUITE_RECEIPT.json')['pass']
for name in names:
    if (S/name).exists():shutil.copy2(S/name,before/name)
    shutil.copy2(downloaded/name,S/name)
write(P/'OBSERVATION_PROVENANCE_CORRECTION.json',{
 'invalid_diagnostic':'INVALID_PRELINEAGE_DIAGNOSTIC.json',
 'reason':'The first observation accidentally treated retained old trace rows as new-run publications. It was invalidated before any benefit claim.',
 'correction':'Compare only records with resumed native_attempt_id and PID at physically completed steps.',
 'valid_real_step600_comparison':'LATEST_RESUME_VERIFICATION.json',
 'scientific_or_acceptance_claim_from_invalid_record':False})
top=[p for p in P.iterdir() if p.is_file() and p.suffix in {'.py','.json','.md','.ps1'}
     and p.name!='INVALID_PRELINEAGE_DIAGNOSTIC.json']
paths=top[:]
for directory in ['agent_candidate','attempt_001','new_controller_package','paused_native_metadata']:
    paths.extend(p for p in (P/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.md','.ps1'}
                 and '__pycache__' not in p.parts)
paths.extend(P/x for x in ['user_steering/RECEIPT.json','prepared_upgrade/PREPARATION_RECEIPT.json',
 'prepared_upgrade/UPGRADE_INSTALL_MANIFEST.json','recovery_actual/STOP_AND_ARCHIVE_RECEIPT.json',
 'recovery_actual/reviewed_request/EVIDENCE_VALIDATION.json'])
paths.extend(R/x for x in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','PROGRAM.json',
 'R65_COST_BENEFIT_REVIEW.md','R65_OPTIMIZATION_RUNNING.md','R65_WAIT_PREPARATION_SNAPSHOT_RECEIPT.json',
 'reviews/record_R65_optimization_steering_root.py'])
paths.extend(S/x for x in ['train_final.py','network_update_check/network_update_check.py',
 'network_update_check/SOURCE_BINDING.json','ROOT_IMPLEMENTATION_REVIEW.json','IMPLEMENTATION_UPGRADE.json',
 'implementation_upgrade/CPU_EQUIVALENCE.json','implementation_upgrade/CUDA_EQUIVALENCE.json',
 'controller_runtime/ROOT_WATCHER_LAUNCH.json','controller_runtime/ROOT_WATCHER_ACTUAL_VERIFICATION.json',
 'controller_runtime/PIPELINE_WATCH_STARTED.json','controller_runtime/PIPELINE_STOP_REQUEST.json',
 'controller_runtime/PIPELINE_WATCH_STOPPED.json',*names])
selected=sorted({rel(p) for p in paths})
assert all((R/p).is_file() for p in selected)
assert not call(['git','diff','--cached','--name-only'])
base=call(['git','rev-parse','HEAD'])
repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert repo['visibility']=='PRIVATE'
selection=P/'DELIVERY_SELECTION.json'
write(selection,{'UTC':datetime.now(timezone.utc).isoformat(),'base_commit':base,'private_repository':repo,
 'purpose':'equivalent_implementation_and_cost_benefit_pause_not_final_model','paths':selected,
 'new_scientific_pilot_executed':False,'model_goal_complete':False})
selected.append(rel(selection));selected.sort()
for offset in range(0,len(selected),40):call(['git','add','-f','--',*selected[offset:offset+40]])
staged=call(['git','diff','--cached','--name-only']).splitlines()
assert set(staged).issubset(set(selected))
ws=subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
 'diff','--cached','--check'],cwd=R,text=True,capture_output=True,encoding='utf8')
assert ws.returncode in (0,2) and not ws.stderr
findings=ws.stdout.splitlines()
assert all(line.endswith(': new blank line at EOF.') for line in findings),findings
check=P/'DELIVERY_STAGED_CHECK.json'
write(check,{'UTC':datetime.now(timezone.utc).isoformat(),'staged_count':len(staged),'paths':staged,
 'whitespace_native_exit':ws.returncode,'bound_source_EOF_only_findings':findings,
 'arrays_weights_and_logs_excluded':True,'resource':dict(resource,**guard.memory())})
call(['git','add','-f','--',rel(check)])
call(['git','commit','-m','Preserve R65 equivalent speedup and pause full training pending benefit evidence'])
commit=call(['git','rev-parse','HEAD'])
call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0]
assert remote==commit
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r65_optimization_cost_hold_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists()
payload=[p for p in P.rglob('*') if p.is_file() and not {'tmp','__pycache__','runtime_cache'}.intersection(p.relative_to(P).parts)]
copies=sorted({rel(p) for p in payload}|set(selected)|{rel(check)})
size=sum((R/p).stat().st_size for p in copies)
assert shutil.disk_usage('D:/').free>size+30*1024**2
backup.mkdir(parents=True)
listing=[]
for name in copies:
    dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(R/name,dest)
    listing.append({'path':name,'bytes':dest.stat().st_size})
bundle=backup/'r65_optimization_cost_hold_incremental.bundle'
call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'RESTORE_README.txt').write_text('Paused incomplete R65 implementation/cost review. Restore previous R65 source and dependency snapshots before this increment. No final model or in-memory process state is contained here. Original and failed native attempts and checkpoint remain on the server; paused state is not successful completion. Never resume blindly. Normal file copy, no separate migration consistency test.\n',encoding='utf8')
result={'UTC':datetime.now(timezone.utc).isoformat(),'purpose':'R65_optimization_and_cost_hold',
 'private_repository':repo,'base_commit':base,'local_commit':commit,'remote_main_SHA':remote,'push_exit':0,
 'backup_root':str(backup),'copied_files':len(listing),'copied_bytes':size,'bundle_bytes':bundle.stat().st_size,
 'D_free_after_bytes':shutil.disk_usage('D:/').free,'training_status':'paused_completed613_inflight614',
 'durable_checkpoint_update':500,'scientific_native_exit':None,'completion_automation':'PAUSED',
 'new_pilot_executed':False,'model_goal_complete':False,'final_weights_delivered':False,
 'resource':dict(resource,**guard.memory())}
write(final,result);shutil.copy2(final,backup/final.name)
write(backup/'BACKUP_FILE_LIST.json',listing)
assert read(final)==read(backup/final.name)
print(json.dumps(result,indent=2),flush=True)
