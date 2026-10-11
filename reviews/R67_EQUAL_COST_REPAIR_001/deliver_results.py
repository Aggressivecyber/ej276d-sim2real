"""Deliver the closed benefit assessment to existing private Git and D backup."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parents[1]
P66=R/'reviews/R66_EQUAL_COST_PILOT_001';S=R/'work/R67_EQUAL_COST_REPAIR_001';S66=R/'work/R66_EQUAL_COST_PILOT_001';S65=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime')
def read(f):return json.loads(f.read_text('utf8'))
def write(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def rel(f):return f.relative_to(R).as_posix()
def call(args,timeout=60):
    if args[0]=='git':args=['git','-c','core.longpaths=true',*args[1:]]
    r=subprocess.run(args,cwd=R,text=True,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
    assert r.returncode==0,dict(args=args,exit=r.returncode,stdout=r.stdout[-6000:],stderr=r.stderr[-3000:])
    return r.stdout.strip()
final=R/'R67_DELIVERY_RECEIPT.json';assert not final.exists()
resume='--resume-staged' in sys.argv
review=read(P/'ROOT_RESULT_REVIEW.json');collection=read(P/'COLLECTION_RECEIPT.json')
assert read(R/'PROJECT_PROGRESS.json')['active_scientific_jobs']==0
assert collection['native']['actual_worker_native_exit'] is not None and not review['original_A_F_complete']
if review['completed_comparison']:
    qa=read(P/'VISUAL_QA_RECEIPT.json');assert qa['actually_opened_and_reviewed']
    assert qa['PNG_SHA256']==sha(S/'R67_COMPARISON.png')
# Replace stale local R65 live metadata with real native terminal files, preserving copies.
download=P66/'R65_terminal_native_metadata';download.mkdir(exist_ok=resume)
old_c=read(S65/'config.json');names=['REMOTE_ACTIVE_JOB.json','train_final.py.NATIVE_RECEIPT.json','R65_COST_HOLD_TERMINATION.json']
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*['rylan@100.82.231.102:'+old_c['remote_root']+'/'+n for n in names],str(download)],30)
assert read(download/'REMOTE_ACTIVE_JOB.json')['actual_worker_native_exit']==-15
assert read(download/'train_final.py.NATIVE_RECEIPT.json')['actual_worker_native_exit']==-15
before=P66/'before_R65_terminal_native_metadata';before.mkdir(exist_ok=resume)
for n in names:
    if(S65/n).exists()and not(before/n).exists():shutil.copy2(S65/n,before/n)
    shutil.copy2(download/n,S65/n)
if(S65/'LATEST_REMOTE_OBSERVATION.json').exists()and not(before/'LATEST_REMOTE_OBSERVATION.json').exists():shutil.copy2(S65/'LATEST_REMOTE_OBSERVATION.json',before/'LATEST_REMOTE_OBSERVATION.json')
write(S65/'LATEST_REMOTE_OBSERVATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='R65_RETIRED_INCOMPLETE_COST_HOLD',
    native_exit=-15,durable_checkpoint_update=500,last_completed_update=613,inflight_update=614,
    in_memory_updates_retained=False,automatic_resume=False,source='reviews/R66_EQUAL_COST_PILOT_001/R65_RETIREMENT_RECEIPT.json'))
for n in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=read(R/n);d.update(latest_delivered_stage='R67_EQUAL_COST_REPAIR_001',delivery_status='see_R67_DELIVERY_RECEIPT_for_actual_delivery',
        delivery_completion_receipt='R67_DELIVERY_RECEIPT.json',latest_completed_backup='R67_DELIVERY_RECEIPT.json',
        latest_stage_publication='R67_DELIVERY_RECEIPT.json',latest_delivery_status='see_R67_DELIVERY_RECEIPT_for_actual_private_Git_D_receipts',
        current_activity='bounded_cost_benefit_diagnostic_closed',next_action='Analyze saved evidence and prepare lower-cost mechanisms only; no automatic R65 restart or new pilot computation under this assessment.')
    write(R/n,d)
selection=[]
for parent in [P66,P,S66,S]:
    selection.extend(f for f in parent.rglob('*')if f.is_file()and f.suffix in {'.py','.json','.md','.svg','.png'}
        and not {'runtime_cache','delivery_runtime','__pycache__','before_registration_metadata','before_closure_metadata'}.intersection(f.relative_to(parent).parts))
selection.extend(R/n for n in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROGRAM.json','PROJECT_PROGRESS.json','R66_REPORT.md','R67_REPORT.md',
    'tasks/R66_EQUAL_COST_PILOT_001.md','tasks/R67_EQUAL_COST_REPAIR_001.md'])
selection.extend(S65/n for n in [*names,'LATEST_REMOTE_OBSERVATION.json'])
selected=sorted({rel(f)for f in selection})
base=call(['git','rev-parse','HEAD'])
already_staged=call(['git','diff','--cached','--name-only']).splitlines()
if resume:
    prior=read(P/'DELIVERY_SELECTION.json');assert prior['base_commit']==base
    assert set(already_staged).issubset(set(prior['paths'])|{rel(P/'DELIVERY_SELECTION.json')})
else:assert not already_staged
repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert repo['visibility']=='PRIVATE'
write(P/'DELIVERY_SELECTION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),base_commit=base,repository=repo,paths=selected,
    purpose='bounded_benefit_evidence_not_final_model',weights_and_arrays_excluded_from_Git=True))
selected.append(rel(P/'DELIVERY_SELECTION.json'))
for lo in range(0,len(selected),35):call(['git','add','-f','--',*selected[lo:lo+35]])
staged=call(['git','diff','--cached','--name-only']).splitlines();assert set(staged).issubset(set(selected))
check=call(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
assert not check
write(P/'DELIVERY_STAGED_CHECK.json',dict(UTC=datetime.now(timezone.utc).isoformat(),staged_paths=staged,whitespace_native_exit=0,
    private_visibility_verified=True,resource=dict(guard,**memory())))
call(['git','add','-f','--',rel(P/'DELIVERY_STAGED_CHECK.json')])
call(['git','commit','-m','Evaluate capped same-start equal-cost benefit and retain R65 cost hold'])
commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert remote==commit
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r66_r67_cost_benefit_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
payload=set(selected)|{rel(P/'DELIVERY_STAGED_CHECK.json')}
for parent in [P66,P,S66,S]:
    payload.update(rel(f)for f in parent.rglob('*')if f.is_file()and not {'runtime_cache','delivery_runtime','__pycache__'}.intersection(f.relative_to(parent).parts))
size=sum((R/n).stat().st_size for n in payload)
assert shutil.disk_usage('D:/').free>size+40*1024**2
backup.mkdir(parents=True);listing=[]
for name in sorted(payload):
    f=R/name;out=backup/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
    listing.append(dict(path=name,bytes=out.stat().st_size,sha256=sha(out)))
    assert sha(f)==listing[-1]['sha256']
bundle=backup/'r66_r67_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'RESTORE_README.txt').write_text('Bounded local benefit diagnostic only. Combine with earlier R65 implementation/dependency snapshots for remote reproduction. R66 is invalid; R67 is a local same-start/fixed-LR comparison, not original A-F acceptance. Diagnostic endpoints must not be promoted to a final method. Full R65 remains incomplete and must not restart automatically. Copies include diagnostic weights/logs; Git excludes tensors and arrays.\n',encoding='utf8')
write(backup/'BACKUP_SHA256_MANIFEST.json',listing)
result=dict(UTC=datetime.now(timezone.utc).isoformat(),purpose='R66_R67_capped_equal_cost_benefit_assessment',
    private_repository=repo,base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,
    backup_root=str(backup),copied_files=len(listing),copied_bytes=size,all_copied_file_SHA256_verified=True,
    bundle_bytes=bundle.stat().st_size,bundle_SHA256=sha(bundle),D_free_after_bytes=shutil.disk_usage('D:/').free,
    R66_status='invalid_execution',R67_status=review['native_status'],R67_native_exit=collection['native']['actual_worker_native_exit'],
    cumulative_native_wall_s=review['cumulative_compute']['cumulative_native_wall_s'],original_cap_s=600,
    R65_status='retired_incomplete_cost_hold',R65_native_exit=-15,active_scientific_jobs=0,
    original_A_F_complete=False,model_goal_complete=False,final_inference_weights_delivered=False,
    diagnostic_weights_delivered=review['completed_comparison'],automatic_resume=False,
    source_reports=['R66_REPORT.md','R67_REPORT.md'],resource=dict(guard,**memory()))
write(final,result);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
print(json.dumps(result,indent=2),flush=True)
