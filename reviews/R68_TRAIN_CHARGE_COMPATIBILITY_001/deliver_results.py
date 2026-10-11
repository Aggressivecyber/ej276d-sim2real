"""Deliver R68 exact scalar evidence and the two completed static audits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
A=R/'reviews/R68_DIRECTION_PREPARATION_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(p):return p.relative_to(R).as_posix()
def call(args,timeout=60):
 if args[0]=='git':args=['git','-c','core.longpaths=true',*args[1:]]
 v=subprocess.run(args,cwd=R,text=True,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
 assert v.returncode==0,dict(command=args,exit=v.returncode,stdout=v.stdout[-4000:],stderr=v.stderr[-2500:])
 return v.stdout.strip()
final=R/'R68_DELIVERY_RECEIPT.json';assert not final.exists()
assert not call(['git','diff','--cached','--name-only'])
assert read(R/'jobs.json')['active_scientific_jobs']==read(R/'PROJECT_PROGRESS.json')['active_scientific_jobs']==0
review=read(P/'ROOT_RESULT_REVIEW.json');assert not review['original_A_F_complete']
assert read(P/'NATIVE_RECEIPT.json')['actual_worker_native_exit']==0
qa=read(P/'VISUAL_QA_RECEIPT.json');assert qa['actually_opened_and_reviewed'] and qa['PNG_SHA256']==sha(S/'R68_CHARGE_ENVELOPE.png')
for f in [P/'reproduce_in_fresh_directory.py',P/'deliver_results.py']:compile(f.read_text('utf8'),str(f),'exec')
base=call(['git','rev-parse','HEAD']);repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert repo['visibility']=='PRIVATE'
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=datetime.now(timezone.utc).isoformat(),latest_delivered_stage=S.name,
  delivery_status='see_R68_DELIVERY_RECEIPT_for_actual_Git_D_delivery',delivery_completion_receipt='R68_DELIVERY_RECEIPT.json',
  latest_completed_backup='R68_DELIVERY_RECEIPT.json',latest_stage_backup='R68_DELIVERY_RECEIPT.json',
  latest_stage_publication='R68_DELIVERY_RECEIPT.json',latest_delivery_status='see_R68_DELIVERY_RECEIPT_for_actual_Git_D_delivery',
  current_activity='R68_closed_charge_compatibility_evidence',
  next_action='Read existing input population/scale provenance and archived evaluation evidence; no automatic training, new pilot or A-F/population changes.')
 write(R/name,d)
jobs=read(R/'jobs.json');next(j for j in jobs['jobs']if j['stage']==S.name)['delivery_receipt']='R68_DELIVERY_RECEIPT.json';write(R/'jobs.json',jobs)
skip={'runtime_cache','__pycache__','report_runtime','plot_layout_runtime','supervisor_runtime','delivery_runtime','reproduction_controller'}
owned=[f for parent in [P,S,A] for f in parent.rglob('*') if f.is_file() and not skip.intersection(f.relative_to(parent).parts)]
selected={rel(f)for f in owned if f.suffix in {'.py','.md','.json','.png','.svg'} and not any(p.startswith('before_')for p in f.relative_to(R).parts)}
selected.update(['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','PROGRAM.json','STATUS.md','jobs.json','R68_REPORT.md','tasks/'+S.name+'.md'])
write(P/'DELIVERY_SELECTION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),base_commit=base,paths=sorted(selected),repository=repo,
      private_arrays_excluded_from_Git=True,scope='R68 diagnostic and static evidence; not a final model'))
selected.add(rel(P/'DELIVERY_SELECTION.json'))
for lo in range(0,len(selected),35):call(['git','add','-f','--',*sorted(selected)[lo:lo+35]])
staged=call(['git','diff','--cached','--name-only']).splitlines();assert set(staged).issubset(selected)
assert not call(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
write(P/'DELIVERY_STAGED_CHECK.json',dict(UTC=datetime.now(timezone.utc).isoformat(),staged_paths=staged,whitespace_native_exit=0,private_visibility_verified=True))
call(['git','add','-f','--',rel(P/'DELIVERY_STAGED_CHECK.json')]);selected.add(rel(P/'DELIVERY_STAGED_CHECK.json'))
call(['git','commit','-m','Bound preserved-charge TRAIN mismatch and explain equal-cost failure'])
commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert commit==remote
reg=read(P/'REGISTRATION.json')
payload=selected|{rel(f)for f in owned}|{r['path']for r in reg['files']}|{reg['input_archive'],'reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
size=sum((R/n).stat().st_size for n in payload)
assert shutil.disk_usage('D:/').free>size+30*1024**2
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r68_charge_compatibility_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
backup.mkdir(parents=True);manifest=[]
for name in sorted(payload):
 f=R/name;out=backup/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 h=sha(out);assert sha(f)==h,name
 manifest.append(dict(path=name,bytes=out.stat().st_size,sha256=h))
bundle=backup/'r68_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
write(backup/'BACKUP_SHA256_MANIFEST.json',manifest)
(backup/'RESTORE_README.txt').write_text('R68 finite TRAIN scalar compatibility certificate and R67 saved-score/static protocol audits. This is not a final response model. Restore the relative paths plus the preceding private repository snapshot. This payload includes all hash-bound R68 scalar inputs and guard dependencies for the explicit fresh-directory replay described in work/R68_TRAIN_CHARGE_COMPATIBILITY_001/REPRODUCE.md. Original A-F remains incomplete; never automatically resume R65 or change source weights/acceptance.\n',encoding='utf8')
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),purpose='R68_finite_TRAIN_charge_compatibility_and_R67_component_audits',
   private_repository=repo,base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,
   backup_root=str(backup),copied_files=len(manifest),copied_bytes=size,all_copied_file_SHA256_verified=True,
   bundle_bytes=bundle.stat().st_size,bundle_SHA256=sha(bundle),D_free_after_bytes=shutil.disk_usage('D:/').free,
   native_exit=0,diagnostic_wall_s=read(P/'NATIVE_RECEIPT.json')['elapsed_s'],active_scientific_jobs=0,
   exact_KS_lower_bound=review['exact_KS_lower_bound'],finite_TRAIN_only=True,original_A_F_complete=False,
   fixed_HGB_or_DEV_impossibility_proven=False,model_goal_complete=False,R65_automatic_resume=False,
   original_acceptance_hashes_verified=True,actually_opened_figure=True,PNG_SHA256=qa['PNG_SHA256'],
   optional_replay_entry_static_checked_not_executed=True,resource=dict(guard,**memory()))
write(final,receipt);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
