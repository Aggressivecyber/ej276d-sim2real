"""Deliver the closed R69 evidence to the existing private Git and D backup."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001';A=R/'reviews/R69_DIRECTION_DECISION_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):
 text=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
 if p==R/'experiments.json':p.write_bytes(text.encode('utf8'))
 else:p.write_text(text,encoding='utf8')
def rel(p):return p.relative_to(R).as_posix()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def call(args,timeout=60):
 if args[0]=='git':args=['git','-c','core.longpaths=true',*args[1:]]
 q=subprocess.run(args,cwd=R,text=True,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
 assert q.returncode==0,dict(command=args,exit=q.returncode,stdout=q.stdout[-3000:],stderr=q.stderr[-2500:])
 return q.stdout.strip()
final=R/'R69_DELIVERY_RECEIPT.json';assert not final.exists()
review=read(P/'ROOT_RESULT_REVIEW.json');col=read(P/'COLLECTION_RECEIPT.json');qa=read(P/'VISUAL_QA_RECEIPT.json')
assert qa['actually_opened_and_reviewed']and qa['PNG_SHA256']==sha(S/'R69_COMPARISON.png')
resume='--resume-staged'in sys.argv
previous_staged=call(['git','diff','--cached','--name-only']).splitlines()
if not resume:assert not previous_staged
assert read(R/'jobs.json')['active_scientific_jobs']==read(R/'PROJECT_PROGRESS.json')['active_scientific_jobs']==0
assert not review['original_A_F_complete']and col['native']['actual_worker_native_exit']is not None
for row in read(P/'REGISTRATION.json')['local_files']:assert sha(R/row['path'])==row['sha256'],row['path']
base=call(['git','rev-parse','HEAD']);repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert repo['visibility']=='PRIVATE'
if resume:
 prior=read(P/'DELIVERY_SELECTION.json');assert prior['base_commit']==base
 assert set(previous_staged).issubset(set(prior['paths'])|{rel(P/'DELIVERY_SELECTION.json')})
for n in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/n);d.update(updated_UTC=datetime.now(timezone.utc).isoformat(),latest_delivered_stage=S.name,
  delivery_status='see_R69_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery',delivery_completion_receipt='R69_DELIVERY_RECEIPT.json',
  latest_completed_backup='R69_DELIVERY_RECEIPT.json',latest_stage_backup='R69_DELIVERY_RECEIPT.json',latest_stage_publication='R69_DELIVERY_RECEIPT.json',
  latest_delivery_status='see_R69_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery',current_activity='R69_fixed_candidate_evidence_closed',
  current_route_decision='reviews/'+S.name+'/ROOT_RESULT_REVIEW.json',
  current_engineering_addendum='R69 fixed K4 gradient admission passed but joint equal-cost response gate failed; historical R65 pack optimization remains separate')
 write(R/n,d)
for n,key,field in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 d=read(R/n);next(x for x in d[key]if x[field]==S.name)['delivery_receipt']='R69_DELIVERY_RECEIPT.json';write(R/n,d)
skip={'runtime_cache','__pycache__','registration_runtime','collection_runtime','report_runtime','delivery_runtime','verify_runtime'}
owned=[f for parent in [P,S,A]for f in parent.rglob('*')if f.is_file()and not skip.intersection(f.relative_to(parent).parts)]
selected={rel(f)for f in owned if f.suffix in{'.py','.md','.json','.png','.svg'}and not any(x.startswith('before_')for x in f.relative_to(R).parts)}
selected.update(['AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','PROGRAM.json','jobs.json','experiments.json','R69_REPORT.md','tasks/'+S.name+'.md'])
write(P/'DELIVERY_SELECTION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),base_commit=base,paths=sorted(selected),repository=repo,weights_and_arrays_excluded_from_Git=True,final_model_delivered=False))
selected.add(rel(P/'DELIVERY_SELECTION.json'))
for lo in range(0,len(selected),35):call(['git','add','-f','--',*sorted(selected)[lo:lo+35]])
staged=call(['git','diff','--cached','--name-only']).splitlines();assert set(staged).issubset(selected)
assert not call(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
write(P/'DELIVERY_STAGED_CHECK.json',dict(UTC=datetime.now(timezone.utc).isoformat(),staged_paths=staged,whitespace_native_exit=0,private_visibility_verified=True))
call(['git','add','-f','--',rel(P/'DELIVERY_STAGED_CHECK.json')]);selected.add(rel(P/'DELIVERY_STAGED_CHECK.json'))
call(['git','commit','-m','Test full-law sampled-gradient cost and preserve joint response evidence'])
commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert remote==commit
payload=selected|{rel(f)for f in owned}|{'reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
size=sum((R/n).stat().st_size for n in payload);assert shutil.disk_usage('D:/').free>size+40*1024**2
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r69_sampled_gradient_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
backup.mkdir(parents=True);listing=[]
for name in sorted(payload):
 f=R/name;out=backup/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 h=sha(out);assert sha(f)==h,name
 listing.append(dict(path=name,bytes=out.stat().st_size,sha256=h))
bundle=backup/'r69_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
write(backup/'BACKUP_SHA256_MANIFEST.json',listing)
(backup/'RESTORE_README.txt').write_text('R69 fixed K4 diagnostic evidence, not final A-F response weights. Restore these relative paths on E/D with the preceding repository snapshot. Diagnostic endpoint tensors and sampling logs are included here, excluded from Git. Frozen R65 source/data/constants and original update500 checkpoint remain separately bound dependencies; use earlier R65 snapshots or the registered retained server root. Exact saved-result gate verification needs no model run. New scientific execution needs a fresh separate registration; never overwrite this attempt or automatically resume R65.\n',encoding='utf8')
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),purpose='R69_capped_full_law_sampled_gradient_evidence',private_repository=repo,
 base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,backup_root=str(backup),copied_files=len(listing),copied_bytes=size,
 all_copied_file_SHA256_verified=True,bundle_bytes=bundle.stat().st_size,bundle_SHA256=sha(bundle),D_free_after_bytes=shutil.disk_usage('D:/').free,
 result_status=review['result_status'],native_exit=col['native']['actual_worker_native_exit'],native_wall_s=col['native']['elapsed_s'],worker_cap_s=600,
 active_scientific_jobs=0,original_A_F_complete=False,model_goal_complete=False,final_inference_weights_delivered=False,
 diagnostic_weights_delivered=review['completed_comparison'],actually_opened_figure=True,PNG_SHA256=qa['PNG_SHA256'],R65_automatic_resume=False,resource=dict(guard,**memory()))
write(final,receipt);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
