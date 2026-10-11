"""Seal this cost-only stage to private Git and verified incremental D backup."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime');mode=sys.argv[1];final=R/'R76_DELIVERY_RECEIPT.json'
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def rel(p):return p.relative_to(R).as_posix()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def call(args,timeout=60,binary=False):
    if args[0]=='git':args=['git','-c','core.longpaths=true',*args[1:]]
    q=subprocess.run(args,cwd=R,capture_output=True,text=not binary,encoding=None if binary else'utf8',errors=None if binary else'replace',timeout=timeout)
    assert q.returncode==0,dict(args=args,exit=q.returncode,stderr=q.stderr[-2000:])
    return q.stdout if binary else q.stdout.strip()
def owned(base):
    return [f for f in base.rglob('*')if f.is_file()and not any(v in {'runtime_cache','cache','__pycache__','evaluation_preparation','gap_preparation'}or v.endswith('_runtime')for v in f.relative_to(base).parts)]
assert not final.exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(P/'ROOT_RESULT_REVIEW.json')['root_cost_evidence_review_pass']
roots=['AGENTS.md','STATUS.md','PLAN.md','PROJECT_PROGRESS.json','PROGRAM.json','jobs.json','experiments.json','R76_REPORT.md','R75_DELIVERY_RECEIPT.json','tasks/'+P.name+'.md']
if mode=='prepare':
    assert not call(['git','diff','--cached','--name-only'])
    repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']));assert repo['visibility']=='PRIVATE'
    base=call(['git','rev-parse','HEAD'])
    for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
        d=read(R/name);d.update(latest_stage_publication=final.name,latest_stage_backup=final.name,delivery_status='R76_cost_delivery_pending_actual_receipt')
        write(R/name,d)
    for name in roots[:7]:
        prior=call(['git','show','HEAD:'+name],binary=True);data=(R/name).read_bytes()
        if prior.count(b'\r\n')>prior.count(b'\n')//2:(R/name).write_bytes(data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    selected=set(roots)
    for f in owned(S)+owned(P):
        if f.suffix not in {'.py','.md','.json','.png'}or '_PRIVATE' in f.name:continue
        if any(v.startswith('before_')for v in f.relative_to(R).parts):continue
        if f.name in {'DELIVERY_SELECTION.json','DELIVERY_STAGED_CHECK.json'}:continue
        selected.add(rel(f))
    write(P/'DELIVERY_SELECTION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),base_commit=base,repository=repo,paths=sorted(selected),
      files_SHA256={name:sha(R/name)for name in sorted(selected)},scope='R76 cost stage source and aggregate evidence; original arrays/weights remain E-D'))
    selected.add(rel(P/'DELIVERY_SELECTION.json'));paths=sorted(selected)
    for i in range(0,len(paths),25):call(['git','add','-f','--',*paths[i:i+25]])
    staged=call(['git','diff','--cached','--name-only']).splitlines();assert set(staged).issubset(selected)
    whitespace=subprocess.run(['git','-c','core.longpaths=true','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=R,capture_output=True,text=True,encoding='utf8')
    findings=whitespace.stdout.splitlines()
    allowed={'work/'+P.name+'/check_combined_loss.py:62: new blank line at EOF.','work/'+P.name+'/combined_loss.py:49: new blank line at EOF.'}
    assert whitespace.returncode in {0,2}and set(findings).issubset(allowed),(whitespace.returncode,findings)
    write(P/'DELIVERY_STAGED_CHECK.json',dict(UTC=datetime.now(timezone.utc).isoformat(),staged_paths=staged,private_visibility_verified=True,whitespace_exit=whitespace.returncode,accepted_exact_whitespace_findings=findings,reason='Preserve already bound R74 module bytes; only an inherited final empty line allowed',other_whitespace_findings=[]))
    call(['git','add','-f','--',rel(P/'DELIVERY_STAGED_CHECK.json')])
    print(json.dumps(dict(staged_count=len(staged)+1,base_commit=base,private_repository=repo,D_free_bytes=shutil.disk_usage('D:/').free)),flush=True)
elif mode=='finalize':
    selection=read(P/'DELIVERY_SELECTION.json');base=selection['base_commit'];assert call(['git','rev-parse','HEAD'])==base
    for name,digest in selection['files_SHA256'].items():assert sha(R/name)==digest,name
    selected=set(selection['paths'])|{rel(P/'DELIVERY_SELECTION.json'),rel(P/'DELIVERY_STAGED_CHECK.json')}
    assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(selected)
    payload=selected|{rel(f)for f in owned(S)+owned(P)if f.suffix not in {'.tar','.tmp'}}|{'CURRENT_ACCEPTANCE.json',
      'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md','PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json','reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
    prior=read(R/'R71_DELIVERY_RECEIPT.json');previous=Path(prior['backup_root']);assert (previous/'BACKUP_SHA256_MANIFEST.json').exists()
    previous_paths={row['path']:row for row in read(previous/'BACKUP_SHA256_MANIFEST.json')}
    required=['work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz','work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/UNIT_CHARGE_CONSTANTS.npz',
      'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/NOISE_CALIBRATION.npz','work/R23_ONLINE_STANDARDIZED_CRITIC_001/VALLEY_DEFINITION.json',
      'work/R71_REAL_CHARGE_FULL_TRAIN_001/FINAL_HEAD.pt','work/R71_REAL_CHARGE_FULL_TRAIN_001/HEAD_STATISTICS.npz']
    reused=[]
    for name in required:
        row=previous_paths[name];assert sha(previous/name)==row['sha256'];reused.append(dict(path=str(previous/name),sha256=row['sha256']))
    size=sum((R/name).stat().st_size for name in payload);assert shutil.disk_usage('D:/').free>size+40*1024**2
    call(['git','commit','-m','Verify retained-graph gradients and fix next full-data endpoint from cost'])
    commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
    remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert remote==commit
    backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r76_retained_graph_cost_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    backup.mkdir(parents=True);manifest=[]
    for name in sorted(payload):
        src,dst=R/name,backup/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        digest=sha(dst);assert digest==sha(src),name;manifest.append(dict(path=name,bytes=dst.stat().st_size,sha256=digest))
    bundle=backup/'r76_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
    write(backup/'BACKUP_SHA256_MANIFEST.json',manifest);write(backup/'REUSED_PRIOR_INPUTS.json',reused)
    (backup/'RESTORE_README.txt').write_text('R76 cost-only stage: 50 candidate plus 3 original discarded engineering updates, main 0. Restore relative paths on E/D. Immutable raw inputs and frozen Q weights are in the prior R71 backup and listed with verified hashes in REUSED_PRIOR_INPUTS.json. Full-waveform A-F remains incomplete. Git bundle requires base '+base+'.\n','utf8')
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=P.name,purpose='engineering_retained_graph_cost_delivery',private_repository=selection['repository'],
      base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,backup_root=str(backup),copied_files=len(manifest),copied_bytes=size,
      all_copied_file_SHA256_verified=True,reused_previous_backup_inputs=reused,bundle_SHA256=sha(bundle),D_free_after_bytes=shutil.disk_usage('D:/').free,
      complete_main_updates=0,engineering_updates=50,extra_original_reference_updates=3,native_exit=0,application_goal_status='active',model_goal_complete=False,
      no_quality_verdict=True,all_local_new_artifacts_on_E_D=True,resource={**guard,**memory()})
    write(final,result);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
    assert not call(['git','diff','--cached','--name-only']);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
else:raise ValueError(mode)
