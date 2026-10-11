"""Explicit private Git source/aggregate publication and complete D artifact backup."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime');mode=sys.argv[1];final=R/'R71_DELIVERY_RECEIPT.json'
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
    result=subprocess.run(args,cwd=R,capture_output=True,text=not binary,encoding=None if binary else'utf8',errors=None if binary else'replace',timeout=timeout)
    assert result.returncode==0,dict(args=args,returncode=result.returncode,stderr=result.stderr[-2000:])
    return result.stdout if binary else result.stdout.strip()
def owned(base):
    return [f for f in base.rglob('*')if f.is_file()and not any(v in {'runtime_cache','cache','__pycache__'}or v.endswith('_runtime')for v in f.relative_to(base).parts)]
assert not final.exists()
assert read(R/'jobs.json')['active_scientific_jobs']==0
assert read(P/'ROOT_RESULT_REVIEW.json')['root_saved_evidence_recomputation_pass']
assert read(P/'inference_cli_check/CLI_EXECUTION_RECEIPT.json')['native_exit']==0
assert read(P/'VISUAL_QA.json')['actual_images_opened']
roots=['AGENTS.md','STATUS.md','PLAN.md','PROJECT_OBJECTIVE.md','LONG_TERM_GOAL_20261010.md','PROJECT_PROGRESS.json','PROGRAM.json',
       'jobs.json','experiments.json','CURRENT_ACCEPTANCE.json','R71_REPORT.md','R70_DELIVERY_RECEIPT.json','tasks/'+P.name+'.md']
if mode=='prepare':
    assert not call(['git','diff','--cached','--name-only'])
    repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']));assert repo['visibility']=='PRIVATE'
    base=call(['git','rev-parse','HEAD'])
    for name in ['PROGRAM.json','PROJECT_PROGRESS.json']:
        d=read(R/name);d.update(latest_delivered_stage=P.name,latest_stage_publication=final.name,latest_stage_backup=final.name,
          latest_completed_backup=final.name,delivery_completion_receipt=final.name,
          delivery_status='see_R71_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery',latest_delivery_status='see_R71_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery')
        write(R/name,d)
    # Preserve prior tracked newline conventions for mutable governance files.
    for name in roots:
        if name=='CURRENT_ACCEPTANCE.json' or name.startswith('tasks/')or name.startswith('R71_')or name=='R70_DELIVERY_RECEIPT.json':continue
        prior=call(['git','show','HEAD:'+name],binary=True);data=(R/name).read_bytes()
        if prior.count(b'\r\n')>prior.count(b'\n')//2:
            data=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n');(R/name).write_bytes(data)
    selected=set(roots)
    for f in owned(S)+owned(P):
        if f.suffix not in {'.py','.md','.json','.png'}or '_PRIVATE' in f.name:continue
        if any(v.startswith('before_')for v in f.relative_to(R).parts):continue
        if f.name in {'DELIVERY_SELECTION.json','DELIVERY_STAGED_CHECK.json'}:continue
        selected.add(rel(f))
    write(P/'DELIVERY_SELECTION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),base_commit=base,repository=repo,
        paths=sorted(selected),files_SHA256={name:sha(R/name)for name in sorted(selected)},
        scope='R71 scientific source and aggregate evidence only; raw arrays and weights stay E/D'))
    selected.add(rel(P/'DELIVERY_SELECTION.json'));paths=sorted(selected)
    for i in range(0,len(paths),25):call(['git','add','-f','--',*paths[i:i+25]])
    staged=call(['git','diff','--cached','--name-only']).splitlines();assert set(staged).issubset(selected)
    assert not call(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
    write(P/'DELIVERY_STAGED_CHECK.json',dict(UTC=datetime.now(timezone.utc).isoformat(),staged_paths=staged,private_visibility_verified=True,whitespace_exit=0))
    call(['git','add','-f','--',rel(P/'DELIVERY_STAGED_CHECK.json')])
    print(json.dumps(dict(staged_count=len(staged)+1,base_commit=base,private_repository=repo,D_free_bytes=shutil.disk_usage('D:/').free)),flush=True)
elif mode=='finalize':
    selection=read(P/'DELIVERY_SELECTION.json');base=selection['base_commit'];assert call(['git','rev-parse','HEAD'])==base
    for name,digest in selection['files_SHA256'].items():assert sha(R/name)==digest,name
    selected=set(selection['paths'])|{rel(P/'DELIVERY_SELECTION.json'),rel(P/'DELIVERY_STAGED_CHECK.json')}
    assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(selected)
    payload=selected|{rel(f)for f in owned(S)+owned(P)if f.suffix not in{'.tar','.tmp'}}|{
        'PROJECT_ACCEPTANCE.md','PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md','PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
        'reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
    local_inputs={
      'TRAIN_path':R/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz',
      'constants_path':R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/UNIT_CHARGE_CONSTANTS.npz',
      'noise_calibration_path':R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/NOISE_CALIBRATION.npz',
      'valley_definition':R/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/VALLEY_DEFINITION.json'}
    bindings=read(S/'INPUT_BINDING.json')['inputs']
    for key,path in local_inputs.items():assert sha(path)==bindings[key]['sha256'],key;payload.add(rel(path))
    size=sum((R/name).stat().st_size for name in payload);assert shutil.disk_usage('D:/').free>size+40*1024**2
    call(['git','commit','-m','Train full-parent REAL-charge response and seal R71 component evidence'])
    commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
    remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert remote==commit
    backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r71_full_train_charge_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    backup.mkdir(parents=True);manifest=[]
    for name in sorted(payload):
        src,dst=R/name,backup/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        digest=sha(dst);assert digest==sha(src),name;manifest.append(dict(path=name,bytes=dst.stat().st_size,sha256=digest))
    bundle=backup/'r71_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
    write(backup/'BACKUP_SHA256_MANIFEST.json',manifest)
    (backup/'RESTORE_README.txt').write_text('R71 is a completed full-TRAIN charge component, not a completed whole-waveform A-F model. Restore these relative paths on E/D. Weights, complete stage outputs and original bound TRAIN/constants are included. Use work/R71_REAL_CHARGE_FULL_TRAIN_001/REPRODUCTION.md and infer.py. The Git bundle requires preceding base '+base+'. Do not resume old R65 or select recovery checkpoints by quality.\n','utf8')
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=P.name,purpose='full_TRAIN_charge_component_delivery',private_repository=selection['repository'],
      base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,backup_root=str(backup),copied_files=len(manifest),copied_bytes=size,
      all_copied_file_SHA256_verified=True,bundle_SHA256=sha(bundle),D_free_after_bytes=shutil.disk_usage('D:/').free,
      complete_main_updates=50000,engineering_updates=50,main_native_exit=0,application_goal_status='active',model_goal_complete=False,
      inference_CLI_native_exit=0,TRAIN_replay4000_bitwise=True,final_heldout4000_replay_complete=False,
      historical_acceptance_v4_unchanged=True,all_local_new_artifacts_on_E_D=True,resource={**guard,**memory()})
    write(final,result);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
    assert not call(['git','diff','--cached','--name-only']);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
else:raise ValueError(mode)
