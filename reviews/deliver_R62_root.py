from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,shutil
R=Path(__file__).resolve().parents[1];STAGE='R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';S=R/'work'/STAGE
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def call(args):
 p=subprocess.run(args,cwd=R,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
 if p.returncode:raise RuntimeError({'argv':args,'native_exit':p.returncode,'stdout':p.stdout[:3000],'stderr':p.stderr[:3000]})
 return p.stdout.strip()
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']
rootfiles=['reviews/R62_ROOT_PREPARATION_FAILURES_001.json','R62_REPORT.md','tasks/'+STAGE+'.md','reviews/R62_ROOT_DIRECTION_001.json']+[str(p.relative_to(R)).replace('\\','/')for p in (R/'reviews').glob('*R62*root*.py')]
storage=R/'reviews/D_BACKUP_TEXT_NTFS_20261011_001'
assert json.loads((storage/'RECEIPT.json').read_text('utf8'))['native_exit']==0
rootfiles += [str(p.relative_to(R)).replace('\\','/')for p in storage.glob('*.json')]
rel=lambda p:str(p.relative_to(R)).replace('\\','/')
stage=[rel(p) for p in S.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.png'] and 'cache' not in p.parts]
reg=[rel(p) for d in ['R62_REGISTRATION_001'] for p in (R/'reviews'/d).iterdir() if p.is_file()]
selected=sorted(set(meta+rootfiles+stage+reg))
assert all((R/p).is_file() for p in selected) and not any(Path(p).suffix in ['.npz','.pt','.log'] for p in selected)
selectionfile='reviews/R62_PUBLICATION_SELECTION_001.json'
assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(set(selected+[selectionfile]))
private=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']));assert private['visibility']=='PRIVATE'
base=call(['git','rev-parse','HEAD'])
write(selectionfile,{'UTC':utc(),'paths':selected,'base_commit':base,'private_repository':private,'explicit_paths_only':True,'payload_arrays_weights_logs_excluded':True})
call(['git','add','-f','--',*selected,selectionfile]);staged=call(['git','diff','--cached','--name-only']).splitlines()
assert set(staged).issubset(set(selected+[selectionfile]))
whitespace=['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check']
w=subprocess.run(whitespace,cwd=R,capture_output=True,text=True,encoding='utf8')
expected=set()
assert w.returncode in [0,2] and set(w.stdout.strip().splitlines()).issubset(expected) and not w.stderr,(w.returncode,w.stdout,w.stderr)
write('reviews/R62_STAGED_CHECK_001.json',{'UTC':utc(),'staged_paths':staged,'staged_count':len(staged),'whitespace_command':whitespace,'whitespace_native_exit':w.returncode,'whitespace_exact_findings_preserved':w.stdout.splitlines(),'whitespace_exception_reason':'No whitespace exceptions allowed for this stage.','arrays_weights_logs_staged':False,'preserved_sealed_bytes':True})
print(json.dumps({'step':'staged','path_count':len(staged),'D_free_bytes':shutil.disk_usage('D:/').free},indent=2),flush=True)
if os.environ.get('R62_DELIVERY_PHASE')!='publish':raise SystemExit(0)
call(['git','commit','-m','Record frozen TRAIN waveform residuals and objective coverage'])
science=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert science==remote
write('R62_PUBLICATION_RECEIPT.json',{'UTC':utc(),'repository':private,'branch':'main','local_commit':science,'remote_SHA':remote,'push_exit':0,'base_commit':base,'selection':selectionfile})
previous=read('R61_BACKUP_RECEIPT.json')
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r62_frozen_residual_final_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists();backup.mkdir(parents=True)
payload=sorted(set(selected+[selectionfile,'reviews/R62_STAGED_CHECK_001.json','R62_PUBLICATION_RECEIPT.json']+[rel(p) for p in S.rglob('*') if p.is_file() and 'cache' not in p.parts]))
payload=sorted(set(payload+[rel(p)for p in storage.iterdir()if p.is_file()]))
required=sum((R/p).stat().st_size for p in payload);free=shutil.disk_usage(backup).free
assert free>required+20*1024**2,('actual_capacity_insufficient',required,free)
copied=[]
for name in payload:
 dest=backup/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(R/name,dest);copied.append({'source_relative_path':name,'bytes':(R/name).stat().st_size})
bundle=backup/'r62_science_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'BACKUP_FILE_LIST.json').write_bytes((json.dumps({'files':copied,'previous_R61_backup_receipt':previous,'incremental_git_base':base},ensure_ascii=False,indent=2)+'\n').encode())
(backup/'RESTORE_README.txt').write_text('R62 frozen TRAIN waveform residual diagnostic incremental backup. Restore prior R61 backup and its dependencies, then overlay these relative paths. R62 compact per-parent path statistics, all coordinate distances, baseline joint moments, 32-batch objective tables, actual indices, RNG states, assertion repair, figures and receipts are included. The original TRAIN, R61 frozen weights, constants, training configuration and saved loss trace remain historical dependencies bound by INPUT_BINDING.json and config.json; this stage neither trains a model nor constructs a teacher. Git bundle is incremental from '+base+'. Normal copying only; no migration content-consistency test and no historical deletion.\n','utf8')
write('R62_BACKUP_RECEIPT.json',{'UTC':utc(),'status':'normal_incremental_copy_complete','backup_root':str(backup),'file_count':len(copied),'copied_bytes':required,'git_bundle_bytes':bundle.stat().st_size,'D_free_before_bytes':free,'D_free_after_bytes':shutil.disk_usage(backup).free,'previous_R61_backup_receipt':'R61_BACKUP_RECEIPT.json','migration_content_consistency_test':False,'old_backups_removed_or_overwritten':False})
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage_progress='sealed_and_delivered',latest_completed_report='R62_REPORT.md',latest_completed_review='reviews/R62_ROOT_DIRECTION_001.json',latest_numeric_receipt='work/'+STAGE+'/RESOURCE_RECEIPT.json',latest_completed_backup='R62_BACKUP_RECEIPT.json',current_stage_registration='reviews/R62_REGISTRATION_001/RECEIPT.json',current_route_decision='reviews/R62_ROOT_DIRECTION_001.json',reproduction_guide='work/'+STAGE+'/REPRODUCTION.md',delivery_index='R62_DELIVERY_COMPLETION_RECEIPT.json',publication_receipt='R62_PUBLICATION_RECEIPT.json',backup_receipt='R62_BACKUP_RECEIPT.json',delivery_completion_receipt='R62_DELIVERY_COMPLETION_RECEIPT.json',model_goal_complete=False)
 x.update(status='R62_delivered_long_term_goal_active',latest_delivered_stage=STAGE,latest_stage_publication='R62_PUBLICATION_RECEIPT.json',last_native_observation='work/'+STAGE+'/LIVE_OBSERVATION.json')
 x['execution_location'].update(current_stage_GPU_used=True,preflight_required=False)
 x['current_job']['status']='sealed_and_delivered_no_live_job';x['remote_job_state']='R62_native_complete_old_R49_process_absent_after_restart_exit_unknown'
 if fn=='PROGRAM.json':x.update(status='R62_delivered_long_term_goal_active',final_report='R62_REPORT.md',updated_at_utc=utc(),latest_permission_status='Root-only long-term goal active; R62 delivered, original A-F and no compute quotas retained.')
 else:x.update(project_status='R62_delivered_long_term_goal_active',current_activity='root_following_R62_evidence_backed_direction',updated_UTC=utc())
 write(fn,x)
for fn,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);a=next(v for v in x[key] if v[identity]==STAGE);a.update(status='sealed_and_delivered',delivery_receipt='R62_DELIVERY_COMPLETION_RECEIPT.json');write(fn,x)
for fn in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/fn;p.write_bytes(p.read_text('utf8').replace('当前完成私有 Git 与 D 盘交付。','私有 Git 与 D 盘实际交付已完成，下一轮由主助手按联合目标协议继续。',1).encode())
finalfiles=meta+['R62_PUBLICATION_RECEIPT.json','R62_BACKUP_RECEIPT.json','reviews/R62_STAGED_CHECK_001.json']
call(['git','add','-f','--',*finalfiles]);call(whitespace);call(['git','commit','-m','Record R62 delivery and evidence-backed next action'])
final=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert final==remote
overlay=backup/'delivery_completion';overlay.mkdir()
for name in finalfiles:
 dest=overlay/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dest)
call(['git','bundle','create',str(overlay/'delivery_metadata.bundle'),science+'..HEAD'])
receipt={'UTC':utc(),'stage':STAGE,'status':'delivered','model_goal_complete':False,'application_goal_status':'active','owner':'/root','science_commit':science,'final_metadata_commit':final,'verified_remote_SHA':remote,'private_repository':private,'backup_root':str(backup),'final_metadata_overlay':str(overlay),'report':'R62_REPORT.md','scientific_receipt':'work/'+STAGE+'/FINAL_RECEIPT.json','backup_receipt':'R62_BACKUP_RECEIPT.json','D_free_final_bytes':shutil.disk_usage(backup).free,'active_scientific_jobs':0,'next_action':read('PROGRAM.json')['next_action']}
write('R62_DELIVERY_COMPLETION_RECEIPT.json',receipt);shutil.copy2(R/'R62_DELIVERY_COMPLETION_RECEIPT.json',overlay/'R62_DELIVERY_COMPLETION_RECEIPT.json')
assert call(['git','diff','--name-only'])=='' and call(['git','diff','--cached','--name-only'])==''
print(json.dumps(receipt,ensure_ascii=False,indent=2))
