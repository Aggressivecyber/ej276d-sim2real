from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,shutil
R=Path(__file__).resolve().parents[1];STAGE='R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';S=R/'work'/STAGE
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def call(args):
 p=subprocess.run(args,cwd=R,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
 if p.returncode:raise RuntimeError({'argv':args,'native_exit':p.returncode,'stdout':p.stdout[:3000],'stderr':p.stderr[:3000]})
 return p.stdout.strip()
final=read('work/'+STAGE+'/FINAL_RECEIPT.json')
assert final['status']=='scientific_endpoint_complete_delivery_pending'
assert read('work/'+STAGE+'/FIGURE_QA.json')['all_figures_opened']
assert read('work/'+STAGE+'/FROZEN_ARTIFACT_REVIEW.json')['pass']
assert read('work/'+STAGE+'/TRAINING_FROZEN_AUDIT.json')['pass']
assert not read('work/'+STAGE+'/LIVE_OBSERVATION.json')['actual_process']['exists']
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json','RESOURCE_BUDGET.json','jobs.json','experiments.json']
rootfiles=['R63_REPORT.md','tasks/'+STAGE+'.md','reviews/R63_ROOT_DIRECTION_001.json','reviews/FROZEN_A_F_VALIDATION_PREPARATION_001.md']+[str(p.relative_to(R)).replace('\\','/')for p in (R/'reviews').glob('*R63*root*.py')]
rel=lambda p:str(p.relative_to(R)).replace('\\','/')
stage=[rel(p) for p in S.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.png'] and 'cache' not in p.parts]
reg=[rel(p) for d in ['R63_REGISTRATION_001','R63_POST_AUDIT_ARTIFICIAL_001'] for p in (R/'reviews'/d).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.png'] and 'cache' not in p.parts]
rootfiles+=['reviews/R63_POSTPROCESS_READY_001.json','reviews/R63_ARTIFICIAL_ASSERTION_REPAIR_001.json','reviews/R63_CLOSEOUT_PREPARATION_001.json']
rootfiles+=['LONG_TERM_GOAL_20261010.md','README.md','reviews/rewrite_goal_parallel_20261011.py']
def is_cache(p):return any(k in ['cache','__pycache__'] or k.endswith('_cache') for k in p.parts)
auxiliary_payload=[rel(p) for d in ['PROJECT_GOAL_PARALLEL_20261011_001','R63_WAIT_PARALLEL_001'] for p in (R/'reviews'/d).rglob('*') if p.is_file() and 'before' not in p.relative_to(R/'reviews'/d).parts and not is_cache(p)]
auxiliary=[p for p in auxiliary_payload if Path(p).suffix in ['.py','.ps1','.json','.md','.png']]
goal_originals=[rel(p) for p in (R/'reviews/PROJECT_GOAL_PARALLEL_20261011_001/before').rglob('*') if p.is_file()]
goal_originals += [rel(p) for p in (R/'reviews/R63_WAIT_PARALLEL_001/root_analysis/before').rglob('*') if p.is_file()]
selected=sorted(set(meta+rootfiles+stage+reg+auxiliary))
assert all((R/p).is_file() for p in selected) and not any(Path(p).suffix in ['.npz','.pt','.log'] for p in selected)
selectionfile='reviews/R63_PUBLICATION_SELECTION_001.json'
assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(set(selected+[selectionfile]))
private=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']));assert private['visibility']=='PRIVATE'
base=call(['git','rev-parse','HEAD'])
write(selectionfile,{'UTC':utc(),'paths':selected,'base_commit':base,'private_repository':private,'explicit_paths_only':True,'payload_arrays_weights_logs_excluded':True})
call(['git','add','-f','--',*selected,selectionfile]);staged=call(['git','diff','--cached','--name-only']).splitlines()
assert set(staged).issubset(set(selected+[selectionfile]))
whitespace=['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check']
w=subprocess.run(whitespace,cwd=R,capture_output=True,text=True,encoding='utf8')
expected={f'work/{STAGE}/{name}:{line}: new blank line at EOF.' for name,line in [('common_math.py',50),('data_support.py',73),('evaluation.py',422),('network.py',51)]}
expected.add('reviews/R63_WAIT_PARALLEL_001/validation_prep/report.json:1298: new blank line at EOF.')
assert w.returncode in [0,2] and set(w.stdout.strip().splitlines()).issubset(expected) and not w.stderr,(w.returncode,w.stdout,w.stderr)
write('reviews/R63_STAGED_CHECK_001.json',{'UTC':utc(),'staged_paths':staged,'staged_count':len(staged),'whitespace_command':whitespace,'whitespace_native_exit':w.returncode,'whitespace_exact_findings_preserved':w.stdout.splitlines(),'whitespace_exception_reason':'Historical EOF blank lines in frozen helpers and the sealed validation-preparation report retain their reviewed or executed bytes; only these exact findings are permitted.','arrays_weights_logs_staged':False,'preserved_sealed_bytes':True})
print(json.dumps({'step':'staged','path_count':len(staged),'D_free_bytes':shutil.disk_usage('D:/').free},indent=2),flush=True)
if os.environ.get('R63_DELIVERY_PHASE')!='publish':raise SystemExit(0)
call(['git','commit','-m','Record full-waveform temporal coverage extension and unchanged joint acceptance'])
science=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert science==remote
write('R63_PUBLICATION_RECEIPT.json',{'UTC':utc(),'repository':private,'branch':'main','local_commit':science,'remote_SHA':remote,'push_exit':0,'base_commit':base,'selection':selectionfile})
previous=read('R62_BACKUP_RECEIPT.json')
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r63_coverage_response_final_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists();backup.mkdir(parents=True)
payload=sorted(set(selected+goal_originals+auxiliary_payload+[selectionfile,'reviews/R63_STAGED_CHECK_001.json','R63_PUBLICATION_RECEIPT.json']+[rel(p) for p in S.rglob('*') if p.is_file() and 'cache' not in p.parts]))
required=sum((R/p).stat().st_size for p in payload);free=shutil.disk_usage(backup).free
assert free>required+20*1024**2,('actual_capacity_insufficient',required,free)
copied=[]
for name in payload:
 dest=backup/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(R/name,dest);copied.append({'source_relative_path':name,'bytes':(R/name).stat().st_size})
bundle=backup/'r63_science_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'BACKUP_FILE_LIST.json').write_bytes((json.dumps({'files':copied,'previous_R62_backup_receipt':previous,'incremental_git_base':base},ensure_ascii=False,indent=2)+'\n').encode())
(backup/'RESTORE_README.txt').write_text('R63 full-waveform coverage extension incremental backup. Restore prior R62 backup and its dependencies, then overlay these relative paths. R63 portable inference weights and constants, final random state, R61-matched parent exposure counts, original precision-order traces, new coverage traces and TRAIN temporal statistics, private outputs, descriptions and predictions are included. Original full TRAIN and frozen charge/noise constants remain training-reproduction dependencies identified in INPUT_BINDING.json and config.json; Frozen R61 training records and distribution statistics are required for the registered objective-extension audit; they are retained through the prior R62/R61 backup chain. No teacher coupling is used. Git bundle is incremental from '+base+'. Normal copying only; no migration content-consistency test and no historical deletion.\n','utf8')
write('R63_BACKUP_RECEIPT.json',{'UTC':utc(),'status':'normal_incremental_copy_complete','backup_root':str(backup),'file_count':len(copied),'copied_bytes':required,'git_bundle_bytes':bundle.stat().st_size,'D_free_before_bytes':free,'D_free_after_bytes':shutil.disk_usage(backup).free,'previous_R62_backup_receipt':'R62_BACKUP_RECEIPT.json','migration_content_consistency_test':False,'old_backups_removed_or_overwritten':False})
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_stage_progress='sealed_and_delivered',latest_completed_report='R63_REPORT.md',latest_completed_review='reviews/R63_ROOT_DIRECTION_001.json',latest_numeric_receipt='work/'+STAGE+'/RESOURCE_RECEIPT.json',latest_completed_backup='R63_BACKUP_RECEIPT.json',current_stage_registration='reviews/R63_REGISTRATION_001/RECEIPT.json',current_route_decision='reviews/R63_ROOT_DIRECTION_001.json',reproduction_guide='work/'+STAGE+'/REPRODUCTION.md',delivery_index='R63_DELIVERY_COMPLETION_RECEIPT.json',publication_receipt='R63_PUBLICATION_RECEIPT.json',backup_receipt='R63_BACKUP_RECEIPT.json',delivery_completion_receipt='R63_DELIVERY_COMPLETION_RECEIPT.json',model_goal_complete=False)
 x['current_job']['status']='sealed_and_delivered_no_live_job';x['remote_job_state']='R63_native_complete_old_R49_process_absent_after_restart_exit_unknown'
 if fn=='PROGRAM.json':x.update(status='R63_delivered_long_term_goal_active',final_report='R63_REPORT.md',updated_at_utc=utc(),latest_permission_status='Root-led long-term goal active with optional multi-agent auxiliary research and analysis; R63 delivered, original A-F, shared resource limits, one scientific job and no compute quotas retained.')
 else:x.update(project_status='R63_delivered_long_term_goal_active',current_activity='root_following_R63_evidence_backed_direction',updated_UTC=utc())
 if fn=='PROJECT_PROGRESS.json':
  for row in x.get('requirements',[]):
   if row['id']=='final_research_delivery':
    row['status']='R63_stage_delivery_complete_full_model_goal_unmet'
    row['latest_evidence']=['R63_REPORT.md','R63_PUBLICATION_RECEIPT.json','R63_BACKUP_RECEIPT.json']
    row['candidate_stage']=STAGE
 write(fn,x)
for fn,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);a=next(v for v in x[key] if v[identity]==STAGE);a.update(status='sealed_and_delivered',delivery_receipt='R63_DELIVERY_COMPLETION_RECEIPT.json');write(fn,x)
for fn in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/fn;p.write_bytes(p.read_text('utf8').replace('私有 Git 与 D 盘交付待完成。','私有 Git 与 D 盘实际交付已完成，下一轮由主助手按联合目标协议继续。',1).encode())
finalfiles=meta+['R63_PUBLICATION_RECEIPT.json','R63_BACKUP_RECEIPT.json','reviews/R63_STAGED_CHECK_001.json']
call(['git','add','-f','--',*finalfiles]);call(whitespace);call(['git','commit','-m','Record R63 delivery and evidence-backed next action'])
final=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert final==remote
overlay=backup/'delivery_completion';overlay.mkdir()
for name in finalfiles:
 dest=overlay/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dest)
call(['git','bundle','create',str(overlay/'delivery_metadata.bundle'),science+'..HEAD'])
receipt={'UTC':utc(),'stage':STAGE,'status':'delivered','model_goal_complete':False,'application_goal_status':'active','owner':'/root','science_commit':science,'final_metadata_commit':final,'verified_remote_SHA':remote,'private_repository':private,'backup_root':str(backup),'final_metadata_overlay':str(overlay),'report':'R63_REPORT.md','scientific_receipt':'work/'+STAGE+'/FINAL_RECEIPT.json','backup_receipt':'R63_BACKUP_RECEIPT.json','D_free_final_bytes':shutil.disk_usage(backup).free,'active_scientific_jobs':0,'next_action':read('PROGRAM.json')['next_action']}
write('R63_DELIVERY_COMPLETION_RECEIPT.json',receipt);shutil.copy2(R/'R63_DELIVERY_COMPLETION_RECEIPT.json',overlay/'R63_DELIVERY_COMPLETION_RECEIPT.json')
assert call(['git','diff','--name-only'])=='' and call(['git','diff','--cached','--name-only'])==''
print(json.dumps(receipt,ensure_ascii=False,indent=2))
