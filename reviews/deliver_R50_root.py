from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,shutil,re
ROOT=Path(__file__).resolve().parents[1];STAGE='R50_FROZEN_VOLTAGE_ATTRIBUTION_001';S=ROOT/'work'/STAGE
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((ROOT/p).read_text('utf8'))
def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def call(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
    if p.returncode:raise RuntimeError({'argv':args,'native_exit':p.returncode,'stdout':p.stdout[:4000],'stderr':p.stderr[:4000]})
    return p.stdout.strip()
previous_staged=call(['git','diff','--cached','--name-only']).splitlines()
private=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert private['visibility']=='PRIVATE'
base=call(['git','rev-parse','HEAD'])
task='tasks/'+STAGE+'.md'
metadata=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json',
          'PROJECT_ACCEPTANCE.md','PROJECT_ACCEPTANCE.json','RESOURCE_BUDGET.json','SERVER_RESOURCE_POLICY.md','jobs.json','experiments.json']
newroot=['LONG_TERM_GOAL_20261010.md','R50_REPORT.md',task,
 'reviews/ROOT_LONG_TERM_GOAL_20261010_001.json','reviews/R50_ROOT_DIRECTION_001.json',
 'reviews/SERVER_RETURN_READONLY_20261010_001.json','reviews/SERVER_RETURN_PROCESS_RECONCILIATION_20261010_001.json',
 'reviews/register_R50_root_goal.py','reviews/deploy_R50_root.py','reviews/finalize_R50_root.py','reviews/deliver_R50_root.py']
stage_files=[str(p.relative_to(ROOT)).replace('\\','/') for p in S.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.png'] and 'cache' not in p.parts]
reg_files=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'reviews/R50_ROOT_REGISTRATION_001').iterdir() if p.is_file()]
selected=sorted(set(metadata+newroot+stage_files+reg_files))
assert set(previous_staged).issubset(set(selected+['reviews/R50_PUBLICATION_SELECTION_001.json']))
assert all((ROOT/n).is_file() for n in selected)
assert not any(Path(n).suffix in ['.npz','.pt','.log'] for n in selected)
selection={'UTC':utc(),'paths':selected,'file_count':len(selected),'bytes':sum((ROOT/n).stat().st_size for n in selected),
           'base_commit':base,'private_repository':private,'explicit_paths_only':True,'payload_arrays_weights_logs_excluded':True}
write('reviews/R50_PUBLICATION_SELECTION_001.json',selection)
call(['git','add','-f','--',*selected,'reviews/R50_PUBLICATION_SELECTION_001.json'])
staged=call(['git','diff','--cached','--name-only']).splitlines()
assert set(staged).issubset(set(selected+['reviews/R50_PUBLICATION_SELECTION_001.json']))
whitespace_command=['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check']
call(whitespace_command)
write('reviews/R50_STAGED_CHECK_001.json',{'UTC':utc(),'staged_paths':staged,'staged_count':len(staged),
    'whitespace_check':'pass_with_CRLF_recognized','whitespace_command':whitespace_command,
    'initial_default_check_exit':2,'initial_default_check_reason':'CRLF line terminators flagged under repository * -text byte-preservation rule',
    'historical_and_sealed_bytes_changed_to_satisfy_check':False,'arrays_weights_logs_staged':False})
print(json.dumps({'step':'staged','file_count':len(staged),'private':True,'D_free_bytes':shutil.disk_usage('D:/').free},indent=2),flush=True)
if os.environ.get('R50_DELIVERY_PHASE')!='publish':raise SystemExit(0)
call(['git','commit','-m','Diagnose ADC representation confound and activate root-only long-term goal'])
science=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert science==remote
write('R50_PUBLICATION_RECEIPT.json',{'UTC':utc(),'repository':private,'branch':'main','local_commit':science,'remote_SHA':remote,'push_exit':0,'base_commit':base,'selection':'reviews/R50_PUBLICATION_SELECTION_001.json'})

# Normal incremental copy, with the existing R49 backup as the historical restore dependency.
previous=read('R49_BACKUP_RECEIPT.json')
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r50_adc_attribution_final_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists();backup.mkdir(parents=True)
payload=sorted(set(selected+['reviews/R50_PUBLICATION_SELECTION_001.json','reviews/R50_STAGED_CHECK_001.json','R50_PUBLICATION_RECEIPT.json']+
    [str(p.relative_to(ROOT)).replace('\\','/') for p in S.rglob('*') if p.is_file() and 'cache' not in p.parts]))
required=sum((ROOT/n).stat().st_size for n in payload)
free=shutil.disk_usage(backup).free
assert free>required+20*1024**2,('actual_capacity_insufficient',required,free)
copied=[]
for name in payload:
    source=ROOT/name;dest=backup/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,dest)
    copied.append({'source_relative_path':name,'bytes':source.stat().st_size})
bundle=backup/'r50_science_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'BACKUP_FILE_LIST.json').write_text(json.dumps({'files':copied,'previous_R49_backup_receipt':previous,'incremental_git_base':base},ensure_ascii=False,indent=2)+'\n','utf8')
(backup/'RESTORE_README.txt').write_text('R50 incremental backup. Restore prior R49 backup and its dependencies, then overlay this directory by relative paths. The Git bundle is incremental from '+base+'. No migration content-consistency test was requested or performed. Local R49 input payloads used by R50 are included in work/R50_FROZEN_VOLTAGE_ATTRIBUTION_001/INPUT_PRIVATE.npz.\n','utf8')
write('R50_BACKUP_RECEIPT.json',{'UTC':utc(),'status':'normal_incremental_copy_complete','backup_root':str(backup),
    'file_count':len(copied),'copied_bytes':required,'git_bundle_bytes':bundle.stat().st_size,
    'D_free_before_bytes':free,'D_free_after_bytes':shutil.disk_usage(backup).free,
    'previous_R49_backup_receipt':'R49_BACKUP_RECEIPT.json','migration_content_consistency_test':False,'old_backups_removed_or_overwritten':False})

for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
    obj=read(fn);obj['current_stage_progress']='sealed_and_delivered';obj['current_job']['status']='sealed_and_delivered_no_live_job'
    obj['latest_completed_report']='R50_REPORT.md';obj['latest_completed_review']='reviews/R50_ROOT_DIRECTION_001.json'
    obj['latest_numeric_receipt']='work/'+STAGE+'/RESOURCE_RECEIPT.json';obj['latest_completed_backup']='R50_BACKUP_RECEIPT.json'
    obj['current_stage_registration']='reviews/R50_ROOT_REGISTRATION_001/RECEIPT.json'
    obj['latest_user_revision']='reviews/ROOT_LONG_TERM_GOAL_20261010_001.json'
    obj['goal_status_receipt']='reviews/ROOT_LONG_TERM_GOAL_20261010_001.json'
    obj['current_route_decision']='reviews/R50_ROOT_DIRECTION_001.json'
    obj['remote_job_state']='R50_native_complete_old_R49_process_absent_after_restart_exit_unknown'
    obj['current_policy_reconciliation']='reviews/R50_ROOT_REGISTRATION_001/RECEIPT.json'
    obj['reproduction_guide']='work/'+STAGE+'/REPRODUCTION.md'
    obj['delivery_index']='R50_DELIVERY_COMPLETION_RECEIPT.json'
    obj['publication_receipt']='R50_PUBLICATION_RECEIPT.json';obj['backup_receipt']='R50_BACKUP_RECEIPT.json'
    obj['delivery_completion_receipt']='R50_DELIVERY_COMPLETION_RECEIPT.json';obj['model_goal_complete']=False
    if fn=='PROGRAM.json':obj.update(status='R50_delivered_long_term_goal_active',final_report='R50_REPORT.md',updated_at_utc=utc(),latest_permission_status='Root-only long-term goal active; R50 delivered, original A-F and no compute quotas retained.')
    else:obj.update(project_status='R50_delivered_long_term_goal_active',current_activity='next_protocol_preparation_root_only',updated_UTC=utc())
    write(fn,obj)
for fn,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    obj=read(fn);row=next(x for x in obj[key] if x[identity]==STAGE);row['status']='sealed_and_delivered';row['delivery_receipt']='R50_DELIVERY_COMPLETION_RECEIPT.json';write(fn,obj)
for fn in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','SERVER_RESOURCE_POLICY.md']:
    p=ROOT/fn;t=p.read_text('utf8');t=t.replace('当前完成本轮交付。','本轮私有Git与D盘交付已完成，后续由主助手按独立协议继续。',1);p.write_text(t,'utf8')
budget=read('RESOURCE_BUDGET.json');import hashlib
budget['policy_sha256']=hashlib.sha256((ROOT/'SERVER_RESOURCE_POLICY.md').read_bytes()).hexdigest();write('RESOURCE_BUDGET.json',budget)
finalfiles=metadata+['R50_PUBLICATION_RECEIPT.json','R50_BACKUP_RECEIPT.json','reviews/R50_STAGED_CHECK_001.json']
call(['git','add','-f','--',*finalfiles]);call(whitespace_command);call(['git','commit','-m','Record R50 delivery and ongoing scientific goal'])
final=call(['git','rev-parse','HEAD']);call(['git','push','origin','main']);remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert final==remote
overlay=backup/'delivery_completion';overlay.mkdir()
for name in finalfiles:
    dest=overlay/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
call(['git','bundle','create',str(overlay/'delivery_metadata.bundle'),science+'..HEAD'])
receipt={'UTC':utc(),'stage':STAGE,'status':'delivered','model_goal_complete':False,'application_goal_status':'active',
    'owner':'/root','science_commit':science,'final_metadata_commit':final,'verified_remote_SHA':remote,
    'private_repository':private,'backup_root':str(backup),'final_metadata_overlay':str(overlay),
    'report':'R50_REPORT.md','scientific_receipt':'work/'+STAGE+'/FINAL_RECEIPT.json','backup_receipt':'R50_BACKUP_RECEIPT.json',
    'D_free_final_bytes':shutil.disk_usage(backup).free,'active_scientific_jobs':0,'next_action':read('PROGRAM.json')['next_action']}
write('R50_DELIVERY_COMPLETION_RECEIPT.json',receipt);shutil.copy2(ROOT/'R50_DELIVERY_COMPLETION_RECEIPT.json',overlay/'R50_DELIVERY_COMPLETION_RECEIPT.json')
assert call(['git','diff','--name-only'])=='' and call(['git','diff','--cached','--name-only'])==''
print(json.dumps(receipt,ensure_ascii=False,indent=2))
