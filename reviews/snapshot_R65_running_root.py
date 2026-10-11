"""Publish and back up a frozen running candidate, without implying completion."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import json
import os
import shutil
import subprocess
import sys

R=Path(__file__).resolve().parents[1]
ID='R65_FULL_EMPIRICAL_D_RESPONSE_001'
S=R/'work'/ID
cache=S/'cache/running_snapshot'
cache.mkdir(parents=True,exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):
    os.environ[key]=str(cache)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h=k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
read=lambda p:json.loads((R/p).read_text('utf8'))
write=lambda p,x:(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
rel=lambda p:p.relative_to(R).as_posix()
utc=lambda:datetime.now(timezone.utc).isoformat()

def call(argv):
    value=subprocess.run(argv,cwd=R,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
    assert value.returncode==0,{'argv':argv,'native_exit':value.returncode,'stdout':value.stdout,'stderr':value.stderr}
    return value.stdout.strip()

def excluded(path):
    return any(x in ('cache','__pycache__','.pytest_cache','fixtures') or x.endswith('_cache') for x in path.parts)

assert read('work/'+ID+'/ROOT_LAUNCH_AND_ARTIFICIAL_REVIEW.json')['server_artificial_suite_passed']
assert read('PROGRAM.json')['current_stage']==ID
meta=['AGENTS.md','PLAN.md','PROGRAM.json','PROJECT_OBJECTIVE.md','PROJECT_PROGRESS.json',
      'RESOURCE_BUDGET.json','jobs.json','experiments.json','README.md','R65_RUNNING.md',
      'tasks/'+ID+'.md']
review_dirs=[R/'reviews'/name for name in ('R65_PREPARATION_001','R65_REGISTRATION_001','R65_LAUNCH_OBSERVATION_001')]
all_payload=[rel(p) for top in [S,*review_dirs] for p in top.rglob('*') if p.is_file() and not excluded(p)]
extensions={'.py','.md','.json','.ps1','.txt','.png'}
controllers=[rel(p) for p in (R/'reviews').glob('*R65*root*.py')]
selected=sorted(set(meta+controllers+[p for p in all_payload if Path(p).suffix in extensions]))
assert all((R/p).is_file() for p in selected)
assert not any(Path(p).suffix in ('.pt','.npz','.tar','.log') for p in selected)
selection='reviews/R65_RUNNING_SNAPSHOT_SELECTION_001.json'
receipt='R65_RUNNING_SNAPSHOT_RECEIPT.json'
assert not (R/receipt).exists(),'Inspect existing publication receipt; never republish blindly.'
private=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']))
assert private['visibility']=='PRIVATE'
base=call(['git','rev-parse','HEAD'])
assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(set(selected+[selection]))
write(selection,{'UTC':utc(),'stage':ID,'purpose':'frozen_running_candidate_snapshot_not_final_delivery',
    'paths':selected,'base_commit':base,'private_repository':private,'scientific_candidate_complete':False})
call(['git','add','-f','--',*selected,selection])
staged=call(['git','diff','--cached','--name-only']).splitlines()
assert set(staged).issubset(set(selected+[selection]))
whitespace=subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
    'diff','--cached','--check'],cwd=R,capture_output=True,text=True,encoding='utf8')
check={'UTC':utc(),'staged_count':len(staged),'staged_paths':staged,'whitespace_native_exit':whitespace.returncode,
    'whitespace_findings':whitespace.stdout.splitlines(),'whitespace_stderr':whitespace.stderr,
    'arrays_weights_logs_excluded':True,'source_bytes_preserved':True,'phase':'staged_for_review'}
write('reviews/R65_RUNNING_STAGED_CHECK_001.json',check)
print(json.dumps({'phase':'staged','files':len(staged),'whitespace_native_exit':whitespace.returncode,
                  'whitespace_findings':whitespace.stdout.splitlines()},indent=2),flush=True)
if len(sys.argv)<2 or sys.argv[1]!='publish':
    raise SystemExit(0)
waiver=read('reviews/R65_RUNNING_WHITESPACE_REVIEW_001.json')
assert whitespace.returncode in (0,2) and not whitespace.stderr
assert set(whitespace.stdout.splitlines())==set(waiver['exact_findings_inspected'])
assert waiver['sealed_bytes_retained'] and waiver['no_general_whitespace_suppression']
call(['git','add','-f','--','reviews/R65_RUNNING_STAGED_CHECK_001.json','reviews/R65_RUNNING_WHITESPACE_REVIEW_001.json'])
call(['git','commit','-m','Freeze full empirical response candidate and record passing CPU CUDA checks'])
commit=call(['git','rev-parse','HEAD'])
call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0]
assert commit==remote
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r65_running_frozen_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists()
payload=sorted(set(selected+all_payload+controllers+[selection,
    'reviews/R65_RUNNING_STAGED_CHECK_001.json','reviews/R65_RUNNING_WHITESPACE_REVIEW_001.json']))
required=sum((R/name).stat().st_size for name in payload)
free=shutil.disk_usage('D:/').free
assert free>required+20*1024**2
backup.mkdir(parents=True)
copied=[]
for name in payload:
    target=backup/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(R/name,target)
    copied.append({'path':name,'bytes':target.stat().st_size})
bundle=backup/'r65_running_source_incremental.bundle'
call(['git','bundle','create',str(bundle),base+'..HEAD'])
(backup/'RESTORE_README.txt').write_text(
    'Running R65 source/protocol/artificial-check snapshot; NOT a completed trained candidate. '
    'Restore previous R64 backup and its recorded R63/input dependency chain, then these relative paths. '
    'Contains frozen constants and reviewed source, not a current remote recovery checkpoint or final weight. '
    'Do not automatically redeploy or start another run. Resolve actual remote process identity first. '
    'Final scientific evaluation and final Git/D delivery remain pending. Ordinary copy only; no separate '
    'migration consistency test. Previous backups remain untouched.\n','utf8')
result={'UTC':utc(),'purpose':'frozen_running_candidate_snapshot_not_final_delivery','stage':ID,
    'private_repository':private,'base_commit':base,'local_commit':commit,'remote_main_SHA':remote,'push_exit':0,
    'backup_root':str(backup),'copied_file_count':len(copied),'copied_bytes':required,
    'git_bundle_bytes':bundle.stat().st_size,'D_free_after_bytes':shutil.disk_usage(backup).free,
    'actual_remote_scientific_native_exit':None,'final_weight_delivered':False,'model_goal_complete':False,
    'old_backups_changed':False,'migration_content_consistency_test':False,
    'CPU_affinity':[3],'RSS_hard_max_bytes':1024**3}
write(receipt,result)
shutil.copy2(R/receipt,backup/receipt)
(backup/'BACKUP_FILE_LIST.json').write_bytes((json.dumps(copied,indent=2)+'\n').encode())
assert read(receipt)==json.loads((backup/receipt).read_text('utf8'))
print(json.dumps(result,indent=2))
