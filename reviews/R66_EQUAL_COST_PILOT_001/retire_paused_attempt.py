"""Retire the user-held R65 process; preserve its durable evidence before R66."""
from pathlib import Path
import sys,json,subprocess

P=Path(__file__).resolve().parent
R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
guard=apply(P)
C=json.loads((R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/config.json').read_text('utf8'))
pause=json.loads((R/'reviews/R65_THROUGHPUT_OPTIMIZATION_001/BENEFIT_EVIDENCE_PAUSE_RECEIPT.json').read_text('utf8'))
assert not (P/'R65_RETIREMENT_RECEIPT.json').exists()
collect='--collect-terminal' in sys.argv
code='from pathlib import Path\nimport os,json,signal,time,hashlib,shutil\nfrom datetime import datetime,timezone\np=Path('+repr(C['remote_root'])+')\npause='+repr(pause)+'\ncollect='+repr(collect)+'\n'+r'''
os.sched_setaffinity(0,{11})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proc(pid):
 d=Path('/proc')/str(pid)
 if not d.exists():return None
 a=(d/'stat').read_text().rsplit(')',1)[1].split()
 if a[0]=='Z':return dict(PID=pid,state='Z',start_ticks=int(a[19]))
 return dict(PID=pid,state=a[0],start_ticks=int(a[19]),cwd=str((d/'cwd').resolve()),command=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode())
if collect:
 before=pause['after']
 assert json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())['actual_worker_native_exit']==-15
 for v in before.values():
  actual=proc(v['PID']);assert actual is None or actual['state']=='Z'
else:
 before={key:proc(pause['after'][key]['PID'])for key in ['worker','supervisor']}
 for key,v in before.items():
  assert v['state']=='T' and v['start_ticks']==pause['after'][key]['start_ticks']
  assert v['cwd']==str(p) and v['command']==pause['after'][key]['command']
cp=p/'cache/RECOVERY_SAME_RUN.pt'
assert sha(cp)==pause['checkpoint_SHA256']
physical=json.loads((p/'PHYSICAL_PROGRESS.json').read_text())
assert physical['last_completed_update']==613 and physical['current_update']==614
archive=p/'native_attempts'/pause['native_attempt_id']/'stage'
assert not archive.exists()
# SIGTERM is pending while stopped; SIGCONT permits its default termination,
# not a planned scientific update. The supervisor then collects native exit.
if not collect:
 os.kill(before['worker']['PID'],signal.SIGTERM)
 os.kill(before['worker']['PID'],signal.SIGCONT)
 os.kill(before['supervisor']['PID'],signal.SIGCONT)
deadline=time.monotonic()+20
while time.monotonic()<deadline:
 a=proc(before['worker']['PID']);b=proc(before['supervisor']['PID'])
 if (a is None or a['state']=='Z')and(b is None or b['state']=='Z'):break
 time.sleep(.2)
else:raise RuntimeError('R65 processes not terminal after pending SIGTERM')
active=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert active['actual_worker_native_exit']==-15,active
assert sha(cp)==pause['checkpoint_SHA256']
after_physical=json.loads((p/'PHYSICAL_PROGRESS.json').read_text())
assert after_physical==physical
archive.mkdir(parents=True)
manifest=[]
for f in sorted(p.rglob('*')):
 if not f.is_file()or'native_attempts'in f.relative_to(p).parts:continue
 target=archive/f.relative_to(p);target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(f,target)
 a=sha(f);assert a==sha(target)
 manifest.append(dict(name=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=a))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='R65_RETIRED_FOR_CAPPED_BENEFIT_PILOT',
 before=before,after={key:proc(v['PID'])for key,v in before.items()},native_exit=-15,
 last_completed_update=613,inflight_update=614,durable_checkpoint_update=500,
 checkpoint_SHA256=sha(cp),in_memory_updates_501_to_613_no_longer_retained=True,
 scientific_terminal_50000_completed=False,automatic_long_resume=False,
 unchanged_physical_progress=after_physical,archive=str(archive),manifest=manifest)
(p/'R65_COST_HOLD_TERMINATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
'''
run=subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=50)
suffix='_collect' if collect else ''
(P/f'R65_RETIREMENT_NATIVE{suffix}.stdout.log').write_text(run.stdout,encoding='utf8')
(P/f'R65_RETIREMENT_NATIVE{suffix}.stderr.log').write_text(run.stderr,encoding='utf8')
assert run.returncode==0,run.stderr
v=json.loads(run.stdout);v['local_guard']=guard
(P/'R65_RETIREMENT_RECEIPT.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v[k]for k in ['UTC','status','native_exit','durable_checkpoint_update','checkpoint_SHA256','archive']}))
print(json.dumps({'archived_files':len(v['manifest']),'bytes':sum(x['bytes']for x in v['manifest'])}))
