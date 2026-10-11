"""Preserve the live run in memory while the user requests evidence before more cost."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import json
import subprocess

P=Path(__file__).resolve().parent
S=P.parents[1]/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
cfg=json.loads((S/'config.json').read_text('utf8'))
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
assert not (P/'BENEFIT_EVIDENCE_PAUSE_RECEIPT.json').exists()
code='from pathlib import Path\nimport json,os,signal,time,hashlib\nfrom datetime import datetime,timezone\np=Path('+repr(cfg['remote_root'])+')\n'+r'''
os.sched_setaffinity(0,{11})
def read(name):return json.loads((p/name).read_text())
def proc(pid):
    d=Path('/proc')/str(pid)
    stat=(d/'stat').read_text().rsplit(')',1)[1].split()
    return {'PID':pid,'cwd':str((d/'cwd').resolve()),'command':(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
      'start_ticks':int(stat[19]),'state':stat[0],'CPU_ticks':int(stat[11])+int(stat[12])}
active=read('REMOTE_ACTIVE_JOB.json');restore=read('SAME_RUN_RESTORE_RECEIPT.json');hs=read('train_final.py.SCIENCE_PID.json')
assert active['actual_worker_PID']==restore['PID']==hs['PID']==330411 and active['supervisor_PID']==330406
assert active.get('actual_worker_native_exit') is None and restore['native_attempt_id']==hs['nonce']
before={key:proc(pid) for key,pid in [('worker',330411),('supervisor',330406)]}
assert before['worker']['cwd']==before['supervisor']['cwd']==str(p)
assert 'linux_owned_worker.py train_final.py' in before['worker']['command']
assert 'linux_supervise.py train_final.py' in before['supervisor']['command']
assert before['worker']['state'] not in ('T','t','Z') and before['supervisor']['state'] not in ('T','t','Z')
for pid in [330411,330406]:os.kill(pid,signal.SIGSTOP)
time.sleep(1)
after={key:proc(pid) for key,pid in [('worker',330411),('supervisor',330406)]}
for key in before:assert after[key]['state']=='T' and after[key]['start_ticks']==before[key]['start_ticks']
time.sleep(1)
later={key:proc(pid) for key,pid in [('worker',330411),('supervisor',330406)]}
for key in after:assert later[key]['state']=='T' and later[key]['CPU_ticks']==after[key]['CPU_ticks']
checkpoint=read('RECOVERY_CHECKPOINT.json')
result={'UTC':datetime.now(timezone.utc).isoformat(),'status':'PAUSED_FOR_USER_REQUESTED_BENEFIT_EVIDENCE',
 'reason':'User asks to establish improvement before the excessive full-distribution training cost.',
 'native_attempt_id':hs['nonce'],'before':before,'after':later,'CPU_ticks_unchanged_during_confirmation':True,
 'physical_progress':read('PHYSICAL_PROGRESS.json'),'last_committed_checkpoint_update':checkpoint['committed_update'],
 'checkpoint_SHA256':hashlib.sha256((p/'cache/RECOVERY_SAME_RUN.pt').read_bytes()).hexdigest(),
 'native_exit':None,'in_memory_state_retained':True,'GPU_context_and_memory_retained':True,
 'automatic_resume_authorized':False,'long_run_benefit_proven':False,'new_scientific_computation_launched':False}
print(json.dumps(result),flush=True)
'''
run=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',
 '/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=20)
(P/'BENEFIT_EVIDENCE_PAUSE_NATIVE.stdout.log').write_text(run.stdout,encoding='utf8')
(P/'BENEFIT_EVIDENCE_PAUSE_NATIVE.stderr.log').write_text(run.stderr,encoding='utf8')
assert run.returncode==0,run.stderr
record=json.loads(run.stdout)
(P/'BENEFIT_EVIDENCE_PAUSE_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
stop=S/'controller_runtime/PIPELINE_STOP_REQUEST.json'
assert not stop.exists()
stop.write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),
 'reason':'User requests evidence of benefit before further full-run cost; remote worker and supervisor are paused.',
 'remote_worker_kill_requested':False,'automatic_resume':False},indent=2)+'\n',encoding='utf8')
print(json.dumps({k:record[k] for k in ['UTC','status','native_attempt_id','last_committed_checkpoint_update',
 'in_memory_state_retained','CPU_ticks_unchanged_during_confirmation','long_run_benefit_proven']},indent=2))
print(json.dumps({'last_completed_update':record['physical_progress']['last_completed_update'],
 'current_update':record['physical_progress']['current_update'],'phase':record['physical_progress']['phase']}))
