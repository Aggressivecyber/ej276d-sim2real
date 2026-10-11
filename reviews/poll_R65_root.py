"""Read R65 owned PID/progress/receipts; never resubmit or alter the job."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import json
import os
import subprocess

R=Path(__file__).resolve().parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
C=json.loads((S/'config.json').read_text('utf8'))
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h=k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
prefix='from pathlib import Path\nimport json,os,subprocess\np=Path('+repr(C['remote_root'])+')\n'
code=prefix+'''
result={}
for name in ('REMOTE_ACTIVE_JOB.json','ARTIFICIAL_SUITE_PROGRESS.json','ARTIFICIAL_SUITE_RECEIPT.json',
             'TRAIN_STARTED.json','TRAIN_PROGRESS.json','PHYSICAL_PROGRESS.json','RECOVERY_CHECKPOINT.json',
             'TRAINING_FAILURE.json','train_final.py.NATIVE_RECEIPT.json','train_final.py.WORKER_RECEIPT.json',
             'FULL_EMPIRICAL_RESPONSE/TRAINING_RECEIPT.json'):
    path=p/name
    if path.exists():
        try:result[name]=json.loads(path.read_text())
        except json.JSONDecodeError:result[name]={'concurrent_write_observed':True}
active=result.get('REMOTE_ACTIVE_JOB.json',{})
pid=active.get('actual_worker_PID')
result['actual_PID_present']=bool(pid and Path('/proc',str(pid)).exists())
if result['actual_PID_present']:
    result['actual_CPU_affinity']=sorted(os.sched_getaffinity(pid))
    result['actual_PID_command']=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace')
    result['actual_PID_cwd']=str(Path('/proc',str(pid),'cwd').resolve())
    result['actual_PID_memory_now']={l.split(':')[0]:int(l.split()[1])*1024
        for l in Path('/proc',str(pid),'status').read_text().splitlines()
        if l.startswith(('VmRSS:','VmHWM:'))}
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
        text=True,capture_output=True,timeout=5)
    result['GPU_process_query_now']={'native_exit':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr}
for name in ('train_final.py.stderr.log','train_final.py.supervisor.stderr.log'):
    path=p/name
    if path.exists():result[name]=path.read_text(errors='replace')[-10000:]
print(json.dumps(result))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
                  'rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
assert p.returncode==0,p.stderr
observed=json.loads(p.stdout)
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),'read_only':True,**observed}
(S/'LATEST_REMOTE_OBSERVATION.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
active=observed.get('REMOTE_ACTIVE_JOB.json',{})
compact={'actual_PID_present':observed['actual_PID_present'],'entry':active.get('entry'),
    'PID':active.get('actual_worker_PID'),'native_exit':active.get('actual_worker_native_exit'),
    'termination_reason':active.get('termination_reason')}
for name in ('actual_PID_memory_now','GPU_process_query_now','TRAINING_FAILURE.json'):
    if name in observed:compact[name]=observed[name]
if 'TRAIN_PROGRESS.json' in observed:
    progress=observed['TRAIN_PROGRESS.json']
    compact['training_log']={k:progress[k] for k in ('UTC','global_step','elapsed_s_this_attempt',
       'GPU_allocated_peak_bytes','GPU_reserved_peak_bytes','max_output_relative_q_error_f32') if k in progress}
    means=progress.get('last_up_to100_mean',{})
    compact['training_log']['last_up_to100_mean']={k:means[k] for k in ('total_loss','D_total','C_total',
        'overall_PSD_W1','Q1_Q7_mean_PSD_W1','D_parameter_gradient_L2','C_parameter_gradient_L2') if k in means}
if 'ARTIFICIAL_SUITE_PROGRESS.json' in observed:
    suite=observed['ARTIFICIAL_SUITE_PROGRESS.json']
    compact['artificial_suite']={k:suite[k] for k in ('current_check','completed_checks','status','pass','elapsed_s') if k in suite}
if 'PHYSICAL_PROGRESS.json' in observed:
    physical=observed['PHYSICAL_PROGRESS.json']
    compact['physical_progress']={k:physical[k] for k in ('UTC','phase','current_update','last_completed_update','last_checkpoint_update',
      'known_completed_physical_operations_this_attempt') if k in physical}
print(json.dumps(compact,indent=2))
