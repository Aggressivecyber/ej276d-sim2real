"""Read only actual R64 worker/progress/native receipts; no science array I/O."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
C = json.loads((S/'config.json').read_text('utf8'))
code = 'from pathlib import Path\nimport json,os\np=Path('+repr(C['remote_root'])+')\n'+'''
def read(name):
    f=p/name
    if not f.exists():return None
    try:return json.loads(f.read_text('utf8'))
    except json.JSONDecodeError:return {'temporarily_incomplete_JSON':True}
active=read('REMOTE_ACTIVE_JOB.json')
pid=(active or {}).get('actual_worker_PID')
identity=None
if pid and Path('/proc/'+str(pid)).exists():
    proc=Path('/proc/'+str(pid))
    try:
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (proc/'status').read_text().splitlines()if ':'in line}
        identity={'PID':pid,'cwd':str((proc/'cwd').resolve()),'argv':(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace'),
                  'current_RSS_bytes':int(fields['VmRSS'].split()[0])*1024,
                  'native_highwater_RSS_bytes':int(fields['VmHWM'].split()[0])*1024,
                  'allowed_CPU_list':fields['Cpus_allowed_list']}
    except OSError:identity=None
history=read('REMOTE_RUN_HISTORY.json')
answer={'active_job':active,'actual_PID_current_identity':identity,
        'progress':read('DIAGNOSTIC_PROGRESS.json'),'native_history_last':history[-1]if isinstance(history,list)and history else history,
        'worker_receipt':read('diagnose_train.py.WORKER_RECEIPT.json'),'failure':read('DIAGNOSTIC_FAILURE.json'),
        'result':read('RESULT.json')}
for n in ['DISTRIBUTION_ARTIFICIAL_CPU.json','DISTRIBUTION_ARTIFICIAL_CUDA.json']:
    data=read(n)
    answer[n]=data.get('compact')if data else None
for n in ['NETWORK_ARTIFICIAL_CPU.json','NETWORK_ARTIFICIAL_CUDA.json']:
    data=read(n)
    answer[n]=({k:data.get(k)for k in ['pass','device','chunk','relative_parameter_gradient_L2_error','max_absolute_parameter_gradient_error','active_Box_rail_samples']}if data else None)
for n in ['diagnose_train.py.stderr.log','supervisor.stderr.log']:
    f=p/n
    answer[n]=f.read_text('utf8',errors='replace')[-9000:]if f.exists()else None
print(json.dumps(answer))
'''
p = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    'rylan@100.82.231.102', '/usr/bin/python3 -B -'], input=code, text=True,
                   capture_output=True, timeout=45)
if p.returncode:
    raise RuntimeError(dict(native_exit=p.returncode, stderr=p.stderr))
value = json.loads(p.stdout)
value.update(queried_UTC=datetime.now(timezone.utc).isoformat(), SSH_native_exit=p.returncode)
folder = S/'monitor'
folder.mkdir(exist_ok=True)
name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC.json')
(folder/name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
active, progress, result = value.get('active_job') or {}, value.get('progress') or {}, value.get('result') or {}
compact = dict(queried_UTC=value['queried_UTC'], phase=progress.get('phase'), counter=progress.get('counter'),
               elapsed_s=progress.get('elapsed_s'), current_actual_PID_identity=value['actual_PID_current_identity'],
               supervisor_record_RSS_peak_bytes=active.get('actual_worker_RSS_peak_bytes'),
               supervisor_record_is_initial_until_native_finish=active.get('actual_worker_native_exit') is None,
               GPU_process_memory_peak_bytes=active.get('GPU_process_memory_peak_bytes'),
               worker_receipt=value['worker_receipt'], failure=value['failure'],
               result_status=result.get('status'), primary=result.get('primary_parameter_gradient_comparison'))
for n in ['DISTRIBUTION_ARTIFICIAL_CPU.json', 'DISTRIBUTION_ARTIFICIAL_CUDA.json', 'NETWORK_ARTIFICIAL_CPU.json', 'NETWORK_ARTIFICIAL_CUDA.json',
          'diagnose_train.py.stderr.log', 'supervisor.stderr.log']:
    compact[n] = value[n]
if value.get('native_history_last'):
    compact['native_history_last'] = value['native_history_last']
print(json.dumps(compact, ensure_ascii=False, indent=2))
