from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

R = Path(__file__).resolve().parents[1]
S = R / 'work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001'
C = json.loads((S / 'config.json').read_text('utf8'))
code = 'from pathlib import Path\nimport json,os\np=Path(' + repr(C['remote_root']) + ')\n' + '''
names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','LP_CONSTRUCTION.json','LP_SOLVER_RECEIPT.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']
out={n:json.loads((p/n).read_text()) for n in names if(p/n).is_file()}
a=out.get('REMOTE_ACTIVE_JOB.json',{});pid=a.get('actual_worker_PID');q=Path('/proc',str(pid))
actual={'PID':pid,'exists':bool(pid and q.exists())}
if actual['exists']:
 actual.update(cwd=str((q/'cwd').resolve()),argv=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace'),status=[s for s in(q/'status').read_text().splitlines()if s.startswith(('State:','VmRSS:','VmHWM:'))])
 stat=(q/'stat').read_text().rsplit(')',1)[1].split();ticks=os.sysconf('SC_CLK_TCK')
 actual.update(process_CPU_s=(int(stat[11])+int(stat[12]))/ticks,process_threads=int(stat[17]),process_elapsed_s=float(Path('/proc/uptime').read_text().split()[0])-int(stat[19])/ticks)
out['actual_process']=actual
print(json.dumps(out))
'''
p = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    'rylan@100.82.231.102', '/usr/bin/python3 -B -'], input=code, text=True, capture_output=True, timeout=30)
assert p.returncode == 0, p.stderr
out = json.loads(p.stdout)
now = datetime.now(timezone.utc).isoformat()
for name, value in out.items():
    if name.endswith('.json'):
        (S / name).write_text(json.dumps(value, indent=2) + '\n', 'utf8')
observation = {'UTC': now, **out}
(S / 'LIVE_OBSERVATION.json').write_text(json.dumps(observation, indent=2) + '\n', 'utf8')
a = out.get('REMOTE_ACTIVE_JOB.json', {})
live = out['actual_process']['exists']
terminal = 'actual_worker_native_exit' in a and not live
status = 'native_complete' if terminal and a['actual_worker_native_exit'] == 0 else 'native_failed' if terminal else 'running' if live else 'not_bound_observe_again'
job = {'stage': C['stage'], 'status': status, 'actual_worker_PID': a.get('actual_worker_PID'),
       'supervisor_PID': a.get('supervisor_PID'), 'actual_process': out['actual_process'],
       'actual_native_exit': a.get('actual_worker_native_exit'), 'observation_UTC': now, 'active_scientific_jobs': int(live)}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    path = R / name
    value = json.loads(path.read_text('utf8'))
    value.update(current_job=job, current_stage_progress=status, active_scientific_jobs=int(live),
                 active_remote_scientific_jobs=int(live), active_local_scientific_jobs=0)
    if name == 'PROGRAM.json':
        value['updated_at_utc'] = now
    else:
        value['updated_UTC'] = now
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf8')
print(json.dumps({'UTC': now, 'status': status, 'actual_process': out['actual_process'],
    'progress': out.get('PROGRESS.json'), 'failure': out.get('FAILURE.json'), 'native': a.get('actual_worker_native_exit'),
    'solver': out.get('LP_SOLVER_RECEIPT.json'), 'numeric': out.get('NUMERIC_RECEIPT.json')}, indent=2))
