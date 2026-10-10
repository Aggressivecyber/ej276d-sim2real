from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

R = Path(__file__).resolve().parents[1]
S = R / 'work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001'
C = json.loads((S / 'config.json').read_text('utf8'))
code = 'from pathlib import Path\nimport json\np=Path(' + repr(C['remote_root']) + ')\n' + '''
names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','COVERAGE.json','NUMERIC_RECEIPT.json','FAILURE.json','diagnose_train.py.NATIVE_RECEIPT.json']
out={n:json.loads((p/n).read_text()) for n in names if(p/n).is_file()}
a=out.get('REMOTE_ACTIVE_JOB.json',{});pid=a.get('actual_worker_PID');q=Path('/proc',str(pid))
actual={'PID':pid,'exists':bool(pid and q.exists())}
if actual['exists']:
 actual.update(cwd=str((q/'cwd').resolve()),argv=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace'),status=[s for s in(q/'status').read_text().splitlines()if s.startswith(('State:','VmRSS:','VmHWM:'))])
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
for name, key, ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    path=R/name
    value=json.loads(path.read_text('utf8'))
    row=next(v for v in value[key] if v[ident]==C['stage'])
    row.update(status=status,actual_worker_PID=a.get('actual_worker_PID'),supervisor_PID=a.get('supervisor_PID'),last_observed_UTC=now,actual_native_exit=a.get('actual_worker_native_exit'),active_scientific_jobs=int(live))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf8')
print(json.dumps({'UTC': now, 'status': status, 'actual_process': out['actual_process'],
    'progress': out.get('PROGRESS.json'), 'failure': out.get('FAILURE.json'), 'native': a.get('actual_worker_native_exit'),
    'coverage': {k:out.get('COVERAGE.json',{}).get(k) for k in ['SOURCE_total','REAL_total','SOURCE_seen','REAL_seen']}, 'numeric': out.get('NUMERIC_RECEIPT.json')}, indent=2))
