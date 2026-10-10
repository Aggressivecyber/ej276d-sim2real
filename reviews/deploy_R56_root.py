from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess

R = Path(__file__).resolve().parents[1]
S = R / 'work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001'
C = json.loads((S / 'config.json').read_text('utf8'))
remote = C['remote_root']
host = 'rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R56_ROOT_')
files = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'diagnose_train.py', 'render_diagnostic.py', 'network.py', 'noise_core.py', 'source_flow_core.py', 'teacher_core.py', 'canonical.py', 'linux_supervise.py', 'linux_owned_worker.py']
for name in files:
    assert (S / name).is_file()
    if name.endswith('.py'):
        ast.parse((S / name).read_text('utf8'), filename=name)
assert C['response_training_runs'] == C['HGB_fits'] == C['DEV_read'] == C['EVAL_read'] == 0
assert all(C[k] is None for k in ['science_budget', 'deadline_UTC', 'output_limit_bytes'])
review = {'UTC': datetime.now(timezone.utc).isoformat(), 'owner': '/root', 'AST_pass': True,
    'reviewed_definitions': ['original source/REAL coverage and frozen exposure counts', 'same full/partial coupling and original R54 teacher', 'same PCG64 first-cycle path with no updates', 'PSD derivative and tangent residual energy in original coordinate metric', 'all6593 original source parents one noise draw Heun128', 'all invalid/outside rows retained and exposure repeats disclosed', 'predefined six-bin direction markers are not A-F acceptance'],
    'scientific_execution_started': False, 'new_test_or_fitting': 0,
    'files': [{'name': n, 'bytes': (S / n).stat().st_size, 'sha256': hashlib.sha256((S / n).read_bytes()).hexdigest()} for n in files]}
(S / 'ROOT_IMPLEMENTATION_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n', 'utf8')
files.append('ROOT_IMPLEMENTATION_REVIEW.json')
events = []


def invoke(argv, code=None):
    p = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=60)
    events.append({'argv': argv, 'native_exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
    (S / 'DEPLOYMENT_RECEIPT.json').write_text(json.dumps({'UTC': datetime.now(timezone.utc).isoformat(),
        'control_PID': os.getpid(), 'events': events}, indent=2) + '\n', 'utf8')
    assert p.returncode == 0, events[-1]
    return p.stdout


ssh = ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, '/usr/bin/python3 -B -']
prefix = 'from pathlib import Path\nimport json,subprocess,time,os\np=Path(' + repr(remote) + ')\n'
checks = prefix + 'inputs=' + repr(C['input_paths']) + '\n'
checks += 'assert all(Path(v).is_file() for v in inputs.values())\n'
checks += 'assert Path(' + repr(C['server_python']) + ').is_file()\nassert p.parent.is_dir()\np.mkdir(exist_ok=False)\nprint(str(p))\n'
invoke(ssh, checks)
invoke(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', *[str(S / n) for n in files], host + ':' + remote + '/'])
code = prefix + 'python=' + repr(C['server_python']) + '\n' + '''
marker=p/'LAUNCH_ONCE.json'
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'diagnose_train.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())if(p/'REMOTE_ACTIVE_JOB.json').exists()else None
record['immediate_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
print(invoke(ssh, code))
