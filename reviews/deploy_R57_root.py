from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import subprocess

R = Path(__file__).resolve().parents[1]
S = R / 'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001'
C = json.loads((S / 'config.json').read_text('utf8'))
remote = C['remote_root']
host = 'rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R57_ROOT_')
files = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'numeric_worker.py', 'render_teacher.py', 'coupling_core.py', 'weighted_metrics.py', 'noise_core.py', 'teacher_core.py', 'canonical.py', 'linux_supervise_post.py', 'linux_owned_worker.py']
for name in files:
    assert (S / name).is_file()
    if name.endswith('.py'):
        ast.parse((S / name).read_text('utf8'), filename=name)
assert C['response_training_runs'] == C['HGB_fits'] == C['ODE_generation'] == C['DEV_read'] == C['EVAL_read'] == 0
assert all(C[k] is None for k in ['science_budget', 'deadline_UTC', 'output_limit_bytes'])
review = {'UTC': datetime.now(timezone.utc).isoformat(), 'owner': '/root', 'AST_pass': True,
    'reviewed_definitions': ['ten original-Q regions with all6593 SOURCE and21047 REAL TRAIN parents', 'exact integer split masses and fixed16 Q groups', 'within-group monotone signed-PSD coupling with original index ties', 'global SOURCE equal margins and conditional REAL equal margins', 'unchanged frozen pulse-only charge correction and canonical ADC', 'all failed and overrange mass retained', 'source_h used only for frozen TRAIN valley diagnostic', 'all480 waveform moments and fixed first edges', 'fixed TRAIN conditions do not replace A-F'],
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
child=subprocess.Popen([python,'-B',str(p/'linux_supervise_post.py'),'numeric_worker.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())if(p/'REMOTE_ACTIVE_JOB.json').exists()else None
record['immediate_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
print(invoke(ssh, code))
