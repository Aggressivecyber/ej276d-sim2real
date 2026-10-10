from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

R = Path(__file__).resolve().parents[1]
S = R / 'work/R55_PSD_SOURCE_COMPATIBILITY_001'
C = json.loads((S / 'config.json').read_text('utf8'))
remote = C['remote_root']
host = 'rylan@100.82.231.102'
events = []


def call(argv, code=None):
    p = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=60)
    events.append({'argv': argv, 'native_exit': p.returncode, 'stderr': p.stderr})
    assert p.returncode == 0, events[-1]
    return p.stdout


code = 'from pathlib import Path\nimport json\np=Path(' + repr(remote) + ')\n' + '''
a=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert 'actual_worker_native_exit' in a and a['actual_worker_native_exit']==0 and a['termination_reason'] is None
assert not Path('/proc',str(a['actual_worker_PID'])).exists()
assert json.loads((p/'NUMERIC_RECEIPT.json').read_text())['status']=='complete'
print(json.dumps([f.name for f in sorted(p.iterdir())if f.is_file()]))
'''
names = json.loads(call(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, '/usr/bin/python3 -B -'], code))
protected = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'ROOT_IMPLEMENTATION_REVIEW.json']
selected = [n for n in names if (not n.endswith('.py') and n not in protected) or not (S / n).exists()]
assert all(Path(n).name == n for n in selected)
call(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', *[host + ':' + remote + '/' + n for n in selected], str(S) + '/'])
(S / 'FETCH_RECEIPT.json').write_text(json.dumps({'UTC': datetime.now(timezone.utc).isoformat(),
    'events': events, 'copied': selected, 'normal_transfer': True, 'migration_consistency_test': False}, indent=2) + '\n', 'utf8')
print(json.dumps({'files': len(selected), 'bytes': sum((S / n).stat().st_size for n in selected)}, indent=2))
