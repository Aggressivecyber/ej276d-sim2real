"""Keep old tracked content/newline styles while retaining the new R70 header/data."""
from pathlib import Path
import json
import sys
P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
apply(P / 'format_runtime')
for name in ('AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md'):
    old = (P / 'before_target_correction' / name).read_bytes()
    old_normal = old.decode('utf8').replace('\r\n', '\n')
    current = (R / name).read_text(encoding='utf8')
    assert current.endswith(old_normal), name
    prefix = current[:-len(old_normal)]
    newline = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    (R / name).write_bytes(prefix.encode('utf8').replace(b'\n', newline) + old)
for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json', 'experiments.json'):
    old = (P / 'before_target_correction' / name).read_bytes()
    data = json.loads((R / name).read_text(encoding='utf8'))
    newline = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    trailer = old[len(old.rstrip(b'\r\n')):] or newline
    (R / name).write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode('utf8').replace(b'\n', newline) + trailer)
print('Preserved five historical text suffixes and three JSON newline styles; values unchanged.')
