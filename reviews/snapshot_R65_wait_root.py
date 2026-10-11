"""Explicit-path private Git and normal D copy for waiting preparation."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys

R = Path(__file__).resolve().parents[1]
W = R / 'reviews/R65_WAIT_ANALYSIS_001'
S = R / 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
import local_resource_guard as guard
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)
resource = guard.apply(W / 'root_snapshot')
read = lambda p: json.loads((R / p).read_text('utf8'))
rel = lambda p: p.relative_to(R).as_posix()
utc = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    (R / p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())

def call(argv):
    if argv[0] == 'git':
        argv = ['git', '-c', 'core.longpaths=true', *argv[1:]]
    result = subprocess.run(argv, cwd=R, capture_output=True, text=True, encoding='utf8', errors='replace', timeout=60)
    assert result.returncode == 0, {'argv': argv, 'native_exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    return result.stdout.strip()

def excluded(p):
    return any(part in ('cache', '__pycache__', 'runtime_cache', 'tmp', '.git') for part in p.parts)

closure = read('reviews/R65_WAIT_ANALYSIS_001/ROOT_CLOSURE_REVIEW.json')
assert closure['status'] == 'ROOT_WAIT_PREPARATION_REVIEW_COMPLETE'
assert read('work/R65_FULL_EMPIRICAL_D_RESPONSE_001/controller_runtime/ROOT_WATCHER_ACTUAL_VERIFICATION.json')['latest_state'] == 'RUNNING_OR_FINALIZING'
receipt = 'R65_WAIT_PREPARATION_SNAPSHOT_RECEIPT.json'
assert not (R / receipt).exists(), 'Inspect the existing completed receipt; do not republish blindly.'
payload = sorted(rel(p) for p in W.rglob('*') if p.is_file() and not excluded(p))
extras = ['R65_WAIT_PREPARATION.md', 'R65_RUNNING_SNAPSHOT_RECEIPT.json',
    'reviews/close_R65_wait_analysis_root.py', 'reviews/start_R65_watch_root.ps1',
    'reviews/snapshot_R65_wait_root.py', 'reviews/review_R65_wait_operations_root.py',
    'reviews/prepare_R65_wait_synthesis_root.py', 'reviews/observe_R65_runtime_root.py',
    'reviews/poll_R65_root.py',
    'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/LATEST_REMOTE_OBSERVATION.json',
    'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/controller_runtime/ROOT_WATCHER_LAUNCH.json',
    'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/controller_runtime/ROOT_WATCHER_ACTUAL_VERIFICATION.json',
    'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/controller_runtime/PIPELINE_WATCH_STARTED.json']
extensions = {'.py', '.md', '.json', '.ps1', '.txt'}
selected = sorted(set(extras + [p for p in payload if Path(p).suffix in extensions
                               and not {'fixtures', 'cases'}.intersection(Path(p).parts)]))
assert all((R / p).is_file() for p in selected)
selection = 'reviews/R65_WAIT_PREPARATION_SELECTION_001.json'
check_path = 'reviews/R65_WAIT_PREPARATION_STAGED_CHECK_001.json'
base = call(['git', 'rev-parse', 'HEAD'])
private = json.loads(call(['gh', 'repo', 'view', 'Aggressivecyber/ej276d-sim2real', '--json', 'visibility,url']))
assert private['visibility'] == 'PRIVATE'
already = set(call(['git', 'diff', '--cached', '--name-only']).splitlines())
previous_selection = read(selection) if (R / selection).exists() else {'paths': []}
assert already.issubset(set(selected + previous_selection['paths'] + [selection, check_path]))
obsolete = sorted(already - set(selected + [selection, check_path]))
if obsolete:
    assert all('cases' in Path(name).parts for name in obsolete)
    repair = W / 'root_snapshot/STAGING_REPAIR.json'
    assert not repair.exists(), 'Inspect the recorded staging repair before retrying.'
    repair.write_bytes((json.dumps({'UTC': utc(), 'cause': 'nested synthetic fixture path exceeds Windows Git filename limit',
        'initial_selection_SHA256': sha(R / selection), 'unstaged_owned_fixture_paths': obsolete,
        'scientific_files_modified': False, 'files_deleted': False,
        'synthetic_results_and_generators_retained_in_Git': True,
        'complete_synthetic_fixtures_retained_on_E_and_D': True}, indent=2) + '\n').encode())
    for offset in range(0, len(obsolete), 60):
        call(['git', 'restore', '--staged', '--', *obsolete[offset:offset + 60]])
write(selection, {'UTC': utc(), 'purpose': 'wait_preparation_not_final_model', 'base_commit': base,
                  'private_repository': private, 'paths': selected, 'model_goal_complete': False})
for offset in range(0, len(selected), 60):
    call(['git', 'add', '-f', '--', *selected[offset:offset + 60]])
call(['git', 'add', '-f', '--', selection])
staged = call(['git', 'diff', '--cached', '--name-only']).splitlines()
assert set(staged).issubset(set(selected + [selection, check_path]))
ws = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                     'diff', '--cached', '--check'], cwd=R, capture_output=True, text=True, encoding='utf8')
assert ws.returncode in (0, 2) and not ws.stderr
eof = []
for line in ws.stdout.splitlines():
    assert line.endswith(': new blank line at EOF.'), line
    name, number, _ = line.rsplit(':', 2)
    raw = (R / name).read_bytes()
    assert raw.rstrip(b'\r\n') + b'\n' != raw and raw.endswith((b'\n\n', b'\n\r\n')), name
    eof.append({'finding': line, 'path': name, 'source_SHA256': hashlib.sha256(raw).hexdigest(),
                'last_80_bytes_hex': raw[-80:].hex(), 'bound_bytes_preserved': True})
check = {'UTC': utc(), 'staged_count': len(staged), 'staged_paths': staged,
         'whitespace_native_exit': ws.returncode, 'exact_EOF_findings': eof,
         'other_whitespace_findings': [], 'scientific_arrays_weights_logs_excluded_from_Git': True,
         'source_bytes_preserved': True, 'resource': dict(resource, **guard.memory())}
write(check_path, check)
print(json.dumps({'phase': 'staged', 'files': len(staged), 'EOF_only_findings': len(eof),
                  'findings': [row['finding'] for row in eof], 'check': check_path}, indent=2), flush=True)
if len(sys.argv) < 2 or sys.argv[1] != 'publish':
    raise SystemExit(0)
call(['git', 'add', '-f', '--', check_path])
call(['git', 'commit', '-m', 'Preserve R65 waiting analysis and reviewed serial post controllers'])
commit = call(['git', 'rev-parse', 'HEAD'])
call(['git', 'push', 'origin', 'main'])
remote = call(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
assert remote == commit
backup = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305') / ('r65_wait_preparation_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists()
copies = sorted(set(payload + extras + [selection, check_path]))
needed = sum((R / p).stat().st_size for p in copies)
assert shutil.disk_usage('D:/').free > needed + 30 * 1024 ** 2
backup.mkdir(parents=True)
copied = []
for name in copies:
    dest = backup / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R / name, dest)
    copied.append({'path': name, 'bytes': dest.stat().st_size})
bundle = backup / 'r65_wait_source_incremental.bundle'
call(['git', 'bundle', 'create', str(bundle), base + '..HEAD'])
(backup / 'RESTORE_README.txt').write_text(
    'R65 waiting preparation and running-watcher start evidence only. Restore prior R65 running-source '
    'snapshot and dependency chain before this incremental copy. No final weight or recovery checkpoint '
    'is contained here. Saved PID/status is historical, so observe actual remote/local process identity '
    'before acting. Never run a second watcher blindly. Original A-F and scientific method unchanged. '
    'Normal copy; no separate migration consistency test. Previous backups untouched.\n', 'utf8')
result = {'UTC': utc(), 'purpose': 'wait_preparation_not_final_model', 'private_repository': private,
          'base_commit': base, 'local_commit': commit, 'remote_main_SHA': remote, 'push_exit': 0,
          'backup_root': str(backup), 'copied_file_count': len(copied), 'copied_bytes': needed,
          'git_bundle_bytes': bundle.stat().st_size, 'D_free_after_bytes': shutil.disk_usage(backup).free,
          'scientific_training_native_exit': None, 'final_weight_delivered': False,
          'model_goal_complete': False, 'old_backups_changed': False,
          'migration_content_consistency_test': False, 'resource': dict(resource, **guard.memory())}
write(receipt, result)
shutil.copy2(R / receipt, backup / receipt)
(backup / 'BACKUP_FILE_LIST.json').write_bytes((json.dumps(copied, indent=2) + '\n').encode())
assert read(receipt) == json.loads((backup / receipt).read_text('utf8'))
print(json.dumps(result, indent=2), flush=True)
