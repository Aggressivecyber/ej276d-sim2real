"""Explicitly publish local-recovery authorization and preserved remote observations."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'reviews/R49_LOCAL_REGISTRATION_PUBLICATION_SELECTION.json'
RECEIPT = 'reviews/R49_LOCAL_REGISTRATION_PUBLICATION_RECEIPT.json'


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def run(*args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if p.returncode:
        raise RuntimeError(json.dumps({'command': args, 'native_exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}))
    return p.stdout.strip()


assert not (ROOT / RECEIPT).exists()
assert read('reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json')['stage'] == 'R49_LOCAL_HEUN128_RECOVERY_001'
assert run('git', 'branch', '--show-current') == 'main'
if sys.argv[1:] == ['--stage']:
    assert not (ROOT / SELECTION).exists() and not run('git', 'diff', '--cached', '--name-only')
    names = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
        'RESOURCE_BUDGET.json', 'SERVER_RESOURCE_POLICY.md', 'jobs.json', 'experiments.json',
        'tasks/R49_LOCAL_HEUN128_RECOVERY_001.md', 'reviews/register_R49_local_recovery.py',
        'reviews/publish_R49_local_registration.py', 'reviews/R49_LOCAL_REGISTRATION_NATIVE_RECEIPT.json',
        'reviews/R49_REMOTE_INTERRUPTION_001.json', 'reviews/R49_ACCEPTANCE_OBSERVATION_001.json',
        'reviews/R49_ROOT_DIRECTION_001.json', 'reviews/R49_LIVE_OBSERVATION.json',
        'reviews/observe_R49_acceptance.py', 'reviews/refresh_R49_live_state.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'reviews/R49_LOCAL_REGISTRATION_001').iterdir()) if p.is_file()]
    names = sorted(set(names))
    assert all((ROOT / n).is_file() and Path(n).suffix in {'.json', '.md', '.py'} for n in names)
    run(sys.executable, '-B', 'reviews/select_continuation_publication.py', '--receipt', SELECTION, '--root', *names)
    selected = read(SELECTION)
    assert not selected['issues']
    run('git', 'add', '-f', '--', *names, SELECTION)
    staged = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(staged) == set(names + [SELECTION])
    run('git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
    print(json.dumps({'staged_paths': staged, 'files': len(staged), 'bytes': selected['total_bytes'], 'issues': []}))
elif sys.argv[1:] == ['--publish']:
    selected = read(SELECTION)
    names = [row['path'] for row in selected['files']] + [SELECTION]
    assert set(run('git', 'diff', '--cached', '--name-only').splitlines()) == set(names)
    assert not run('git', 'diff', '--name-only', '--', *names)
    for row in selected['files']:
        data = (ROOT / row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    info = json.loads(run('gh', 'repo', 'view', '--json', 'nameWithOwner,isPrivate,visibility'))
    assert info['isPrivate'] and info['nameWithOwner'] == 'Aggressivecyber/ej276d-sim2real'
    run('git', 'commit', '-m', 'Register local R49 recovery during server maintenance')
    head = run('git', 'rev-parse', 'HEAD')
    run('git', 'push', 'origin', 'main')
    remote = run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert head == remote
    receipt = {'status': 'registration_private_Git_verified', 'UTC': datetime.now(timezone.utc).isoformat(),
        'actual_control_PID': os.getpid(), 'repository': info['nameWithOwner'], 'visibility': info['visibility'],
        'local_HEAD': head, 'GitHub_main': remote, 'match': True, 'all_native_exits': 0,
        'registration': 'reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json', 'selection': SELECTION,
        'compute_budget_enforced': False, 'scientific_arrays_or_weights_published': False,
        'remote_interrupted_stage_claimed_complete': False}
    (ROOT / RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', 'utf8')
    print(json.dumps(receipt, ensure_ascii=False))
else:
    raise SystemExit('Use --stage then inspect the whitelist, then --publish.')
