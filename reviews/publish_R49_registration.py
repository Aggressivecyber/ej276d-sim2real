"""Publish an explicit registration/authorization whitelist, then verify PRIVATE/main."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'reviews/R49_REGISTRATION_PUBLICATION_SELECTION.json'
RECEIPT = 'reviews/R49_REGISTRATION_PUBLICATION_RECEIPT.json'


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def run(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(json.dumps({'command': args, 'native_exit': result.returncode,
                                      'stdout': result.stdout, 'stderr': result.stderr}))
    return result.stdout.strip()


assert not (ROOT / RECEIPT).exists()
assert read('reviews/R49_REGISTRATION_001/RECEIPT.json')['candidate_steps'] == 128
assert run('git', 'branch', '--show-current') == 'main'
mode = sys.argv[1:]
if mode == ['--stage']:
    assert not (ROOT / SELECTION).exists()
    names = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
        'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json', 'RESOURCE_BUDGET.json', 'SERVER_RESOURCE_POLICY.md',
        'jobs.json', 'experiments.json', 'tasks/R49_FROZEN_HEUN128_RESPONSE_001.md',
        'reviews/apply_no_compute_budget_20261010.py', 'reviews/register_R49.py', 'reviews/publish_R49_registration.py',
        'reviews/R49_NUMERICAL_DESIGN_CONSULT_001.md', 'reviews/R49_REGISTRATION_NATIVE_RECEIPT.json']
    for folder in ['reviews/NO_COMPUTE_BUDGET_20261010_001', 'reviews/R49_REGISTRATION_001']:
        names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / folder).iterdir()) if p.is_file())
    names = sorted(set(names))
    assert all(Path(n).suffix in {'.json', '.md', '.py'} for n in names)
    assert not run('git', 'diff', '--cached', '--name-only')
    run(sys.executable, '-B', 'reviews/select_continuation_publication.py', '--receipt', SELECTION, '--root', *names)
    selected = read(SELECTION)
    assert not selected['issues']
    run('git', 'add', '-f', '--', *names, SELECTION)
    staged = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(staged) == set(names + [SELECTION])
    run('git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
    print(json.dumps({'actual_PID': os.getpid(), 'staged_paths': staged, 'files': len(staged),
        'bytes_excluding_selection': selected['total_bytes'], 'credential_or_event_payload_issues': []}, ensure_ascii=False))
elif mode == ['--publish']:
    selected = read(SELECTION)
    names = [row['path'] for row in selected['files']] + [SELECTION]
    assert set(run('git', 'diff', '--cached', '--name-only').splitlines()) == set(names)
    assert not run('git', 'diff', '--name-only', '--', *names)
    for row in selected['files']:
        raw = (ROOT / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    info = json.loads(run('gh', 'repo', 'view', '--json', 'nameWithOwner,isPrivate,visibility'))
    assert info['isPrivate'] and info['nameWithOwner'] == 'Aggressivecyber/ej276d-sim2real'
    run('git', 'commit', '-m', 'Authorize unbudgeted R49 frozen-weight response evaluation')
    head = run('git', 'rev-parse', 'HEAD')
    run('git', 'push', 'origin', 'main')
    remote = run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert head == remote
    receipt = {'status': 'registration_private_Git_verified', 'UTC': datetime.now(timezone.utc).isoformat(),
        'actual_control_PID': os.getpid(), 'repository': info['nameWithOwner'], 'visibility': info['visibility'],
        'local_HEAD': head, 'GitHub_main': remote, 'match': True, 'all_native_exits': 0,
        'registration': 'reviews/R49_REGISTRATION_001/RECEIPT.json', 'selection': SELECTION,
        'compute_budget_enforced': False, 'scientific_arrays_or_weights_published': False}
    (ROOT / RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', 'utf8')
    print(json.dumps(receipt, ensure_ascii=False))
else:
    raise SystemExit('Use --stage then --publish; inspect the staged whitelist between them.')
