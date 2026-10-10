"""Stage an inspected R48 whitelist, then publish and verify the private branch."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
SELECTION = 'reviews/R48_PUBLICATION_SELECTION_001.json'
RECEIPT = 'R48_PUBLICATION_RECEIPT.json'


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def run(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(json.dumps({'command': args, 'native_exit': result.returncode,
                                      'stdout': result.stdout, 'stderr': result.stderr}))
    return result.stdout.strip()


def check_selection(selection):
    assert not selection['issues'] and selection['status'] == 'ready_for_explicit_staging'
    paths = []
    for row in selection['files']:
        path = (ROOT / row['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert path.suffix.lower() in {'.py', '.md', '.json', '.png', '.txt', '.ps1'}
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
        paths.append(row['path'])
    assert len(paths) == len(set(paths)) and SELECTION not in paths
    return paths + [SELECTION]


assert not (ROOT / RECEIPT).exists()
assert read('work/' + STAGE + '/FINAL_RECEIPT.json')['status'] == 'sealed'
assert run('git', 'branch', '--show-current') == 'main'
selection = read(SELECTION)
names = check_selection(selection)

if sys.argv[1:] == ['--stage']:
    partial = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(partial) <= set(names), partial
    run('git', 'add', '-f', '--', *names)
    staged = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert staged and set(staged) <= set(names)
    selection.update(actual_staged_paths=staged, unchanged_selected_paths=sorted(set(names) - set(staged)),
                     selection_itself_also_staged=SELECTION,
                     staged_UTC=datetime.now(timezone.utc).isoformat())
    save(SELECTION, selection)
    run('git', 'add', '-f', '--', SELECTION)
    assert set(run('git', 'diff', '--cached', '--name-only').splitlines()) == set(staged)
    run('git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
    print(json.dumps({'staged_count': len(staged), 'unchanged_selected_count': len(selection['unchanged_selected_paths']),
                      'paths': staged, 'actual_PID': os.getpid()}, ensure_ascii=False))
elif sys.argv[1:] == ['--publish']:
    assert set(run('git', 'diff', '--cached', '--name-only').splitlines()) == set(selection['actual_staged_paths'])
    assert not run('git', 'diff', '--name-only', '--', *names)
    info = json.loads(run('gh', 'repo', 'view', '--json', 'nameWithOwner,isPrivate,visibility'))
    assert info['isPrivate'] and info['nameWithOwner'] == 'Aggressivecyber/ej276d-sim2real'
    run('git', 'commit', '-m', 'Record R48 paired guided noise-completion flow evidence')
    head = run('git', 'rev-parse', 'HEAD')
    run('git', 'push', 'origin', 'main')
    remote = run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert head == remote
    receipt = {'status': 'verified_published', 'stage': STAGE,
        'completed_UTC': datetime.now(timezone.utc).isoformat(), 'actual_control_PID': os.getpid(),
        'repository': info['nameWithOwner'], 'visibility': info['visibility'], 'visibility_verified_live': True,
        'local_HEAD': head, 'GitHub_main': remote, 'match': True,
        'files': selection['actual_staged_paths'], 'unchanged_selected_paths': selection['unchanged_selected_paths'],
        'selection': SELECTION, 'all_git_commands_native_exit': 0,
        'scientific_payloads_in_Git': False, 'scientific_rerun_for_publication': False,
        'whitespace': 'Native CRLF accepted and inherited blank-at-eof excluded; sealed source unchanged.'}
    save(RECEIPT, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {'files', 'unchanged_selected_paths'}}, ensure_ascii=False))
else:
    raise SystemExit('Use --stage or --publish after final review and explicit selection.')
