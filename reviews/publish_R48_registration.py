"""Publish an explicit administrative registration whitelist, never stage arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
stage = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
selection_name = 'reviews/R48_REGISTRATION_PUBLICATION_SELECTION.json'

def read(name):
    return json.loads((root / name).read_text('utf-8-sig'))

def save(name, obj):
    (root / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', 'utf8')

def run(*args):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(json.dumps({'command': args, 'exit': result.returncode,
                                      'stdout': result.stdout, 'stderr': result.stderr}))
    return result.stdout.strip()

def binding(path):
    raw = (root / path).read_bytes()
    return {'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

if sys.argv[1:] == ['--prepare']:
    obs = {'stage': stage, 'actual_tool_chunk': '48aea5', 'actual_native_exit': 0,
           'actual_PowerShell_PID': 53540, 'actual_UTC': '2026-10-10T07:34:59.6356323Z',
           'phase': 'implementation_control_reading_registered_protocol',
           'scientific_started': False, 'reported_by': 'existing sim2real_sol_max_executor'}
    save('reviews/R48_EXECUTOR_START_OBSERVATION.json', obs)
    for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
        data = read(name)
        assert data['current_stage'] == stage
        data.update(status='R48_implementing_model_goal_unmet', current_stage_progress='implementing_no_science_yet',
                    current_activity='R48_executor_implementation_before_scientific_start',
                    executor_start_observation='reviews/R48_EXECUTOR_START_OBSERVATION.json')
        save(name, data)
    names = ['AGENTS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json', 'PROJECT_OBJECTIVE.md',
             'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json', 'RESOURCE_BUDGET.json', 'jobs.json',
             'experiments.json', 'tasks/' + stage + '.md', 'reviews/register_R48.py',
             'reviews/publish_R48_registration.py', 'reviews/R48_DESIGN_CONSULT_001.md',
             'reviews/R48_REGISTRATION_NATIVE_RECEIPT.json', 'reviews/R48_EXECUTOR_START_OBSERVATION.json',
             'reviews/R48_REGISTRATION_PUBLICATION_RETRY.json']
    for folder in ['reviews/R48_REGISTRATION_001', 'reviews/PROJECT_GOAL_ACTIVATION_20261010_001']:
        names += [p.relative_to(root).as_posix() for p in sorted((root / folder).iterdir()) if p.is_file()]
    assert len(names) == len(set(names))
    partial = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(partial) <= set(names + [selection_name]), partial
    bindings = [binding(p) for p in names]
    assert all(Path(p['path']).suffix in ['.md', '.json', '.py'] for p in bindings)
    selection = {'stage': stage, 'purpose': 'Protocol and administrative status before first scientific run',
                 'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': bindings,
                 'data_arrays_weights_logs_included': False,
                 'selection_itself_also_staged': selection_name}
    save(selection_name, selection)
    print(json.dumps({'file_count': len(names) + 1, 'selected_bytes': sum(p['bytes'] for p in bindings), 'paths': names}, ensure_ascii=False))
elif sys.argv[1:] == ['--publish']:
    selection = read(selection_name)
    names = [r['path'] for r in selection['files']] + [selection_name]
    for row in selection['files']:
        assert row == binding(row['path']), row['path']
    visibility = json.loads(run('gh', 'repo', 'view', '--json', 'nameWithOwner,isPrivate,visibility'))
    assert visibility['isPrivate'] and visibility['nameWithOwner'] == 'Aggressivecyber/ej276d-sim2real'
    partial = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(partial) <= set(names), partial
    before = run('git', 'rev-parse', 'HEAD')
    branch = run('git', 'branch', '--show-current')
    assert branch == 'main'
    run('git', 'add', '-f', '--', *names)
    staged = run('git', 'diff', '--cached', '--name-only').splitlines()
    assert set(staged) == set(names), {'missing': sorted(set(names) - set(staged)), 'extra': sorted(set(staged) - set(names))}
    run('git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
    run('git', 'commit', '-m', 'Register R48 guided noise-completion paired protocol')
    sha = run('git', 'rev-parse', 'HEAD')
    run('git', 'push', 'origin', 'main')
    remote = run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert remote == sha
    receipt = {'status': 'registration_published', 'stage': stage,
               'actual_UTC': datetime.now(timezone.utc).isoformat(), 'actual_control_PID': os.getpid(),
               'before_SHA': before, 'commit_SHA': sha, 'remote_SHA': remote, 'branch': branch,
               'repository': visibility, 'all_git_commands_native_exit': 0,
               'exact_staged_paths': staged, 'scientific_completion_claimed': False,
               'D_backup_in_this_operation': False, 'data_arrays_weights_logs_included': False}
    save('reviews/R48_REGISTRATION_PUBLICATION_RECEIPT.json', receipt)
    print(json.dumps({'commit': sha, 'remote': remote, 'files': len(staged), 'PID': os.getpid()}))
else:
    raise SystemExit('use --prepare or --publish')
