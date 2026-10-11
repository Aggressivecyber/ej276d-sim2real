"""Prepare an explicit stage, then publish and verify the static R70 delivery."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'delivery_runtime')
mode = sys.argv[1]
final = R / 'R70_DELIVERY_RECEIPT.json'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p, data):
    old = p.read_bytes() if p.exists() else b''
    newline = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    trailer = old[len(old.rstrip(b'\r\n')):] or newline
    p.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode('utf8').replace(b'\n', newline) + trailer)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return p.relative_to(R).as_posix()


def call(args, timeout=60):
    if args[0] == 'git':
        args = ['git', '-c', 'core.longpaths=true', *args[1:]]
    result = subprocess.run(args, cwd=R, text=True, capture_output=True,
                            encoding='utf8', errors='replace', timeout=timeout)
    assert result.returncode == 0, {'command': args, 'exit': result.returncode,
                                   'stdout': result.stdout[-2000:], 'stderr': result.stderr[-2000:]}
    return result.stdout.strip()


def owned():
    return [f for f in P.rglob('*') if f.is_file() and f.suffix in {'.py', '.md', '.json'}
            and not any(part.endswith('_runtime') or part in {'runtime_cache', '__pycache__'}
                        for part in f.relative_to(P).parts)]


assert not final.exists(), 'Delivery already recorded.'
assert read(R / 'jobs.json')['active_scientific_jobs'] == 0
if mode == 'prepare':
    assert not call(['git', 'diff', '--cached', '--name-only']), 'Unrelated staged files exist.'
    repository = json.loads(call(['gh', 'repo', 'view', 'Aggressivecyber/ej276d-sim2real', '--json', 'visibility,url']))
    assert repository['visibility'] == 'PRIVATE'
    base = call(['git', 'rev-parse', 'HEAD'])
    for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
        data = read(R / name)
        data.update({
            'updated_UTC': datetime.now(timezone.utc).isoformat(),
            'latest_delivered_stage': P.name,
            'latest_stage_publication': final.name, 'latest_stage_backup': final.name,
            'latest_completed_backup': final.name, 'delivery_completion_receipt': final.name,
            'delivery_status': 'see_R70_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery',
            'latest_delivery_status': 'see_R70_DELIVERY_RECEIPT_for_actual_private_Git_D_delivery',
            'execution_plan': 'tasks/' + P.name + '.md', 'current_plan': 'PLAN.md',
            'numerical_protocol_status': 'v5_candidate_specific_reference_not_registered',
            'latest_acceptance_observation_role': 'historical_v4_result_no_v5_model_evaluation',
        })
        write(R / name, data)
    print(call([sys.executable, '-B', str(P / 'verify_saved_protocol.py')]), flush=True)
    selected = {rel(f) for f in owned() if 'before_target_correction' not in f.parts}
    selected.update(['AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md',
                     'LONG_TERM_GOAL_20261010.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
                     'experiments.json', 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md',
                     'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json', 'CURRENT_ACCEPTANCE.json',
                     'R70_REPORT.md', 'tasks/' + P.name + '.md'])
    selected.discard(rel(P / 'DELIVERY_SELECTION.json'))
    selected.discard(rel(P / 'DELIVERY_STAGED_CHECK.json'))
    selection = {'UTC': datetime.now(timezone.utc).isoformat(), 'base_commit': base,
                 'repository': repository, 'paths': sorted(selected),
                 'files_SHA256': {name: sha(R / name) for name in sorted(selected)},
                 'scope': 'Static protocol and audit delivery; no model weights or arrays.'}
    write(P / 'DELIVERY_SELECTION.json', selection)
    selected.add(rel(P / 'DELIVERY_SELECTION.json'))
    ordered = sorted(selected)
    for start in range(0, len(ordered), 30):
        call(['git', 'add', '-f', '--', *ordered[start:start+30]])
    staged = call(['git', 'diff', '--cached', '--name-only']).splitlines()
    assert set(staged).issubset(selected)
    assert not call(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    write(P / 'DELIVERY_STAGED_CHECK.json', {'UTC': datetime.now(timezone.utc).isoformat(),
          'staged_paths': staged, 'whitespace_native_exit': 0, 'private_visibility_verified': True})
    call(['git', 'add', '-f', '--', rel(P / 'DELIVERY_STAGED_CHECK.json')])
    print(json.dumps({'mode': mode, 'base_commit': base, 'staged_paths': staged,
                      'extra_staged_receipt': rel(P / 'DELIVERY_STAGED_CHECK.json'),
                      'D_free_bytes': shutil.disk_usage('D:/').free}, ensure_ascii=False, indent=2), flush=True)
elif mode == 'finalize':
    selection = read(P / 'DELIVERY_SELECTION.json')
    base = selection['base_commit']
    assert call(['git', 'rev-parse', 'HEAD']) == base
    for name, digest in selection['files_SHA256'].items():
        assert sha(R / name) == digest, name
    selected = set(selection['paths']) | {rel(P / 'DELIVERY_SELECTION.json'), rel(P / 'DELIVERY_STAGED_CHECK.json')}
    staged = set(call(['git', 'diff', '--cached', '--name-only']).splitlines())
    assert staged.issubset(selected)
    assert not call(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    payload = selected | {rel(f) for f in owned()} | {'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
                'jobs.json', 'R69_DELIVERY_RECEIPT.json', 'reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
    size = sum((R / name).stat().st_size for name in payload)
    assert shutil.disk_usage('D:/').free > size + 40 * 1024 ** 2
    call(['git', 'commit', '-m', 'Align response target with REAL charge and retain simulation gap support'])
    commit = call(['git', 'rev-parse', 'HEAD'])
    call(['git', 'push', 'origin', 'main'])
    remote = call(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
    assert remote == commit
    backup = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305') / ('r70_real_charge_target_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    backup.mkdir(parents=True)
    manifest = []
    for name in sorted(payload):
        src, dst = R / name, backup / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(dst)
        assert sha(src) == digest, name
        manifest.append({'path': name, 'bytes': dst.stat().st_size, 'sha256': digest})
    bundle = backup / 'r70_incremental.bundle'
    call(['git', 'bundle', 'create', str(bundle), base + '..HEAD'])
    write(backup / 'BACKUP_SHA256_MANIFEST.json', manifest)
    (backup / 'RESTORE_README.txt').write_text('R70 is a prospective REAL-charge target correction and static method audit, not a trained model. Restore these relative paths on E/D over the preceding repository snapshot. Original v4 acceptance and old results are retained. The incremental Git bundle depends on base commit ' + base + '. No weights, new scientific arrays, or new acceptance scores were produced. New scientific execution requires a separately registered fixed protocol; do not resume R65 automatically.\n', encoding='utf8')
    result = {
        'UTC': datetime.now(timezone.utc).isoformat(), 'stage': P.name,
        'purpose': 'prospective_REAL_charge_target_and_static_audit_delivery',
        'private_repository': selection['repository'], 'base_commit': base,
        'local_commit': commit, 'remote_main_SHA': remote, 'push_exit': 0,
        'backup_root': str(backup), 'copied_files': len(manifest), 'copied_bytes': size,
        'all_copied_file_SHA256_verified': True, 'bundle_SHA256': sha(bundle),
        'D_free_after_bytes': shutil.disk_usage('D:/').free,
        'target_acceptance_version': 5, 'historical_v4_acceptance_unchanged': True,
        'old_results_reclassified': False, 'science_array_reads': 0,
        'new_model_fits': 0, 'new_model_inferences': 0, 'new_scientific_jobs': 0,
        'R65_resumed': False, 'model_goal_complete': False,
        'application_goal_status_observed': 'paused', 'application_goal_changed': False,
        'static_verification': 'reviews/' + P.name + '/ROOT_STATIC_VERIFICATION.json',
        'resource': {**guard, **memory()},
    }
    write(final, result)
    shutil.copy2(final, backup / final.name)
    assert sha(final) == sha(backup / final.name)
    assert not call(['git', 'diff', '--cached', '--name-only'])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
else:
    raise ValueError(mode)
