"""Publish only R77/R78 code and aggregate evidence; verify a private E/D backup."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
R = P.parents[1]
P77 = R / 'reviews/R77_FIXED25000_FULL_DATA_TRAIN_001'
S77 = R / 'work/R77_FIXED25000_FULL_DATA_TRAIN_001'
S = R / 'work' / P.name
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'delivery_runtime')
final = R / 'R78_DELIVERY_RECEIPT.json'

def read(path):
    return json.loads(path.read_text('utf8'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def rel(path):
    return path.relative_to(R).as_posix()

def call(args, timeout=90, binary=False):
    if args[0] == 'git':
        args = ['git', '-c', 'core.longpaths=true', *args[1:]]
    result = subprocess.run(args, cwd=R, capture_output=True, text=not binary,
                            encoding=None if binary else 'utf8',
                            errors=None if binary else 'replace', timeout=timeout)
    assert result.returncode == 0, dict(command=args, native_exit=result.returncode,
                                        stderr=result.stderr[-2500:])
    return result.stdout if binary else result.stdout.strip()

def owned(base):
    excluded = {'cache', 'runtime', 'runtime_cache', '__pycache__'}
    for path in base.rglob('*'):
        if any(part in excluded or part.endswith('_runtime') for part in path.relative_to(base).parts):
            continue
        assert not path.is_symlink(), path
        if path.is_file() and path.suffix not in {'.tar', '.tmp', '.pyc'}:
            yield path

def git_selection():
    selected = set(['AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
                    'jobs.json', 'experiments.json', 'R77_REPORT.md', 'R78_REPORT.md',
                    'R78_RESULT_SUMMARY.json', 'R76_DELIVERY_RECEIPT.json',
                    'tasks/R77_FIXED25000_FULL_DATA_TRAIN_001.md',
                    'tasks/R78_FULL_RESPONSE_EVALUATION_001.md'])
    for base, excluded in [
        (S77, {'LOSS_STATISTICS.json', 'TRAIN_TRACE.json'}),
        (S, {'metrics.json', 'METRICS_COMPACT.json', 'CHARGE_DISTRIBUTION_DIAGNOSTICS.json', 'C2ST_RUNNING.json'}),
        (P77, {'root_review.py'}),
        (P, {'config.disabled.json', 'config_post.disabled.json', 'DELIVERY_SELECTION.json', 'DELIVERY_STAGED_CHECK.json'})]:
        for path in base.iterdir():
            if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.json'} and path.name not in excluded:
                assert '_PRIVATE' not in path.name
                selected.add(rel(path))
    selected.add(rel(S77 / 'FULL_TRAIN_LOSSES.png'))
    for path in (S / 'figures_repaired_003').iterdir():
        if path.suffix in {'.png', '.svg'}:
            selected.add(rel(path))
    for path in (S / 'results').glob('WIDTH_*.json'):
        selected.add(rel(path))
    for name in ['plot_repair_001', 'plot_repair_002', 'plot_repair_003',
                 'summary_receipt_only', 'evaluator_candidate', 'runtime_candidate',
                 'static_adapter_review', 'static_summary_review']:
        for path in (P / name).iterdir():
            if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.json'} and '_PRIVATE' not in path.name:
                selected.add(rel(path))
    for name in selected:
        assert (R / name).is_file(), name
        assert Path(name).suffix in {'.py', '.sh', '.md', '.json', '.png', '.svg'}, name
        assert '_PRIVATE' not in Path(name).name, name
        assert not name.endswith('/metrics.json'), name
    return selected

assert not final.exists(), 'Use existing delivery receipt; do not repeat publication.'
assert read(R / 'jobs.json')['active_scientific_jobs'] == 0
assert read(P / 'ROOT_RESULT_REVIEW.json')['primary_A_C_pass'] is False
assert read(P / 'ROOT_VISUAL_QA.json')['status'] == 'PASS_RENDERED_VISUAL_REVIEW'
mode = sys.argv[1]

if mode == 'prepare':
    pending_index = set(call(['git', 'diff', '--cached', '--name-only']).splitlines())
    if pending_index:
        prior_selection = read(P / 'DELIVERY_SELECTION.json')
        assert call(['git', 'rev-parse', 'HEAD']) == prior_selection['base_commit']
        assert pending_index.issubset(set(prior_selection['paths']) | {
            rel(P / 'DELIVERY_SELECTION.json'), rel(P / 'DELIVERY_STAGED_CHECK.json')})
    repository = json.loads(call(['gh', 'repo', 'view', 'Aggressivecyber/ej276d-sim2real', '--json', 'visibility,url']))
    assert repository['visibility'] == 'PRIVATE'
    base = call(['git', 'rev-parse', 'HEAD'])
    for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
        data = read(R / name)
        data.update(current_activity='R78_evaluation_complete_no_active_scientific_job',
                    next_action='Fixed R77 failed and is sealed; application goal remains paused',
                    delivery_status='See_R78_DELIVERY_RECEIPT_json_for_actual_Git_D_confirmation')
        write(R / name, data)
    for name in ['AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json', 'jobs.json', 'experiments.json']:
        old = call(['git', 'show', 'HEAD:' + name], binary=True)
        current = (R / name).read_bytes()
        if old.count(b'\r\n') > old.count(b'\n') // 2:
            (R / name).write_bytes(current.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    selected = git_selection()
    selection = dict(UTC=datetime.now(timezone.utc).isoformat(), base_commit=base,
                     repository=repository, paths=sorted(selected),
                     files_SHA256={name: sha(R / name) for name in sorted(selected)},
                     scope='R77 training endpoint and negative R78 response evaluation; code, configuration, aggregate evidence and actual figures only',
                     event_arrays_weights_logs_and_full_CDF_json_private_E_D=True,
                     unrelated_existing_changes_included=False)
    write(P / 'DELIVERY_SELECTION.json', selection)
    selected.add(rel(P / 'DELIVERY_SELECTION.json'))
    paths = sorted(selected)
    for start in range(0, len(paths), 25):
        call(['git', 'add', '-f', '--', *paths[start:start + 25]])
    staged = call(['git', 'diff', '--cached', '--name-only']).splitlines()
    assert set(staged).issubset(selected)
    result = subprocess.run(['git', '-c', 'core.longpaths=true', '-c',
                             'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                             'diff', '--cached', '--check'], cwd=R, capture_output=True, text=True,
                            encoding='utf8', errors='replace')
    findings = result.stdout.splitlines()
    allowed = {
        'work/R77_FIXED25000_FULL_DATA_TRAIN_001/check_combined_loss.py:62: new blank line at EOF.',
        'work/R77_FIXED25000_FULL_DATA_TRAIN_001/combined_loss.py:49: new blank line at EOF.',
    }
    # Matplotlib emits trailing spaces inside SVG path coordinates. Keep the
    # already-rendered files byte-identical; source files retain the strict check.
    generated_svg_findings = []
    source_findings = []
    for line in findings:
        if line.startswith('+'):
            continue
        if line.startswith(rel(S / 'figures_repaired_003') + '/') and '.svg:' in line and line.endswith(': trailing whitespace.'):
            generated_svg_findings.append(line)
        else:
            source_findings.append(line)
    assert result.returncode in {0, 2} and set(source_findings).issubset(allowed), (result.returncode, source_findings[:20])
    write(P / 'DELIVERY_STAGED_CHECK.json', dict(
        UTC=datetime.now(timezone.utc).isoformat(), staged_paths=staged,
        staged_files=len(staged), private_repository_verified=True,
        whitespace_native_exit=result.returncode, accepted_inherited_findings=source_findings,
        generated_Matplotlib_SVG_path_whitespace_count=len(generated_svg_findings),
        reason='Already bound sources and rendered figures remain byte-identical; only inherited final empty lines and generated SVG path whitespace allowed',
        source_arrays_weights_and_raw_CDF_json_in_index=False))
    call(['git', 'add', '-f', '--', rel(P / 'DELIVERY_STAGED_CHECK.json')])
    print(json.dumps(dict(staged_files=len(staged) + 1, base_commit=base,
                          D_free_bytes=shutil.disk_usage('D:/').free, private_repository=repository)), flush=True)

elif mode == 'finalize':
    selection = read(P / 'DELIVERY_SELECTION.json')
    base = selection['base_commit']
    assert call(['git', 'rev-parse', 'HEAD']) == base
    for name, digest in selection['files_SHA256'].items():
        assert sha(R / name) == digest, name
    selected = set(selection['paths']) | {rel(P / 'DELIVERY_SELECTION.json'), rel(P / 'DELIVERY_STAGED_CHECK.json')}
    assert set(call(['git', 'diff', '--cached', '--name-only']).splitlines()).issubset(selected)
    payload = selected | {rel(path) for folder in [P77, S77, P, S] for path in owned(folder)}
    payload |= {'CURRENT_ACCEPTANCE.json', 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md',
                'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
                'reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py'}
    frozen = read(P / 'LOCAL_FROZEN_INPUT_DELIVERY_BINDING.json')['files']
    for entry in frozen:
        assert sha(R / entry['local_relative_path']) == entry['sha256']
        payload.add(entry['local_relative_path'])
    previous = Path(read(R / 'R71_DELIVERY_RECEIPT.json')['backup_root'])
    previous_manifest = {entry['path']: entry for entry in read(previous / 'BACKUP_SHA256_MANIFEST.json')}
    reused = []
    for name in ['work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz',
                 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/UNIT_CHARGE_CONSTANTS.npz',
                 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/NOISE_CALIBRATION.npz',
                 'work/R23_ONLINE_STANDARDIZED_CRITIC_001/VALLEY_DEFINITION.json',
                 'work/R71_REAL_CHARGE_FULL_TRAIN_001/FINAL_HEAD.pt',
                 'work/R71_REAL_CHARGE_FULL_TRAIN_001/HEAD_STATISTICS.npz']:
        row = previous_manifest[name]
        assert sha(previous / name) == row['sha256']
        reused.append(dict(original_relative_path=name, path=str(previous / name), sha256=row['sha256']))
    size = sum((R / name).stat().st_size for name in payload)
    assert shutil.disk_usage('D:/').free > size + 80 * 1024 ** 2
    call(['git', 'commit', '-m', 'Complete fixed full-data R77 training and close failed R78 response evaluation'])
    commit = call(['git', 'rev-parse', 'HEAD'])
    call(['git', 'push', 'origin', 'main'])
    remote = call(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
    assert remote == commit
    publication = dict(UTC=datetime.now(timezone.utc).isoformat(), base_commit=base, local_commit=commit,
                       remote_main_SHA=remote, push_native_exit=0, private_repository=selection['repository'])
    write(P / 'PUBLICATION_RECEIPT.json', publication)
    payload.add(rel(P / 'PUBLICATION_RECEIPT.json'))
    backup = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305') / (
        'r77_r78_full_response_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    backup.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name in sorted(payload):
        src, dst = R / name, backup / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(dst)
        assert digest == sha(src), name
        manifest.append(dict(path=name, bytes=dst.stat().st_size, sha256=digest))
    bundle = backup / 'r77_r78_incremental.bundle'
    call(['git', 'bundle', 'create', str(bundle), base + '..HEAD'])
    bundle_verification = call(['git', 'bundle', 'verify', str(bundle)])
    write(backup / 'BACKUP_SHA256_MANIFEST.json', manifest)
    write(backup / 'REUSED_PRIOR_INPUTS.json', reused)
    write(backup / 'REMOTE_LOCAL_FROZEN_INPUT_MAP.json', frozen)
    (backup / 'RESTORE_README.txt').write_text(
        'R77 fixed 25000 full-data training completed; R78 A/B/C failed. Restore listed relative files on E/D. '
        'Final R77 weights, all R78 event outputs, receipts, failed attempts and plots are included. '
        'Frozen evaluation inputs are included; remote-to-local mapping is in REMOTE_LOCAL_FROZEN_INPUT_MAP.json. '
        'Large immutable TRAIN inputs/constants are reused from the verified R71 D backup in REUSED_PRIOR_INPUTS.json. '
        'No science was rerun for this delivery. Final plots are figures_repaired_003. '
        'The original summary exit1 remains preserved; the local renderer completed exit0. '
        'Application goal is paused, not complete. Git bundle requires base ' + base + '.\n', 'utf8')
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), stage=P.name,
                   purpose='R77_complete_training_and_R78_negative_response_evaluation_delivery',
                   **{key: publication[key] for key in ['base_commit', 'local_commit', 'remote_main_SHA', 'private_repository']},
                   push_exit=0, backup_root=str(backup), copied_files=len(manifest),
                   copied_bytes=sum(row['bytes'] for row in manifest), all_copied_file_SHA256_verified=True,
                   immutable_TRAIN_inputs_reused_with_verified_hashes=reused,
                   frozen_evaluation_inputs_copied=len(frozen), bundle_SHA256=sha(bundle),
                   bundle_native_verification_exit=0, backup_manifest_SHA256=sha(backup / 'BACKUP_SHA256_MANIFEST.json'),
                   complete_main_updates=25000, primary_A_C_pass=False, original_numeric_native_exits=[0, 0, 0],
                   original_summary_native_exit=1, repaired_plot_native_exit=0, visually_reviewed_actual_figures=8,
                   response_updates_in_evaluation=0, additional_scientific_recalculation=False,
                   independent_main_replay_rows=4000, private_event_arrays_or_weights_in_Git=False,
                   application_goal_status='paused', model_goal_complete=False, next_scientific_job_started=False,
                   D_free_after_bytes=shutil.disk_usage('D:/').free, all_local_new_artifacts_on_E_D=True,
                   resource=dict(guard, **memory()))
    write(final, receipt)
    shutil.copy2(final, backup / final.name)
    assert sha(final) == sha(backup / final.name)
    assert not call(['git', 'diff', '--cached', '--name-only'])
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
else:
    raise ValueError(mode)
