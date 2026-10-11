"""Explicit private publication and normal D-drive copying after reviewed closure."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import json
import os
import shutil
import subprocess

R = Path(__file__).resolve().parents[1]
STAGE = 'R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
S = R/'work'/STAGE
control_cache = S/'cache/root_delivery'
control_cache.mkdir(parents=True, exist_ok=True)
for key in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME'):
    os.environ[key] = str(control_cache)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
mask, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system))
allowed = [i for i in range(64) if mask.value & (1 << i)]
assert kernel.SetProcessAffinityMask(handle, 1 << allowed[min(3, len(allowed)-1)])


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads((R/p).read_text('utf8'))


def write(p, value):
    (R/p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def call(args):
    p = subprocess.run(args, cwd=R, capture_output=True, text=True, encoding='utf8', errors='replace', timeout=60)
    if p.returncode:
        raise RuntimeError(dict(argv=args, native_exit=p.returncode, stdout=p.stdout[:3000], stderr=p.stderr[:3000]))
    return p.stdout.strip()


def is_cache(path):
    return any(k in ('cache', '__pycache__', '.pytest_cache') or k.endswith('_cache') for k in path.parts)


rel = lambda p: p.relative_to(R).as_posix()
final = read('work/'+STAGE+'/FINAL_RECEIPT.json')
assert final['status'] == 'scientific_endpoint_complete_delivery_pending'
assert read('work/'+STAGE+'/FIGURE_QA.json')['all_figures_opened']
assert read('work/'+STAGE+'/FROZEN_ARTIFACT_REVIEW.json')['pass_review']
assert read('work/'+STAGE+'/FETCH_RECEIPT.json')['remote_completion']['actual_PID_gone']
native = read('work/'+STAGE+'/diagnose_train.py.NATIVE_RECEIPT.json')
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None
meta = ['AGENTS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_OBJECTIVE.md', 'PROJECT_PROGRESS.json',
        'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']
rootfiles = ['R64_REPORT.md', 'tasks/'+STAGE+'.md', 'reviews/R64_ROOT_DIRECTION_001.json', 'README.md']
rootfiles += [rel(p) for p in (R/'reviews').glob('*R64*root*.py')]
text_extensions = {'.py', '.json', '.md', '.png', '.ps1', '.txt'}
stage_payload = [rel(p) for p in S.rglob('*') if p.is_file() and not is_cache(p)]
auxiliary_payload = [rel(p) for directory in ('R64_REGISTRATION_001', 'R64_PREPARATION_001',
                    'R64_DEPLOYMENT_REPAIR_001', 'R64_RESULT_ANALYSIS_001')
                    for p in (R/'reviews'/directory).rglob('*') if p.is_file() and not is_cache(p)]
stage = [p for p in stage_payload if Path(p).suffix in text_extensions and '/monitor/' not in p]
auxiliary = [p for p in auxiliary_payload if Path(p).suffix in text_extensions]
selected = sorted(set(meta+rootfiles+stage+auxiliary))
assert all((R/p).is_file() for p in selected)
assert not any(Path(p).suffix in ('.npz', '.pt', '.log', '.tar') for p in selected)
selectionfile = 'reviews/R64_PUBLICATION_SELECTION_001.json'
assert set(call(['git', 'diff', '--cached', '--name-only']).splitlines()).issubset(set(selected+[selectionfile]))
private = json.loads(call(['gh', 'repo', 'view', 'Aggressivecyber/ej276d-sim2real', '--json', 'visibility,url']))
assert private['visibility'] == 'PRIVATE'
base = call(['git', 'rev-parse', 'HEAD'])
write(selectionfile, dict(UTC=utc(), paths=selected, base_commit=base, private_repository=private,
      explicit_paths_only=True, payload_arrays_weights_logs_excluded=True))
call(['git', 'add', '-f', '--', *selected, selectionfile])
staged = call(['git', 'diff', '--cached', '--name-only']).splitlines()
assert set(staged).issubset(set(selected+[selectionfile]))
whitespace = ['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
              'diff', '--cached', '--check']
w = subprocess.run(whitespace, cwd=R, capture_output=True, text=True, encoding='utf8')
# Sealed historical helper bytes are retained. Add any exact expected finding
# only after inspecting it; never suppress a general whitespace category.
expected = {
    'work/'+STAGE+'/network.py:51: new blank line at EOF.',
    'work/'+STAGE+'/distribution_adapter/r63_original/network.py:51: new blank line at EOF.',
    'reviews/R64_PREPARATION_001/distribution_gradient/r63_original/network.py:51: new blank line at EOF.',
    'reviews/R64_PREPARATION_001/worker_review/REVIEW.json:619: new blank line at EOF.',
    'reviews/R64_PREPARATION_001/worker_review/REVIEW.md:13: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/audit_script_review/REVIEW.json:330: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/audit_script_review/REVIEW.md:16: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/audit_script_review/STATIC_RECEIPT.json:221: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/gradient/ANALYSIS.json:943: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/gradient/BINDING_RECEIPT.json:98: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/gradient/REPORT.md:143: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/full_reference_implementation_review/BINDING_RECEIPT.json:158: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/full_reference_implementation_review/REPORT.md:73: new blank line at EOF.',
    'reviews/R64_RESULT_ANALYSIS_001/full_reference_implementation_review/REVIEW.json:355: new blank line at EOF.',
}
assert w.returncode in (0, 2) and set(w.stdout.strip().splitlines()).issubset(expected) and not w.stderr, (w.returncode, w.stdout, w.stderr)
write('reviews/R64_STAGED_CHECK_001.json', dict(UTC=utc(), staged_paths=staged, staged_count=len(staged),
      whitespace_command=whitespace, whitespace_native_exit=w.returncode,
      whitespace_exact_findings_preserved=w.stdout.splitlines(), preserved_sealed_bytes=True,
      whitespace_exception_reason='Only individually inspected and listed EOF blank lines in sealed helper/review artifacts are allowed; bound bytes remain unchanged.',
      arrays_weights_logs_staged=False))
print(json.dumps(dict(step='staged', path_count=len(staged), D_free_bytes=shutil.disk_usage('D:/').free)), flush=True)
if os.environ.get('R64_DELIVERY_PHASE') != 'publish':
    raise SystemExit(0)

call(['git', 'commit', '-m', 'Record frozen full-reference geometry and paired distribution gradients'])
science = call(['git', 'rev-parse', 'HEAD'])
call(['git', 'push', 'origin', 'main'])
remote = call(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
assert science == remote
write('R64_PUBLICATION_RECEIPT.json', dict(UTC=utc(), repository=private, branch='main', local_commit=science,
      remote_SHA=remote, push_exit=0, base_commit=base, selection=selectionfile))
previous = read('R63_BACKUP_RECEIPT.json')
backup = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/(
    'r64_frozen_reference_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
assert not backup.exists()
backup.mkdir(parents=True)
payload = sorted(set(selected+stage_payload+auxiliary_payload+[
    selectionfile, 'reviews/R64_STAGED_CHECK_001.json', 'R64_PUBLICATION_RECEIPT.json']))
required = sum((R/p).stat().st_size for p in payload)
free = shutil.disk_usage(backup).free
assert free > required+20*1024**2, ('actual_capacity_insufficient', required, free)
copied = []
for i, name in enumerate(payload):
    dest = backup/name
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(R/name, dest)
    copied.append(dict(source_relative_path=name, bytes=(R/name).stat().st_size))
    if i % 32 == 31:
        print(json.dumps(dict(step='normal_D_copy', completed=i+1, total=len(payload))), flush=True)
bundle = backup/'r64_science_incremental.bundle'
call(['git', 'bundle', 'create', str(bundle), base+'..HEAD'])
(backup/'BACKUP_FILE_LIST.json').write_bytes((json.dumps(dict(files=copied,
    previous_R63_backup_receipt=previous, incremental_git_base=base), ensure_ascii=False, indent=2)+'\n').encode())
(backup/'RESTORE_README.txt').write_text(
    'R64 frozen-reference diagnostic incremental backup. Restore prior R63 backup and its dependencies, '
    'then overlay these relative paths. Includes completed direct TRAIN endpoints, frozen parent IDs, '
    'all nine random streams and indices, geometry, full reference endpoint and parameter gradients, '
    '32 pairs of float64 parameter gradients, endpoint/parameter means and bootstrap records, '
    'source code, original failed prelaunch packaging records, native receipts and reviewed figures. '
    'No new model weight exists: the unchanged R63 weight and eight original inputs remain dependencies '
    'identified in INPUT_BINDING.json and config.json and retained through the prior backup chain. '
    'Macro endpoint gradients are reconstructed from saved endpoints and indices; they were not saved. '
    'Git bundle is incremental from '+base+'. Normal copying only; no migration content-consistency '
    'test and no historical deletion.\n', 'utf8')
write('R64_BACKUP_RECEIPT.json', dict(UTC=utc(), status='normal_incremental_copy_complete', backup_root=str(backup),
      file_count=len(copied), copied_bytes=required, git_bundle_bytes=bundle.stat().st_size,
      D_free_before_bytes=free, D_free_after_bytes=shutil.disk_usage(backup).free,
      previous_R63_backup_receipt='R63_BACKUP_RECEIPT.json', migration_content_consistency_test=False,
      old_backups_removed_or_overwritten=False))
for fn in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
    x = read(fn)
    x.update(current_stage_progress='sealed_and_delivered', latest_completed_report='R64_REPORT.md',
      latest_completed_review='reviews/R64_ROOT_DIRECTION_001.json', latest_numeric_receipt='work/'+STAGE+'/RESOURCE_RECEIPT.json',
      latest_completed_backup='R64_BACKUP_RECEIPT.json', current_stage_registration='reviews/R64_REGISTRATION_001/RECEIPT.json',
      current_route_decision='reviews/R64_ROOT_DIRECTION_001.json', reproduction_guide='work/'+STAGE+'/REPRODUCTION.md',
      delivery_index='R64_DELIVERY_COMPLETION_RECEIPT.json', publication_receipt='R64_PUBLICATION_RECEIPT.json',
      backup_receipt='R64_BACKUP_RECEIPT.json', delivery_completion_receipt='R64_DELIVERY_COMPLETION_RECEIPT.json', model_goal_complete=False)
    x['current_job']['status'] = 'sealed_and_delivered_no_live_job'
    x['remote_job_state'] = 'R64_native_complete_old_R49_process_absent_after_restart_exit_unknown'
    if fn == 'PROGRAM.json':
        x.update(status='R64_delivered_long_term_goal_active', final_report='R64_REPORT.md', updated_at_utc=utc(),
          latest_permission_status='Root-led long-term goal active with same-model/effort parallel assistance; R64 delivered; original A-F, resource limits, one scientific job and no compute quotas retained.')
    else:
        x.update(project_status='R64_delivered_long_term_goal_active', current_activity='root_following_R64_evidence_backed_direction', updated_UTC=utc())
        for row in x.get('requirements', []):
            if row['id'] == 'final_research_delivery':
                row.update(status='R64_stage_delivery_complete_full_model_goal_unmet', candidate_stage=STAGE,
                  latest_evidence=['R64_REPORT.md', 'R64_PUBLICATION_RECEIPT.json', 'R64_BACKUP_RECEIPT.json'])
    write(fn, x)
for fn, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    x = read(fn)
    a = next(v for v in x[key] if v[identity] == STAGE)
    a.update(status='sealed_and_delivered', delivery_receipt='R64_DELIVERY_COMPLETION_RECEIPT.json')
    write(fn, x)
for fn in ('AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md'):
    p = R/fn
    p.write_bytes(p.read_text('utf8').replace('私有 Git 与 D 盘交付待完成。',
       '私有 Git 与 D 盘实际交付已完成，下一轮由主助手按联合目标协议继续。', 1).encode())
finalfiles = meta+['R64_PUBLICATION_RECEIPT.json', 'R64_BACKUP_RECEIPT.json', 'reviews/R64_STAGED_CHECK_001.json']
call(['git', 'add', '-f', '--', *finalfiles])
call(whitespace)
call(['git', 'commit', '-m', 'Record R64 diagnostic delivery and evidence-backed next action'])
final_commit = call(['git', 'rev-parse', 'HEAD'])
call(['git', 'push', 'origin', 'main'])
remote = call(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
assert final_commit == remote
overlay = backup/'delivery_completion'
overlay.mkdir()
for name in finalfiles:
    dest = overlay/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R/name, dest)
call(['git', 'bundle', 'create', str(overlay/'delivery_metadata.bundle'), science+'..HEAD'])
receipt = dict(UTC=utc(), stage=STAGE, status='delivered', model_goal_complete=False, application_goal_status='active',
    owner='/root', science_commit=science, final_metadata_commit=final_commit, verified_remote_SHA=remote,
    private_repository=private, backup_root=str(backup), final_metadata_overlay=str(overlay), report='R64_REPORT.md',
    scientific_receipt='work/'+STAGE+'/FINAL_RECEIPT.json', backup_receipt='R64_BACKUP_RECEIPT.json',
    D_free_final_bytes=shutil.disk_usage(backup).free, active_scientific_jobs=0, next_action=read('PROGRAM.json')['next_action'])
write('R64_DELIVERY_COMPLETION_RECEIPT.json', receipt)
shutil.copy2(R/'R64_DELIVERY_COMPLETION_RECEIPT.json', overlay/'R64_DELIVERY_COMPLETION_RECEIPT.json')
assert call(['git', 'diff', '--name-only']) == '' and call(['git', 'diff', '--cached', '--name-only']) == ''
print(json.dumps(receipt, ensure_ascii=False, indent=2))
