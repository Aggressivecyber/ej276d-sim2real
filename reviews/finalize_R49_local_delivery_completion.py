"""Publish actual R49 delivery metadata and copy a small terminal D overlay."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
PARENT = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305').resolve()
NOW = datetime.now(timezone.utc).isoformat()
TERMINAL = 'R49_DELIVERY_COMPLETION_RECEIPT.json'


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


assert not (ROOT / TERMINAL).exists()
pub, backup = read('R49_PUBLICATION_RECEIPT.json'), read('R49_BACKUP_RECEIPT.json')
assert pub['status'] == 'verified_published' and pub['match']
assert backup['status'] == 'copy_completed' and backup['copy_operations_completed_without_exception']
science_head = pub['local_HEAD']
assert git('rev-parse', 'HEAD') == science_head == backup['git_commit']
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
assert read(f'work/{STAGE}/FINAL_RECEIPT.json')['status'] == 'sealed'
base = Path(backup['backup_root']).resolve()
assert base.parent == PARENT and base.is_dir()
dest = base / 'delivery_completion'
assert dest.resolve().parent == base and not dest.exists()
assert shutil.disk_usage(base).free > 0

program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
assert program['current_stage'] == progress['current_stage'] == STAGE
assert program['active_local_scientific_jobs'] == 0
assert program['active_scientific_jobs'] is None and program['active_remote_scientific_jobs'] is None
assert not program['project_model_goal_complete']
assert not progress['model_goal_complete']
terminal_next_action = (
    'R49 local stage delivered; retain the fixed-recipe negative result and unknown old remote terminal state. '
    'Under the existing no-compute-budget authorization, record a distinct next protocol before new scientific '
    'work, beginning with the unresolved same-treated REAL voltage representation and source-conditional PSD '
    'problems. No A-C-qualified candidate exists for D/E validation.')
# R49 is a stage result; even A-C success still requires separately registered D/E.
program.update(updated_at_utc=NOW, updated_UTC=NOW, delivery_status='R49_scientific_stage_delivered',
    status='R49_delivered_model_goal_unmet',
    current_activity='R49_local_delivered_no_live_local_job_remote_state_unknown',
    next_action=terminal_next_action,
    last_native_observation=TERMINAL,
    executor_start_observation=f'work/{STAGE}/EXECUTOR_ACK.json',
    current_stage_progress='sealed_and_delivered', active_scientific_jobs=None,
    active_local_scientific_jobs=0, active_remote_scientific_jobs=None,
    latest_completed_delivery_index='R49_DELIVERY_INDEX.json',
    latest_completed_publication='R49_PUBLICATION_RECEIPT.json', latest_completed_publication_SHA=science_head,
    latest_completed_backup='R49_BACKUP_RECEIPT.json', latest_backup_verification_pending=False,
    latest_backup_completed_UTC=backup['completed_UTC'], latest_completed_backup_mode=backup['backup_mode'],
    latest_backup_base_dependencies=backup['base_backup_roots_required'],
    delivery_completion_receipt=TERMINAL)
program['current_job'].update(status='sealed_and_delivered_no_live_job')
save(ROOT / 'PROGRAM.json', program)
progress.update(updated_UTC=NOW, latest_delivered_stage=STAGE,
    status=program['status'], project_status='R49_stage_delivered_model_goal_unmet',
    current_activity=program['current_activity'], next_action=program['next_action'],
    last_native_observation=TERMINAL,
    executor_start_observation=f'work/{STAGE}/EXECUTOR_ACK.json',
    delivery_status=program['delivery_status'], current_stage_progress='sealed_and_delivered',
    latest_stage_publication='R49_PUBLICATION_RECEIPT.json', latest_stage_backup='R49_BACKUP_RECEIPT.json',
    latest_backup_mode=backup['backup_mode'], latest_backup_base_dependencies=backup['base_backup_roots_required'],
    delivery_completion_receipt=TERMINAL, active_scientific_jobs=None,
    active_local_scientific_jobs=0, active_remote_scientific_jobs=None)
for row in progress['requirements']:
    if row['id'] == 'final_research_delivery':
        row.update(status='R1_R49_stage_evidence_delivered_model_goal_unmet',
                   latest_evidence=['R49_REPORT.md', 'R49_PUBLICATION_RECEIPT.json', 'R49_BACKUP_RECEIPT.json'])
save(ROOT / 'PROJECT_PROGRESS.json', progress)
for filename, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(filename)
    next(row for row in data[key] if row.get(identity) == STAGE).update(
        status='sealed_and_delivered_no_live_job',
        delivery_status='sealed_and_delivered_no_live_stage_job',
        publication_receipt='R49_PUBLICATION_RECEIPT.json', backup_receipt='R49_BACKUP_RECEIPT.json',
        active_scientific_jobs=0)
    save(ROOT / filename, data)

delivery_text = ('\n\nR49 科学产物已完成私有 Git 推送和 D 盘正常增量复制。'
                 '最终状态与元数据副本见 [R49 交付回执](R49_DELIVERY_COMPLETION_RECEIPT.json)。'
                 '本机科学作业已结束，旧远端重放终态仍未知。模型目标尚未完成；本轮科学判定见R49_REPORT，'
                 '原A–F及不设计算预算的政策继续有效。\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    path.write_text('# R49 阶段交付已完成' + delivery_text + '\n' + path.read_text('utf8'), 'utf8')
report = ROOT / 'R49_REPORT.md'
report.write_text(report.read_text('utf8') + '\n## 实际阶段交付\n\n'
    f'科学产物私有 Git 提交 `{science_head}` 已推送且远端一致。D 盘新增正常复制 '
    f'{backup["copied_file_count"]} 文件、{backup["copied_source_bytes"]:,} B，增量 Git bundle '
    f'{backup["bundle_bytes"]:,} B；目录 `{base.as_posix()}`。'
    '恢复需要 BACKUP_FILE_LIST.json 指定的全部历史副本及 Git bundle 链；该目录单独不是完整快照。'
    '最终状态文件另存于 delivery_completion 小型增量目录。既有副本未覆盖，没有执行迁移内容一致性测试或恢复测试。'
    '\n\n最终回执：[R49_DELIVERY_COMPLETION_RECEIPT.json](R49_DELIVERY_COMPLETION_RECEIPT.json)。\n', 'utf8')
index = read('R49_DELIVERY_INDEX.json')
index.update(status='scientific_artifacts_published_and_backed_up_final_receipt_authoritative',
             delivery_completion_receipt=TERMINAL)
index['private_Git'].update(status='verified_published', scientific_artifact_commit=science_head)
index['backup'].update(status='copy_completed', mode=backup['backup_mode'],
    base_dependencies=backup['base_backup_roots_required'], standalone_full_snapshot=False,
    metadata_overlay_directory=dest.as_posix(), metadata_overlay_receipt=TERMINAL)
for row in index['bindings']:
    if row['path'] == 'R49_REPORT.md':
        raw = report.read_bytes()
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
save(ROOT / 'R49_DELIVERY_INDEX.json', index)

names = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
         'jobs.json', 'experiments.json', 'R49_REPORT.md', 'R49_DELIVERY_INDEX.json',
         'R49_PUBLICATION_RECEIPT.json', 'R49_BACKUP_RECEIPT.json']
git('add', '-f', '--', *names)
staged = git('diff', '--cached', '--name-only').splitlines()
assert set(staged) == set(names)
git('-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
print(json.dumps({'terminal_metadata_staging': staged, 'scientific_artifact_commit': science_head}), flush=True)
git('commit', '-m', 'Finalize R49 stage delivery receipts and original-goal status')
head = git('rev-parse', 'HEAD')
git('push', 'origin', 'main')
remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
assert head == remote and not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')

required_copy_bytes = sum((ROOT / name).stat().st_size for name in names + ['reviews/finalize_R49_local_delivery_completion.py'])
assert shutil.disk_usage(base).free > required_copy_bytes
dest.mkdir()
copied = []
for name in names + ['reviews/finalize_R49_local_delivery_completion.py']:
    target = dest / 'project' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    copied.append({'path': 'project/' + name, 'source_bytes': (ROOT / name).stat().st_size})
bundle = dest / 'delivery-metadata.bundle'
git('bundle', 'create', str(bundle), 'main', '^' + science_head)
(dest / 'RESTORE_DELIVERY_OVERLAY.txt').write_text(
    'Restore R49 using the parent BACKUP_FILE_LIST.json and every named storage root.\n'
    'Apply delivery_completion/project into the same NEW E/D destination last.\n'
    'For Git, apply the parent required bundle chain, parent r49-incremental.bundle, then this delivery-metadata.bundle.\n'
    'Earlier backup files were not overwritten. Normal-copy completion only; no content-equality, migration or restore test.\n', 'utf8')
save(dest / 'COPIED_FILES.json', {'files': copied, 'normal_copy_completed': True, 'content_identity_test': False})
receipt = {'status': 'R49_delivery_complete_model_goal_unmet',
    'completed_UTC': datetime.now(timezone.utc).isoformat(), 'actual_control_PID': os.getpid(),
    'stage': STAGE, 'scientific_artifact_commit': science_head, 'terminal_metadata_commit': head,
    'GitHub_main': remote, 'private_repository': pub['repository'], 'remote_SHA_match': True,
    'tracked_worktree_clean': True, 'historical_untracked_files_left_in_place': True,
    'scientific_publication_receipt': 'R49_PUBLICATION_RECEIPT.json',
    'scientific_backup_receipt': 'R49_BACKUP_RECEIPT.json', 'scientific_backup_root': str(base),
    'required_base_roots': backup['base_backup_roots_required'],
    'required_existing_Git_bundle_chain_in_order': backup['required_existing_Git_bundle_chain_in_order'],
    'standalone_full_snapshot': False, 'metadata_overlay_root': str(dest),
    'metadata_copied_files': len(copied), 'metadata_copied_source_bytes': sum(row['source_bytes'] for row in copied),
    'incremental_bundle_bytes': bundle.stat().st_size, 'incremental_bundle_required_base_commit': science_head,
    'copy_operations_completed_without_exception': True, 'bundle_create_native_exit': 0,
    'prior_backup_files_overwritten': False, 'migration_identity_tests_performed': False,
    'restore_execution_performed': False, 'D_free_after_bytes': shutil.disk_usage(base).free,
    'project_model_goal_complete': False, 'active_scientific_jobs': None,
    'active_local_scientific_jobs': 0, 'active_remote_scientific_jobs': None,
    'remote_previous_replay_terminal': 'unknown', 'global_single_physical_job_confirmed': False,
    'remote_generated_arrays_returned': False,
    'scientific_decision': program['latest_scientific_decision'], 'next_action': program['next_action'],
    'original_acceptance_A_F_unchanged': True, 'sealed_scientific_stage_modified': False}
assert receipt['D_free_after_bytes'] > 0
save(ROOT / TERMINAL, receipt)
save(dest / TERMINAL, receipt)
print(json.dumps(receipt, ensure_ascii=False), flush=True)
