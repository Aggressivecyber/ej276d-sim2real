"""Publish R47 delivery status and copy its terminal metadata into a D overlay."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R47_POPULATION_NOISE_DIAGNOSTIC_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


assert not (ROOT / 'R47_DELIVERY_COMPLETION_RECEIPT.json').exists()
publication, backup = read('R47_PUBLICATION_RECEIPT.json'), read('R47_BACKUP_RECEIPT.json')
assert publication['status'] == 'verified_published' and publication['match']
assert backup['status'] == 'copy_completed' and backup['copy_operations_completed_without_exception']
science_head = publication['local_HEAD']
assert git('rev-parse', 'HEAD') == science_head == backup['git_commit']
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
base = Path(backup['backup_root']).resolve()
parent = Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305').resolve()
assert base.parent == parent and base.is_dir()
dest = base / 'delivery_completion'
assert dest.resolve().parent == base and not dest.exists()
assert shutil.disk_usage(base).free > 268435456 + 4 * 1024**2

program = read('PROGRAM.json')
program.update(updated_at_utc=NOW, status='R47_diagnostic_delivered_next_method_design_model_goal_unmet',
    current_stage_progress='sealed_and_delivered_no_response_candidate',
    current_activity='R47_delivered_next_method_design', active_scientific_jobs=0,
    latest_completed_delivery_index='R47_DELIVERY_INDEX.json',
    latest_completed_publication='R47_PUBLICATION_RECEIPT.json', latest_completed_publication_SHA=science_head,
    latest_completed_backup='R47_BACKUP_RECEIPT.json', latest_backup_verification_pending=False,
    latest_backup_completed_UTC=backup['completed_UTC'], latest_completed_backup_mode=backup['backup_mode'],
    latest_backup_base_dependencies=backup['base_backup_roots_required'],
    delivery_completion_receipt='R47_DELIVERY_COMPLETION_RECEIPT.json',
    next_action='Define a fixed input-noise-aware complete-source response operator and matched control before registering the final method plan. No response training is running.',
    project_model_goal_complete=False)
program['current_job'].update(status='sealed_and_delivered_no_live_job')
save(ROOT / 'PROGRAM.json', program)
progress = read('PROJECT_PROGRESS.json')
progress.update(updated_UTC=NOW, status=program['status'], latest_delivered_stage=STAGE,
    current_activity=program['current_activity'], current_stage_progress=program['current_stage_progress'],
    latest_stage_publication='R47_PUBLICATION_RECEIPT.json', latest_stage_backup='R47_BACKUP_RECEIPT.json',
    latest_backup_mode=backup['backup_mode'], latest_backup_base_dependencies=backup['base_backup_roots_required'],
    delivery_completion_receipt='R47_DELIVERY_COMPLETION_RECEIPT.json', active_scientific_jobs=0,
    model_goal_complete=False)
for row in progress['requirements']:
    if row['id'] == 'final_research_delivery':
        row.update(status='R1_R47_stage_evidence_delivered_model_goal_unmet',
            latest_evidence=['R47_REPORT.md', 'R47_PUBLICATION_RECEIPT.json', 'R47_BACKUP_RECEIPT.json'])
save(ROOT / 'PROJECT_PROGRESS.json', progress)
for filename, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(filename)
    next(row for row in data[key] if row.get(identity) == STAGE).update(
        status='sealed_and_delivered_no_live_stage_job', publication_receipt='R47_PUBLICATION_RECEIPT.json',
        backup_receipt='R47_BACKUP_RECEIPT.json', active_scientific_jobs=0)
    save(ROOT / filename, data)
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    content = path.read_text('utf8')
    content = content.replace('# R47 已封存：完成新计划首轮诊断', '# R47 已交付：完成新计划首轮诊断', 1)
    content = content.replace('根正在完成私有 Git 与 D 盘增量交付，实际状态见 ',
                              '私有 Git 科学产物与 D 盘增量复制已完成，最终元数据回执见 ', 1)
    path.write_text(content, 'utf8')
report = ROOT / 'R47_REPORT.md'
report.write_text(report.read_text('utf8') +
    f'\n## 实际阶段交付\n\nR47 科学作业累计 {read("work/" + STAGE + "/RESOURCE_RECEIPT.json")["scientific_elapsed_s"]:.6f} 秒；封存 112 文件、6,553,416 B，3 个科学进程均以原生退出码 0 完成。改进块累计科学 3198.497257 秒，方法名额仍使用 2/3，响应训练仍使用 1/6。\n\n'
    f'私有 Git 科学产物提交 `{science_head}` 已推送且远端一致。D 盘新增正常复制 {backup["copied_file_count"]} 文件、{backup["copied_source_bytes"]:,} B，Git 增量 bundle {backup["bundle_bytes"]:,} B；目录 `{base.as_posix()}`。恢复需要 BACKUP_FILE_LIST.json 列出的 R44/R45/R46 副本和 Git 依赖，单独此目录不是完整快照；既有副本未覆盖。最终状态文件在 delivery_completion 中作为小型增量保存，结果以 [最终交付回执](R47_DELIVERY_COMPLETION_RECEIPT.json) 为准。没有执行迁移内容一致性测试。\n\n'
    '本轮完成计划构建和首阶段实证推进；下一方法仍需固定算子与对照，尚未登记新的响应训练。模型联合目标未完成。\n', 'utf8')
index = read('R47_DELIVERY_INDEX.json')
index.update(status='scientific_artifacts_published_and_backed_up_final_receipt_authoritative',
             delivery_completion_receipt='R47_DELIVERY_COMPLETION_RECEIPT.json')
index['private_Git'].update(status='verified_published', scientific_artifact_commit=science_head)
index['backup'].update(status='copy_completed', mode=backup['backup_mode'],
    base_dependencies=backup['base_backup_roots_required'], standalone_full_snapshot=False,
    metadata_overlay_directory=dest.as_posix(), metadata_overlay_receipt='R47_DELIVERY_COMPLETION_RECEIPT.json')
for row in index['bindings']:
    if row['path'] == 'R47_REPORT.md':
        data = report.read_bytes()
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
save(ROOT / 'R47_DELIVERY_INDEX.json', index)
names = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
         'jobs.json', 'experiments.json', 'R47_REPORT.md', 'R47_DELIVERY_INDEX.json',
         'R47_PUBLICATION_RECEIPT.json', 'R47_BACKUP_RECEIPT.json']
git('add', '-f', '--', *names)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(names)
git('-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check')
print(json.dumps({'terminal_metadata_staging': names, 'scientific_artifact_commit': science_head}), flush=True)
git('commit', '-m', 'Finalize R47 diagnostic delivery and next-method design status')
head = git('rev-parse', 'HEAD')
git('push', 'origin', 'main')
remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
assert head == remote and not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
dest.mkdir()
copied = []
for name in names + ['reviews/finalize_R47_delivery_completion.py']:
    target = dest / 'project' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    copied.append({'path': 'project/' + name, 'source_bytes': (ROOT / name).stat().st_size})
bundle = dest / 'delivery-metadata.bundle'
git('bundle', 'create', str(bundle), 'main', '^' + science_head)
(dest / 'RESTORE_DELIVERY_OVERLAY.txt').write_text(
    'Restore R47 using the parent BACKUP_FILE_LIST.json and every named R44/R45/R46 storage root.\n'
    'Apply delivery_completion/project into the same NEW E/D destination last.\n'
    'Git requires the R46 full bundle, R46 metadata bundle, parent r47-incremental.bundle, then this delivery-metadata.bundle.\n'
    'Earlier backup files were not overwritten. Normal-copy completion only, with no content-equality or migration test.\n', 'utf8')
save(dest / 'COPIED_FILES.json', {'files': copied, 'normal_copy_completed': True, 'content_identity_test': False})
receipt = {'status': 'R47_delivery_complete_model_goal_unmet', 'completed_UTC': datetime.now(timezone.utc).isoformat(),
    'stage': STAGE, 'scientific_artifact_commit': science_head, 'terminal_metadata_commit': head,
    'GitHub_main': remote, 'private_repository': publication['repository'], 'remote_SHA_match': True,
    'tracked_worktree_clean': True, 'historical_untracked_files_left_in_place': True,
    'scientific_publication_receipt': 'R47_PUBLICATION_RECEIPT.json',
    'scientific_backup_receipt': 'R47_BACKUP_RECEIPT.json', 'scientific_backup_root': str(base),
    'required_base_roots': backup['base_backup_roots_required'],
    'required_R46_Git_metadata_overlay': backup['base_delivery_overlay_required_for_Git'],
    'standalone_full_snapshot': False, 'metadata_overlay_root': str(dest),
    'metadata_copied_files': len(copied), 'metadata_copied_source_bytes': sum(row['source_bytes'] for row in copied),
    'incremental_bundle_bytes': bundle.stat().st_size, 'incremental_bundle_required_base_commit': science_head,
    'copy_operations_completed_without_exception': True, 'bundle_create_native_exit': 0,
    'prior_backup_files_overwritten': False, 'migration_identity_tests_performed': False,
    'D_free_after_bytes': shutil.disk_usage(base).free, 'project_model_goal_complete': False,
    'active_scientific_jobs': 0, 'new_response_candidate': False, 'next_response_method_registered': False,
    'original_acceptance_A_F_unchanged': True, 'sealed_scientific_stage_modified': False}
save(ROOT / 'R47_DELIVERY_COMPLETION_RECEIPT.json', receipt)
save(dest / 'R47_DELIVERY_COMPLETION_RECEIPT.json', receipt)
print(json.dumps(receipt, ensure_ascii=False), flush=True)
