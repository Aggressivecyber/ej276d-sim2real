"""Prepare metadata-only delivery helpers for the authorized local recovery."""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / 'reviews'
LOCAL = 'R49_LOCAL_HEUN128_RECOVERY_001'
REMOTE = 'R49_FROZEN_HEUN128_RESPONSE_001'


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def save(name, text):
    target = REVIEWS / name
    assert not target.exists(), target
    ast.parse(text)
    target.write_text(text, 'utf8')
    return {'path': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
            'AST_parse': True, 'executed': False}


written = []
text = (REVIEWS / 'finalize_R49_scientific_review.py').read_text('utf-8-sig')
text = text.replace(REMOTE, LOCAL)
for old, new in [
    ('ROOT_R49_KEY_RESULT_REVIEW_001', 'ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001'),
    ('R49_ACCEPTANCE_OBSERVATION_001', 'R49_LOCAL_ACCEPTANCE_OBSERVATION_001'),
    ('R49_ROOT_FIGURE_QA_001', 'R49_LOCAL_ROOT_FIGURE_QA_001'),
    ('R49_SEAL_NATIVE_RECEIPT', 'R49_LOCAL_SEAL_NATIVE_RECEIPT'),
    ('R49_ROOT_DIRECTION_001', 'R49_LOCAL_ROOT_DIRECTION_001'),
    ('R49_RESOURCE_RECONCILIATION_001', 'R49_LOCAL_RESOURCE_RECONCILIATION_001'),
    ('6380.315270759122', '8487.652792954119'),
    ('r49_heun128_final_', 'r49_local_heun128_final_'),
    ("final['new_training_updates']", "final['new_response_training_updates']"),
    ("resource['new_scientific_failure_receipts']", "resource['scientific_failure_files']"),
]:
    text = text.replace(old, new)
text = replace_once(text,
    "assert all(job['handshake_before_imports'] and job['limit_s'] is None for job in jobs)",
    "assert all(job['handshake_before_imports'] and job['science_budget_s'] is None\n"
    "           and job['wall_budget_s'] is None and job['deadline_UTC'] is None\n"
    "           and job['compute_budget_enforced'] is False for job in jobs)")
text = replace_once(text,
    "job['actual_worker_RSS_peak_bytes'] <= job['RSS_limit_bytes']",
    "job['job_tree_RSS_peak_bytes'] <= job['resource_admission']['RSS_safe_limit_bytes']")
text = text.replace("'active_scientific_jobs': 0, 'stage_immutable': True,",
    "'active_scientific_jobs': None, 'active_local_scientific_jobs': 0,\n"
    "    'active_remote_scientific_jobs': None, 'remote_replay_terminal': 'unknown',\n"
    "    'global_single_physical_job_confirmed': False, 'stage_immutable': True,")
text = text.replace("current_activity='R49_private_Git_and_D_delivery', active_scientific_jobs=0,",
    "current_activity='R49_local_private_Git_and_D_delivery_remote_state_unknown',\n"
    "        active_scientific_jobs=None, active_local_scientific_jobs=0, active_remote_scientific_jobs=None,\n"
    "        remote_previous_replay_terminal='unknown', global_single_physical_job_confirmed=False,")
text = text.replace("latest_user_steering_receipt='reviews/NO_COMPUTE_BUDGET_20261010_001/RECEIPT.json'",
    "latest_user_steering_receipt='reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json'")
text = text.replace("'R49 completed; user-authorized unbudgeted project policy remains active; original A-F unchanged.'",
    "'Authorized R49 local recovery completed; remote replay terminal unknown; unbudgeted project policy and original A-F retained.'")
text = text.replace("'CPU12、RSS48GiB安全设置及单科学作业约束继续有效。\\n\\n'",
    "'本机CPU4、RSS4GiB及单本机科学作业约束保持；旧远端重放终态未知，全局在运行作业数未知。\\n\\n'")
# The original concatenated f-string contains the CPU sentence in the same literal.
text = text.replace('CPU12、RSS48GiB安全设置及单科学作业约束继续有效。',
    '本机CPU4、RSS4GiB及单本机科学作业约束保持；旧远端重放终态未知，全局在运行作业数未知。')
text = replace_once(text,
    "(ROOT / 'R49_REPORT.md').write_text(header + (ROOT / PREFIX / 'REPORT.md').read_text('utf8'), 'utf8')",
    "header += ('远端原尝试已完成数值、生成与评价，共2107.3375221949973秒；旧重放未取得终端回执，'\n"
    "           '其额外全程耗时未知，未猜测归零。累计数值只包含已经取得原生终态的工作。'\n"
    "           '本机A–C依据本机重新生成及评价的数组；远端指标仅作历史。'\n"
    "           '远端生成数组未回传，本次交付不声称包含它们。'\n"
    "           '详见[远端中断记录](reviews/R49_REMOTE_INTERRUPTION_001.json)。\\n\\n')\n"
    "(ROOT / 'R49_REPORT.md').write_text(header + (ROOT / PREFIX / 'REPORT.md').read_text('utf8'), 'utf8')")
text = replace_once(text, "'model_goal_complete': False, 'primary_A_C_pass': ABC, 'scientific_decision': decision,",
    "'model_goal_complete': False, 'primary_A_C_pass': ABC, 'scientific_decision': decision,\n"
    "    'local_recovery_stage': STAGE, 'remote_attempt_stage': '" + REMOTE + "',\n"
    "    'remote_previous_replay_terminal': 'unknown', 'remote_generated_arrays_returned': False,\n"
    "    'active_local_scientific_jobs': 0, 'active_remote_scientific_jobs': None,\n"
    "    'active_scientific_jobs': None, 'global_single_physical_job_confirmed': False,")
text = replace_once(text, "'active_scientific_jobs': 0, 'model_goal_complete': False, 'primary_A_C_pass': ABC,",
    "'active_local_scientific_jobs': 0, 'active_scientific_jobs': None,\n"
    "    'remote_previous_replay_terminal': 'unknown', 'model_goal_complete': False, 'primary_A_C_pass': ABC,")
written.append(save('finalize_R49_local_scientific_review.py', text))

text = (REVIEWS / 'publish_R49.py').read_text('utf-8-sig').replace(REMOTE, LOCAL)
text = text.replace('Record R49 frozen Heun128 response comparison evidence',
    'Record R49 local Heun128 recovery and interrupted remote evidence')
text = text.replace("    run('git', 'add', '-f', '--', *names)",
    "    for offset in range(0, len(names), 64):\n"
    "        run('git', 'add', '-f', '--', *names[offset:offset + 64])")
text = text.replace("    assert not run('git', 'diff', '--name-only', '--', *names)",
    "    assert not run('git', 'diff', '--name-only')")
written.append(save('publish_R49_local.py', text))

text = (REVIEWS / 'backup_R49_incremental.py').read_text('utf-8-sig').replace(REMOTE, LOCAL)
text = text.replace('r49_heun128_final_', 'r49_local_heun128_final_')
text = text.replace("STAGE = '" + LOCAL + "'", "STAGE = '" + LOCAL + "'\nREMOTE_STAGE = '" + REMOTE + "'")
text = replace_once(text,
    "reuse = (relative.startswith('work/') and not relative.startswith(f'work/{STAGE}/')",
    "reuse = (relative.startswith('work/') and not relative.startswith(f'work/{STAGE}/')\n"
    "                 and not relative.startswith(f'work/{REMOTE_STAGE}/')")
text = replace_once(text, "'new_stage_bytes': stage_bytes, 'copy_operations_completed_without_exception': True,",
    "'new_stage_bytes': stage_bytes, 'copy_operations_completed_without_exception': True,\n"
    "    'local_stage_sealed': True, 'remote_attempt_copied_as_incomplete': REMOTE_STAGE,\n"
    "    'remote_previous_replay_terminal': 'unknown', 'remote_generated_arrays_returned': False,")
written.append(save('backup_R49_local_incremental.py', text))

text = (REVIEWS / 'finalize_R49_delivery_completion.py').read_text('utf-8-sig').replace(REMOTE, LOCAL)
text = text.replace('reviews/finalize_R49_delivery_completion.py', 'reviews/finalize_R49_local_delivery_completion.py')
text = text.replace('parent r48-incremental.bundle', 'parent r49-incremental.bundle')
text = text.replace("assert program['active_scientific_jobs'] == 0",
    "assert program['active_local_scientific_jobs'] == 0\n"
    "assert program['active_scientific_jobs'] is None and program['active_remote_scientific_jobs'] is None")
text = text.replace("current_activity='R49_stage_delivered_no_live_job'",
    "current_activity='R49_local_delivered_no_live_local_job_remote_state_unknown'")
text = text.replace("current_stage_progress='sealed_and_delivered', active_scientific_jobs=0,",
    "current_stage_progress='sealed_and_delivered', active_scientific_jobs=None,\n"
    "    active_local_scientific_jobs=0, active_remote_scientific_jobs=None,")
text = text.replace("delivery_completion_receipt=TERMINAL, active_scientific_jobs=0)",
    "delivery_completion_receipt=TERMINAL, active_scientific_jobs=None,\n"
    "    active_local_scientific_jobs=0, active_remote_scientific_jobs=None)")
text = text.replace("'project_model_goal_complete': False, 'active_scientific_jobs': 0,",
    "'project_model_goal_complete': False, 'active_scientific_jobs': None,\n"
    "    'active_local_scientific_jobs': 0, 'active_remote_scientific_jobs': None,\n"
    "    'remote_previous_replay_terminal': 'unknown', 'global_single_physical_job_confirmed': False,\n"
    "    'remote_generated_arrays_returned': False,")
text = text.replace("'模型目标尚未完成；本轮科学判定见R49_REPORT，原A–F及不设计算预算的政策继续有效。\\n')",
    "'本机科学作业已结束，旧远端重放终态仍未知。模型目标尚未完成；本轮科学判定见R49_REPORT，'\n"
    "                 '原A–F及不设计算预算的政策继续有效。\\n')")
written.append(save('finalize_R49_local_delivery_completion.py', text))
print(json.dumps({'status': 'metadata_helpers_prepared_not_executed', 'files': written}, ensure_ascii=False))
