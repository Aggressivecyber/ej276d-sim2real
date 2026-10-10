"""Register user-authorized local failover; record unknown remote state honestly."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = 'R49_FROZEN_HEUN128_RESPONSE_001'
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
REG = 'reviews/R49_LOCAL_REGISTRATION_001'
TASK = 'tasks/' + STAGE + '.md'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def bind(name):
    raw = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


assert not (ROOT / REG).exists()
program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
budget = read('RESOURCE_BUDGET.json')
assert program['current_stage'] == progress['current_stage'] == OLD
assert not budget['compute_budget_enforced'] and budget['improvement_block']['method_plans_max'] is None
assert not (ROOT / 'work' / OLD / 'FINAL_RECEIPT.json').exists()
native = read('work/' + OLD + '/REMOTE_RUN_HISTORY.json')
science = sum(row['elapsed_s'] for row in native)
assert abs(science - 2107.3375221949973) < 1e-8
assert {row['entry'] for row in native} == {'check_models.py', 'generate_all.py', 'evaluate_saved.py'}
remote = read('work/' + OLD + '/replay_all.py.LOCAL_ACTIVE_JOB.json')
assert remote['actual_remote_handshake']['actual_worker_PID'] == 218727
assert 'SSH_native_exit' in remote and not (ROOT / 'work' / OLD / 'replay_all.py.NATIVE_RECEIPT.json').exists()
assert read('work/' + OLD + '/EVALUATION_NUMERIC_RECEIPT.json')['counters']['HGB_attempts'] == 40
reg = ROOT / REG
reg.mkdir()
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
             'RESOURCE_BUDGET.json', 'SERVER_RESOURCE_POLICY.md', 'jobs.json', 'experiments.json']:
    shutil.copy2(ROOT / name, reg / name)

remote_observation = {
    'recorded_UTC': NOW, 'root_control_PID': os.getpid(), 'stage': OLD,
    'status': 'incomplete_remote_replay_terminal_unknown_user_reported_maintenance',
    'user_reported_server_state': '正在维护，稍后恢复',
    'user_authorized_execution_change': '换本地继续运行',
    'actual_completed_native_jobs': native, 'actual_completed_native_science_s': science,
    'unknown_replay': remote, 'last_observed_replay_elapsed_s': 120.15029214299284,
    'unfinished_replay_full_science_elapsed_s': None,
    'unknown_replay_cost_not_included_in_completed_total': True,
    'local_full_metrics_path': 'work/' + OLD + '/metrics.json',
    'full_generated_DEV_arrays_returned': False, 'native_replay_receipt_returned': False,
    'full_stage_delivery_complete': False, 'remote_process_cancelled': False,
    'remote_replay_relaunched': False, 'new_remote_jobs_authorized': False,
    'original_remote_A_B_observation': 'reviews/R49_ACCEPTANCE_OBSERVATION_001.json',
    'current_remote_PROCESS_count': None, 'global_single_job_verified': False,
    'explicit_local_failover_authorized_despite_unknown_remote_terminal_state': True,
    'no_scientific_failure_inferred_from_transport_exit': True,
}
write('reviews/R49_REMOTE_INTERRUPTION_001.json', remote_observation)
block = budget['improvement_block']
assert block['method_plans_registered'] == 4
old_row = next(row for row in block['actual_stage_receipts'] if row['stage'] == OLD)
old_row.update(receipt='reviews/R49_REMOTE_INTERRUPTION_001.json', science_s=science,
    status='incomplete_remote_replay_terminal_unknown', stage_science_complete=False,
    inflight_science_not_yet_in_total=True, unfinished_replay_science_s=None)
next(row for row in block['registered_stages'] if row['stage'] == OLD).update(
    status='remote_incomplete_local_failover_authorized', actual_completed_native_science_s=science,
    unknown_remote_replay=True, recovery_stage=STAGE)
block['registered_stages'].append({'stage': STAGE, 'parent_stage': OLD,
    'method_plan_index': 4, 'counts_as_new_method': False, 'response_runs_reserved': 0,
    'science_budget_s': None, 'status': 'registered_local_preflight_pending',
    'registration': REG + '/RECEIPT.json',
    'scientific_design': 'Same fixed R49 protocol; distinct local recovery attempt and own-output evaluation/replay.',
    'user_authorization': '换本地继续运行'})
block['actual_stage_receipts'].append({'stage': STAGE, 'parent_stage': OLD,
    'receipt': REG + '/RECEIPT.json', 'science_s': 0.0, 'response_training_runs_started': 0,
    'new_response_updates': 0, 'stage_science_complete': False,
    'inflight_science_not_yet_in_total': False, 'status': 'registered_local_preflight_pending'})
block['actual_science_seconds_recorded'] = sum(row['science_s'] for row in block['actual_stage_receipts'])
assert abs(block['actual_science_seconds_recorded'] - 8487.652792954119) < 1e-8
block.update(current_family_status='R49_same_recipe_local_recovery_registered_remote_terminal_unknown',
    latest_direction_decision=TASK, exhausted_limit=None, further_scientific_execution_authorized=True)
budget.update(updated_UTC=NOW, current_scientific_execution_location='local_CPU',
    local_scientific_limits={'CPU_threads': 4, 'job_tree_RSS_bytes': 4 * 2**30,
        'max_concurrent_local_scientific_jobs': 1, 'compute_budget_enforced': False,
        'science_budget_s': None, 'wall_budget_s': None, 'deadline_UTC': None,
        'stage_output_budget_bytes': None, 'local_files': 'E/D only'},
    unknown_remote_job_observation='reviews/R49_REMOTE_INTERRUPTION_001.json')
write('RESOURCE_BUDGET.json', budget)

location = {'mode': 'local_CPU_frozen_numerical_generation_evaluation_replay', 'CPU_threads': 4,
    'job_tree_RSS_bytes': 4 * 2**30, 'GPU_count': 0, 'current_stage_GPU_used': False,
    'current_stage_response_training': False, 'local_files': 'E/D only',
    'preflight_required': True, 'remote_previous_attempt_terminal': 'unknown',
    'server_user_reported_state': 'maintenance', 'single_local_scientific_job': True,
    'global_single_physical_job_confirmed': False}
for data in [program, progress]:
    data.update(updated_UTC=NOW, updated_at_utc=NOW,
        status='R49_local_recovery_registered_model_goal_unmet', current_stage=STAGE,
        current_task=TASK, current_stage_registration=REG + '/RECEIPT.json',
        current_stage_scope='same_R49_fixed_protocol_local_recovery_with_separate_actual_evidence',
        current_activity='local_dependency_mapping_and_preflight', current_stage_progress='registered_not_started',
        current_direction='Continue the same fixed method locally after user-authorized failover; preserve remote unknown state.',
        current_route_decision=TASK, current_route_research_closed=False,
        current_job={'stage': STAGE, 'status': 'registered_local_preflight_pending'},
        active_scientific_jobs=None, active_local_scientific_jobs=0, active_remote_scientific_jobs=None,
        remote_job_state='unknown_user_reports_maintenance', unknown_remote_job_receipt='reviews/R49_REMOTE_INTERRUPTION_001.json',
        execution_location=location, current_executor_ack='work/' + STAGE + '/EXECUTOR_ACK.json',
        latest_user_steering_receipt=REG + '/RECEIPT.json', last_native_observation=REG + '/RECEIPT.json',
        next_action='Existing executor continuously completes local fixed checks, own-output generation/evaluation/replay and delivery.',
        next_prepared_task=None, queued_stage_registration=None, numeric_execution_released=True,
        current_stage_deadline_UTC=None, stage_deadline_UTC=None, executor_seal_deadline_UTC=None,
        model_goal_complete=False, compute_budget_enforced=False, improvement_block_closed=False)
program['project_model_goal_complete'] = False
write('PROGRAM.json', program)
write('PROJECT_PROGRESS.json', progress)
for filename, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(filename)
    next(row for row in data[key] if row.get(identity) == OLD).update(
        status='remote_replay_terminal_unknown_local_recovery_authorized', active_scientific_jobs=None,
        remote_terminal_state='unknown', recovery_stage=STAGE,
        interruption_receipt='reviews/R49_REMOTE_INTERRUPTION_001.json', stage_science_complete=False)
    assert not any(row.get(identity) == STAGE for row in data[key])
    data[key].append({identity: STAGE, 'parent_stage': OLD, 'method_plan_index': 4,
        'counts_as_new_method': False, 'status': 'registered_local_preflight_pending',
        'registered_UTC': NOW, 'registration': REG + '/RECEIPT.json', 'task': TASK,
        'execution_location': location, 'max_concurrent_local_scientific_jobs': 1,
        'science_budget_s': None, 'compute_budget_enforced': False,
        'planned_new_response_training_runs': 0, 'planned_new_response_updates': 0,
        'actual_science_seconds': 0, 'actual_native_jobs': [], 'EVAL_access': False})
    write(filename, data)
notice = ('# 当前执行：R49 已获授权切换本机恢复\n\n'
    '用户明确要求“换本地继续运行”，并说明服务器正在维护。按 tasks/R49_LOCAL_HEUN128_RECOVERY_001.md '
    '在E盘独立目录执行同一冻结协议：本机CPU4/RSS4GiB、单本机科学作业，计算预算字段仍为null。'
    '既有执行器完成本机数值门后，自行补生成并评价、重放本机产物；新增训练/校准0，原A–F不变。'
    '远端已完成结果保留，旧重放终态未知，不重复远端提交，也不声称已确认全局只有一个物理作业。'
    '登记与旧远端状态见 reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json 和 '
    'reviews/R49_REMOTE_INTERRUPTION_001.json。模型目标尚未完成。\n\n## 以下为历史执行状态\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    p = ROOT / name
    p.write_text(notice + p.read_text('utf-8-sig'), 'utf8')
policy = ROOT / 'SERVER_RESOURCE_POLICY.md'
policy.write_text(notice + policy.read_text('utf-8-sig'), 'utf8')
budget['policy_sha256'] = bind('SERVER_RESOURCE_POLICY.md')['sha256']
write('RESOURCE_BUDGET.json', budget)
receipt = {'stage': STAGE, 'parent_stage': OLD, 'registered_UTC': NOW,
    'actual_root_control_PID': os.getpid(), 'user_execution_instruction': '换本地继续运行',
    'user_server_status': '正在维护，稍后恢复', 'task': bind(TASK),
    'compute_budget_enforced': False, 'science_budget_s': None, 'wall_budget_s': None,
    'deadline_UTC': None, 'stage_output_budget_bytes': None,
    'location': location, 'counts_as_new_method': False, 'method_plans_registered': 4,
    'planned_new_response_training_runs': 0, 'planned_new_response_updates': 0,
    'calibration_refits': 0, 'original_scientific_thresholds_changed': False,
    'remote_interruption': 'reviews/R49_REMOTE_INTERRUPTION_001.json',
    'completed_remote_science_s_preserved': science,
    'unknown_remote_science_s_invented_or_zeroed': False,
    'root_submitted_scientific_job': False,
    'executor_continuous_authorization_after_local_preflight': True,
    'no_stepwise_approval_required': True, 'historical_stage_files_modified': False,
    'snapshots': sorted(p.name for p in reg.iterdir())}
write(REG + '/RECEIPT.json', receipt)
print(json.dumps({'stage': STAGE, 'registered_UTC': NOW, 'root_PID': os.getpid(),
    'receipt': REG + '/RECEIPT.json', 'method_plans_registered': 4,
    'remote_terminal_state': 'unknown', 'local_science_started_by_root': False,
    'completed_block_science_s': block['actual_science_seconds_recorded']}, ensure_ascii=False))
