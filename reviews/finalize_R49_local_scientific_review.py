"""Reconcile sealed R49 and root observations; no inference, fits or resampling."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
PREFIX = 'work/' + STAGE + '/'
NOW = datetime.now(timezone.utc)


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def write(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def binding(name):
    data = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


assert not (ROOT / 'reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json').exists()
final, resource, qa, gates, child_acceptance = [read(PREFIX + name) for name in [
    'FINAL_RECEIPT.json', 'RESOURCE_RECEIPT.json', 'FIGURE_QA.json', 'GATE_METRICS.json', 'ACCEPTANCE_A_C.json']]
acceptance = read('reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json')
root_qa = read('reviews/R49_LOCAL_ROOT_FIGURE_QA_001.json')
seal = read('reviews/R49_LOCAL_SEAL_NATIVE_RECEIPT.json')
direction = read('reviews/R49_LOCAL_ROOT_DIRECTION_001.json')
assert final['status'] == 'sealed' and final['no_further_stage_writes']
assert final['new_response_training_updates'] == resource['new_response_updates'] == 0
assert resource['new_response_training_runs'] == resource['new_calibration_fits'] == 0
assert final['candidate_steps'] == 128 and final['reference_steps'] == 256
assert final['both_numerical_pass'] and final['D_E_not_run']
assert final['model_project_target_complete'] is False
assert resource['compute_budget_enforced'] is False
assert all(resource[key] is None for key in ['science_budget_s', 'wall_budget_s', 'deadline_UTC', 'stage_output_budget_bytes'])
jobs = resource['native_jobs']
required = {'check_models.py', 'generate_all.py', 'evaluate_saved.py', 'replay_all.py', 'summarize_and_plot.py', 'render_train.py'}
assert required.issubset({job['entry'] for job in jobs if job['actual_worker_native_exit'] == 0})
assert all(job['handshake_before_imports'] and job['science_budget_s'] is None
           and job['wall_budget_s'] is None and job['deadline_UTC'] is None
           and job['compute_budget_enforced'] is False for job in jobs)
science = resource['actual_scientific_elapsed_s']
assert abs(science - sum(job['elapsed_s'] for job in jobs)) < 1e-8
assert science == final['actual_scientific_elapsed_s']
assert resource['EVAL_read'] == resource['CAL_waveform_decode'] == resource['new_source_acquisition'] == 0
assert [resource['generation'][key] for key in ['main_generation_rows', 'repeat_noise_rows', 'source_probe_rows', 'extra_guide_probe_rows']] == [8000, 2048, 1664, 256]
assert resource['counters']['HGB_attempts'] == final['actual_HGB_fits'] == 40
assert resource['counters']['density_description_slots'] == final['density_slots'] == 32
assert final['total_generated_rows'] == 11968
replay = read(PREFIX + 'REPLAY_RECEIPT.json')
assert final['independent_main_replay_rows'] == resource['replay_rows'] == replay['rows'] == 8000
assert final['independent_main_replay_pass'] == resource['replay_pass'] == replay['pass']
all_figures = {path.name for path in (ROOT / PREFIX / 'results/figures').glob('*.png')}
assert all_figures == {row['file'] for row in qa['figures'] if row['actually_opened']}
assert all_figures == {row['file'] for row in root_qa['figures'] if row['actually_opened']}
for row in root_qa['figures']:
    assert binding(PREFIX + 'results/figures/' + row['file'])['sha256'] == row['SHA256'].lower()
seals = [binding(PREFIX + name) for name in ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json']]
assert seals[0]['sha256'] == final['manifest_SHA256'].lower()
assert seal['executor_native_exit'] == 0 and seal['executor_seal_PID'] == final['PID']
assert seal['FINAL_RECEIPT_SHA256'].lower() == seals[1]['sha256']
files = [path for path in (ROOT / PREFIX).rglob('*') if path.is_file()]
stage_count, stage_bytes = len(files), sum(path.stat().st_size for path in files)
assert stage_count == final['listed_files'] + 2 == seal['stage_file_count']
assert stage_bytes == final['listed_bytes'] + sum(row['bytes'] for row in seals) == seal['stage_bytes']

primary = 'GUIDED_NOISE_COMPLETION'
gate_names = ['gate1_complete_response', 'gate2_PSD_width_centers_noise', 'gate3_source_preservation',
              'gate4_boundary_source', 'gate5_same_treated_REAL_control', 'gate6_numerical_replay']
arm_results = {}
for arm in ['GUIDED_FIXED_NOISE_CONTROL', primary]:
    row = acceptance['arms'][arm]
    six = [gates['models'][arm][name]['pass'] for name in gate_names]
    C = all(six)
    assert row['A_pass'] == child_acceptance[arm]['A']['pass']
    assert C == child_acceptance[arm]['C']['pass']
    arm_results[arm] = {'A_pass': row['A_pass'], 'B_pass': row['B_pass'], 'C_pass': C,
                        'all_six_original_gates': dict(zip(gate_names, six)),
                        'A_C_pass': row['A_pass'] and row['B_pass'] and C}
ABC = arm_results[primary]['A_C_pass']
assert ABC == gates['primary_A_C_pass'] == final['primary_A_C_pass']
assert direction['primary_A_C_pass'] == ABC and direction['original_A_F_unchanged']
assert direction['compute_budget_enforced'] is False
decision = gates['decision']
observed_peaks_ok = all(job['job_tree_RSS_peak_bytes'] <= job['resource_admission']['RSS_safe_limit_bytes'] for job in jobs)
assert observed_peaks_ok
review = {
    'stage': STAGE, 'reviewed_UTC': NOW.isoformat(), 'actual_root_control_PID': os.getpid(),
    'status': 'sealed_full_response_reviewed_model_goal_unmet',
    'review_scope': 'Saved aggregates, implementation, actual native receipts and rendered figures; no scientific recomputation.',
    'numerical_results': read(PREFIX + 'TRAIN_SOLVER_CHECK.json'),
    'arms': arm_results, 'primary_A_C_pass': ABC, 'primary_candidate': primary,
    'fixed_control_not_substituted': True, 'original_A_F_unchanged': True,
    'A_B_receipt': 'reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json',
    'C_gate_receipt': PREFIX + 'GATE_METRICS.json', 'independent_replay': replay,
    'D_E': 'not_run_separate_fixed_protocol_required_for_A_C_accepted_candidate',
    'F': 'stage_science_sealed_private_Git_D_delivery_pending_no_A_F_model',
    'new_training_runs': 0, 'new_training_updates': 0, 'new_calibration_fits': 0,
    'historical_training_each_updates': 50000, 'historical_training_charged_again': False,
    'scientific_jobs': jobs, 'scientific_wall_s': science,
    'scientific_counts': {'generation': resource['generation'], 'evaluation': resource['counters'], 'independent_main_replay_rows': 8000},
    'stage_file_count': stage_count, 'stage_bytes': stage_bytes,
    'figures_actually_opened_by_root': len(root_qa['figures']), 'root_figure_QA': 'reviews/R49_LOCAL_ROOT_FIGURE_QA_001.json',
    'observed_resource_peaks_below_registered_limits': observed_peaks_ok,
    'continuous_RSS_observation_complete': not resource['RSS_query_missing_events'],
    'memory_query_gaps': resource['RSS_query_missing_events'],
    'actual_native_failures_preserved': resource['scientific_failure_files'],
    'compute_budget_enforced': False, 'time_or_output_budget_stop': False,
    'scientific_decision': decision, 'current_fixed_recipe_closed': not ABC,
    'whole_method_family_declared_ineffective': False, 'project_model_goal_complete': False,
    'active_scientific_jobs': None, 'active_local_scientific_jobs': 0,
    'active_remote_scientific_jobs': None, 'remote_replay_terminal': 'unknown',
    'global_single_physical_job_confirmed': False, 'stage_immutable': True, 'sealed_bindings': seals,
    'next_direction': direction['next_direction'], 'next_action': direction['next_action'],
}
write('reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json', review)

budget = read('RESOURCE_BUDGET.json')
block = budget['improvement_block']
assert budget['compute_budget_enforced'] is False and block['method_plans_max'] is None
assert block['total_science_seconds_max'] is None and block['method_plans_registered'] == 4
matches = [index for index, row in enumerate(block['actual_stage_receipts']) if row['stage'] == STAGE]
assert len(matches) == 1
previous_provisional = block['actual_stage_receipts'][matches[0]]
assert not previous_provisional['stage_science_complete']
old_sum = sum(row['science_s'] for row in block['actual_stage_receipts'] if row['stage'] != STAGE)
assert abs(old_sum - 8487.652792954119) < 1e-8
block['actual_stage_receipts'][matches[0]] = {
    'stage': STAGE, 'receipt': PREFIX + 'RESOURCE_RECEIPT.json', 'science_s': science,
    'response_training_runs_started': 0, 'new_response_updates': 0,
    'status': 'sealed_full_response_evaluated', 'stage_science_complete': True,
    'inflight_science_not_yet_in_total': False}
block['actual_science_seconds_recorded'] = old_sum + science
block['response_training_runs_started'] = sum(row['response_training_runs_started'] for row in block['actual_stage_receipts'])
assert block['response_training_runs_started'] == 3
next(row for row in block['registered_stages'] if row['stage'] == STAGE).update(
    status='sealed_full_response_evaluated', actual_completed_native_science_s=science,
    actual_science_seconds=science, response_training_runs_started=0, new_response_updates=0,
    stage_science_complete=True, resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json')
block.update(current_family_status=direction['current_family_status'],
    latest_direction_decision='reviews/R49_LOCAL_ROOT_DIRECTION_001.json', exhausted_limit=None,
    further_scientific_execution_authorized=True)
budget['updated_UTC'] = NOW.isoformat()
write('RESOURCE_BUDGET.json', budget)

program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
assert program['current_stage'] == progress['current_stage'] == STAGE
for data in [program, progress]:
    data.update(updated_at_utc=NOW.isoformat(), updated_UTC=NOW.isoformat(),
        status='R49_sealed_full_response_delivery_in_progress_model_goal_unmet',
        current_stage_progress='sealed_reviewed_publication_backup_pending',
        current_activity='R49_local_private_Git_and_D_delivery_remote_state_unknown',
        active_scientific_jobs=None, active_local_scientific_jobs=0, active_remote_scientific_jobs=None,
        remote_previous_replay_terminal='unknown', global_single_physical_job_confirmed=False,
        latest_completed_stage=STAGE, latest_response_model_stage=STAGE,
        current_stage_scope='frozen_guided_noise_pair_common_Heun128_full_response',
        latest_response_conclusion=decision, latest_scientific_decision=decision,
        current_direction=direction['next_direction'], current_route_decision='reviews/R49_LOCAL_ROOT_DIRECTION_001.json',
        current_route_research_closed=not ABC, improvement_block_closed=False,
        further_scientific_execution_authorized=True, next_action=direction['next_action'],
        next_prepared_task=None, queued_stage_registration=None,
        current_executor_ack=PREFIX + 'EXECUTOR_ACK.json',
        latest_user_steering_receipt='reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json',
        latest_permission_status='Authorized R49 local recovery completed; remote replay terminal unknown; unbudgeted project policy and original A-F retained.',
        model_goal_complete=False,
        current_job={'stage': STAGE, 'status': 'sealed_no_live_job', 'scientific_start_observed': True,
                     'all_actual_native_exits_zero': resource['all_native_exits_zero'], 'active_scientific_jobs': 0})
program.update(latest_completed_report='R49_REPORT.md',
    latest_completed_review='reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json',
    latest_key_result_review='reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json',
    latest_numeric_receipt=PREFIX + 'RESOURCE_RECEIPT.json', final_report='R49_REPORT.md',
    delivery_index='R49_DELIVERY_INDEX.json', reproduction_guide=PREFIX + 'REPRODUCTION.md',
    publication_receipt='R49_PUBLICATION_RECEIPT.json', backup_receipt='R49_BACKUP_RECEIPT.json',
    delivery_completion_receipt='R49_DELIVERY_COMPLETION_RECEIPT.json', project_model_goal_complete=False)
progress.update(project_status='R49_full_response_reviewed_model_goal_unmet_delivery_pending',
    next_round_policy='User-authorized no compute budgets; retain original A-F and register any distinct next scientific protocol.',
    latest_acceptance_observation='reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json')
for row in progress['requirements']:
    if row['id'] in ['target_distribution_learning', 'G4_conditioned_full_response', 'charge_conditional_width_noise_waveform']:
        row.update(status='R49_A_C_passed_D_E_pending' if ABC else 'R49_joint_A_C_failure_original_thresholds_retained',
                   latest_evidence=['R49_REPORT.md', 'reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json', PREFIX + 'GATE_METRICS.json'])
    elif row['id'] == 'reproducible_final_model':
        row.update(status='R49_8000_main_replay_passed_no_A_F_final_model' if replay['pass'] else 'R49_8000_main_replay_failed_no_A_F_final_model',
                   latest_evidence=[PREFIX + 'FINAL_RECEIPT.json', PREFIX + 'REPLAY_RECEIPT.json'])
    elif row['id'] == 'final_research_delivery':
        row.update(status='R1_R48_delivered_R49_delivery_pending_model_goal_unmet',
                   latest_evidence=['R49_REPORT.md', PREFIX + 'FINAL_RECEIPT.json'])
write('PROGRAM.json', program)
write('PROJECT_PROGRESS.json', progress)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(name)
    row = next(item for item in data[key] if item.get(identity) == STAGE)
    row.update(status='sealed_full_response_root_delivery_in_progress', active_scientific_jobs=0,
        actual_native_jobs=jobs, scientific_elapsed_s=science, actual_completed_native_science_s=science,
        actual_science_seconds=science, scientific_counts=review['scientific_counts'],
        response_training_runs_started=0, response_training_runs_completed=0,
        last_observed_UTC=NOW.isoformat(), last_observed_running_jobs=[], awaiting_native_receipt_jobs=[],
        resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json', final_receipt=PREFIX + 'FINAL_RECEIPT.json',
        root_review='reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json', project_model_goal_complete=False,
        scientific_decision=decision, stage_science_complete=True)
    write(name, data)

current = '# R49 已封存：完整响应评价完成\n\n' + direction['Chinese_current_state'] + (
    ' 原A–F与不设计算预算的政策保持；模型目标尚未完成。当前完成私有Git及D盘交付。'
    '\n\n## 以下为历史执行记录\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    path.write_text(current + path.read_text('utf-8-sig'), 'utf8')
header = '# R49 根复核结果\n\n' + direction['Chinese_result_summary'] + '\n\n'
header += (f'两组复用R48的50000更新末点，本轮新增训练与校准均为0。实际科学耗时{science:.6f}秒，'
           f'累计记录{block["actual_science_seconds_recorded"]:.6f}秒；没有计算时间、次数或阶段体积预算上限。'
           f'保存{stage_count}文件、{stage_bytes:,} B。本机CPU4、RSS4GiB及单本机科学作业约束保持；旧远端重放终态未知，全局在运行作业数未知。\n\n')
header += ('[原A/B复核](reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json) · '
           '[完整根复核](reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json) · '
           '[后续方向](reviews/R49_LOCAL_ROOT_DIRECTION_001.json) · '
           '[交付索引](R49_DELIVERY_INDEX.json)\n\n')
header += f'[本机阶段报告]({PREFIX}REPORT.md) · [复现步骤]({PREFIX}REPRODUCTION.md)\n\n'
header += ('远端原尝试已完成数值、生成与评价，共2107.3375221949973秒；旧重放未取得终端回执，'
           '其额外全程耗时未知，未猜测归零。累计数值只包含已经取得原生终态的工作。'
           '本机A–C依据本机重新生成及评价的数组；远端指标仅作历史。'
           '远端生成数组未回传，本次交付不声称包含它们。'
           '详见[远端中断记录](reviews/R49_REMOTE_INTERRUPTION_001.json)。\n\n')
figure_labels = {
    'c2st.png': 'SOURCE 十视图判别',
    'paired_noise_rule_C2ST.png': '两臂配对判别差及区间',
    'charge_psd_main_body.png': '电荷与 PSD 主体分布',
    'charge_psd_full_range.png': '电荷与 PSD 完整范围',
    'conditional_psd_CDF.png': '各电荷区 PSD 累积分布',
    'energy_peak_width.png': '条件峰位与宽度及失败区',
    'noise_curvature.png': '噪声与波形曲率',
    'conditional_waveforms.png': '条件波形比较',
    'same_treated_REAL_controls.png': '同处理 REAL 三项相对判别与波形改变量',
    'source_preservation.png': '源信息保留',
    'boundary_differences.png': '边界扰动差异',
    'initial_and_random_noise.png': '初始噪声与重复随机噪声',
    'training_source_identity.png': '既有训练轨迹',
    'TRAIN_numerical_check.png': 'TRAIN 数值检查',
    'TRAIN_noise_operator.png': 'TRAIN 噪声算子',
    'TRAIN_initial_noise_rule.png': 'TRAIN 初始噪声规则',
}
figure_index = '\n## 已复核图件\n\n' + '\n'.join(
    f'- [{figure_labels.get(name, name)}]({PREFIX}results/figures/{name})'
    for name in sorted(all_figures)) + '\n'
(ROOT / 'R49_REPORT.md').write_text(
    header + (ROOT / PREFIX / 'REPORT.md').read_text('utf8') + figure_index, 'utf8')
backup_dir = ('D:/GAN_backups/research/sim2real_two_papers_20261008_0305/'
              'r49_local_heun128_final_' + NOW.strftime('%Y%m%d_%H%M%S_UTC'))
index = {'stage': STAGE, 'status': 'science_sealed_publication_and_backup_pending',
    'created_UTC': NOW.isoformat(), 'report': 'R49_REPORT.md',
    'execution_plan': 'tasks/' + STAGE + '.md', 'resource_receipt': PREFIX + 'RESOURCE_RECEIPT.json',
    'model_goal_complete': False, 'primary_A_C_pass': ABC, 'scientific_decision': decision,
    'local_recovery_stage': STAGE, 'remote_attempt_stage': 'R49_FROZEN_HEUN128_RESPONSE_001',
    'remote_previous_replay_terminal': 'unknown', 'remote_generated_arrays_returned': False,
    'active_local_scientific_jobs': 0, 'active_remote_scientific_jobs': None,
    'active_scientific_jobs': None, 'global_single_physical_job_confirmed': False,
    'original_acceptance_A_F_unchanged': True, 'compute_budget_enforced': False,
    'private_Git': {'status': 'pending', 'repository': 'Aggressivecyber/ej276d-sim2real',
                  'visibility': 'PRIVATE', 'receipt': 'R49_PUBLICATION_RECEIPT.json'},
    'backup': {'status': 'pending', 'directory': backup_dir, 'receipt': 'R49_BACKUP_RECEIPT.json',
               'mode': 'incremental_with_explicit_R44_R45_R46_R47_R48_dependencies', 'standalone_full_snapshot': False},
    'bindings': [binding(name) for name in ['R49_REPORT.md', 'tasks/' + STAGE + '.md',
        'reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json', 'reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json',
        'reviews/R49_LOCAL_ROOT_DIRECTION_001.json', PREFIX + 'FINAL_RECEIPT.json', PREFIX + 'FINAL_MANIFEST.json',
        PREFIX + 'RESOURCE_RECEIPT.json', PREFIX + 'TRAIN_SOLVER_CHECK.json', PREFIX + 'REPLAY_RECEIPT.json',
        PREFIX + 'REPRODUCTION.md']]}
write('R49_DELIVERY_INDEX.json', index)
write('reviews/R49_LOCAL_RESOURCE_RECONCILIATION_001.json', {
    'UTC': NOW.isoformat(), 'actual_root_control_PID': os.getpid(),
    'replaced_provisional_R49_entry': previous_provisional, 'final_R49_science_s': science,
    'block_science_s': block['actual_science_seconds_recorded'], 'method_plans_registered': 4,
    'training_runs_total': 3, 'new_R49_training_runs': 0,
    'compute_budget_enforced': False, 'provisional_plus_final_double_counted': False,
    'historical_training_charged_again': False, 'new_scientific_computation': False})
print(json.dumps({'status': 'R49_root_review_and_state_reconciled', 'PID': os.getpid(),
    'stage_science_s': science, 'block_science_s': block['actual_science_seconds_recorded'],
    'active_local_scientific_jobs': 0, 'active_scientific_jobs': None,
    'remote_previous_replay_terminal': 'unknown', 'model_goal_complete': False, 'primary_A_C_pass': ABC,
    'backup_destination': backup_dir}, ensure_ascii=False))
