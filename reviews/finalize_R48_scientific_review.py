"""Reconcile sealed R48 metadata and record the bounded negative result; no science."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
PREFIX = 'work/' + STAGE + '/'
NOW = datetime.now(timezone.utc)


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def binding(name):
    raw = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


assert not (ROOT / 'reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json').exists()
final, resource, qa = [read(PREFIX + n) for n in ['FINAL_RECEIPT.json', 'RESOURCE_RECEIPT.json', 'FIGURE_QA.json']]
acceptance = read('reviews/R48_ACCEPTANCE_OBSERVATION_001.json')
training = read(PREFIX + 'TRAINING_COMPACT.json')['training']
assert final['status'] == 'sealed' and final['no_further_stage_writes']
assert final['training_runs'] == resource['completed_new_training_runs'] == 2
assert final['updates_per_arm'] == resource['updates_per_arm'] == 50000
assert final['numerical_prerequisite_pass'] is False and acceptance['C_pass'] is False
assert acceptance['A_pass'] is None and acceptance['B_pass'] is None
assert qa['all_final_images_actually_opened'] and len(qa['figures']) == 4
jobs = resource['native_scientific_jobs']
assert len(jobs) == 6 and all(j['actual_worker_native_exit'] == j['launcher_native_exit'] == 0 for j in jobs)
science = resource['scientific_wall_s']
assert abs(science - sum(j['elapsed_s'] for j in jobs)) < 1e-9
assert science == final['scientific_wall_s'] and science <= resource['science_limit_s'] == 14400
assert resource['training_guard_wall_s'] <= resource['training_total_limit_s'] == 10800
assert all(j['elapsed_s'] <= 5400 for j in jobs if j['entry'].startswith('train_'))
counts = resource['scientific_inference_and_fit_counts']
assert counts == {'completed_training_updates': 100000, 'TRAIN_solver_field_rows': 98304,
                  'DEV_neural_inference': 0, 'new_HGB': 0, 'new_EM': 0, 'main_replay': 0}
assert resource['EVAL_read'] == resource['CAL_waveform_decode'] == 0
assert len(resource['memory_query_gaps']) == 2
assert all(g['cause'] == 'unknown' for g in resource['memory_query_gaps'])
seals = [binding(PREFIX + n) for n in ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json']]
assert seals[0]['sha256'] == final['manifest_sha256']
assert seals[1]['sha256'] == '8cbe65f05a6b1e7c7f8f99c9e0d7b069318a2b4109d9a4e1184176f56cd60c08'
files = [p for p in (ROOT / 'work' / STAGE).rglob('*') if p.is_file()]
stage_count, stage_bytes = len(files), sum(p.stat().st_size for p in files)
assert stage_count == final['files_before_final_receipt'] + 1 == 186
assert stage_bytes == final['bytes_before_final_receipt'] + seals[1]['bytes'] == 44449915
assert stage_bytes <= 160 * 2**20
assert all(binding(PREFIX + f['path'])['sha256'] == f['sha256'] for f in qa['figures'])

decision = 'close_fixed_R48_Heun64_recipe_numerical_failure_before_DEV'
direction = ('The three-method improvement block is exhausted without an A-C accepted candidate. '
             'Preserve R48 as a fixed-recipe numerical failure; its noise-completion efficacy remains unevaluated. '
             'No fourth method, higher-step search, new training or DEV execution is authorized in this block.')
next_action = ('Finish the negative-result delivery; then start no new scientific jobs. Further scientific '
               'execution requires an explicitly revised bounded plan, not reuse of remaining time or fit slots.')
review = {
    'stage': STAGE, 'reviewed_UTC': NOW.isoformat(), 'actual_root_control_PID': os.getpid(),
    'status': 'sealed_numerical_failure_reviewed_model_goal_unmet',
    'review_scope': 'Metadata, source, native receipts and rendered figures; no scientific recomputation.',
    'actual_read_tools': ['2add82', 'e9fdda', '303029', '52f950'],
    'figures_actually_opened_by_root': 4, 'corrected_noise_rule_figure_reopened_by_root': True,
    'figure_receipt': PREFIX + 'FIGURE_QA.json', 'numerical_results': acceptance['numerical_results'],
    'same_initial_state_and_all_seven_streams': True, 'fixed_endpoint_updates_each': 50000,
    'fresh_training_runs_completed': 2, 'scientific_jobs': jobs, 'scientific_counts': counts,
    'scientific_wall_s': science, 'stage_file_count': stage_count, 'stage_bytes': stage_bytes,
    'observed_resource_peaks_below_registered_limits': True,
    'continuous_RSS_observation_complete': False, 'memory_query_gaps': resource['memory_query_gaps'],
    'memory_gap_causes_resolved': False,
    'A': 'not_evaluated', 'B': 'not_evaluated', 'C': 'failed_numerical_prerequisite',
    'other_C_requirements': 'not_evaluated', 'D_E': 'not_registered_no_A_C_accepted_model',
    'F': 'stage_delivery_pending_no_accepted_final_model_or_main_output_replay',
    'noise_completion_scientific_efficacy': 'unknown_not_evaluated',
    'identity_loss_comparability': 'Different velocity targets; lower loss does not prove output stability.',
    'current_improvement_block_closed': True, 'method_plans_consumed': 3, 'method_plans_max': 3,
    'higher_step_search_performed': False, 'automatic_budget_extension': False,
    'original_A_F_unchanged': True, 'project_model_goal_complete': False,
    'active_scientific_jobs': 0, 'stage_immutable': True, 'sealed_bindings': seals,
    'scientific_decision': decision, 'next_direction': direction, 'next_action': next_action,
}
write('reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json', review)
write('reviews/R48_SEAL_NATIVE_RECEIPT.json', {
    'stage': STAGE, 'recorded_UTC': NOW.isoformat(), 'actual_root_control_PID': os.getpid(),
    'executor_seal_PID': 2772, 'executor_native_tool': '787d05', 'executor_native_exit': 0,
    'root_post_seal_read_tool': '2add82', 'root_post_seal_read_native_exit': 0,
    'stage_last_write_observed_UTC': '2026-10-10T08:55:37.764516+00:00',
    'stage_file_count': stage_count, 'stage_bytes': stage_bytes, 'bindings': seals,
    'stage_no_further_writes': True, 'finalization_was_control_not_scientific_job': True})

budget = read('RESOURCE_BUDGET.json')
block = budget['improvement_block']
assert block['method_plans_registered'] == block['method_plans_max'] == 3
matches = [i for i, row in enumerate(block['actual_stage_receipts']) if row['stage'] == STAGE]
assert len(matches) == 1
previous_provisional = block['actual_stage_receipts'][matches[0]]
assert not previous_provisional['stage_science_complete']
block['actual_stage_receipts'][matches[0]] = {
    'stage': STAGE, 'receipt': PREFIX + 'RESOURCE_RECEIPT.json', 'science_s': science,
    'response_training_runs_started': 2, 'new_response_updates': 100000,
    'status': 'sealed_numerical_failure_before_DEV', 'stage_science_complete': True,
    'inflight_science_not_yet_in_total': False}
block['actual_science_seconds_recorded'] = sum(row['science_s'] for row in block['actual_stage_receipts'])
block['response_training_runs_started'] = sum(row['response_training_runs_started'] for row in block['actual_stage_receipts'])
assert abs(block['actual_science_seconds_recorded'] - 6380.315270759122) < 1e-8
assert block['response_training_runs_started'] == 3
next(row for row in block['registered_stages'] if row['stage'] == STAGE).update(
    status='sealed_numerical_failure_before_DEV', actual_completed_native_science_s=science,
    actual_science_seconds=science, response_training_runs_started=2, new_response_updates=100000,
    stage_science_complete=True, resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json')
block.update(current_family_status='three_method_block_closed_no_A_C_accepted_candidate',
    latest_direction_decision='reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json',
    exhausted_limit='method_plans', further_scientific_execution_authorized=False)
budget['updated_UTC'] = NOW.isoformat()
write('RESOURCE_BUDGET.json', budget)

program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
assert program['current_stage'] == progress['current_stage'] == STAGE
for data in [program, progress]:
    data.update(updated_at_utc=NOW.isoformat(), updated_UTC=NOW.isoformat(),
        status='R48_sealed_negative_result_delivery_in_progress_model_goal_unmet',
        current_stage_progress='sealed_reviewed_publication_backup_pending',
        current_activity='R48_private_Git_and_D_delivery', active_scientific_jobs=0,
        latest_completed_stage=STAGE, latest_response_model_stage=STAGE,
        current_stage_scope='fixed_guided_noise_pair_closed_at_TRAIN_numerical_gate',
        latest_response_conclusion=decision, latest_scientific_decision=decision,
        current_direction=direction, current_route_decision='reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json',
        current_route_research_closed=True, improvement_block_closed=True,
        further_scientific_execution_authorized=False, next_action=next_action,
        next_prepared_task=None, queued_stage_registration=None,
        current_executor_ack=PREFIX + 'EXECUTOR_ACK.json',
        latest_user_steering_receipt='reviews/PROJECT_GOAL_ACTIVATION_20261010_001/RECEIPT.json',
        latest_permission_status='Registered R48 completed; original three-method budget exhausted; no automatic extension.',
        model_goal_complete=False,
        current_job={'stage': STAGE, 'status': 'sealed_no_live_job', 'scientific_start_observed': True,
                     'all_six_actual_native_exits_zero': True, 'active_scientific_jobs': 0})
program.update(latest_completed_report='R48_REPORT.md',
    latest_completed_review='reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json',
    latest_key_result_review='reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json',
    latest_numeric_receipt=PREFIX + 'RESOURCE_RECEIPT.json',
    final_report='R48_REPORT.md', delivery_index='R48_DELIVERY_INDEX.json',
    reproduction_guide=PREFIX + 'REPRODUCTION.md', publication_receipt='R48_PUBLICATION_RECEIPT.json',
    backup_receipt='R48_BACKUP_RECEIPT.json', delivery_completion_receipt='R48_DELIVERY_COMPLETION_RECEIPT.json',
    project_model_goal_complete=False)
progress.update(project_status='three_method_block_closed_model_goal_unmet_delivery_pending',
    next_round_policy='No automatic new method after the original three-method cap; retain original A-F.',
    latest_acceptance_observation='reviews/R48_ACCEPTANCE_OBSERVATION_001.json')
for row in progress['requirements']:
    if row['id'] in ['target_distribution_learning', 'G4_conditioned_full_response', 'charge_conditional_width_noise_waveform']:
        row.update(status='R48_not_evaluated_numerical_prerequisite_failed_R46_joint_failure_retained',
                   latest_evidence=['R48_REPORT.md', 'reviews/R48_ACCEPTANCE_OBSERVATION_001.json', 'R46_REPORT.md'])
    elif row['id'] == 'reproducible_final_model':
        row.update(status='R48_weights_and_random_states_sealed_no_accepted_model_or_main_replay',
                   latest_evidence=[PREFIX + 'FINAL_RECEIPT.json', PREFIX + 'TRAIN_SOLVER_CHECK.json'])
    elif row['id'] == 'final_research_delivery':
        row.update(status='R1_R47_delivered_R48_delivery_pending_model_goal_unmet',
                   latest_evidence=['R48_REPORT.md', PREFIX + 'FINAL_RECEIPT.json'])
write('PROGRAM.json', program)
write('PROJECT_PROGRESS.json', progress)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(name)
    row = next(x for x in data[key] if x.get(identity) == STAGE)
    row.update(status='sealed_numerical_failure_root_delivery_in_progress', active_scientific_jobs=0,
        actual_native_jobs=jobs, scientific_elapsed_s=science, actual_completed_native_science_s=science,
        scientific_counts=counts, response_training_runs_started=2, response_training_runs_completed=2,
        last_observed_UTC=NOW.isoformat(), last_observed_running_jobs=[],
        resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json', final_receipt=PREFIX + 'FINAL_RECEIPT.json',
        root_review='reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json', project_model_goal_complete=False,
        scientific_decision=decision, stage_science_complete=True)
    write(name, data)

current = ('# R48 已封存：数值门失败，三方法预算结束\n\n'
    '两臂各完成50000次更新；64/128步检查p95为1.332799/1.449824 LSB，均超过1 LSB。'
    'A/B未评价，C数值前提失败，D/E未启动。噪声补全机制有效性未知；不改原A–F，不选对照替代。'
    '方法名额3/3已耗尽，停止新增科学作业，不自动提高步数或开第四方法。'
    '本轮科学3181.818013秒，改进块累计6380.315271秒；训练次数3/6，剩余时间/次数不授权扩展方法。'
    '当前仅进行私有Git与D盘交付；实际结果见R48_DELIVERY_COMPLETION_RECEIPT.json。模型目标未完成。\n\n'
    '## 以下为历史执行记录\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    path.write_text(current + path.read_text('utf-8-sig'), 'utf8')

report = f'''# R48 结果：数值前提失败，模型目标未完成

固定主候选 GUIDED_NOISE_COMPLETION 与等预算对照 GUIDED_FIXED_NOISE_CONTROL 均完成从零训练50000次更新。两臂未通过事先固定的64/128步积分一致性门，按协议在DEV之前关闭。第三个方法名额已经使用，本改进块停止新增科学作业。

| 固定末点 | RMS中位数（LSB，≤0.25） | RMS P95（LSB，≤1） | 最大相对电荷误差（≤1e-5） | 数值判定 |
|---|---:|---:|---:|---|
| 等预算对照 | 0.158880 | 1.332799 | 1.040213e-8 | 失败 |
| 主候选 | 0.116197 | 1.449824 | 9.337949e-9 | 失败 |

每臂检查同一组SOURCE/REAL TRAIN各64条，64步是候选，128步仅为参考。没有更高步数搜索，也没有选择中间checkpoint或对照赢家。上述是固定数值配方的负结果，尚不能判定噪声补全机制无效。

两臂共享初始化、全部七项训练索引/随机流、guide、REAL身份路径、谱核和各0.5的SOURCE/IDENTITY损失。唯一臂间差别为初态创新方差 tau² 与 max(tau²−v(input),0)。两臂的SOURCE首/末批MSE分别为1.488745→0.571175和1.488745→0.571149；IDENTITY分别为0.219879→0.000632975和0.000821378→0.000056315。IDENTITY速度目标不同，较低MSE不能证明输出稳定。首末批数值也不是完整人口的收敛保证。

校准及训练实际使用3752个unique SOURCE父事件和18739条unique REAL；加载容器6593/21047另行记录。每臂SOURCE、matched REAL及REAL身份路径各有160万行训练曝光，重复曝光不增加独立人口。真实逐事件跨域对应关系和物理噪声识别均未获得。

| 原验收项 | R48结果 |
|---|---|
| A 完整SOURCE和REAL响应 | 未评价；无DEV输出或HGB拟合 |
| B PSD、条件峰宽、噪声 | 未评价；无DEV输出或EM拟合 |
| C 源信息、边界、数值可信 | 数值前提失败；其余本轮未评价 |
| D 下游效用 | 未登记执行；没有A–C合格模型 |
| E FULL/GAP适用范围 | 未登记执行；没有A–C合格模型 |
| F 可复现交付 | 本轮负结果与两组末点已封存；私有Git与D复制以实际交付回执为准；尚无满足A–F的最终模型 |

R48新增DEV推理、HGB、EM和主输出独立重放均为0。已完成TRAIN求解器前向98304行，不能把它记作8000条主输出重放。EVAL读取与CAL波形解码均为0，原源组成/权重和旧失败结论不变。

本阶段6个计费作业均原生退出0（原生执行成功与科学判定通过分别记录）。科学总耗时{science:.9f}秒，训练守护累计3103.261651秒。改进块累计{block['actual_science_seconds_recorded']:.9f}/90000秒，训练3/6次，方法3/3项。阶段186文件、44,449,915 B。观察到的worker RSS峰值1,991,344,128 B，GPU allocated/reserved峰值681,300,992/746,586,112 B；两次RSS查询缺口原因仍为unknown，不能称连续监控无缺口。

四张TRAIN图已实际打开复核，噪声规则图统一0–0.60 mV坐标后重新打开，原图另名保留。未为缺失的DEV结果补画成功图。

![数值门](work/{STAGE}/results/figures/TRAIN_numerical_check.png)

[训练曲线](work/{STAGE}/results/figures/training_source_identity.png) · [噪声算子](work/{STAGE}/results/figures/TRAIN_noise_operator.png) · [初态噪声规则](work/{STAGE}/results/figures/TRAIN_initial_noise_rule.png)

下一次科学执行须有另行明确的有限计划及资源范围；不能借尚余的科学时间/训练次数绕过已耗尽的方法上限。新证据首先需要说明如何同时满足数值可信和原完整响应要求，再决定是否另行开展实验。本轮没有预先授权新方法、重训或更大步数搜索。

可复核入口：[执行器报告](work/{STAGE}/REPORT.md)、[数值结果](work/{STAGE}/TRAIN_SOLVER_CHECK.json)、[训练摘要](work/{STAGE}/TRAINING_COMPACT.json)、[资源回执](work/{STAGE}/RESOURCE_RECEIPT.json)、[原生退出链](work/{STAGE}/NATIVE_CHAIN.json)、[复现说明](work/{STAGE}/REPRODUCTION.md)、[根复核](reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json)、[原A–F](PROJECT_ACCEPTANCE.md)。config.json继承的new_response_training_runs=0仅为非生效历史元数据；真实两次训练以原生回执及TRAINING_COMPACT为准，冻结配置未为美化计数而改写。

所有本机新增数据/源码/日志/缓存/临时文件保留E/D；原始和封存资料未覆盖。科学产物在E及已用服务器新目录保存；原始输入为既有依赖，D增量不应宣称重新归档了所有外部原始数据。恢复入口并不代表执行过恢复测试。模型目标仍未完成。
'''
(ROOT / 'R48_REPORT.md').write_text(report, 'utf8')
backup_dir = ('D:/GAN_backups/research/sim2real_two_papers_20261008_0305/'
              'r48_guided_noise_final_' + NOW.strftime('%Y%m%d_%H%M%S_UTC'))
index = {'stage': STAGE, 'status': 'science_sealed_publication_and_backup_pending',
    'created_UTC': NOW.isoformat(), 'report': 'R48_REPORT.md',
    'execution_plan': 'tasks/' + STAGE + '.md',
    'resource_receipt': PREFIX + 'RESOURCE_RECEIPT.json', 'model_goal_complete': False,
    'scientific_decision': decision, 'original_acceptance_A_F_unchanged': True,
    'private_Git': {'status': 'pending', 'repository': 'Aggressivecyber/ej276d-sim2real',
        'visibility': 'PRIVATE', 'receipt': 'R48_PUBLICATION_RECEIPT.json'},
    'backup': {'status': 'pending', 'directory': backup_dir, 'receipt': 'R48_BACKUP_RECEIPT.json',
        'mode': 'incremental_with_explicit_R44_R45_R46_R47_dependencies', 'standalone_full_snapshot': False},
    'bindings': [binding(n) for n in ['R48_REPORT.md', 'tasks/' + STAGE + '.md',
        'reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json', 'reviews/R48_ACCEPTANCE_OBSERVATION_001.json',
        PREFIX + 'FINAL_RECEIPT.json', PREFIX + 'FINAL_MANIFEST.json', PREFIX + 'RESOURCE_RECEIPT.json',
        PREFIX + 'TRAINING_COMPACT.json', PREFIX + 'TRAIN_SOLVER_CHECK.json', PREFIX + 'REPRODUCTION.md']]}
write('R48_DELIVERY_INDEX.json', index)
write('reviews/R48_RESOURCE_RECONCILIATION_001.json', {
    'UTC': NOW.isoformat(), 'actual_root_control_PID': os.getpid(),
    'replaced_provisional_R48_entry': previous_provisional,
    'final_R48_science_s': science, 'block_science_s': block['actual_science_seconds_recorded'],
    'method_plans': [3, 3], 'training_runs': [3, 6], 'provisional_plus_final_double_counted': False,
    'new_scientific_computation': False})
print(json.dumps({'status': 'R48_root_review_and_state_reconciled', 'PID': os.getpid(),
    'stage_science_s': science, 'block_science_s': block['actual_science_seconds_recorded'],
    'active_scientific_jobs': 0, 'model_goal_complete': False, 'backup_destination': backup_dir}, ensure_ascii=False))
