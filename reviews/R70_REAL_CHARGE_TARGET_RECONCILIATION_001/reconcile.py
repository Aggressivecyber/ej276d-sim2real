"""Apply the explicit REAL-charge target correction; metadata only, no science."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'reconciliation_runtime')

STAGE = P.name
now = datetime.now(timezone.utc).isoformat()
quote = '保留实际数据的电荷分布 因为他是真实的 仿真的目的是为了提供一个趋势参考 真实数据可以看作是仿真数据的扭曲，在数据缺少能区通过仿真可以弥补数据不足 所以符合真实电荷分布是我们要保留的'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_bytes() if p.exists() else b''
    newline = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    trailer = old[len(old.rstrip(b'\r\n')):] or newline
    body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf8').replace(b'\n', newline)
    p.write_bytes(body + trailer)


assert not (P / 'ROOT_RECONCILIATION_RECEIPT.json').exists(), 'Already applied; verify instead.'
parent = read(R / 'PROJECT_ACCEPTANCE.json')
old_hash = {
    'PROJECT_ACCEPTANCE.json': 'b0e4c13302dc67b0eec0799e77fdbab603e29a293cb32bbc26fa0734f025ac9e',
    'PROJECT_ACCEPTANCE.md': 'd6f68a1e902d901bcceafaa1d23f97b1a8ff5ad9f52acf46b00b82d7a6bdb04d',
}
for name, digest in old_hash.items():
    assert sha(R / name) == digest, name
assert parent['version'] == 4
goal = read(P / 'APP_GOAL_OBSERVATION.json')['result']['goal']
assert goal['status'] == 'paused'
for name in ('jobs.json', 'PROGRAM.json', 'PROJECT_PROGRESS.json'):
    assert read(R / name)['active_scientific_jobs'] == 0
managed = ['AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md',
           'LONG_TERM_GOAL_20261010.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
           'experiments.json', 'jobs.json', 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json']
snapshot = P / 'before_target_correction'
snapshot.mkdir()
before = []
for name in managed:
    shutil.copy2(R / name, snapshot / name)
    assert sha(R / name) == sha(snapshot / name)
    before.append({'path': name, 'bytes': (R / name).stat().st_size, 'sha256': sha(R / name)})
frozen = ['R69_REPORT.md', 'R69_DELIVERY_RECEIPT.json',
          'tasks/R69_SAMPLED_FULL_GRADIENT_PILOT_001.md',
          'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/source_flow_core.py',
          'tasks/R6_CHARGE_GAP_GENERALIZATION_001.md']
frozen_hash = {name: sha(R / name) for name in frozen}
write(P / 'BEFORE_MANIFEST.json', before)
write(P / 'USER_STEERING_RECEIPT.json', {
    'recorded_UTC': now, 'user_message_UTC': '2026-10-11T08:51:20.244571Z',
    'exact_user_text': quote, 'authority': 'latest direct human instruction in this conversation',
    'interpretation': 'REAL empirical Q is the target; SOURCE Q may be transformed by a learned response; simulations supply conditional trends and missing-region hypotheses.',
    'supersedes': ['hard per-event preservation of original SOURCE Q as a final response target',
                   'the historical two-domain gap as the sole test of REAL-only missing-region supplementation'],
    'does_not_authorize': ['changing historical results', 'changing REAL reference values or weights',
                          'dropping SOURCE parents', 'using held-out REAL for fitting',
                          'claiming measured frequency in truly unobserved regions', 'automatically resuming R65'],
    'before_manifest': 'reviews/' + STAGE + '/BEFORE_MANIFEST.json',
    'science_runs_started': 0,
})

md = R / 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md'
current = {
    'version': 5, 'defined_UTC': now, 'status': 'prospective_REAL_charge_target_not_yet_model_validated',
    'authority': 'latest direct user correction recorded in USER_STEERING_RECEIPT.json',
    'document': md.name, 'document_sha256': sha(md),
    'base_acceptance': {'version': 4, 'registry': 'PROJECT_ACCEPTANCE.json',
                        'document': 'PROJECT_ACCEPTANCE.md', 'sha256': old_hash,
                        'unchanged_historical_bytes': True, 'retroactive_reclassification': False},
    'scientific_threshold_version': parent['scientific_threshold_version'],
    'response_target_version': 5,
    'user_supplied_numeric_thresholds': False,
    'physical_equivalence_claim': False, 'historically_exposed_development_data': True,
    'final_response': copy.deepcopy(parent['final_response']),
    'utility': copy.deepcopy(parent['utility']),
    'charge_target': {
        'reference': 'unchanged legal REAL identity, raw canonical Q and original empirical weights',
        'fit_reference_split': 'TRAIN only; holdout and GAP-hidden REAL excluded',
        'SOURCE_original_Q_hard_preservation': False,
        'SOURCE_parent_identity_composition_order_and_weight_preserved': True,
        'same_public_response_path_for_SOURCE_and_REAL': True,
        'REAL_identity_bypass_or_domain_flag_allowed': False,
        'target_condition_axis': 'canonical output Q under unchanged REAL bin boundaries',
        'required_row_fields': ['parent_ID', 'input_Q', 'predicted_Q_target', 'preADC_Q', 'canonical_ADC_Q',
                                'original_bin', 'predicted_target_bin', 'preADC_bin', 'ADC_bin',
                                'input_metadata', 'weight', 'random_stream'],
        'mandatory_Q_diagnostics': ['empirical_CDF', 'KS', 'W1', 'counts', 'weights', 'outside_support_mass'],
        'Q_acceptance_binding': 'original fixed Q_PSD_2D BA gate plus full original five-view joint gate',
        'new_Q_diagnostic_threshold_chosen_after_results': False,
        'true_missing_region_event_rate': 'unknown absent independent physical evidence',
    },
    'numerical_precision': {
        'protocol': 'new candidate-specific protocol must bind reference before scientific execution',
        'candidate_steps': None, 'reference_steps': None,
        'median_LSB_max': 0.25, 'p95_LSB_max': 1.0, 'LSB_V': 1 / 16383,
        'preADC_relative_Q_target_error_max': 1e-5,
        'relative_Q_target_error_definition': 'abs(Q_preADC-Q_predicted)/abs(Q_predicted)',
        'Q_predicted_independent_of_output_integral': True,
        'nonpositive_or_nonfinite_Q_predicted_is_failure': True,
        'old_SOURCE_Q_conservation_gate_replaced': True,
        'old_GATE_reclassified_as_pass': False,
        'independent_main_replay_rows_per_model': 4000,
        'iterative_solver_reference': 'preregistered N/2N, unchanged tolerances',
        'noniterative_module_reference': 'separately preregistered independent precision reference; no fictional N/2N result',
        'science_execution_registered': False,
    },
    'source_and_boundary': {
        'original_source_parent_count': 2000, 'original_outside_support_count_retained': 417,
        'Q_and_PSD_Spearman_max_decline': 0.05,
        'original_purehad_valley_max_increase': 0.02,
        'source_audit_condition': 'unchanged original nominal/source-parent conditions',
        'original_perturbation_nine_boundary_and_remaining_stage_gates_required': True,
        'target_bin_migration_additionally_required': True,
    },
    'scope': {
        'comparison': 'one frozen FULL versus REAL-only GAP study with complete legal SOURCE retained',
        'gap_half_open_mV_ns': [3703.7782948464155, 6711.804904974998],
        'hide_GAP_REAL_TRAIN_and_all_derived_statistics': True,
        'retain_all_legal_SOURCE_TRAIN_including_nominal_gap': True,
        'GAP_may_borrow_FULL_weights_or_gap_REAL_statistics': False,
        'unobserved_mask_means_zero_probability': False,
        'query_mass_mask_and_loss_require_new_protocol_binding': True,
        'matched_method_training_and_compute_except_REAL_availability': True,
        'required_reporting': ['all_original_nominal_gap_parents', 'all_self_assigned_output_gap',
                               'complete_migration', 'counts_and_weights', 'support_failures', 'outside_gap_controls'],
        'validated_region_requires_A_C': True,
        'failed_gap_can_be_reported_as_successful_completion': False,
        'historical_R6_reclassified': False,
        'physical_incident_energy_calibration': 'unknown',
    },
    'frozen_validation': copy.deepcopy(parent['frozen_validation']),
    'improvement_block': copy.deepcopy(parent['improvement_block']),
    'resource_policy': 'SERVER_RESOURCE_POLICY.md', 'resource_registry': 'RESOURCE_BUDGET.json',
    'resource_version': parent['resource_version'],
    'positive_completion': 'Same frozen method satisfies prospectively corrected A-F and actual delivery; report gap success separately and require evidence for any supplementation claim.',
    'negative_closeout': parent['negative_closeout'],
    'historical_R1_R69_results_changed': False, 'model_goal_complete': False,
    'steering_receipt': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
}
current['final_response']['noise_and_source_and_boundary_and_replay'] = 'Original requirements remain except the explicitly replaced SOURCE-Q conservation target; see source_and_boundary and numerical_precision.'
write(R / 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json', current)
write(R / 'CURRENT_ACCEPTANCE.json', {
    'version': 5, 'effective_UTC': now,
    'document': md.name, 'document_sha256': sha(md),
    'registry': 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
    'registry_sha256': sha(R / 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json'),
    'historical_base_document': 'PROJECT_ACCEPTANCE.md',
    'historical_base_registry': 'PROJECT_ACCEPTANCE.json',
    'latest_human_correction_receipt': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
    'application_goal_text_may_still_name_v4': True,
    'latest_direct_user_instruction_controls_conflicting_historical_goal_text': True,
})

header = '''# 当前目标修正：实测电荷谱为准，仿真提供条件趋势

2026-10-11 用户明确指定保留实测电荷分布。后续新方法按 [验收 v5](PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md) 与 `CURRENT_ACCEPTANCE.json` 执行：允许仿真事件经响应改变 Q，保留原父事件/组成/权重；输出条件按 canonical Q 归区，原名义条件另作来源审计。缺口验证只隐藏 REAL，保留该区仿真。原 A/B/D 等数字保持，原 SOURCE 逐事件 Q 守恒门明确替换；v4 和所有旧失败不改。

R70 完成静态目标协调，下一方向为一个公共条件 Q-head 小试验，尚未预登记或运行。R65 保持停止，R69 固定 K4 负结果已交付。原模型目标及 v5 模型目标均未完成。应用目标本轮实际读取为 paused，本轮未改变状态或启动科学作业。当前源码/数据/日志/cache/temp 只写 E/D。主助手负责方向、审查和 Git；既有辅助静态审计按用户最新模型政策使用 gpt-6.1-sol/max，覆盖下文旧同模型/单助手条款。

证据：`R70_REPORT.md` 与 `reviews/R70_REAL_CHARGE_TARGET_RECONCILIATION_001/ROOT_RECONCILIATION_RECEIPT.json`；实际交付见 `R70_DELIVERY_RECEIPT.json`。新科学工作须在一次性明确输入、方法、端点和资源后登记；无需逐步骤重新请求确认。

## 以下为历史记录

'''
for name in ('AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md'):
    old = (R / name).read_bytes()
    newline = b'\r\n' if old.count(b'\r\n') > old.count(b'\n') // 2 else b'\n'
    (R / name).write_bytes(header.encode('utf8').replace(b'\n', newline) + old)

for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
    data = read(R / name)
    data['last_completed_scientific_job'] = data.get('current_job')
    data['latest_completed_scientific_stage'] = 'R69_SAMPLED_FULL_GRADIENT_PILOT_001'
    data['historical_frozen_method_reference'] = {'path': data.get('frozen_method'), 'sha256': data.get('frozen_method_sha256')}
    data.update({
        'updated_UTC': now, 'updated_at_utc': now,
        'status': 'R70_static_target_reconciliation_complete_model_unmet',
        'project_status': 'incomplete_user_target_corrected', 'application_goal_status': goal['status'],
        'application_goal_status_observed_UTC': read(P / 'APP_GOAL_OBSERVATION.json')['observed_UTC'],
        'application_goal_status_note': 'Fresh tool returns paused; no state change by this R70 turn.',
        'goal_status_note': 'Latest direct user REAL-charge target correction is implemented prospectively; no completed response model.',
        'current_stage': STAGE, 'current_task': 'tasks/' + STAGE + '.md',
        'current_stage_scope': 'Static user-target and acceptance reconciliation only',
        'current_stage_progress': 'completed_static_not_scientific_success',
        'current_stage_registration': None, 'current_stage_response_training_runs_started': 0,
        'current_activity': 'R70_target_correction_and_static_preparation_complete',
        'current_priority': 'REAL empirical charge is the response target; SOURCE provides conditional trend',
        'current_direction': 'One common conditional charge-response head, then conditional waveform response if supported',
        'next_action': 'Preregister one bounded TRAIN-only common-Q-head compatibility experiment before any new scientific execution',
        'next_prepared_task': 'common_conditional_Q_head_compatibility_not_registered',
        'current_route_research_closed': False,
        'current_route_decision': 'reviews/' + STAGE + '/ROOT_RECONCILIATION_RECEIPT.json',
        'current_policy_reconciliation': 'reviews/' + STAGE + '/ROOT_RECONCILIATION_RECEIPT.json',
        'latest_user_revision': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
        'latest_user_steering': quote,
        'latest_user_steering_receipt': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
        'latest_permission_status': 'Explicit target correction; current task is static and launches no science',
        'acceptance_document': md.name, 'acceptance_registry': 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
        'acceptance_version': 5, 'acceptance_pointer': 'CURRENT_ACCEPTANCE.json',
        'original_acceptance_version': 4, 'original_acceptance_bytes_unchanged': True,
        'original_A_F_complete': False, 'full_A_F_proven': False,
        'model_goal_complete': False, 'project_model_goal_complete': False,
        'complete_response_joint_scientific_acceptance': False,
        'active_scientific_jobs': 0, 'active_local_scientific_jobs': 0, 'active_remote_scientific_jobs': 0,
        'current_job': None, 'numeric_execution_released': False,
        'current_model_fit_role': 'No v5 model fit or inference performed',
        'frozen_method': None, 'frozen_method_sha256': None,
        'latest_completed_stage': STAGE, 'latest_completed_report': 'R70_REPORT.md',
        'latest_completed_review': 'reviews/' + STAGE + '/ROOT_STATIC_VERIFICATION.json',
        'latest_scientific_decision': 'No new scientific result; explicit user target correction supersedes original SOURCE-Q hard preservation.',
        'latest_diagnostic_stage': 'R69_SAMPLED_FULL_GRADIENT_PILOT_001',
        'latest_diagnostic_conclusion': 'R69 gradient admission passed; joint equal-cost benefit failed (NEGATIVE_LOCAL_SUPPORT).',
        'latest_response_model_stage_role': 'Historical R63 frozen response under v4; no v5 candidate yet',
        'current_engineering_addendum': 'R70 Q-response target and conditioning migration; R65 remains retired on cost hold',
        'execution_policy': 'No scientific execution in R70; register one fixed bounded benefit test before future launch',
        'execution_coordination': 'Root target synthesis and delivery; two scoped static audits',
        'latest_user_agent_policy': 'Auxiliary analysis uses gpt-6.1-sol/max with root leading and reviewing.',
        'latest_user_agent_policy_receipt': 'reviews/R65_THROUGHPUT_OPTIMIZATION_001/user_steering/RECEIPT.json',
        'subagents_allowed': True,
        'delivery_status': 'R70_pending_actual_Git_and_D_delivery',
    })
    data['latest_target_reconciliation'] = {
        'stage': STAGE, 'science_runs': 0, 'old_SOURCE_Q_hard_preservation_replaced': True,
        'REAL_reference_changed': False, 'SOURCE_parent_weights_changed': False,
        'historical_results_changed': False, 'new_model_acceptance_evaluated': False,
    }
    for item in data.get('requirements', []):
        item['evidence_acceptance_version'] = 4
        item['current_v5_status'] = 'not_evaluated_no_v5_candidate'
    write(R / name, data)

exp = read(R / 'experiments.json')
assert not any(x.get('id') == STAGE for x in exp['experiments'])
exp['experiments'].append({
    'id': STAGE, 'kind': 'static_target_reconciliation', 'scientific_execution': False,
    'status': 'completed_static', 'defined_UTC': now, 'completed_UTC': now,
    'task': 'tasks/' + STAGE + '.md', 'report': 'R70_REPORT.md',
    'acceptance_registry': 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
    'user_steering_receipt': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
    'science_array_reads': 0, 'model_fits': 0, 'model_inferences': 0,
    'model_goal_complete': False, 'delivery_receipt': 'R70_DELIVERY_RECEIPT.json',
})
exp['updated_UTC'] = now
write(R / 'experiments.json', exp)
for name, digest in {**old_hash, **frozen_hash}.items():
    assert sha(R / name) == digest, name
assert sha(R / 'jobs.json') == sha(snapshot / 'jobs.json')
receipt = {
    'UTC': now, 'stage': STAGE, 'status': 'static_target_reconciliation_applied',
    'authority_receipt': 'reviews/' + STAGE + '/USER_STEERING_RECEIPT.json',
    'new_registry': 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json', 'new_registry_SHA256': sha(R / 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json'),
    'new_document': md.name, 'new_document_SHA256': sha(md),
    'original_acceptance_SHA256': old_hash, 'historical_frozen_SHA256': frozen_hash,
    'historical_files_unchanged': True, 'jobs_registry_unchanged': True,
    'source_parent_weights_changed': False, 'real_reference_changed': False,
    'old_source_Q_conservation_replaced_prospectively': True,
    'old_double_domain_gap_replaced_prospectively': True,
    'original_response_and_utility_numeric_thresholds_retained': True,
    'no_scientific_arrays_read': True, 'new_fit_inference_and_science_jobs': 0,
    'app_goal_observation': 'APP_GOAL_OBSERVATION.json', 'app_goal_changed_by_this_turn': False,
    'model_goal_complete': False, 'next_scientific_task_registered': False,
    'R65_resumed': False, 'resource': {**guard, **memory()},
}
write(P / 'ROOT_RECONCILIATION_RECEIPT.json', receipt)
print(json.dumps({'status': receipt['status'], 'version': 5, 'old_files_unchanged': True,
                  'new_science_jobs': 0, 'application_goal_status': goal['status'], 'resource': memory()}, ensure_ascii=False))
