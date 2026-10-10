"""Record completed R47 evidence and reconcile current control ledgers."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R47_POPULATION_NOISE_DIAGNOSTIC_001'
PREFIX = 'work/' + STAGE + '/'
NOW = datetime.now(timezone.utc)


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def binding(name):
    payload = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


assert not (ROOT / 'reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json').exists()
final = read(PREFIX + 'FINAL_RECEIPT.json')
resource = read(PREFIX + 'RESOURCE_RECEIPT.json')
qa = read(PREFIX + 'FIGURE_QA.json')
assert final['status'] == 'sealed' and final['no_further_stage_writes']
assert final['stage_files_total'] == 112 and final['stage_bytes_total'] == 6553416
assert all(job['native_exit'] == 0 for job in resource['scientific_jobs'])
assert resource['scientific_counts']['HGB_completed'] == 5
assert resource['scientific_counts']['response_training_updates'] == 0
assert resource['scientific_elapsed_s'] <= 900 and not resource['GPU_used']
assert qa['ROOT_actually_opened_all_six_and_reopened_two_repaired_figures']
seals = [binding(PREFIX + name) for name in ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json']]
assert seals[0]['sha256'] == final['FINAL_MANIFEST_sha256']
assert seals[1]['sha256'] == '807654762f059c767b24eb7b08d33d258c295eb276ea8494053095a4c7b848e5'
science = resource['scientific_elapsed_s']
direction = ('Define one input-noise-aware complete-source response operator and its matched control before '
             'registering the final method plan. Preserve original A-F and original source weights; '
             'R47 is diagnostic evidence, not an accepted response model.')
review = {
    'stage': STAGE, 'reviewed_UTC': NOW.isoformat(), 'root_control_PID': os.getpid(),
    'status': 'diagnostic_evidence_reviewed_no_response_candidate',
    'review_scope': 'Read actual metrics, native/resource receipts and six rendered figures; no scientific rerun.',
    'root_read_native_tools': ['0b714d', '8d269b', 'fcfbe7', '6dff03', 'ecec27', '308bd6'],
    'root_read_exception': {'tool': '8d269b', 'native_exit': 1,
        'reason': 'Read-only summary assumed REAL_DEV had the same nesting as generated paths.',
        'corrected_tool': 'fcfbe7', 'corrected_native_exit': 0, 'scientific_data_changed': False},
    'scientific_jobs': resource['scientific_jobs'], 'scientific_elapsed_s': science,
    'scientific_counts': resource['scientific_counts'], 'resource_limits_satisfied': True,
    'all_original_rows_retained': True, 'source_particle_composition_is_not_truth': True,
    'charge_only_BA': {'SOURCE': 0.586, 'LONG': 0.590, 'RAW_TRAIN_REFERENCE': 0.478},
    'charge_only_source_and_long_decision': 'uncertain_under_preregistered_95pct_CI_rule',
    'charge_only_not_a_lower_bound_on_joint_classifier': True,
    'REAL_reference_baseline_RMS_mV': {'RAW': 0.645992, 'INITIAL_POST_ADC': 1.193383,
                                      'SHORT': 1.234196, 'LONG': 1.537006},
    'noise_finding': 'Initialization changes REAL noise amplitude; learned paths further alter amplitude and correlation.',
    'initial_REAL_minus_RAW_baseline_BA': {'estimate': 0.05, 'CI95': [-0.0198266, 0.1100447],
        'interpretation': 'Conditional paired interval crosses zero; normalized local classifier omits absolute RMS.'},
    'functional_variance_decomposition': 'All four variances and six signed 2Cov terms retained; not independent physical noise or causal attribution.',
    'metadata_only_correction': PREFIX + 'METADATA_CORRECTION_RECEIPT.json',
    'metadata_correction_refit_count': 0, 'figures_actually_opened': 6, 'repaired_figures_reopened': 2,
    'figure_QA_receipt': PREFIX + 'FIGURE_QA.json', 'sealed_bindings': seals,
    'stage_file_count': final['stage_files_total'], 'stage_bytes': final['stage_bytes_total'],
    'old_method_family_stays_closed': True, 'original_A_F_unchanged': True,
    'optional_scope_question_unanswered_original_scope_retained': True,
    'new_response_model_candidate': False, 'project_model_goal_complete': False,
    'next_direction': direction, 'next_method_registered': False, 'active_scientific_jobs': 0,
    'scientific_publication_pending': True, 'D_incremental_backup_pending': True,
}
write('reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json', review)
write('reviews/R47_SEAL_NATIVE_RECEIPT.json', {
    'stage': STAGE, 'root_recorded_UTC': NOW.isoformat(),
    'executor_seal_PID': 21284, 'executor_native_tool_result': '89d335', 'executor_native_exit': 0,
    'native_completion_basis': 'Executor completed finalization; root separately read final receipts and recorded file bindings.',
    'root_post_seal_read_native_tool': '6dff03', 'root_post_seal_read_native_exit': 0,
    'finalization_was_control_not_scientific_job': True, 'stage_no_further_writes': True,
    'bindings': seals, 'stage_file_count': final['stage_files_total'], 'stage_bytes': final['stage_bytes_total']})

budget = read('RESOURCE_BUDGET.json')
block = budget['improvement_block']
assert not any(x['stage'] == STAGE for x in block['actual_stage_receipts'])
block['actual_stage_receipts'].append({'stage': STAGE, 'receipt': PREFIX + 'RESOURCE_RECEIPT.json',
    'science_s': science, 'response_training_runs_started': 0, 'new_response_updates': 0,
    'status': 'sealed_diagnostic_no_response_candidate', 'stage_science_complete': True})
block['actual_science_seconds_recorded'] = sum(x['science_s'] for x in block['actual_stage_receipts'])
block['diagnostic_science_seconds_recorded'] = sum(x['science_s'] for x in block['actual_stage_receipts']
    if x['stage'] in ['R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001', STAGE])
next(x for x in block['diagnostic_stages_registered'] if x['stage'] == STAGE).update(
    status='sealed_diagnostic_no_response_candidate', actual_science_seconds=science,
    HGB_completed=5, stage_science_complete=True, resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json')
block['latest_direction_decision'] = 'reviews/R47_ROUTE_DECISION_001.md'
budget['updated_UTC'] = NOW.isoformat()
write('RESOURCE_BUDGET.json', budget)

program = read('PROGRAM.json')
program.update(updated_at_utc=NOW.isoformat(), status='R47_sealed_root_delivery_in_progress_model_goal_unmet',
    current_stage_progress='sealed_key_results_and_figures_reviewed_publication_backup_pending',
    current_activity='R47_private_Git_and_D_incremental_delivery', active_scientific_jobs=0,
    latest_completed_stage=STAGE, latest_completed_report='R47_REPORT.md',
    latest_completed_review='reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json',
    latest_key_result_review='reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json',
    latest_numeric_receipt=PREFIX + 'RESOURCE_RECEIPT.json', latest_diagnostic_stage=STAGE,
    latest_diagnostic_conclusion='charge_classifier_uncertain_REAL_noise_changes_at_initialization_and_learned_path',
    latest_scientific_decision='R47_diagnostic_complete_no_new_response_candidate',
    current_direction=direction, current_route_decision='reviews/R47_ROUTE_DECISION_001.md',
    next_prepared_task=None, queued_stage_registration=None,
    next_action='Complete R47 delivery, then define the next fixed mechanism under the unchanged acceptance protocol.',
    final_report='R47_REPORT.md', delivery_index='R47_DELIVERY_INDEX.json',
    reproduction_guide=PREFIX + 'REPRODUCTION.md', publication_receipt='R47_PUBLICATION_RECEIPT.json',
    backup_receipt='R47_BACKUP_RECEIPT.json', delivery_completion_receipt='R47_DELIVERY_COMPLETION_RECEIPT.json',
    current_job={'stage': STAGE, 'status': 'sealed_no_live_job', 'scientific_start_observed': True,
                 'scientific_jobs_all_actual_native_exit_0': True},
    original_fixed_charge_flow_family_closed=True, current_route_research_closed=False,
    project_model_goal_complete=False)
program['git']['visibility_verified_at_utc'] = NOW.isoformat()
program['git']['visibility_verification_basis'] = 'Live gh repo view tool ecec27: PRIVATE'
write('PROGRAM.json', program)
progress = read('PROJECT_PROGRESS.json')
progress.update(updated_UTC=NOW.isoformat(), status=program['status'], latest_completed_stage=STAGE,
    current_activity=program['current_activity'], current_stage_progress=program['current_stage_progress'],
    latest_diagnostic_stage=STAGE, latest_diagnostic_conclusion=program['latest_diagnostic_conclusion'],
    current_direction=direction, current_route_decision='reviews/R47_ROUTE_DECISION_001.md',
    active_scientific_jobs=0, model_goal_complete=False)
write('PROJECT_PROGRESS.json', progress)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(name)
    row = next(x for x in data[key] if x.get(identity) == STAGE)
    row.update(status='sealed_diagnostic_root_delivery_in_progress', scientific_start_observed=True,
        active_scientific_jobs=0, scientific_jobs=resource['scientific_jobs'], scientific_elapsed_s=science,
        scientific_counts=resource['scientific_counts'], response_training_runs=0,
        resource_receipt=PREFIX + 'RESOURCE_RECEIPT.json', final_receipt=PREFIX + 'FINAL_RECEIPT.json',
        root_review='reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json', project_model_goal_complete=False)
    write(name, data)

current_text = ('# R47 已封存：完成新计划首轮诊断\n\n'
    'R47 已完成 5 个诊断 HGB、6 张实际复核图，科学 9.997501 秒，0 响应训练/ODE/EM。'
    '电荷一维判别按原规则为不确定；REAL 噪声在初始化与学习路径均出现变化。'
    '原 A–F、全部源权重和旧家族关闭保留。下一方法先固定已有噪声处理算子与等预算对照；'
    '尚无 R48 登记或后台响应训练。根正在完成私有 Git 与 D 盘增量交付，实际状态见 '
    'R47_DELIVERY_COMPLETION_RECEIPT.json。模型目标未完成。\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    path.write_text(current_text + path.read_text('utf-8-sig'), 'utf8')

backup_dir = ('D:/GAN_backups/research/sim2real_two_papers_20261008_0305/'
              'r47_population_noise_final_' + NOW.strftime('%Y%m%d_%H%M%S_UTC'))
index = {'stage': STAGE, 'status': 'science_sealed_publication_and_backup_pending',
    'created_UTC': NOW.isoformat(), 'report': 'R47_REPORT.md',
    'execution_plan': 'PROJECT_EXECUTION_PLAN_20261010_002.md',
    'next_direction': 'reviews/R47_ROUTE_DECISION_001.md', 'resource_receipt': PREFIX + 'RESOURCE_RECEIPT.json',
    'model_goal_complete': False, 'new_response_candidate': False, 'original_acceptance_unchanged': True,
    'private_Git': {'status': 'pending', 'repository': 'Aggressivecyber/ej276d-sim2real',
        'visibility': 'PRIVATE', 'receipt': 'R47_PUBLICATION_RECEIPT.json'},
    'backup': {'status': 'pending', 'directory': backup_dir, 'receipt': 'R47_BACKUP_RECEIPT.json',
        'mode': 'incremental_with_explicit_R44_R45_R46_dependencies', 'standalone_full_snapshot': False},
    'bindings': [binding(n) for n in ['R47_REPORT.md', 'PROJECT_EXECUTION_PLAN_20261010_002.md',
        'reviews/R47_ROUTE_DECISION_001.md', 'reviews/ROOT_R47_KEY_RESULT_REVIEW_001.json',
        PREFIX + 'FINAL_RECEIPT.json', PREFIX + 'FINAL_MANIFEST.json', PREFIX + 'RESOURCE_RECEIPT.json',
        PREFIX + 'METRICS_COMPACT.json', PREFIX + 'REPRODUCTION.md']]}
write('R47_DELIVERY_INDEX.json', index)
print(json.dumps({'status': 'root_science_review_recorded', 'PID': os.getpid(), 'science_s': science,
    'block_science_s': block['actual_science_seconds_recorded'],
    'diagnostic_science_s': block['diagnostic_science_seconds_recorded'],
    'stage_bytes': final['stage_bytes_total'], 'backup_destination': backup_dir}, ensure_ascii=False), flush=True)
