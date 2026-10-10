"""Register the fixed guided noise-completion comparison; administrative only."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import shutil

root = Path(__file__).resolve().parents[1]
stage = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
task = 'tasks/' + stage + '.md'
regname = 'reviews/R48_REGISTRATION_001'
reg = root / regname
now = datetime.now(timezone.utc)

def read(name):
    return json.loads((root / name).read_text('utf-8-sig'))

def save(name, obj):
    (root / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', 'utf8')

def bind(name):
    raw = (root / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

program = read('PROGRAM.json')
progress = read('PROJECT_PROGRESS.json')
budget = read('RESOURCE_BUDGET.json')
acceptance = read('PROJECT_ACCEPTANCE.json')
block = budget['improvement_block']
assert program['current_stage'] == 'R47_POPULATION_NOISE_DIAGNOSTIC_001'
assert program['active_scientific_jobs'] == 0
assert program['application_goal_status'] == 'active'
assert block['method_plans_registered'] == 2
assert block['response_training_runs_registered'] == 1
assert block['response_training_runs_started'] == 1
assert block['actual_science_seconds_recorded'] + 14400 <= block['total_science_seconds_max']
assert acceptance['scientific_threshold_version'] == 1
assert acceptance['numerical_protocol_version'] == 2
assert not reg.exists() and not (root / 'work' / stage).exists()
dfree = shutil.disk_usage('D:/').free
efree = shutil.disk_usage('E:/').free
assert dfree >= (256 + 160 + 50) * 2**20
assert efree >= 8 * 2**30
reg.mkdir()
snapshots = ['AGENTS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
             'PROJECT_OBJECTIVE.md', 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
             'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']
for name in snapshots:
    shutil.copy2(root / name, reg / name)
shutil.copy2(root / task, reg / 'PROTOCOL_DRAFT_BEFORE_REGISTRATION.md')

doc = (root / task).read_text('utf8')
old = '本任务在登记回执产生前仅为协议草案，不授权科学执行。'
new = ('本任务由 reviews/R48_REGISTRATION_001/RECEIPT.json 绑定后授权执行，'
       '以该回执的实际时间与预算为准。')
assert doc.count(old) == 1
doc = doc.replace(old, new)
doc += ('\n登记解释：本任务正文优先于只读咨询稿。损失固定为两项各0.5；'
        'PCG64 20261010481/482/483/484依次对应SOURCE t、SOURCE epsilon、'
        'IDENTITY t、IDENTITY epsilon。咨询稿的系数1和身份流次序不采用。'
        '控制先训练，随后主候选；两臂均为从零的50000更新末点。'
        '源流为 work/R43_SCALAR_TIME_NEURAL_FLOW_001/SCALAR_TIME_FLOW_FINAL/'
        'ACTUAL_CFM_STREAM_PRIVATE.npz；沿用其实际已耦合source/matched_target/ordinal，'
        '不再次排列。校准及实现检查也计入本轮科学秒。\n')
(root / task).write_text(doc, 'utf8')

# Refresh only current-status references; every scientific number is unchanged.
adoc = (root / 'PROJECT_ACCEPTANCE.md').read_text('utf-8-sig')
adoc = adoc.replace('当前 R46 固定 N=64', '当前 R48 沿用固定 N=64')
stale = ('应用内持续目标状态与科学状态分开。2026-10-10 本次读取应用返回无活动 goal；'
         '本文件不虚报 active，也不自动重建应用 goal。用户的继续指令仍授权当前明确范围的工作。')
fresh = ('应用内持续目标状态与科学状态分开。2026-10-10 07:10:59 UTC，用户明确要求'
         '“设置目标”后应用目标已实际创建为 active，回执见 '
         'reviews/PROJECT_GOAL_ACTIVATION_20261010_001/RECEIPT.json。'
         '模型仍须满足原 A–F；应用 active 不代表科学成功。')
assert adoc.count(stale) == 1
(root / 'PROJECT_ACCEPTANCE.md').write_text(adoc.replace(stale, fresh), 'utf8')
acceptance['document_sha256'] = bind('PROJECT_ACCEPTANCE.md')['sha256']
acceptance['stage_screen'] = ('Original R43 scientific thresholds; fixed64/reference128 '
    'numerical scheme is retained by R48 for both new endpoints. No tolerance lowered.')
acceptance['previous_application_goal_observation'] = acceptance['application_goal_observation']
acceptance['application_goal_observation'] = {
    'value': 'active', 'created_UTC': '2026-10-10T07:10:59+00:00',
    'observed_UTC': '2026-10-10T07:30:27+00:00', 'explicit_user_request': '设置目标',
    'receipt': 'reviews/PROJECT_GOAL_ACTIVATION_20261010_001/RECEIPT.json'}
acceptance['numerical_precision']['previous_protocol'] = acceptance['numerical_precision']['protocol']
acceptance['numerical_precision']['protocol'] = task
acceptance['current_status_reconciliation_UTC'] = now.isoformat()
save('PROJECT_ACCEPTANCE.json', acceptance)

inputs = [task, 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
          'reviews/R48_DESIGN_CONSULT_001.md', 'reviews/R47_ROUTE_DECISION_001.md',
          'reviews/PROJECT_GOAL_ACTIVATION_20261010_001/RECEIPT.json',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/FINAL_MANIFEST.json',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/config.json',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/network.py',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/source_flow_core.py',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/UNIT_CHARGE_CONSTANTS.json',
          'work/R43_SCALAR_TIME_NEURAL_FLOW_001/UNIT_CHARGE_CONSTANTS.npz',
          'work/R46_FIXED_INTEGRATION_BUDGET_001/FINAL_MANIFEST.json']
stream = 'work/R43_SCALAR_TIME_NEURAL_FLOW_001/SCALAR_TIME_FLOW_FINAL/ACTUAL_CFM_STREAM_PRIVATE.npz'
assert (root / stream).is_file()
receipt = {
    'stage': stage, 'registered_UTC': now.isoformat(), 'root_control_PID': os.getpid(),
    'task': task, 'task_sha256': bind(task)['sha256'],
    'method_plan_index_in_current_block': 3, 'response_runs_reserved': 2,
    'primary_candidate': 'GUIDED_NOISE_COMPLETION', 'control': 'GUIDED_FIXED_NOISE_CONTROL',
    'execution_order': ['GUIDED_FIXED_NOISE_CONTROL', 'GUIDED_NOISE_COMPLETION'],
    'mechanism': 'Only initial innovation variance differs: tau2 versus max(tau2-v(input),0).',
    'common_framework': 'Full480 fixed guide, REAL identity trajectories, TRAIN colored kernel, fresh same initialization.',
    'loss_weights_source_identity': [0.5, 0.5],
    'PCG64_seed_map': {'source_t': 20261010481, 'source_epsilon': 20261010482,
                      'identity_t': 20261010483, 'identity_epsilon': 20261010484},
    'torch_initialization_seed': 20261010480,
    'training': {'updates_each': 50000, 'response_training_runs_max': 2,
                 'source_rows_per_update': 32, 'same_REAL_identity_rows_per_update': 32,
                 'original_stream_rows': 94400, 'original_batches_per_cycle': 2950,
                 'training_science_seconds_each_max': 5400,
                 'training_science_seconds_total_max': 10800, 'fresh_Adam_each': True,
                 'checkpoint_or_DEV_selection': False},
    'science_budget_s': 14400, 'executor_wall_budget_s': 19800, 'stage_wall_budget_s': 21600,
    'executor_seal_deadline_UTC': (now + timedelta(seconds=19800)).isoformat(),
    'stage_deadline_UTC': (now + timedelta(seconds=21600)).isoformat(),
    'output_limit_bytes_per_local_or_server_stage': 160 * 2**20,
    'local_disk_snapshot_bytes': {'D_free': dfree, 'E_free': efree},
    'D_backup_reserve_bytes': 256 * 2**20,
    'server': 'rylan@100.82.231.102',
    'server_project': '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305',
    'server_limits': budget['server_limits'], 'local_auxiliary_limits': budget['local_auxiliary_limits'],
    'training_device': 'CUDA', 'generation_evaluation_replay_device': 'CPU12',
    'source_stream_reference': {'path': stream, 'bytes': (root / stream).stat().st_size,
                                'provenance': 'R43 sealed manifest; no migration consistency test.'},
    'calibration_REAL': 'Only unique actual matched_target identities in the saved training stream.',
    'extra_REAL_G4_CAL_EVAL_decode': False,
    'solver': {'candidate_steps': 64, 'reference_steps': 128, 'solver': 'Heun',
               'TRAIN_rows_each_model': 128, 'chunk': 64,
               'median_LSB_max': 0.25, 'p95_LSB_max': 1.0, 'preADC_relative_q_max': 1e-5,
               'both_endpoints_required_before_DEV': True, 'additional_steps_search': False},
    'conditional_evaluation': {'SOURCE_each': 2000, 'same_treated_REAL_each': 2000,
               'joint_each': 1024, 'probe_each': 832, 'full_source_perturb_each': 128,
               'HGB_max': 40, 'density_slots_and_EM_max': 32,
               'classifier_fit_holdout_rows': [600, 250], 'bootstrap_replicates': 500,
               'independent_CPU_main_replay_rows': 8000},
    'input_bindings': [bind(p) for p in inputs],
    'acceptance_scientific_version': 1, 'numerical_protocol_version': 2,
    'original_A_F_and_source_population_unchanged': True,
    'historical_closed_recipes_reopened': False, 'unanswered_scope_question_assumed_approved': False,
    'max_concurrent_scientific_jobs': 1, 'migration_consistency_tests': False,
    'implementation_consult_deviations': 'Task overrides consult loss normalization and RNG ordering; consult is not execution authority.',
    'consult_closeout': 'Read-only final edit was reported 19.59 s late; zero scientific execution, retained as administrative overrun.',
    'success_condition': 'Primary passes original six stage gates and A-C, then separately register one frozen D/E.',
    'failure_condition': 'Close the fixed last method without parameter/architecture expansion; full model goal remains unmet.'}
save(regname + '/RECEIPT.json', receipt)
block['method_plans_registered'] = 3
block['response_training_runs_registered'] = 3
block['registered_stages'].append({'stage': stage, 'method_plan_index': 3,
    'response_runs_reserved': 2, 'science_budget_s': 14400, 'status': 'registered_not_started',
    'registration': regname + '/RECEIPT.json'})
block['current_family_status'] = 'new_fixed_guide_identity_noise_comparison_registered_old_recipes_closed'
block['latest_direction_decision'] = task
budget['updated_UTC'] = now.isoformat()
save('RESOURCE_BUDGET.json', budget)
location = {'mode': 'server_CUDA_training_CPU12_generation_evaluation_replay_local_auxiliary',
    'server': receipt['server'], 'CPU_threads': 12, 'GPU_count': 1,
    'job_tree_RSS_bytes': 48 * 2**30, 'GPU_allocated_and_reserved_max_bytes': 10 * 2**30,
    'preflight_required': True, 'local_auxiliary_CPU_threads': 4, 'local_files': 'E/D only'}
common = dict(current_stage=stage, current_task=task, current_activity='R48_registered_for_continuous_execution',
    status='R48_registered_model_goal_unmet', current_stage_progress='registered_for_executor',
    current_stage_registration=regname + '/RECEIPT.json', active_scientific_jobs=0,
    current_task_addendum=None, current_engineering_addendum=None,
    execution_location=location, current_stage_deadline_UTC=receipt['stage_deadline_UTC'],
    stage_deadline_UTC=receipt['stage_deadline_UTC'], executor_seal_deadline_UTC=receipt['executor_seal_deadline_UTC'],
    next_action='Existing executor implements and completes registered paired runs and conditional full evaluation; root reviews and delivers.',
    model_goal_complete=False, application_goal_status='active', next_prepared_task=None)
program.update(common, updated_at_utc=now.isoformat(),
    current_stage_scope='last_fixed_method_two_equal_budget_fresh_models',
    current_job={'stage': stage, 'status': 'registered_not_started', 'scientific_jobs': 0},
    latest_scientific_decision=receipt['mechanism'])
progress.update(common, updated_UTC=now.isoformat(), current_direction=receipt['mechanism'])
save('PROGRAM.json', program)
save('PROJECT_PROGRESS.json', progress)
experiments = read('experiments.json')
jobs = read('jobs.json')
assert not any(x['id'] == stage for x in experiments['experiments'])
assert not any(x.get('stage') == stage for x in jobs['jobs'])
experiments['experiments'].append({'id': stage, 'status': 'registered_for_executor',
    'task': task, 'registered_UTC': now.isoformat(), 'registration': regname + '/RECEIPT.json',
    'method_plan_index': 3, 'response_fits_max': 2, 'updates_each_max': 50000,
    'science_budget_s': 14400, 'scientific_question': receipt['mechanism'],
    'DEV_used_for': 'One fixed endpoint per arm only after both TRAIN numerical gates pass.', 'EVAL_access': False})
jobs['jobs'].append({'stage': stage, 'kind': 'paired_fresh_guided_noise_completion_models',
    'status': 'registered_for_executor', 'registered_UTC': now.isoformat(),
    'registration': regname + '/RECEIPT.json', 'max_concurrent_scientific_jobs': 1, 'actual_native_jobs': []})
save('experiments.json', experiments)
save('jobs.json', jobs)
intro = ('# R48 已登记：完整 guide、REAL 身份轨迹与噪声补全对照\n\n'
    '应用目标 active；当前执行 '+task+'。固定两臂各从零50000更新，只有已有噪声扣除规则不同。'
    '方法名额3/3，新增训练预留2次；科学4小时、执行器5.5小时、整轮6小时，新增产物各160MiB。'
    '仅既有 sim2real_sol_max_executor 连续执行，一次一个科学作业。'
    '原 A–F、完整源人口、历史负结果和 E/D 存储约束保持；固定主候选未达 A–C 则关闭本方案，'
    '不扩预算或选对照替代。实际登记时间及截止见 '+regname+'/RECEIPT.json。模型目标尚未完成。\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    (root / name).write_text(intro + (root / name).read_text('utf-8-sig'), 'utf8')
print(json.dumps({'stage': stage, 'registered_UTC': now.isoformat(), 'PID': os.getpid(),
    'task_sha256': receipt['task_sha256'], 'executor_deadline_UTC': receipt['executor_seal_deadline_UTC'],
    'stage_deadline_UTC': receipt['stage_deadline_UTC'], 'method_count': 3, 'reserved_runs_total': 3,
    'started_runs_unchanged': block['response_training_runs_started'],
    'science_seconds_unchanged': block['actual_science_seconds_recorded'], 'D_free_bytes': dfree}))
