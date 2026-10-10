"""Register the fixed 128/256 protocol under the user's no-compute-budget instruction."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_FROZEN_HEUN128_RESPONSE_001'
TASK = 'tasks/' + STAGE + '.md'
CONSULT = 'reviews/R49_NUMERICAL_DESIGN_CONSULT_001.md'
OLD = 'work/R48_GUIDED_NOISE_COMPLETION_FLOW_001/'
REG = 'reviews/R49_REGISTRATION_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def bind(name):
    raw = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
budget, acceptance = read('RESOURCE_BUDGET.json'), read('PROJECT_ACCEPTANCE.json')
assert (ROOT / CONSULT).is_file() and (ROOT / CONSULT).stat().st_size > 0
assert program['active_scientific_jobs'] == 0 and program['current_stage'] == 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
assert budget['resource_version'] == 3 and not budget['compute_budget_enforced']
assert budget['improvement_block']['method_plans_max'] is None
assert read(OLD + 'FINAL_RECEIPT.json')['status'] == 'sealed'
assert not (ROOT / REG).exists() and not (ROOT / 'work' / STAGE).exists()
manifest = {r['path']: r for r in read(OLD + 'FINAL_MANIFEST.json')['files']}
frozen = []
for name in ['GUIDED_FIXED_NOISE_CONTROL/FINAL_FLOW.pt', 'GUIDED_NOISE_COMPLETION/FINAL_FLOW.pt',
             'NOISE_CALIBRATION.npz', 'network.py', 'noise_core.py', 'source_flow_core.py']:
    row = bind(OLD + name)
    assert row['sha256'] == manifest[name]['sha256'] and row['bytes'] == manifest[name]['bytes']
    frozen.append(row)

(ROOT / REG).mkdir()
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
             'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json', 'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']:
    shutil.copy2(ROOT / name, ROOT / REG / name)
shutil.copy2(ROOT / TASK, ROOT / REG / 'PROTOCOL_DRAFT_BEFORE_REGISTRATION.md')
doc = (ROOT / TASK).read_text('utf8')
old_text = '本任务在登记回执产生前为方案草案，尚未授权科学执行。'
assert doc.count(old_text) == 1
(ROOT / TASK).write_text(doc.replace(old_text,
    '本任务由 reviews/R49_REGISTRATION_001/RECEIPT.json 绑定后授权现有执行器连续完成，不设计算预算或截止时间。'), 'utf8')

adoc = ROOT / 'PROJECT_ACCEPTANCE.md'
content = adoc.read_text('utf8').replace('当前 R48 沿用固定 N=64', '当前 R49 固定 N=128')
content = content.replace('当前R49数值方案须单独登记后才执行。', '当前R49的128/256数值方案已单独登记，按其协议执行。')
content += ('\n## 2026-10-10 R49 前瞻性数值方案\n\n'
    '用户明确要求下一轮且不设置预算。R48两组固定50000更新权重保持不变，R49共同采用Heun128候选，'
    '256步仅作同一组128条TRAIN输入的参考；中位0.25 LSB、P95 1 LSB、preADC相对电荷1e-5容差不变。'
    '两组都通过才执行完整DEV评价和8000条独立主输出重放。R48原64/128失败保留，不追改历史。'
    '原A/B/D/E、六项科学门和源人口/权重保持；本轮没有新训练、重新校准、步数扫描或对照赢家选择。'
    '任务与登记回执：tasks/R49_FROZEN_HEUN128_RESPONSE_001.md、reviews/R49_REGISTRATION_001/RECEIPT.json。\n')
adoc.write_text(content, 'utf8')
previous_numeric = acceptance['numerical_precision'].copy()
acceptance.update(version=4, numerical_protocol_version=3, numerical_revision_UTC=NOW,
    numerical_revision_reason='New user-authorized frozen R48 weight comparison at common128/reference256 before R49 science.',
    previous_numerical_precision=previous_numeric,
    stage_screen='Original R43 scientific gates; R49 common Heun128/reference256 and original numerical tolerances.',
    document_sha256=bind('PROJECT_ACCEPTANCE.md')['sha256'], current_status_reconciliation_UTC=NOW)
acceptance['numerical_precision'].update(protocol=TASK, previous_protocol=previous_numeric['protocol'],
    candidate_steps=128, reference_steps=256, further_step_search_authorized=False,
    historical64step_conclusions_changed=False, numeric_tolerances_lowered=False)
save('PROJECT_ACCEPTANCE.json', acceptance)

block = budget['improvement_block']
block['method_plans_registered'] += 1
assert block['method_plans_registered'] == 4
block['registered_stages'].append({'stage': STAGE, 'method_plan_index': 4, 'response_runs_reserved': 0,
    'science_budget_s': None, 'status': 'registered_not_started', 'registration': REG + '/RECEIPT.json',
    'scientific_design': 'Frozen R48 pair, common128/reference256; no fresh response fit.',
    'user_authorization': '推进下一轮 不设置预算'})
block.update(current_family_status='R49_frozen_numerical_scheme_registered_without_compute_budget',
    latest_direction_decision=TASK, further_scientific_execution_authorized=True)
budget['updated_UTC'] = NOW
save('RESOURCE_BUDGET.json', budget)

mechanism = 'Only common numerical integration changes to Heun128; retain the frozen R48 primary/control and all scientific gates.'
for data in [program, progress]:
    data.update(updated_UTC=NOW, updated_at_utc=NOW,
        status='R49_registered_unbudgeted_science_not_started_model_goal_unmet',
        current_stage=STAGE, current_task=TASK, current_stage_registration=REG + '/RECEIPT.json',
        current_stage_scope='frozen_R48_pair_common_Heun128_numerical_and_conditional_full_response',
        current_activity='R49_executor_implementation_and_preflight', current_stage_progress='registered_not_started',
        current_direction=mechanism, current_route_decision=TASK, current_route_research_closed=False,
        next_action='Existing executor continuously implements and completes the registered R49 protocol without compute quotas.',
        next_prepared_task=None, queued_stage_registration=None, active_scientific_jobs=0,
        current_job={'stage': STAGE, 'status': 'registered_not_started'},
        current_executor_ack='work/' + STAGE + '/EXECUTOR_ACK.json',
        stage_deadline_UTC=None, current_stage_deadline_UTC=None, executor_seal_deadline_UTC=None,
        acceptance_version=4, numerical_protocol_version=3,
        model_goal_complete=False, compute_budget_enforced=False)
program.update(project_model_goal_complete=False, numeric_execution_released=True,
    latest_scientific_decision='R48 failure preserved; R49 prospective128/256 numerical scheme registered before execution.')
save('PROGRAM.json', program)
save('PROJECT_PROGRESS.json', progress)
job = {'stage': STAGE, 'kind': 'frozen_response_numerical_scheme_and_conditional_full_evaluation',
    'status': 'registered_not_started', 'registered_UTC': NOW, 'registration': REG + '/RECEIPT.json',
    'max_concurrent_scientific_jobs': 1, 'compute_budget_enforced': False,
    'actual_native_jobs': [], 'actual_science_seconds': 0, 'active_scientific_jobs': 0}
data = read('jobs.json')
assert all(row.get('stage') != STAGE for row in data['jobs'])
data['jobs'].append(job)
save('jobs.json', data)
data = read('experiments.json')
assert all(row.get('id') != STAGE for row in data['experiments'])
data['experiments'].append({'id': STAGE, 'status': 'registered_not_started', 'task': TASK,
    'registered_UTC': NOW, 'registration': REG + '/RECEIPT.json', 'method_plan_index': 4,
    'planned_new_response_training_runs': 0, 'planned_new_response_updates': 0,
    'science_budget_s': None, 'scientific_question': mechanism,
    'DEV_used_for': 'Fixed endpoint comparison only after both original TRAIN numerical checks pass.',
    'EVAL_access': False, 'compute_budget_enforced': False})
save('experiments.json', data)

notice = ('# R49 已登记：冻结两组权重，以128/256步检查推进\n\n'
    '用户授权“推进下一轮 不设置预算”已生效。现执行tasks/R49_FROZEN_HEUN128_RESPONSE_001.md，'
    '只改变R48两组固定权重的共同积分精度，不重新训练或校准。两组TRAIN数值门通过后由既有执行器'
    '连续完成原完整响应/源保真/REAL控制和重放；原A–F不降，旧失败不改。'
    '没有方法/训练次数、科学/墙钟耗时、截止或阶段体积配额；安全资源、单作业和E/D约束保留。'
    '实际登记见reviews/R49_REGISTRATION_001/RECEIPT.json，模型目标尚未完成。\n\n'
    '## 以下为授权与历史状态\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    p = ROOT / name
    p.write_text(notice + p.read_text('utf8'), 'utf8')

receipt = {'stage': STAGE, 'registered_UTC': NOW, 'actual_root_control_PID': os.getpid(),
    'task': TASK, 'task_sha256': bind(TASK)['sha256'], 'consultation': CONSULT,
    'user_instruction': '推进下一轮 不设置预算', 'compute_budget_enforced': False,
    'science_budget_s': None, 'wall_budget_s': None, 'deadline_UTC': None, 'stage_output_budget_bytes': None,
    'numerical_protocol_version': 3, 'scientific_threshold_version': 1, 'resource_version': 3,
    'primary': 'GUIDED_NOISE_COMPLETION', 'control': 'GUIDED_FIXED_NOISE_CONTROL',
    'candidate_steps': 128, 'reference_steps': 256, 'CPU_threads': 12, 'chunk': 64,
    'TRAIN_rows_each_model': 128, 'RMS_median_LSB_max': .25, 'RMS_p95_LSB_max': 1.,
    'preADC_relative_q_max': 1e-5, 'both_required_before_DEV': True,
    'planned_new_training_runs': 0, 'planned_new_response_updates': 0, 'historical_updates_each': 50000,
    'planned_DEV_generated_rows_if_numerical_pass': 11968, 'planned_HGB_fits': 40,
    'planned_density_slots': 32, 'planned_main_replay_rows': 8000,
    'primary_control_switch_allowed': False, 'higher_step_search_in_this_protocol': False,
    'frozen_local_version_bindings': frozen,
    'protocol_bindings': [bind(n) for n in [TASK, CONSULT, 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
        'SERVER_RESOURCE_POLICY.md', 'reviews/NO_COMPUTE_BUDGET_20261010_001/RECEIPT.json',
        OLD + 'FINAL_RECEIPT.json', OLD + 'TRAINING_COMPACT.json', OLD + 'TRAIN_SOLVER_CHECK.json']],
    'local_disk_free_at_registration': {d: shutil.disk_usage(d + ':/').free for d in ['E', 'D']},
    'historical_scientific_stages_modified': False, 'new_scientific_work_started_by_root': False,
    'scientific_thresholds_lowered': False, 'project_model_goal_complete': False}
save(REG + '/RECEIPT.json', receipt)
print(json.dumps({'status': 'R49_registered', 'PID': os.getpid(), 'UTC': NOW, 'task_sha256': receipt['task_sha256'],
    'compute_budget_enforced': False, 'candidate_reference_steps': [128, 256],
    'new_training_runs': 0, 'science_started': False}, ensure_ascii=False))
