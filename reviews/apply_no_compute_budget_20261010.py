"""Apply the user's explicit no-compute-budget instruction; preserve old evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reviews/NO_COMPUTE_BUDGET_20261010_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def bind(name):
    raw = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


assert not DEST.exists()
program, progress = read('PROGRAM.json'), read('PROJECT_PROGRESS.json')
budget, acceptance = read('RESOURCE_BUDGET.json'), read('PROJECT_ACCEPTANCE.json')
assert program['current_stage'] == 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
assert program['active_scientific_jobs'] == 0 and not program['project_model_goal_complete']
assert budget['resource_version'] == 2 and budget['improvement_block']['method_plans_registered'] == 3
assert acceptance['scientific_threshold_version'] == 1 and acceptance['numerical_protocol_version'] == 2
assert read('R48_DELIVERY_COMPLETION_RECEIPT.json')['remote_SHA_match']
DEST.mkdir()
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
             'RESOURCE_BUDGET.json', 'SERVER_RESOURCE_POLICY.md', 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json']:
    shutil.copy2(ROOT / name, DEST / name)

receipt_path = 'reviews/NO_COMPUTE_BUDGET_20261010_001/RECEIPT.json'
policy = '''# 执行政策 v3：取消计算预算，保留科学标准与机器安全

2026-10-10用户明确要求“推进下一轮 不设置预算”。本政策即时覆盖旧的三个方法、六次训练、50000更新、科学/墙钟耗时、截止时间及阶段文件体积等预算停止上限。旧阶段的协议、真实消耗和负结果保持，原A–F科学验收线不降低。授权与旧文件快照见 reviews/NO_COMPUTE_BUDGET_20261010_001/RECEIPT.json。

后续不设置方法方案数、响应训练次数/更新数、科学秒数、墙钟秒数、截止时间或阶段产物字节数预算。配置以null表示未设置，代码必须明确跳过相应预算终止逻辑；不得把null当0、继承旧截止时间或换成极大整数。实际计算量、失败尝试、起止、原生退出和资源使用仍记入RESOURCE_BUDGET.json。该文件现在同时承担历史账目和无预算执行状态登记。

每轮固定科学问题、方法、训练或推理设计及验收范围。固定样本数、原1024 REAL+2000源样本效用任务、拟合方案数和唯一冻结末点属于科学比较设计，不因取消计算配额而改变。没有足够证据时不能用无限重训、事后选优或反复查看最终评价来替代原验收。数值错误、不有限、输入冲突、科学门失败和真实资源不可用仍须如实处理；这些是结果或安全问题，不是计时预算到限。

既有机器安全设置保留：服务器CPU12、实际作业树RSS48GiB、单GPU allocated/reserved各10GiB；本机辅助CPU4/RSS4GiB。一次一个科学作业，启动前核对本项目进程、可用内存/磁盘；GPU作业另查空闲显存和已有GPU计算任务。CPU作业不因另一个安全无关的GPU占用而拒绝执行。本项目不得修改他人作业和共享环境。安全阈值基于既有执行设置，不是累计计算量预算。

远程仍使用既有项目环境和独立阶段目录；未知作业状态先查原PID/会话与真实原生回执，不能因一次观测超时重复提交或杀死仍在正常计算的作业。握手/传输/连接管理超时不等同于科学作业超时。真实PID握手和进程树内存观察保留，观察缺口按unknown记录。

本机全部新增数据、源码、日志、缓存、temp只写E/D。磁盘按实际可用容量及待保存的必要数据安排；旧4GiB/160MiB阶段额度和D额外256MiB不作为新阶段预算停止条件。仍避免写满文件系统，优先引用历史输入、避免重复模型/数组/归档，不删除或覆盖历史成果换空间。必要证据无法容纳时报告实际容量不足，不虚报备份完成。

现有sim2real_sol_max_executor连续完成每轮登记范围的实现、调试、科学执行、绘图和封存；根负责方向、关键结果和Git/D交付，不增加代理或逐子步骤审批。取消预算不授权改源人口、权重或标签，不授权降低A–F门槛。

科学代码/配置/紧凑指标和关键图按明确路径提交私有Git并核对实际远端SHA，原始数据、逐事件数组、权重、日志和缓存不进Git；D正常增量复制保留历史依赖和实际回执，不做迁移内容一致性测试。完整模型目标仍需同一冻结方法同时满足A–F。
'''
(ROOT / 'SERVER_RESOURCE_POLICY.md').write_text(policy, 'utf8')
budget['previous_budgeted_block_at_user_override'] = budget['improvement_block'].copy()
budget['previous_response_training_budget_at_user_override'] = budget['response_training'].copy()
budget['previous_frozen_validation_budget_at_user_override'] = budget['frozen_validation'].copy()
budget.update(resource_version=3, updated_UTC=NOW, reason='User explicitly requests the next round without a compute budget.',
              policy_sha256=bind('SERVER_RESOURCE_POLICY.md')['sha256'], user_override_receipt=receipt_path,
              compute_budget_enforced=False, deadlines_enforced=False)
budget['server_limits']['stage_output_bytes_max'] = None
budget['server_limits']['disk_free_reserve_bytes'] = None
budget['server_limits']['storage_policy'] = 'Actual available capacity and required bytes; avoid duplicate storage and preserve history.'
for key in ['updates_max', 'training_science_seconds_max', 'stage_science_seconds_max', 'stage_wall_seconds_max']:
    budget['response_training'][key] = None
budget['response_training']['same_budget_control_required_for_method_comparison'] = True
budget['response_training']['comparison_note'] = 'Matched exposure, optimization and hardware remain part of scientific design, not a quota.'
block = budget['improvement_block']
for key in ['diagnostic_science_seconds_max', 'method_plans_max', 'all_response_training_runs_max', 'total_science_seconds_max']:
    block[key] = None
block.update(no_automatic_extension_at_limit=False, current_family_status='user_authorized_next_round_without_compute_budget',
    further_scientific_execution_authorized=True, exhausted_limit=None,
    prior_bounded_block_scientific_conclusions_preserved=True, user_override_receipt=receipt_path)
for key in ['utility_science_seconds_max', 'scope_science_seconds_max', 'total_science_seconds_max']:
    budget['frozen_validation'][key] = None
for old, new in [('utility_runs_max', 'planned_utility_studies_per_frozen_candidate'),
                 ('FULL_GAP_studies_max', 'planned_FULL_GAP_studies_per_frozen_candidate')]:
    budget['frozen_validation'][new] = budget['frozen_validation'].pop(old)
save('RESOURCE_BUDGET.json', budget)

notice = ('# 用户新授权：推进下一轮，不设置计算预算\n\n'
    '2026-10-10用户明确要求“推进下一轮 不设置预算”。方法数、训练次数/更新数、科学/墙钟耗时、'
    '截止时间和阶段产物体积不再设预算上限；实际消耗照常记账。原A–F、源人口/权重、历史封存、'
    '单科学作业、既有机器安全设置和本机E/D存储约束保持。当前准备R49：冻结R48两组权重，'
    '共同Heun128候选/256参考，登记后由既有执行器连续运行。旧“三方法耗尽，不再自动继续”'
    '属于已结束的预算块，不能阻止本次新授权。详见SERVER_RESOURCE_POLICY.md及' + receipt_path + '。\n\n'
    '本轮应用接口实际返回无goal；不自动重建应用目标，项目A–F目标及本次执行授权继续有效。\n\n'
    '## 以下为历史状态与协议\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = ROOT / name
    path.write_text(notice + path.read_text('utf-8-sig'), 'utf8')

adoc = ROOT / 'PROJECT_ACCEPTANCE.md'
adoc.write_text('# 当前资源执行修订：不设置计算预算\n\n'
    '2026-10-10用户明确要求“推进下一轮 不设置预算”。以下原A–F科学指标、数据范围和历史结论保持；'
    '文中的历史方法数、训练数、科学秒数/墙钟预算已由SERVER_RESOURCE_POLICY.md v3覆盖，不再作为后续停工条件。'
    '固定科学样本/比较设计和机器安全设置继续有效。当前R49数值方案须单独登记后才执行。'
    '本轮应用接口返回无goal，不能把历史active状态当作当前状态。\n\n' + adoc.read_text('utf-8-sig'), 'utf8')
acceptance.update(resource_version=3, resource_revision_UTC=NOW,
    current_status_reconciliation_UTC=NOW, document_sha256=bind('PROJECT_ACCEPTANCE.md')['sha256'],
    compute_budget_enforced=False, user_override_receipt=receipt_path)
for key in ['diagnostic_science_seconds_max', 'method_plans_max', 'all_response_training_runs_max',
            'total_science_seconds_max', 'updates_per_training_run_max', 'science_seconds_per_training_stage_max']:
    acceptance['improvement_block'][key] = None
acceptance['improvement_block']['no_automatic_extension_at_limit'] = False
for key in ['utility_science_seconds_max', 'scope_science_seconds_max', 'total_science_seconds_max']:
    acceptance['frozen_validation'][key] = None
for old, new in [('utility_runs_max', 'planned_utility_studies_per_frozen_candidate'),
                 ('FULL_GAP_studies_max', 'planned_FULL_GAP_studies_per_frozen_candidate')]:
    acceptance['frozen_validation'][new] = acceptance['frozen_validation'].pop(old)
acceptance['application_goal_observation'] = {'value': None, 'observed_in_current_turn': True,
    'source': 'get_goal tool returned null after the latest user request', 'automatically_recreated': False}
save('PROJECT_ACCEPTANCE.json', acceptance)

for data in [program, progress]:
    data.update(updated_UTC=NOW, updated_at_utc=NOW,
        status='R49_preparation_authorized_without_compute_budget_model_goal_unmet',
        current_activity='prepare_R49_frozen_Heun128_protocol', application_goal_status=None,
        goal_status_note='Current get_goal returned null; no automatic goal creation. Human request authorizes this round.',
        latest_user_steering='推进下一轮 不设置预算', latest_user_steering_receipt=receipt_path,
        latest_permission_status='Next round explicitly authorized without compute quotas; scientific gates preserved.',
        current_direction='Freeze R48 weights and evaluate common Heun128 candidate with 256-step TRAIN reference.',
        next_action='Register R49 after source-only feasibility review; existing executor then runs the complete protocol.',
        current_route_research_closed=False, improvement_block_closed=False,
        further_scientific_execution_authorized=True, active_scientific_jobs=0,
        next_prepared_task='tasks/R49_FROZEN_HEUN128_RESPONSE_001.md',
        resource_version=3, compute_budget_enforced=False,
        stage_deadline_UTC=None, current_stage_deadline_UTC=None, executor_seal_deadline_UTC=None)
progress.update(project_status='next_round_authorized_model_goal_unmet',
    improvement_block_science_seconds_max=None, improvement_block_response_training_runs_max=None,
    next_round_policy='Continue the newly authorized round with fixed scientific design and no compute quota.')
save('PROGRAM.json', program)
save('PROJECT_PROGRESS.json', progress)
receipt = {'UTC': NOW, 'actual_control_PID': os.getpid(), 'user_instruction': '推进下一轮 不设置预算',
    'action': 'remove_compute_quota_stop_conditions_for_future_work', 'resource_version': 3,
    'compute_budget_enforced': False, 'scientific_thresholds_lowered': False,
    'historical_stage_protocols_or_results_modified': False, 'scientific_jobs_started': 0,
    'application_goal_tool_observation': None, 'application_goal_automatically_created': False,
    'retained_safety_limits': budget['server_limits'], 'retained_local_limits': budget['local_auxiliary_limits'],
    'snapshots': [p.name for p in DEST.iterdir() if p.is_file()],
    'new_policy_binding': bind('SERVER_RESOURCE_POLICY.md'),
    'scientific_acceptance_version': acceptance['scientific_threshold_version'],
    'historical_numerical_protocol_version_before_R49': acceptance['numerical_protocol_version']}
save(receipt_path, receipt)
print(json.dumps({'status': 'explicit_no_compute_budget_override_applied', 'PID': os.getpid(),
    'UTC': NOW, 'receipt': receipt_path, 'science_started': False, 'application_goal': None}, ensure_ascii=False))
