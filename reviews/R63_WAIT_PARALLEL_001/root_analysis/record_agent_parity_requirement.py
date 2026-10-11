"""Persist the user's agent-model parity requirement without changing R63 science."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shutil

R = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
S = R / 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
POLICY = ('所有承担执行、调研、分析、规划或审查的子智能体，包括后续派生代理，'
          '必须与主助手当前使用的模型和思考强度完全相同。启动及继续任务时继承主助手配置，'
          '不得因任务类型或效率考虑设置较低模型或思考强度；主助手配置变化后先核对并同步再派工。'
          '2026-10-11已核验本次主助手和三个R63辅助代理均为gpt-6-astra / max。')
HEADING = '# 子智能体模型与思考强度的硬性要求（2026-10-11）'
BLOCK = HEADING + '\n\n' + POLICY + '\n\n'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text('utf8'))

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

observation = load(HERE / 'AGENT_MODEL_PARITY_OBSERVATION.json')
assert observation['all_three_children_match']
configuration = observation['root_current_configuration']
assert (configuration['model'], configuration['effort']) == ('gpt-6-astra', 'max')
names = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md',
         'README.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json', 'reviews/finalize_R63_root.py']
before = HERE / 'before/AGENT_PARITY_001'
before.mkdir(parents=True, exist_ok=False)
protected = [R / 'PROJECT_ACCEPTANCE.md', R / 'PROJECT_ACCEPTANCE.json',
             R / 'tasks/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001.md',
             S / 'config.json', S / 'config_post.json', *sorted(S.glob('*.py'))]
frozen = {str(p.relative_to(R)): sha(p) for p in protected}
changes = {}
for name in names:
    source = R / name
    destination = before / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    changes[name] = {'before_SHA256': sha(source)}

for name in names[:5]:
    path = R / name
    original = path.read_text('utf8')
    assert HEADING not in original
    path.write_text(BLOCK + original, encoding='utf8')

for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    path = R / name
    value = load(path)
    coordination = value.setdefault('execution_coordination', {})
    coordination.update(subagents_same_model_as_root_required=True,
                        subagents_same_reasoning_effort_as_root_required=True,
                        agent_model_policy='inherit_current_root_model_and_effort_without_downgrade_overrides',
                        verify_after_root_configuration_change=True,
                        latest_verified_root_model=configuration['model'],
                        latest_verified_root_reasoning_effort=configuration['effort'],
                        model_parity_receipt='reviews/R63_WAIT_PARALLEL_001/root_analysis/AGENT_MODEL_PARITY_OBSERVATION.json')
    value['latest_user_agent_policy'] = POLICY
    value['latest_user_agent_policy_receipt'] = 'reviews/R63_WAIT_PARALLEL_001/root_analysis/AGENT_PARITY_REQUIREMENT_RECEIPT.json'
    save(path, value)

path = R / 'reviews/finalize_R63_root.py'
source = path.read_text('utf8')
old = '主助手统筹并按需使用多智能体，等待期间主动调研、分析既有证据和准备后续方法；'
new = '主助手统筹并按需使用多智能体；全部执行和规划子智能体须与主助手当前模型及思考强度完全一致，继承配置且不降级覆盖。等待期间主动调研、分析既有证据和准备后续方法；'
assert source.count(old) == 1
source = source.replace(old, new)
ast.parse(source)
path.write_text(source, encoding='utf8')

for name in names:
    changes[name]['after_SHA256'] = sha(R / name)
assert frozen == {str(p.relative_to(R)): sha(p) for p in protected}
receipt = {'UTC': datetime.now(timezone.utc).isoformat(),
           'user_requirement': POLICY,
           'same_current_configuration_verified_for_all_three_agents': True,
           'configuration_observation': 'reviews/R63_WAIT_PARALLEL_001/root_analysis/AGENT_MODEL_PARITY_OBSERVATION.json',
           'changed_files': changes,
           'protected_scientific_files_unchanged': frozen,
           'protected_scientific_file_count': len(protected),
           'application_goal_status': 'active',
           'application_goal_recreated_or_completed': False,
           'new_scientific_jobs': 0,
           'live_scientific_job_changed_or_stopped': False,
           'all_new_outputs_on_E': True}
save(HERE / 'AGENT_PARITY_REQUIREMENT_RECEIPT.json', receipt)
print(json.dumps({'status': 'requirement_recorded', 'root_model': configuration['model'],
                  'root_effort': configuration['effort'], 'all_three_agents_match': True,
                  'changed_files': list(changes), 'frozen_scientific_files_unchanged': len(protected)},
                 ensure_ascii=False, indent=2))
