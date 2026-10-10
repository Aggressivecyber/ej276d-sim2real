from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

R = Path(__file__).resolve().parents[1]
ID = 'R55_PSD_SOURCE_COMPATIBILITY_001'
S = R / 'work' / ID
REG = R / 'reviews/R55_REGISTRATION_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((R / name).read_text('utf8'))


def write(name, value):
    (R / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def bind(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status'] == 'R54_delivered_long_term_goal_active'
assert read('R54_DELIVERY_COMPLETION_RECEIPT.json')['status'] == 'delivered'
assert read('PROGRAM.json')['active_scientific_jobs'] == 0
old = read('work/R54_CONTINUOUS_CHARGE_FLOW_001/config.json')
post = read('work/R54_CONTINUOUS_CHARGE_FLOW_001/config_post.json')
policy = read('work/R52_JOINT_CHARGE_PSD_COUPLING_001/config.json')
cfg = {k: policy[k] for k in ['server_python', 'CPU_threads', 'RSS_limit_bytes', 'GPU_limit_bytes',
       'MemAvailable_min_bytes', 'science_budget', 'deadline_UTC', 'output_limit_bytes', 'compute_budget_enforced']}
inputs = {
    'features': old['remote_root'] + '/results/FEATURES_PRIVATE.npz',
    'metrics': old['remote_root'] + '/metrics.json',
    'actual_inputs': post['frozen_eval_input_root'] + '/ACTUAL_GENERATION_INPUTS.npz',
    'source_metadata': post['frozen_eval_input_root'] + '/DEV_SOURCE_METADATA.npz',
    'response_ids': old['remote_root'] + '/CONTINUOUS_Q_RESPONSE/generated.npz',
    'valley_definition': post['frozen_eval_input_root'] + '/VALLEY_DEFINITION.json',
    'width_definition': post['local_input_paths']['width_definition'],
}
code = 'from pathlib import Path\nimport hashlib,json\npaths=' + repr(inputs) + '\n'
code += 'print(json.dumps({k:{"path":v,"bytes":Path(v).stat().st_size,"sha256":hashlib.sha256(Path(v).read_bytes()).hexdigest()}for k,v in paths.items()}))\n'
p = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    'rylan@100.82.231.102', '/usr/bin/python3 -B -'], input=code, text=True, capture_output=True, timeout=30)
assert p.returncode == 0, p.stderr
input_binding = json.loads(p.stdout)
assert input_binding['features']['sha256'] == bind(R / 'work/R54_CONTINUOUS_CHARGE_FLOW_001/results/FEATURES_PRIVATE.npz')['sha256']
assert input_binding['metrics']['sha256'] == bind(R / 'work/R54_CONTINUOUS_CHARGE_FLOW_001/metrics.json')['sha256']
S.mkdir()
REG.mkdir()
meta = ['AGENTS.md', 'PLAN.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json', 'PROJECT_OBJECTIVE.md',
        'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']
for name in meta:
    shutil.copy2(R / name, REG / name)
cfg.update(stage=ID, owner='/root', subagents_allowed=False, registered_UTC=NOW,
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R55_ROOT_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'),
    protocol='tasks/' + ID + '.md', allowed_scientific_entries=['numeric_worker.py'], input_paths=inputs,
    SOURCE_N=2000, REAL_N=2000, expected_purehad_counts={'bin3': 213, 'bin4': 181},
    original_overall_W1_max=0.01, original_conditional_mean_W1_max=0.01, original_valley_delta_max=0.02,
    diagnostic_PSD_margin=1e-6, q_relative_tolerance=1e-5, ADC_q_bound_LSB=600,
    q_absolute_margin_Vns=1e-6, log_boundary_margin=1e-10, min_common_bin_N=40,
    LP_method='highs-ds', LP_attempts=1, LP_options={'presolve': True, 'threads': 12, 'parallel': False},
    response_training_runs=0, HGB_fits=0, density_fits=0, ODE_generation=0,
    historically_exposed_DEV_read=True, CAL_decode=0, EVAL_read=0)
write('work/' + ID + '/config.json', cfg)
for name in ['linux_supervise_post.py', 'linux_owned_worker.py']:
    shutil.copy2(R / 'work/R52_JOINT_CHARGE_PSD_COUPLING_001' / name, S / name)
p = S / 'linux_supervise_post.py'
p.write_bytes(p.read_bytes().replace(b'R52 requires', b'R55 requires'))
shutil.copy2(R / cfg['protocol'], S / 'PROTOCOL.md')
write('work/' + ID + '/INPUT_BINDING.json', {'UTC': NOW, 'inputs': input_binding,
      'binding_only_no_new_statistics': True, 'SSH_native_exit': 0,
      'acceptance': bind(R / 'PROJECT_ACCEPTANCE.json'), 'source_report': bind(R / 'R54_REPORT.md')})
banner = '# 当前执行：R55 固定源人口的必要条件兼容性诊断\n\n' + (
    'R54 已作为负结果交付。现按 tasks/' + ID + '.md 登记一次 PSD/原始源谷比例的放宽下界诊断，包含数值与 ADC 分箱不确定性；0 新响应训练，尚未启动科学进程。原 A–F、固定 SOURCE 人口、主助手独立、无计算配额及 E/D 存储限制保持。\n\n## 以下为历史记录\n\n')
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    (R / name).write_text(banner + (R / name).read_text('utf8'), 'utf8')
action = 'Execute the one registered conservative PSD/source-valley LP, certify a rigorous lower bound, and distinguish conflict from inconclusive compatibility before proposing new response training.'
job = {'stage': ID, 'status': 'registered_not_launched', 'active_scientific_jobs': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    x = read(name)
    x.update(current_stage=ID, current_task=cfg['protocol'], current_stage_scope='frozen_population_necessary_condition_LP',
        current_stage_progress='registered_not_launched', current_job=job, next_action=action,
        current_stage_registration='reviews/R55_REGISTRATION_001/RECEIPT.json', active_scientific_jobs=0,
        active_remote_scientific_jobs=0, active_local_scientific_jobs=0, subagents_allowed=False,
        application_goal_status='active', model_goal_complete=False, status='R55_registered_long_term_goal_active')
    x['execution_location'] = {**x['execution_location'], 'remote_root': cfg['remote_root'], 'mode': 'server_CPU_compatibility_LP'}
    if name == 'PROGRAM.json':
        x['updated_at_utc'] = NOW
    else:
        x.update(project_status=x['status'], updated_UTC=NOW, current_activity='root_preparing_R55', current_direction=action)
    write(name, x)
b = read('RESOURCE_BUDGET.json')
b['current_registered_stage'] = ID
b['improvement_block']['registered_stages'].append({'stage': ID, 'kind': 'necessary_condition_diagnostic_not_response_method',
    'registered_UTC': NOW, 'science_budget': None, 'new_training_runs': 0, 'new_HGB_fits': 0, 'LP_attempts': 1, 'status': 'registered'})
write('RESOURCE_BUDGET.json', b)
for name, key, ident in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    x = read(name)
    assert not any(v[ident] == ID for v in x[key])
    x[key].append({ident: ID, 'status': 'registered', 'owner': '/root', 'registered_UTC': NOW,
                  'protocol': cfg['protocol'], 'output_root': 'work/' + ID, 'remote_root': cfg['remote_root'], 'model_candidate': False})
    write(name, x)
write('reviews/R55_REGISTRATION_001/RECEIPT.json', {'UTC': NOW, 'stage': ID, 'root_control_PID': os.getpid(),
    'protocol': bind(R / cfg['protocol']), 'config': bind(S / 'config.json'), 'input_binding': bind(S / 'INPUT_BINDING.json'),
    'owner': '/root', 'application_goal_status': 'active', 'original_A_F_unchanged': True,
    'scientific_execution_started': False, 'response_training_runs': 0, 'remote_root': cfg['remote_root']})
print(json.dumps({'stage': ID, 'UTC': NOW, 'remote_root': cfg['remote_root'], 'scientific_execution_started': False}, indent=2))
