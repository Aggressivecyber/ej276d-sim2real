from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, subprocess, os, copy
R = Path(__file__).resolve().parents[1]
ID = 'R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
S = R / 'work' / ID
H = R / 'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
REG = R / 'reviews/R61_REGISTRATION_001'
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads((R / p).read_text('utf8'))

def write(p, value):
    (R / p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8'))

def bind(p):
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status'] == 'R60_delivered_long_term_goal_active'
assert read('R60_DELIVERY_COMPLETION_RECEIPT.json')['status'] == 'delivered'
assert not read('work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/LIVE_OBSERVATION.json')['actual_process']['exists']
old = read('work/R59_DIRECT_DISTRIBUTION_RESPONSE_001/config.json')
historical = read('work/R59_DIRECT_DISTRIBUTION_RESPONSE_001/DIRECT_DISTRIBUTION_RESPONSE/TRAINING_RECEIPT.json')
hr = old['remote_root']
remote = hr.split('/R59_ROOT_')[0] + '/R61_ROOT_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs = dict(old['input_paths'])
inputs.update(distribution_statistics=hr + '/TRAIN_DISTRIBUTION_CONSTANTS.npz',
              R59_training_receipt=hr + '/DIRECT_DISTRIBUTION_RESPONSE/TRAINING_RECEIPT.json',
              R59_parent_exposures=hr + '/DIRECT_DISTRIBUTION_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz',
              R59_final_random_states=hr + '/DIRECT_DISTRIBUTION_RESPONSE/FINAL_RANDOM_STATES.json')
code = 'from pathlib import Path\nimport json,hashlib,subprocess,shutil,os\ninputs=' + repr(inputs) + '\n' + '''
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305';jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace');cwd=str((p/'cwd').resolve())
  if 'python' in argv and (base in argv or cwd.startswith(base)):jobs.append({'PID':int(p.name),'cwd':cwd,'argv':argv})
 except(OSError,PermissionError):pass
assert not jobs,jobs
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,apps
print(json.dumps({'inputs':{k:{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()},'read_only_preflight':{'own_project_processes':jobs,'available_memory_bytes':mem['MemAvailable'],'GPU':gpu,'GPU_compute_processes':apps,'server_disk_free_bytes':shutil.disk_usage(base).free},'scientific_arrays_decoded':False}))
'''
p = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    'rylan@100.82.231.102', '/usr/bin/python3 -B -'], input=code, text=True, capture_output=True, timeout=60)
assert p.returncode == 0, p.stderr
observed = json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes'] >= old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip()) * 1048576 >= 11811160064
for key, name in [('distribution_statistics', 'TRAIN_DISTRIBUTION_CONSTANTS.npz'),
                  ('R59_training_receipt', 'DIRECT_DISTRIBUTION_RESPONSE/TRAINING_RECEIPT.json'),
                  ('R59_parent_exposures', 'DIRECT_DISTRIBUTION_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz'),
                  ('R59_final_random_states', 'DIRECT_DISTRIBUTION_RESPONSE/FINAL_RANDOM_STATES.json')]:
    assert bind(H / name)['sha256'] == observed['inputs'][key]['sha256']
dfree = shutil.disk_usage('D:/').free
delivery_estimate = read('R59_BACKUP_RECEIPT.json')['copied_bytes'] + 5 * 1024**2
assert dfree > delivery_estimate + 20 * 1024**2, ('physical_backup_capacity_insufficient', dfree, delivery_estimate)
S.mkdir(); REG.mkdir()
meta = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json', 'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']
for name in meta:
    shutil.copy2(R / name, REG / name)
C = copy.deepcopy(old)
C.update(stage=ID, arm='PRECISION_RESOLVED_RESPONSE', arms=['PRECISION_RESOLVED_RESPONSE'], registered_UTC=NOW,
         remote_root=remote, protocol='tasks/' + ID + '.md', input_paths=inputs,
         historical_R59_root=hr, historical_R59_report='R59_REPORT.md', historical_R60_report='R60_REPORT.md',
         expected_R59_initial_state_SHA=historical['initial_state_SHA'], expected_R59_stream_hashes=historical['stream_hashes'],
         source_order_rule='float32_PSD_interval_disjoint_only_original_nonzero_denominator',
         sole_scientific_change='Exclude source-order penalties for overlapping conservative float32 PSD intervals; retain all original parents and original nonzero-pair normalization.',
         fixed_endpoint_single_factor_ablation=True, new_response_training_runs=1,
         constants_reused_from_frozen_R59=True, interval_fit_or_threshold_search=False)
assert all(C[k] == old[k] for k in ['model_seed', 'PCG64_seeds', 'updates', 'batch_per_region', 'identity_batch',
           'region_categories', 'lr', 'lr_initial', 'lr_final', 'warmup_updates', 'betas', 'gradient_clip', 'projection_directions',
           'box_projection_iterations', 'solver_steps', 'solver_check_steps', 'generation_chunk', 'loss_weights',
           'PSD_loss_scale', 'valley_delta', 'valley_smoothing', 'feature_names', 'feature_scale_floor', 'noise_scale_floor'])
write('work/' + ID + '/config.json', C)
frozen = ['network.py', 'noise_core.py', 'source_flow_core.py', 'linux_owned_worker.py',
          'NOISE_CALIBRATION.npz', 'NOISE_CALIBRATION.json', 'UNIT_CHARGE_CONSTANTS.npz', 'UNIT_CHARGE_CONSTANTS.json',
          'HISTORICAL_TRAIN_SOLVER_CHECK.json']
for name in frozen + ['train_final.py', 'distribution_loss.py', 'artificial_check.py']:
    shutil.copy2(H / name, S / name)
shutil.copy2(H / 'distribution_loss.py', S / 'R59_distribution_loss.py')
supervisor = (R / 'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/linux_supervise.py').read_text('utf8').replace('R60 requires explicit null', 'R61 requires explicit null')
(S / 'linux_supervise.py').write_bytes(supervisor.encode('utf8'))
shutil.copy2(R / C['protocol'], S / 'PROTOCOL.md')
write('work/' + ID + '/INPUT_BINDING.json', {'UTC': NOW, **observed,
      'original_acceptance': [bind(R / name) for name in ['PROJECT_ACCEPTANCE.json', 'PROJECT_ACCEPTANCE.md']],
      'frozen_R59_helpers': {name: bind(H / name) for name in frozen + ['distribution_loss.py', 'train_final.py', 'artificial_check.py']},
      'R60_evidence': [bind(R / 'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001/RESULT.json'), bind(R / 'reviews/R60_ROOT_DIRECTION_001.json')],
      'trained_weights_loaded': False, 'local_D_free_bytes': dfree, 'delivery_bytes_estimate_not_science_quota': delivery_estimate})
action = 'Implement and run the one-factor precision-resolved source-order ablation at the same R59 initialization, streams and update50000 endpoint; preserve original numerical and full A-F requirements.'
banner = '# 当前执行：R61 存储精度排序单因素消融已登记\n\nR60三个冲突标记未触发的结论保持。本轮独立检验float32 PSD区间重叠排序罚的影响，只改源排序分子，保留原分母、完整人口及其余训练条件。相同初始化/六条随机流，唯一50000步末点；按原数值门和A–C评价，再决定是否进入D/E。主助手独立，无子智能体，无计算配额；CPU12/RSS48GiB/GPU10GiB、单科学作业及本机E/D限制保持。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    path = R / name
    path.write_bytes((banner + path.read_text('utf8')).encode('utf8'))
job = {'stage': ID, 'status': 'registered_not_launched', 'active_scientific_jobs': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    value = read(name)
    value.update(current_stage=ID, current_task=C['protocol'], current_stage_scope='single_factor_precision_resolved_source_order_ablation',
                 current_stage_progress='registered_not_launched', current_job=job, next_action=action,
                 current_stage_registration='reviews/R61_REGISTRATION_001/RECEIPT.json', active_scientific_jobs=0,
                 active_remote_scientific_jobs=0, active_local_scientific_jobs=0, subagents_allowed=False,
                 application_goal_status='active', model_goal_complete=False)
    value['execution_location'].update(remote_root=remote, mode='server_GPU_single_root_neural_training',
                                       preflight_required=True, current_stage_GPU_used=False)
    if name == 'PROGRAM.json':
        value.update(status='R61_registered_long_term_goal_active', updated_at_utc=NOW)
    else:
        value.update(status='R61_registered_long_term_goal_active', project_status='R61_registered_long_term_goal_active',
                     updated_UTC=NOW, current_activity='root_implementing_R61', current_direction=action)
    write(name, value)
b = read('RESOURCE_BUDGET.json')
b['current_registered_stage'] = ID
b['improvement_block']['registered_stages'].append({'stage': ID, 'kind': 'precision_resolved_source_order_ablation',
    'registered_UTC': NOW, 'science_budget': None, 'planned_new_training_runs': 1, 'fixed_terminal_update': 50000, 'status': 'registered'})
write('RESOURCE_BUDGET.json', b)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    value = read(name)
    value[key].append({identity: ID, 'status': 'registered', 'owner': '/root', 'registered_UTC': NOW,
                      'protocol': C['protocol'], 'output_root': 'work/' + ID, 'remote_root': remote,
                      'new_training_runs_started': 0, 'fixed_terminal_update': 50000, 'model_goal_complete': False})
    write(name, value)
write('reviews/R61_REGISTRATION_001/RECEIPT.json', {'UTC': NOW, 'stage': ID, 'root_control_PID': os.getpid(),
      'protocol': bind(R / C['protocol']), 'config': bind(S / 'config.json'), 'owner': '/root', 'application_goal_status': 'active',
      'original_A_F_unchanged': True, 'scientific_execution_started': False, 'response_training_runs_started': 0,
      'remote_root': remote, 'read_only_preflight': observed['read_only_preflight'], 'D_free_bytes': dfree,
      'R60_markers_reinterpreted_as_positive': False, 'same_initialization_and_random_streams_required': True})
print(json.dumps({'stage': ID, 'UTC': NOW, 'remote_root': remote, 'new_training_runs_registered': 1,
                  'scientific_execution_started': False, 'preflight': observed['read_only_preflight'], 'D_free_bytes': dfree}, indent=2))
