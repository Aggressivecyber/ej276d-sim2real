from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

R = Path(__file__).resolve().parents[1]
ID = 'R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
S = R/'work'/ID
REG = R/'reviews/R64_REGISTRATION_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads((R/p).read_text('utf8'))


def write(p, value):
    (R/p).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def bind(p):
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


assert not S.exists() and not REG.exists()
assert read('PROGRAM.json')['status'] == 'R63_delivered_long_term_goal_active'
assert read('R63_DELIVERY_COMPLETION_RECEIPT.json')['status'] == 'delivered'
assert read('reviews/R64_PREPARATION_001/DISPATCH_AND_PARITY.json')['all_match']
H = R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
old = json.loads((H/'config.json').read_text('utf8'))
hr, arm = old['remote_root'], old['arm']
base = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
remote = base+'/R64_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs = {
    'TRAIN_path': old['TRAIN_path'],
    'constants_path': old['constants_path'],
    'noise_calibration_path': old['noise_calibration_path'],
    'distribution_statistics': old['input_paths']['distribution_statistics'],
    'temporal_statistics': hr+'/TRAIN_TEMPORAL_CONSTANTS.npz',
    'frozen_weight': hr+'/'+arm+'/FINAL_FLOW.pt',
    'frozen_training_receipt': hr+'/'+arm+'/TRAINING_RECEIPT.json',
    'frozen_training_config': hr+'/config.json',
}
code = 'from pathlib import Path\nimport json,hashlib,subprocess,shutil,os\ninputs='+repr(inputs)+'\nbase='+repr(base)+'\n'+'''
jobs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace')
        cwd=str((p/'cwd').resolve())
        if 'python' in argv and (base in argv or cwd.startswith(base)):
            jobs.append({'PID':int(p.name),'cwd':cwd,'argv':argv})
    except(OSError,PermissionError):pass
assert not jobs,jobs
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,apps
print(json.dumps({'inputs':{k:{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}for k,p in inputs.items()},'read_only_preflight':{'own_project_processes':jobs,'available_memory_bytes':mem['MemAvailable'],'GPU':gpu,'GPU_compute_processes':apps,'server_disk_free_bytes':shutil.disk_usage(base).free},'scientific_arrays_decoded':False}))
'''
p = subprocess.run(['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o',
                    'ConnectTimeout=8', 'rylan@100.82.231.102', '/usr/bin/python3 -B -'],
                   input=code, text=True, capture_output=True, timeout=60)
assert p.returncode == 0, p.stderr
observed = json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes'] >= old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip())*1048576 >= 11811160064
assert observed['inputs']['frozen_weight']['sha256'] == json.loads((H/'FROZEN_ARTIFACT_REVIEW.json').read_text('utf8'))['weight_file_SHA256']
planned_gradient_bytes = (1+2*32)*5*1369889*8
physical_backup_readiness = planned_gradient_bytes + 700*1024**2
assert shutil.disk_usage('D:/').free > physical_backup_readiness, 'Actual free storage insufficient for conservative uncompressed diagnostic delivery'
assert observed['read_only_preflight']['server_disk_free_bytes'] > physical_backup_readiness

S.mkdir()
REG.mkdir()
meta = ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json',
        'PROJECT_PROGRESS.json', 'RESOURCE_BUDGET.json', 'jobs.json', 'experiments.json']
for name in meta:
    shutil.copy2(R/name, REG/name)
C = {k: old[k] for k in ['server_python', 'CPU_threads', 'RSS_limit_bytes', 'GPU_limit_bytes',
                          'MemAvailable_min_bytes', 'expected_R33_environment']}
seeds = {'full_SOURCE_epsilon': 20261011600, 'full_REAL_epsilon': 20261011601,
         'full_feature_directions': 20261011602, 'temporal_feature_directions': 20261011640,
         'macro_SOURCE_rows': 20261011641, 'macro_REAL_rows': 20261011642,
         'gradient_paired_bootstrap': 20261011644, 'REAL_floor_A_rows': 20261011645,
         'REAL_floor_B_rows': 20261011646}
stream_reference = read('work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/RANDOM_STREAM_RECEIPT.json')
C.update(stage=ID, owner='/root', subagents_allowed=True,
         agent_model_and_reasoning_policy='Exactly inherit current root settings for all execution and planning agents; no overrides',
         registered_UTC=NOW, remote_root=remote, protocol='tasks/'+ID+'.md', historical_R63_root=hr,
         input_paths=inputs, science_budget=None, deadline_UTC=None, output_limit_bytes=None,
         compute_budget_enforced=False, PCG64_seeds=seeds,
         expected_full_stream_hashes={k: stream_reference['chains'][k] for k in list(seeds)[:3]},
         SOURCE_N=6593, REAL_N=21047, full_parent_rows=27640,
         generation_chunk=64, endpoint_definition='original_direct_training_endpoint_projected_before_ADC',
         model_training_mode=True, MHA_fastpath_enabled=False,
         gradient_components=['full_window', 'PSD', 'noise', 'coordinate_RMS', 'temporal_joint'],
         excluded_gradient_components=['source_order', 'source_valley', 'REAL_identity', 'box_correction'],
         full_reference_gradients=1, paired_macro_batches=32, microbatches_per_macro=4,
         microbatch_per_region=32, macro_per_region=128, projection_directions=64,
         temporal_projection_directions=64, temporal_feature_count=31, temporal_power_floor=1e-20,
         PSD_loss_scale=.01, loss_weights=old['loss_weights'],
         matched_REAL_floor_replicates=32, paired_gradient_bootstrap_replicates=1000,
         bootstrap_quantiles=[.025, .975], bootstrap_quantile_method='linear',
         neural_replay_full_SOURCE_passes=33, planned_total_network_rows=245209,
         native_parameter_count=1369889, endpoint_replay_bitwise_required=True,
         canonical_codes_exact_required=True, Box_free_masks_exact_required=True,
         parameter_VJP_accumulation_dtype='float64_CPU_in_original_chunk_order',
         keep_all_parameter_gradient_vectors_float64=True,
         save_full_reference_endpoint_gradients=True,
         save_macro_endpoint_gradients=False,
         macro_endpoint_storage='Summary records and online mean vectors; reproduce exact vectors from fixed endpoints and saved macro IDs',
         geometry_numpy_cuda_bitwise_equivalence_claimed=False,
         charge_bound_nominal_scaled_atol=2e-10,
         supplementary_summaries=['raw_and_canonical_PSD_W1_by_all_original_regions',
                                  'source_Q_and_PSD_rank', 'fixed_nine_boundary_W1',
                                  'waveform_means_and_SDs', 'projection_activity_and_charge_errors'],
         response_training_runs=0, response_updates=0, HGB_fits=0, density_fits=0,
         DEV_read=0, EVAL_read=0, CAL_decode=0,
         allowed_scientific_entries=['diagnose_train.py'], GPU_scientific_entries=['diagnose_train.py'])
write('work/'+ID+'/config.json', C)
frozen = ['network.py', 'noise_core.py', 'source_flow_core.py', 'R61_distribution_loss.py',
          'distribution_loss.py', 'precision_order.py']
for name in frozen+['linux_owned_worker.py']:
    shutil.copy2(H/name, S/name)
(S/'linux_supervise.py').write_bytes((H/'linux_supervise.py').read_bytes().replace(b'R61 requires explicit null', b'R64 requires explicit null'))
shutil.copy2(R/C['protocol'], S/'PROTOCOL.md')
write('work/'+ID+'/INPUT_BINDING.json', {
    'UTC': NOW, **observed, 'original_acceptance': [bind(R/n) for n in ['PROJECT_ACCEPTANCE.json', 'PROJECT_ACCEPTANCE.md']],
    'frozen_R63_helpers': {n: bind(H/n) for n in frozen},
    'frozen_weight_tensor_SHA': json.loads((H/arm/'TRAINING_RECEIPT.json').read_text('utf8'))['final_state_SHA'],
    'source_parent_ID_SHA': json.loads((H/'TRAIN_POPULATION_AND_COORDINATES.json').read_text('utf8'))['SOURCE_ids_SHA256'],
    'R62_full_stream_reference': bind(R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/RANDOM_STREAM_RECEIPT.json'),
    'local_D_free_bytes': shutil.disk_usage('D:/').free,
    'conservative_uncompressed_gradient_bytes': planned_gradient_bytes,
    'physical_storage_readiness_estimate_not_an_output_quota': physical_backup_readiness,
    'scientific_execution_started': False})
action = 'Complete the registered frozen R63 direct-TRAIN endpoint geometry and paired 4x32 versus 128 five-term distribution-gradient diagnostic, preserving all original parents, finite-reference semantics and original A-F; no training update.'
banner = '# 当前执行：R64 冻结 TRAIN 参照与分布梯度诊断已登记\n\nR63固定末点已关闭并完成私有 Git/D 盘交付。R64先在CPU/CUDA人工检查后，按 tasks/'+ID+'.md 对全部原TRAIN父事件采集直接训练端点，描述既有四视图/31维残差、同量REAL参照及名义电荷均值约束，再完成一个完整有限参照和32组4×32/128配对分布梯度比较。0新训练/教师/HGB/密度拟合，0 DEV/EVAL/CAL读取。原A–F不变，长期目标active；全部执行和规划代理继承主助手相同模型与思考强度，主助手负责科学执行和Git。无计算配额，单科学作业和既有资源/E-D存储要求保持。\n\n## 以下为历史记录\n\n'
for name in ['AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md']:
    (R/name).write_bytes((banner+(R/name).read_text('utf8')).encode())
job = {'stage': ID, 'status': 'registered_not_launched', 'active_scientific_jobs': 0}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    value = read(name)
    value.update(current_stage=ID, current_task=C['protocol'], current_stage_scope='frozen_R63_TRAIN_finite_reference_geometry_and_distribution_gradient',
                 current_stage_progress='registered_not_launched', current_job=job, next_action=action,
                 current_stage_registration='reviews/R64_REGISTRATION_001/RECEIPT.json', active_scientific_jobs=0,
                 active_remote_scientific_jobs=0, active_local_scientific_jobs=0, subagents_allowed=True,
                 application_goal_status='active', model_goal_complete=False)
    value['execution_location'] = {**value['execution_location'], 'remote_root': remote,
                                   'mode': 'server_GPU_single_frozen_diagnostic', 'preflight_required': True}
    if name == 'PROGRAM.json':
        value.update(status='R64_registered_long_term_goal_active', updated_at_utc=NOW)
    else:
        value.update(status='R64_registered_long_term_goal_active', project_status='R64_registered_long_term_goal_active',
                     updated_UTC=NOW, current_activity='root_preparing_registered_R64', current_direction=action)
    write(name, value)
B = read('RESOURCE_BUDGET.json')
B['current_registered_stage'] = ID
B['improvement_block']['registered_stages'].append({'stage': ID,
    'kind': 'frozen_TRAIN_finite_reference_geometry_and_gradient_diagnostic', 'registered_UTC': NOW,
    'science_budget': None, 'planned_new_training_runs': 0, 'planned_neural_forward_rows': 245209, 'status': 'registered'})
write('RESOURCE_BUDGET.json', B)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    value = read(name)
    assert not any(v.get(identity) == ID for v in value[key])
    value[key].append({identity: ID, 'status': 'registered', 'owner': '/root', 'registered_UTC': NOW,
                      'protocol': C['protocol'], 'output_root': 'work/'+ID, 'remote_root': remote,
                      'model_candidate': False, 'new_training_runs_started': 0})
    write(name, value)
write('reviews/R64_REGISTRATION_001/RECEIPT.json', {
    'UTC': NOW, 'stage': ID, 'root_control_PID': os.getpid(), 'protocol': bind(R/C['protocol']),
    'config': bind(S/'config.json'), 'owner': '/root', 'application_goal_status': 'active',
    'original_A_F_unchanged': True, 'scientific_execution_started': False, 'response_training_runs_started': 0,
    'remote_root': remote, 'read_only_preflight': observed['read_only_preflight'],
    'all_current_auxiliary_agents_match_root': True})
print(json.dumps({'stage': ID, 'UTC': NOW, 'remote_root': remote,
                  'planned_neural_rows': 245209, 'new_training': 0,
                  'D_free_bytes': shutil.disk_usage('D:/').free}, indent=2))
