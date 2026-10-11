"""Register the fixed full empirical D candidate after delivered R64."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import ctypes
import hashlib
import json
import os
import shutil
import subprocess

R = Path(__file__).resolve().parents[1]
ID = 'R65_FULL_EMPIRICAL_D_RESPONSE_001'
S = R/'work'/ID
REG = R/'reviews/R65_REGISTRATION_001'
H = R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
G = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
NOW = datetime.now(timezone.utc).isoformat()
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
mask, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system))
cpus = [i for i in range(64) if mask.value & (1 << i)]
cpu = cpus[min(3, len(cpus)-1)]
assert kernel.SetProcessAffinityMask(handle, 1 << cpu)
kernel.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
kernel.GetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ulong)]
assert kernel.SetProcessWorkingSetSizeEx(handle, 16*1024**2, 1024**3, 6)
ws_min, ws_max, ws_flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_ulong()
assert kernel.GetProcessWorkingSetSizeEx(handle, ctypes.byref(ws_min), ctypes.byref(ws_max), ctypes.byref(ws_flags))
assert ws_max.value == 1024**3 and ws_flags.value & 6 == 6
cache = R/'reviews/R65_PREPARATION_001/root_registration_cache'
cache.mkdir(parents=True, exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME'):
    os.environ[key] = str(cache)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
read = lambda p: json.loads(p.read_text('utf8'))

def write(p, value):
    p.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())

def bind(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''):
            h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())

assert not S.exists() and not REG.exists()
assert read(R/'R64_DELIVERY_COMPLETION_RECEIPT.json')['status'] == 'delivered'
assert read(R/'PROGRAM.json')['status'] == 'R64_delivered_long_term_goal_active'
assert read(G/'diagnose_train.py.NATIVE_RECEIPT.json')['actual_worker_native_exit'] == 0
assert read(G/'FETCH_RECEIPT.json')['remote_completion']['actual_PID_gone']
root_review = read(R/'reviews/R65_PREPARATION_001/ROOT_PROTOCOL_REVIEW.json')
assert root_review['ready_for_registration'] and root_review['original_A_F_changed'] is False
protocol = R/'tasks'/f'{ID}.md'
assert root_review['protocol_sha256'] == bind(protocol)['sha256']
schema = read(R/'reviews/R65_PREPARATION_001/ROOT_TRAINING_SCHEMA.json')
old = read(H/'config.json')
old_receipt = read(H/'COVERAGE_RESPONSE/TRAINING_RECEIPT.json')
hr = old['remote_root']
base = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
remote = base+'/R65_ROOT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
inputs = {k:old['input_paths'][k] for k in ('TRAIN_path','constants_path','noise_calibration_path',
    'train_solver_input','valley_definition','distribution_statistics')}
inputs.update(temporal_statistics=hr+'/TRAIN_TEMPORAL_CONSTANTS.npz',
    R63_training_receipt=hr+'/COVERAGE_RESPONSE/TRAINING_RECEIPT.json',
    R63_parent_exposures=hr+'/COVERAGE_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz',
    R63_final_random_states=hr+'/COVERAGE_RESPONSE/FINAL_RANDOM_STATES.json',
    R63_config=hr+'/config.json')
code = 'from pathlib import Path\nimport json,hashlib,subprocess,shutil,os\ninputs='+repr(inputs)+'\nbase='+repr(base)+'\n'+'''
jobs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace')
        cwd=str((p/'cwd').resolve())
        if 'python' in argv and(base in argv or cwd.startswith(base)):
            jobs.append(dict(PID=int(p.name),cwd=cwd,argv=argv))
    except(OSError,PermissionError):pass
assert not jobs,jobs
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,apps
def bind(name):
    p=Path(name);h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
print(json.dumps(dict(inputs={k:bind(p)for k,p in inputs.items()},read_only_preflight=dict(
    own_project_processes=jobs,available_memory_bytes=mem['MemAvailable'],GPU=gpu,
    GPU_compute_processes=apps,server_disk_free_bytes=shutil.disk_usage(base).free),scientific_arrays_decoded=False)))
'''
p = subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'rylan@100.82.231.102','/usr/bin/python3 -B -'], input=code, text=True, capture_output=True, timeout=60)
assert p.returncode == 0, p.stderr
observed = json.loads(p.stdout)
assert observed['read_only_preflight']['available_memory_bytes'] >= old['MemAvailable_min_bytes']
assert int(observed['read_only_preflight']['GPU'].split(',')[-1].strip())*1048576 >= 11811160064
local_inputs = dict(temporal_statistics=H/'TRAIN_TEMPORAL_CONSTANTS.npz',
    R63_training_receipt=H/'COVERAGE_RESPONSE/TRAINING_RECEIPT.json',
    R63_parent_exposures=H/'COVERAGE_RESPONSE/PARENT_EXPOSURES_PRIVATE.npz',
    R63_final_random_states=H/'COVERAGE_RESPONSE/FINAL_RANDOM_STATES.json', R63_config=H/'config.json')
for key, path in local_inputs.items():
    assert bind(path)['sha256'] == observed['inputs'][key]['sha256'], key
old_binding = read(G/'INPUT_BINDING.json')['inputs']
for key in ('TRAIN_path','constants_path','noise_calibration_path','distribution_statistics','temporal_statistics'):
    assert observed['inputs'][key]['sha256'] == old_binding[key]['sha256'], key
delivery_estimate = max(256*1024**2, read(R/'R63_BACKUP_RECEIPT.json')['copied_bytes']+64*1024**2)
assert shutil.disk_usage('D:/').free > delivery_estimate+20*1024**2
assert shutil.disk_usage('E:/').free > delivery_estimate+20*1024**2
assert observed['read_only_preflight']['server_disk_free_bytes'] > delivery_estimate+20*1024**2
S.mkdir()
REG.mkdir()
meta = ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','PROGRAM.json','PROJECT_PROGRESS.json',
        'RESOURCE_BUDGET.json','jobs.json','experiments.json']
for name in meta:
    shutil.copy2(R/name, REG/name)
C = copy.deepcopy(old)
for key in ('expected_R61_stream_hashes','expected_R61_initial_state_SHA','fixed_endpoint_objective_extension',
            'new_temporal_statistics_TRAIN_only'):
    C.pop(key, None)
C.update(stage=ID, arm='FULL_EMPIRICAL_RESPONSE', arms=['FULL_EMPIRICAL_RESPONSE'],
    registered_UTC=NOW, remote_root=remote, protocol='tasks/'+ID+'.md', owner='/root', subagents_allowed=True,
    agent_model_and_reasoning_policy='Exactly inherit current root settings, no overrides',
    input_paths=inputs, historical_R63_root=hr, historical_R63_report='R63_REPORT.md', historical_R64_report='R64_REPORT.md',
    expected_R63_initial_state_SHA=old_receipt['initial_state_SHA'], expected_R63_stream_hashes=old_receipt['stream_hashes'],
    sole_scientific_change='Complete finite-parent D with fresh per-parent noise and exact full parameter VJP; one original occurrence-noise C microbatch; joint D-C stochastic coupling explicitly changes.',
    original_distribution_and_temporal_statistics_reused=True, new_statistics_fit=False,
    complete_parent_D=True, original_C_microbatches_per_update=1, neural_replay_chunk=64, W1_feature_chunk=64,
    full_D_scalar_order=schema['full_D_scalar_order'], C_scalar_order=schema['C_scalar_order'],
    parameter_gradient_accumulation='float64 CPU in original64-row chunk and named-parameter order, then add one C, cast once to parameter dtype, clip once, Adam once',
    complete_D_noise_policy='one fresh epsilon per original SOURCE parent per update, reused only within two-pass replay',
    original_C_noise_policy='fresh independent epsilon per occurrence, exact original R63 stream consumption',
    original_D_C_random_coupling_preserved=False, new_REAL_feature_cache=False, extracted_C_implementation=False,
    sampled_parameter_VJP=False, model_training_mode=True, MHA_fastpath_enabled=False,
    loss_columns=schema['loss_columns'], precision_columns=schema['precision_columns'],
    replay_columns=schema['replay_columns'], expected_replay_row=schema['expected_replay_row'],
    expected_completed_update_counts=schema['expected_completed_update_counts'],
    recovery_interval_updates=schema['recovery_interval_updates'], progress_interval_updates=schema['progress_interval_updates'],
    original_streams=schema['original_streams'], new_stream=schema['new_stream'],
    new_response_training_runs=1, unique_terminal_updates=50000, science_budget=None, deadline_UTC=None,
    output_limit_bytes=None, compute_budget_enforced=False)
C['PCG64_seeds']['full_SOURCE_epsilon'] = 20261011650
for key in ('model_seed','updates','batch_per_region','identity_batch','region_categories','lr','lr_initial',
            'lr_final','warmup_updates','betas','gradient_clip','projection_directions','box_projection_iterations',
            'solver_steps','solver_check_steps','generation_chunk','PSD_loss_scale','valley_delta','valley_smoothing',
            'feature_names','feature_scale_floor','noise_scale_floor','source_order_rule','loss_weights'):
    assert C[key] == old[key], key
assert {k:C['PCG64_seeds'][k]for k in old['PCG64_seeds']} == old['PCG64_seeds']
write(S/'config.json', C)
frozen = ['network.py','noise_core.py','source_flow_core.py','precision_order.py','distribution_loss.py',
    'R61_distribution_loss.py','linux_owned_worker.py','linux_supervise.py','NOISE_CALIBRATION.npz',
    'NOISE_CALIBRATION.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json',
    'TRAIN_DISTRIBUTION_CONSTANTS.npz','TRAIN_TEMPORAL_CONSTANTS.npz']
for name in frozen:
    shutil.copy2(H/name, S/name)
shutil.copytree(G/'distribution_adapter', S/'distribution_adapter', ignore=shutil.ignore_patterns('__pycache__','*_cache'))
for name in ('portable_fixture.py','network_replay_artificial.py'):
    shutil.copy2(G/name, S/name)
shutil.copy2(protocol, S/'PROTOCOL.md')
shutil.copy2(R/'reviews/R65_PREPARATION_001/ROOT_TRAINING_SCHEMA.json', S/'TRAINING_SCHEMA.json')
write(S/'INPUT_BINDING.json', dict(UTC=NOW, **observed,
    original_acceptance=[bind(R/n)for n in ('PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md')],
    frozen_R63_helpers={n:bind(H/n)for n in frozen},
    R64_adapter_helpers=[bind(p)for p in (G/'distribution_adapter').rglob('*')if p.is_file()and'__pycache__'not in p.parts],
    original_R63_training_reference=bind(H/'COVERAGE_RESPONSE/TRAINING_RECEIPT.json'),
    R64_evidence=[bind(R/'R64_REPORT.md'),bind(R/'reviews/R64_ROOT_DIRECTION_001.json'),bind(G/'RESULT.json')],
    trained_weights_loaded=False, actual_D_free_bytes=shutil.disk_usage('D:/').free,
    delivery_bytes_estimate_not_science_quota=delivery_estimate, model_goal_complete=False))
action = ('Complete artificial joint D/C and real-network update proofs, run the single full empirical D candidate '
          'from original initialization to update50000, then original numeric/A-C/replay; original D/E only after passing.')
top = ('# 当前执行：R65 完整有限TRAIN分布训练已登记\n\n'
    'R64固定诊断及私有Git/D盘交付已完成。R65保留原初始化、网络、统计常数、Adam/LR、'
    '50000次唯一末点和一项原C微批；每步以全部6593 SOURCE和21047 REAL计算五D，'
    '新每父噪声流和全部64行分块VJP。D-C随机关联与数值聚合顺序变化已明确记录。'
    '先完成CPU/CUDA人工检查，再开始科学训练。原A–F、全部源父及区域质量保持；'
    '主助手负责科学方向与Git，全部代理继承相同模型/思考强度。无计算配额，'
    '单科学作业，服务器CPU12/RSS48GiB/GPU10GiB，本机共享CPU4/RSS4GiB和E/D存储。\n\n')
for name in ('AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md'):
    (R/name).write_bytes((top+'## 以下为历史记录\n\n'+(R/name).read_text('utf8')).encode())
job = dict(stage=ID, status='registered_not_launched', active_scientific_jobs=0)
for name in ('PROGRAM.json','PROJECT_PROGRESS.json'):
    value = read(R/name)
    value.update(current_stage=ID,current_task=C['protocol'],current_stage_scope='full_empirical_D_original_C_training',
        current_stage_progress='registered_not_launched', current_job=job, next_action=action,
        current_stage_registration='reviews/R65_REGISTRATION_001/RECEIPT.json', active_scientific_jobs=0,
        active_remote_scientific_jobs=0,active_local_scientific_jobs=0, subagents_allowed=True,
        application_goal_status='active',model_goal_complete=False,status='R65_registered_long_term_goal_active')
    value['execution_location'].update(remote_root=remote,mode='server_GPU_single_root_neural_training',
                                      preflight_required=True,current_stage_GPU_used=False)
    if name == 'PROGRAM.json':value['updated_at_utc']=NOW
    else:value.update(project_status=value['status'],updated_UTC=NOW,current_activity='root_implementing_R65',current_direction=action)
    write(R/name,value)
b = read(R/'RESOURCE_BUDGET.json')
b['current_registered_stage']=ID
b['improvement_block']['registered_stages'].append(dict(stage=ID,kind='full_empirical_D_original_C_training',
    registered_UTC=NOW,science_budget=None,planned_new_training_runs=1,fixed_terminal_update=50000,status='registered'))
write(R/'RESOURCE_BUDGET.json',b)
for name,key,identity in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
    value=read(R/name)
    value[key].append({identity:ID,'status':'registered','owner':'/root','registered_UTC':NOW,
        'protocol':C['protocol'],'output_root':'work/'+ID,'remote_root':remote,'new_training_runs_started':0,
        'fixed_terminal_update':50000,'model_goal_complete':False})
    write(R/name,value)
write(REG/'RECEIPT.json',dict(UTC=NOW,stage=ID,root_control_PID=os.getpid(),CPU_affinity=[cpu],
    native_working_set_hard_max_bytes=ws_max.value,native_working_set_flags=ws_flags.value,
    protocol=bind(protocol),config=bind(S/'config.json'),root_protocol_review=bind(R/'reviews/R65_PREPARATION_001/ROOT_PROTOCOL_REVIEW.json'),
    owner='/root',application_goal_status='active',original_A_F_unchanged=True,scientific_execution_started=False,
    response_training_runs_started=0,remote_root=remote,read_only_preflight=observed['read_only_preflight'],
    D_free_bytes=shutil.disk_usage('D:/').free,original_seven_streams_preserved=True,new_noise_seed=20261011650,
    full_D_and_C_joint_noise_law_changed=True,delivery_estimate_bytes=delivery_estimate))
print(json.dumps(dict(stage=ID,remote_root=remote,scientific_execution_started=False,
                     preflight=observed['read_only_preflight'],D_free_bytes=shutil.disk_usage('D:/').free),indent=2))
