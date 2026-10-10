from pathlib import Path
from datetime import datetime, timezone
import os, json, ast, hashlib, shutil

R = Path(__file__).resolve().parent
P = R.parent.parent
OLD = R.parent / 'R49_FROZEN_HEUN128_RESPONSE_001'
def write(name, value):
    (R/name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())
def stage(prefix):
    paths = list(R.parent.glob(prefix+'_001'))
    if len(paths) != 1: raise RuntimeError((prefix, [str(p) for p in paths]))
    return paths[0]

cfg = json.loads((OLD/'config.json').read_text('utf8'))
r48 = stage('R48_GUIDED_NOISE_COMPLETION_FLOW')
r42 = stage('R42_SOURCE_INITIALIZED_NEURAL_FLOW')
r43 = stage('R43_SCALAR_TIME_NEURAL_FLOW')
r46 = stage('R46_FIXED_INTEGRATION_BUDGET')
mapping = {
    'R7_root':stage('R7_RESPONSE_RESIDUAL_DIAGNOSIS'),
    'R26_root':stage('R26_CONDITIONAL_BATCH_TRANSLATION'),
    'R27_root':stage('R27_FIXED_PROJECTION_COVERAGE_DIAGNOSTIC'),
    'R32_root':stage('R32_CONDITIONAL_MEAN_TRANSLATION'),
    'R33_root':stage('R33_RESIDUAL_OT_FLOW'),
    'R35_root':stage('R35_PSD_ORDERED_FLOW'),
    'R36_root':stage('R36_CONDITIONAL_DIFFUSION_EDIT'),
    'R37_root':stage('R37_FULL_COVARIANCE_CALIBRATION'),
    'R38_root':stage('R38_SLICED_MONOTONE_TRANSPORT'),
    'R39_root':stage('R39_FULL_JOINT_KERNEL_TRANSLATION'),
}
cfg.update({k:str(v) for k,v in mapping.items()})
cfg.update(stage=R.name, execution={'training':'none; new updates 0',
    'numeric_generation_evaluation_replay':'Windows local CPU4/chunk64/Heun128',
    'actual_PID_nonce_RSS_before_imports':True},
    threads=4, CPU_threads=4, CPU_local_threads=4, RSS_limit_bytes=4*1024**3,
    GPU_count=0, GPU_limit_bytes=None, GPU_allocated_limit_bytes=None, GPU_reserved_limit_bytes=None,
    remote_root=None, MemAvailable_min_bytes=None,
    historical_training_root=str(r48), historical_local_training_root=str(r48),
    frozen_eval_input_root=str(r42), evaluation_code_reference_root=str(OLD),
    constants_path=str(r43/'UNIT_CHARGE_CONSTANTS.npz'),
    constants_from=str(r43/'UNIT_CHARGE_CONSTANTS.npz'),
    train_solver_input=str(r46/'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz'),
    old_BASE_REAL_generated=str(mapping['R39_root']/'BASE_REAL_CONTROL/generated.npz'),
    old_BASE_REAL_metrics=str(mapping['R39_root']/'metrics.json'),
    first_actual_command=json.loads((R/'PREPARATION_ACK.json').read_text('utf-8-sig')),
    parent_remote_stage=str(OLD), remote_original_replay_terminal='unknown',
    global_single_physical_job_confirmed=False, science_budget=None,
    science_budget_s=None, numeric_wall_limit_s=None, wall_budget_s=None,
    deadline_UTC=None, absolute_deadline_UTC=None, output_limit_bytes=None,
    compute_budget_enforced=False, deadlines_enforced=False,
    local_registration=str(P/'reviews/R49_LOCAL_REGISTRATION_001/RECEIPT.json'))
cfg['local_input_paths']['width_definition'] = str(stage('R3B_ENERGY_CONDITIONAL_WIDTH')/'results/DIAGNOSTIC_DEFINITION.json')
cfg['local_training_input_paths']['FINE_CONSTANTS.npz'] = str(stage('R23_ONLINE_STANDARDIZED_CRITIC')/'FINE_CONSTANTS.npz')
cfg['endpoint_weight_paths'] = {n:str(r48/a/'FINAL_FLOW.pt') for a,names in
    zip(cfg['arms'], [[cfg['endpoints'][0],cfg['endpoints'][1]],[cfg['endpoints'][2],cfg['endpoints'][3]]]) for n in names}

names = ['network.py','noise_core.py','source_flow_core.py','flow_runtime.py','check_models.py',
    'generate_all.py','flow_diagnostics.py','evaluate_saved.py','replay_all.py','evaluation.py',
    'diagnostic_core.py','assessment.py','energy_width.py','data_support.py','common_math.py',
    'model.py','legacy_model.py','endpoint_common.py','temporal_common.py','mean_description.py',
    'flow_core.py','summarize_and_plot.py','render_figures.py','render_train.py','make_report.py']
bindings=[]
for name in names:
    text=(OLD/name).read_text('utf8').replace('\r','')
    text=text.replace("torch.set_num_threads(12)","torch.set_num_threads(4)").replace("torch.set_num_interop_threads(12)","torch.set_num_interop_threads(4)")
    text=text.replace("os.environ[key]='12'","os.environ[key]='4'").replace("'CPU_threads':12","'CPU_threads':4")
    if name == 'make_report.py':
        text=text.replace("read('REMOTE_RUN_HISTORY.json')","read('RUN_HISTORY.json')")
        text=text.replace('服务器同一CPU后端、12线程','本机同一CPU后端、4线程')
    ast.parse(text,filename=name)
    (R/name).write_bytes(text.encode())
    bindings.append({'file':name,'source':str(OLD/name),'bytes':len(text.encode()),'SHA256':hashlib.sha256(text.encode()).hexdigest()})
shutil.copyfile(r48/'NOISE_CALIBRATION.npz',R/'NOISE_CALIBRATION.npz')
shutil.copyfile(r48/'NOISE_CALIBRATION.json',R/'NOISE_CALIBRATION.json')
for a in cfg['arms']:(R/a).mkdir(exist_ok=True)
for name in ['config.json','config_cpu.json']:write(name,cfg)

required = [Path(cfg['train_solver_input']),Path(cfg['constants_path']),Path(cfg['local_input_paths']['width_definition']),
    *[Path(cfg['endpoint_weight_paths'][a]) for a in cfg['arms']],
    *[r48/a/'TRAINING_RECEIPT.json' for a in cfg['arms']],
    *[r42/n for n in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz',
       'JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','VALLEY_DEFINITION.json','CAL_REFERENCE_METADATA.json','DATA_ROLES.json']],
    mapping['R26_root']/'results/C2ST_PRIVATE.npz', mapping['R27_root']/'results/PROJECTION_SPACE_PRIVATE.npz',
    mapping['R7_root']/'results/DIAGNOSTIC_SELECTIONS.json',mapping['R7_root']/'results/C2ST_HOLDOUT_PREDICTIONS.npz',
    mapping['R7_root']/'metrics.json',mapping['R39_root']/'metrics.json',Path(cfg['old_BASE_REAL_generated']),
    mapping['R32_root']/'CONTROL_FINAL/generated.npz',mapping['R33_root']/'INDEPENDENT_CFM/generated.npz']
views=cfg['C2ST_views']+['normalized_u480','seen_reconstruction480','complement_reconstruction480']
required += [mapping['R32_root']/'results'/('CONTROL_FINAL__'+v+'_PREDICTIONS_PRIVATE.npz') for v in views]
required += [mapping['R39_root']/'results'/('BASE_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz') for v in views]
required += [mapping['R27_root']/'results'/('TRAIN_REFERENCE__'+v+'_PREDICTIONS_PRIVATE.npz') for v in views[-3:]]
required += [mapping['R27_root']/'results'/('C_TRAIN_REFERENCE__'+v+'.json') for v in views[-3:]]
inventory=[{'path':str(p),'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file()else None}for p in required]
write('LOCAL_INPUT_INVENTORY.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'inputs':inventory,'missing':[str(p)for p in required if not p.is_file()],
    'scientific_arrays_decoded':0,'source_migration_hash_tests':0,'weights_and_large_inputs_referenced_not_copied':True})
write('SOURCE_BINDINGS.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'files':bindings})
write('ENVIRONMENT_IMPORT_FAILURE_001.json',{'tools':['d936b1','0c769e'],'error':"ModuleNotFoundError: No module named 'psutil'",
    'science_started':False,'resolution':'no installation; existing Windows ctypes guard used',
    'successful_followup_tool':'d2d4f3','successful_followup_PID':27664})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'sources':len(bindings),
    'dependencies':len(inventory),'missing':[x['path']for x in inventory if not x['exists']]}),flush=True)
if any(not x['exists']for x in inventory):raise SystemExit(1)
