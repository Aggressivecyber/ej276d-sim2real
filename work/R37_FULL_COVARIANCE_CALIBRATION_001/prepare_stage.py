from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil

ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]
OLD=ROOT.parent/'R36_CONDITIONAL_DIFFUSION_EDIT_001'; R33=ROOT.parent/'R33_RESIDUAL_OT_FLOW_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R37_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
prior=json.loads((OLD/'config.json').read_text('utf8'))
keys=['C2ST','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed','seed','C2ST_views',
      'CPU_python','RSS_limit_bytes','GPU_allocated_limit_bytes','local_input_paths','local_training_input_paths',
      'R26_root','R27_root','R28_root','R32_root','R33_root','R35_root','base_G']
cfg={k:prior[k] for k in keys}
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=reg['stage_deadline_UTC'],science_budget=1800,numeric_wall_limit_s=1800,
    training_wall_s=900,statistics_fit_wall_s=900,updates=2950,batch=32,threads=4,generation_chunk=128,
    arms=['FULL_COVARIANCE_CALIBRATION'],endpoints=['FULL_COVARIANCE_CALIBRATION'],
    HGB_cap=10,HGB_max_fits=10,density_EM_max=8,width_slots=8,main_generation_rows=2000,
    repeated_rows_cap=1024,source_probe_rows=832,extra_guide_probe_rows=128,replay_rows=2000,
    output_limit_bytes=536870912,eigenfloor_trace_fraction=1e-6,covariance_ddof=1,
    original_covariance_negative_relative_limit=1e-8,closure_relative_Frobenius_limit=1e-6,
    full_dimension=480,selection='unique complete statistical fit; no checkpoint selection',
    R36_root=str(OLD),remote_R33_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC',
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R37_REMOTE_20261009_204109_UTC',
    server_python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
    affine_path=str(ROOT/'AFFINE_CONSTANTS.npz'),
    endpoint_weight_paths={'FULL_COVARIANCE_CALIBRATION':str(ROOT/'FULL_COVARIANCE_CALIBRATION/CALIBRATION_MAPS.npz')},
    execution={'statistics_fit':'same R33 Linux environment, CUDA frozen G batch32 then CPU4 float64 moments/maps',
               'generation_evaluation_replay':'local CPU4; G chunk128; calibration float64',
               'CPU_python':prior['CPU_python'],'actual_PID_nonce_RSS_before_imports':True},
    first_actual_command={'tool':'5dc2b1','PID':48248,'native_exit':0,'UTC':'2026-10-09T20:41:42.185129+00:00'},
    new_neural_fit_or_updates=0,new_EVAL=0,new_CAL=0,new_G4=0,new_REAL=0,installation=0,migration_content_tests=0)
code=['model.py','legacy_model.py','evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py',
      'endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py',
      'flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py']
for name in code:shutil.copy2(OLD/name,ROOT/name)
shutil.copy2(OLD/'resource_check_corrected.py',ROOT/'resource_check.py')
shutil.copy2(R33/'reorder_core.py',ROOT/'reorder_core.py')
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz',
         'REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for name in payload:shutil.copy2(OLD/name,ROOT/name)
remote=dict(cfg); previous=json.loads((R33/'remote_config.json').read_text('utf8'))
for k in ['training_input_paths','consumed_R26_stream','consumed_R26_receipt','actual_training_random_states','actual_training_receipt']:
    remote[k]=previous[k]
remote.update(base_G=previous['base_G'],affine_path=cfg['remote_R33_root']+'/AFFINE_CONSTANTS.npz',
    original_training_receipt_path=cfg['remote_R33_root']+'/TRAINING_NUMERIC_RECEIPT.json')
for n,c in [('config.json',cfg),('remote_config.json',remote)]:
    (ROOT/n).write_text(json.dumps(c,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'copied_dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),
     'new_science_started':False,'old_stage_writes':0,'migration_content_tests':0,
     'control_lookup_error':{'tool':'b56192','missing':'R35/generate_all.py','science':0,'reuse':'R36/generate_all.py interface'}}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
