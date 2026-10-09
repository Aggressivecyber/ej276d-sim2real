from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil

ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]
OLD=ROOT.parent/'R37_FULL_COVARIANCE_CALIBRATION_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R38_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
cfg=json.loads((OLD/'config.json').read_text('utf8'))
for key in ['eigenfloor_trace_fraction','covariance_ddof','original_covariance_negative_relative_limit','closure_relative_Frobenius_limit']:
    cfg.pop(key,None)
arm='SLICED_MONOTONE_TRANSPORT'
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=reg['stage_deadline_UTC'],arms=[arm],endpoints=[arm],
    R37_root=str(OLD),remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R38_REMOTE_20261009_211706_UTC',
    endpoint_weight_paths={arm:str(ROOT/arm/'SLICED_TRANSPORT_PARAMS.npz')},affine_path=str(ROOT/'AFFINE_CONSTANTS.npz'),
    layers=32,directions=16,direction_seed=20261010381,direction_gradient_steps=20,
    direction_stepsize=.1,direction_zero_norm=1e-12,direction_subsample=1024,
    quantile_knots=257,quantile_low=.005,quantile_high=.995,orthogonality_tolerance=1e-10,
    execution={'statistics_fit':'R33 Linux environment, CUDA original G32 then CPU4 float64 directions/quantile maps',
               'generation_evaluation_replay':'local CPU4, frozen G128, sliced maps float64',
               'CPU_python':cfg['CPU_python'],'actual_PID_nonce_RSS_before_imports':True},
    first_actual_command={'tool':'6b89d2','PID':60576,'native_exit':0,'UTC':'2026-10-09T21:18:06.922876+00:00'},
    selection='complete32-layer endpoint only, no DEV selection',new_neural_fit_or_updates=0)
code=['model.py','legacy_model.py','evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py',
      'endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py',
      'flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py',
      'resource_check.py','reorder_core.py','covariance_description.py']
for name in code: shutil.copy2(OLD/name,ROOT/name)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz',
         'REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for name in payload: shutil.copy2(OLD/name,ROOT/name)
remote=json.loads((OLD/'remote_config.json').read_text('utf8'))
remote.update(cfg)
remote['base_G']=json.loads((OLD/'remote_config.json').read_text('utf8'))['base_G']
remote['affine_path']=cfg['remote_R33_root']+'/AFFINE_CONSTANTS.npz'
for n,c in [('config.json',cfg),('remote_config.json',remote)]:
    (ROOT/n).write_text(json.dumps(c,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'copied_dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),
     'new_science_started':False,'old_stage_writes':0,'migration_content_tests':0}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
