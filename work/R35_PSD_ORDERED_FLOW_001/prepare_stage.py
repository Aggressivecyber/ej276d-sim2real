from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]; OLD=ROOT.parent/'R33_RESIDUAL_OT_FLOW_001'
reg=json.loads((P/'reviews/R35_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
if (ROOT/'PREPARATION_RECEIPT.json').exists():raise RuntimeError('prepared already')
cfg=json.loads((OLD/'config.json').read_text('utf8'))
remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R35_REMOTE_20261009_190721_UTC'
oldremote=cfg['remote_root']
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=reg['stage_deadline_UTC'],science_budget=1800,numeric_wall_limit_s=1800,
    training_wall_s=900,training_total_limit_s=900,arms=['PSD_ORDERED_CFM'],endpoints=['PSD_ORDERED_CFM'],updates=2950,
    HGB_max_fits=10,HGB_cap=10,HGB_plan=10,density_EM_max=8,width_slots=8,
    main_generation_rows=2000,repeated_rows_cap=1024,source_probe_rows=832,replay_rows=2000,
    output_limit_bytes=402653184,R33_root=str(OLD),remote_R33_root=oldremote,remote_root=remote_root,
    endpoint_weight_paths={'PSD_ORDERED_CFM':str(ROOT/'PSD_ORDERED_CFM/FINAL_FLOW.pt')},
    first_actual_command={'tool':'944ece','PID':61832,'native_exit':0,'UTC':'2026-10-09T19:08:10.958070+00:00'},
    execution={'training':'R33 same server Python/CUDA; one PSD-ordered flow fit','evaluation_replay':'local CPU4',
        'CPU_python':cfg['CPU_python'],'actual_PID_nonce_RSS_before_imports':True},
    sole_training_intervention='stable same-order raw BASE preADC PSD and REAL PSD pairing within each original Q subbatch',
    selection='actual final only',endpoint_selection_uses_DEV=False,projection_applied=False,
    source_PSD_key='raw frozen G preADC x0, each f32 waveform promoted to f64, baseline100, fast100:150/total100:400',
    invalid_sort_key_policy='persist exact step/subbatch and stop; no dropping, fallback, resampling or clamp',
    solver_check_rows_each=32,response_fit_attempts_max=1,
    initial_flow_path=str(OLD/'INITIAL_FLOW.pt'),affine_path=str(OLD/'AFFINE_CONSTANTS.npz'),
    original_training_receipt_path=str(OLD/'TRAINING_NUMERIC_RECEIPT.json'),
    original_base_hashes_path=str(OLD/'INDEPENDENT_CFM/BASE_FORWARD_HASHES.json'))
code=['model.py','legacy_model.py','reorder_core.py','flow_core.py','flow_runtime.py','evaluation.py',
    'diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py',
    'common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','supervise.py',
    'owned_worker.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py']
for name in code:shutil.copy2(OLD/name,ROOT/name)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz',
    'DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json',
    'DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for name in payload:shutil.copy2(OLD/name,ROOT/name)
remote=json.loads(json.dumps(cfg))
oldcfg=json.loads((OLD/'remote_config.json').read_text('utf8'))
for key in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','base_G']:
    remote[key]=oldcfg[key]
remote.update(initial_flow_path=oldremote+'/INITIAL_FLOW.pt',affine_path=oldremote+'/AFFINE_CONSTANTS.npz',
    original_training_receipt_path=oldremote+'/TRAINING_NUMERIC_RECEIPT.json',
    original_base_hashes_path=oldremote+'/INDEPENDENT_CFM/BASE_FORWARD_HASHES.json',
    expected_R33_environment=json.loads((OLD/'remote_receipts/SERVER_ENVIRONMENT.json').read_text('utf8')))
for name,value in [('config.json',cfg),('remote_config.json',remote)]:
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
text=(ROOT/'linux_supervise.py').read_text('utf8').replace("30 if entry=='pid_probe.py' else 1850","960")
(ROOT/'linux_supervise.py').write_text(text,'utf8')
text=(ROOT/'launch_remote.py').read_text('utf8').replace("['pid_probe.py','train_server.py']","['train_server.py']")
(ROOT/'launch_remote.py').write_text(text,'utf8')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'copied_dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),
    'remote_input_payloads':'read-only R33 existing inputs, initial state, affine constants and actual stream receipts',
    'migration_content_tests':0,'science_started':False,'old_stage_writes':0}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt))
