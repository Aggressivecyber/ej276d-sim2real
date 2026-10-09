from pathlib import Path
from datetime import datetime, timezone
import json, shutil, os, ast, hashlib

ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]
OLD=ROOT.parent/'R35_PSD_ORDERED_FLOW_001'; R33=ROOT.parent/'R33_RESIDUAL_OT_FLOW_001'
reg=json.loads((P/'reviews/R36_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
old=json.loads((OLD/'config.json').read_text('utf8'))
keys=['C2ST','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed','seed','C2ST_views',
      'CPU_python','RSS_limit_bytes','GPU_allocated_limit_bytes','local_input_paths','local_training_input_paths',
      'R26_root','R27_root','R28_root','R32_root','R33_root','base_G']
cfg={k:old[k] for k in keys}
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],
    absolute_deadline_UTC=reg['stage_deadline_UTC'],science_budget=3000,numeric_wall_limit_s=3000,
    training_wall_s=900,updates=2950,batch=32,threads=4,generation_chunk=64,
    arms=['CONDITIONAL_DIFFUSION_EDIT'],endpoints=['CONDITIONAL_DIFFUSION_EDIT','REAL_GUIDE_EDIT'],
    HGB_cap=20,HGB_max_fits=20,density_EM_max=16,width_slots=16,
    main_generation_rows=4000,repeated_rows_cap=1024,source_probe_rows=832,extra_guide_probe_rows=128,replay_rows=4000,
    output_limit_bytes=536870912,model_seed=20261010360,training_noise_seed=20261010361,inference_seed=20261010362,
    noise_steps=100,t_start=50,cosine_s=.008,beta_cap=.999,optimizer={'lr':2e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'clip':5},
    training_objective='full480 epsilon MSE only',selection='actual final; no EMA or checkpoint selection',
    architecture={'encoder_lengths':[480,240,120,60],'channels':[32,64,128,128],
        'encoder_resblocks_each':2,'decoder_resblocks_at_120_240_480_each':2,'GN_groups':8,'attention_heads':4,
        'input_channels':['noisy_state','absolute_position'],'condition_embedding_dim':27,'embedding_hidden':128},
    source_support='original actual clean source-Q bin; outside retain frozen-G',
    physical_signal_electronic_split='unknown after editing',ADC='single original float32 14bit',
    no_x0_clipping_projection_smoothing_or_integral_adjustment=True,
    first_actual_command={'tool':'681787','PID':45868,'native_exit':0,'UTC':'2026-10-09T19:48:18.5595425+00:00'},
    execution={'training':'registered Linux CUDA existing environment','generation_evaluation_replay':'local CPU4',
        'CPU_python':old['CPU_python'],'actual_PID_nonce_RSS_before_imports':True},
    R35_root=str(OLD),remote_R33_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC',
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R36_REMOTE_20261009_194711_UTC',
    server_python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
    endpoint_weight_paths={'CONDITIONAL_DIFFUSION_EDIT':str(ROOT/'CONDITIONAL_DIFFUSION_EDIT/FINAL_DENOISER.pt'),
                           'REAL_GUIDE_EDIT':str(ROOT/'CONDITIONAL_DIFFUSION_EDIT/FINAL_DENOISER.pt')},
    affine_path=str(ROOT/'AFFINE_CONSTANTS.npz'),new_EVAL=0,new_CAL=0,new_G4=0,new_REAL=0,installation=0,migration_content_tests=0)
code=['model.py','legacy_model.py','evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py',
      'endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py',
      'flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py']
for n in code:shutil.copy2(OLD/n,ROOT/n)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz',
         'DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json',
         'VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for n in payload:shutil.copy2(OLD/n,ROOT/n)
rcfg=json.loads((OLD/'remote_config.json').read_text('utf8'))
remote=dict(cfg)
for k in ['training_input_paths','consumed_R26_stream','consumed_R26_receipt']:
    remote[k]=rcfg[k]
remote.update(affine_path=cfg['remote_R33_root']+'/AFFINE_CONSTANTS.npz',
    original_training_receipt_path=cfg['remote_R33_root']+'/TRAINING_NUMERIC_RECEIPT.json',
    expected_R33_environment=rcfg['expected_R33_environment'])
for n,c in [('config.json',cfg),('remote_config.json',remote)]:
    (ROOT/n).write_text(json.dumps(c,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'copied_dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),
         'remote_inputs':'read-only R33 REAL target/consumed target-bin stream and TRAIN affine; no source forward during denoiser training',
         'migration_content_tests':0,'science_started':False,'old_stage_writes':0,
         'preparation_read_failure':{'tool':'30ea4d','exit':1,'missing':'R35/replay.py','correct_file':'replay_endpoints.py','scientific_arrays_read':0}}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
