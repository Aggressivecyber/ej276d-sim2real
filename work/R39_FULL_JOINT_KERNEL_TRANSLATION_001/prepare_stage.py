from pathlib import Path
from datetime import datetime,timezone
import os,json,shutil,ast,hashlib
ROOT=Path(__file__).resolve().parent;P=ROOT.parents[1];R32=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001';R38=ROOT.parent/'R38_SLICED_MONOTONE_TRANSPORT_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R39_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
addon=P/'tasks/R39_REAL_CONTROL_SCOPE_ADDENDUM_001.md';addbytes=addon.read_bytes()
assert hashlib.sha256(addbytes).hexdigest()=='58ab02ccc12222d89e2bfff97de09cd29f369c0464db6872d7d8d9275f1fe697'
cfg=json.loads((R38/'config.json').read_text('utf8'));train=json.loads((R32/'config.json').read_text('utf8'))
arm='JOINT_KERNEL_FINAL';remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R39_REMOTE_20261009_214432_UTC'
for k in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','G_initial','reader_initial','D_initial','seed','lr','betas','gradient_clip','loss_weights','style_reconstruction_weight','batch','validation_every','fresh_projection_seed','style_train_seed','old_noise_seed','reorder_seed']:
    cfg[k]=train[k]
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],absolute_deadline_UTC=reg['stage_deadline_UTC'],
    remote_root=remote,arms=[arm],endpoints=[arm,'BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL'],updates=2950,training_wall_s=900,training_total_limit_s=900,
    scientific_package='single joint three-block Gaussian-mixture MMD replaces fine projected training quantile loss',
    kernel_sigmas=[.25,.5,1,2,4],initial_gradient_calibration_batches=64,mean_coefficient=0.,science_budget=1800,
    selection='sole final2950 DEV endpoint; original J20 selected weights recorded only',
    endpoint_weight_paths={arm:str(ROOT/arm/'FINAL_TRANSLATOR.pt'),'BASE_REAL_CONTROL':cfg['base_G'],'JOINT_KERNEL_REAL_CONTROL':str(ROOT/arm/'FINAL_TRANSLATOR.pt')},
    HGB_max_fits=30,HGB_plan=30,HGB_cap=30,density_EM_max=24,width_slots=24,main_generation_rows=6000,repeated_rows_cap=1024,source_probe_rows=832,replay_rows=6000,
    R38_root=str(R38),first_actual_command={'tool':'f0da4c','PID':47824,'native_exit':0,'UTC':'2026-10-09T21:47:03.0078409+00:00'},
    execution={'training':'same R32 Linux CUDA environment one new neural fit','evaluation_replay':'localCPU4','CPU_python':cfg['CPU_python'],'actual_scientific_PID_nonce_RSS_before_imports':True},
    REAL_control_gate='three-view paired candidate-minus-same-treated-BASE95%upper<=.10 and pointdelta<=.01; RAW absolute descriptive',
    REAL_control_scope_addendum={'path':str(addon),'sha256':hashlib.sha256(addbytes).hexdigest(),'bytes':len(addbytes)})
for n in ['model.py','legacy_model.py','prior.py','fine_core.py','reorder_core.py']:
    shutil.copy2(R32/n,ROOT/n)
for n in ['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py','covariance_description.py']:
    shutil.copy2(R38/n,ROOT/n)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for n in payload:shutil.copy2(R38/n,ROOT/n)
t=(ROOT.parent/'R28_FRESH_PROJECTION_TRANSLATION_001/online_training.py').read_text('utf8')
t=t.replace('from fine_core import FreshFineLoss as FineLoss,NAMES','from kernel_core import JointKernelLoss as FineLoss\nfrom fine_core import NAMES')
t=t.replace("'FRESH_FINE']","'FRESH_FINE','JOINT_KERNEL_FINAL']")
t=t.replace("actual.pop('original_noise')","actual['original_noise']").replace("actual.pop('original_style')","actual['original_style']")
t=t.replace("clock=time.monotonic();initial=validate(0)","clock=time.monotonic();initial=validate(0);calibration=fine.calibrate(model,xs,xt,actual,cfg['batch'])")
t=t.replace("counter['training_s']+time.monotonic()-clock>=900-10","counter['training_s']+time.monotonic()-clock>=cfg['training_total_limit_s']-10")
t=t.replace("'fine_train':float(fine_value),","'MMD_raw':fine.rows[-1][0],'MMD_scaled':fine.rows[-1][1],'MMD_coefficient':fine.c,'Gram_summary':fine.rows[-1][2:],'negative_MMD_batches':sum(v[0]<0 for v in fine.rows),'fine_train':float(fine_value),")
t=t.replace("'fresh_projection_calls':fine.calls","'joint_MMD_update_calls':fine.calls,'initial_gradient_coefficient':fine.c,'initial_extra_source_forward_rows':2048,'MMD_negative_batches':sum(v[0]<0 for v in fine.rows),'fresh_projection_calls':0")
t=t.replace("fine.persist();assert fine.calls==updates","fine.persist();assert fine.calls==updates")
t=t.replace("fine.persist();\n        (root/'FAILED_TRAINING.json')","fine.persist();\n        (root/'FAILED_TRAINING.json')")
(ROOT/'online_training.py').write_text(t,'utf8')
remote_cfg=json.loads(json.dumps(cfg));oldremote=json.loads((R32/'remote_config.json').read_text('utf8'))
for k in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','G_initial','reader_initial','D_initial']:
    remote_cfg[k]=oldremote[k]
for n,c in [('config.json',cfg),('remote_config.json',remote_cfg)]:
    (ROOT/n).write_text(json.dumps(c,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
source=[]
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));source.append({'name':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_AST':source,'dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),'science_started':False,'old_stage_writes':0,'migration_content_tests':0}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
