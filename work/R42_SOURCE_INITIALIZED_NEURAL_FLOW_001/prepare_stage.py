from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os,ast,hashlib
ROOT=Path(__file__).resolve().parent;P=ROOT.parents[1];OLD=ROOT.parent/'R41_PATCH_CONTENT_ABLATION_001';R36=ROOT.parent/'R36_CONDITIONAL_DIFFUSION_EDIT_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R42_REGISTRATION_001/RECEIPT.json').read_text('utf8'));prior=json.loads((OLD/'config.json').read_text('utf8'))
keys=['C2ST','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed','seed','threads','RSS_limit_bytes','GPU_allocated_limit_bytes','output_limit_bytes','C2ST_views','CPU_python','local_input_paths','training_input_paths','actual_training_random_states','actual_training_receipt','consumed_R26_stream','consumed_R26_receipt','local_training_input_paths','R27_root','R28_root','R32_root','R33_root','R35_root','R36_root','R37_root','R38_root','R39_root','affine_path','old_BASE_REAL_generated','old_BASE_REAL_metrics','server_python']
cfg={k:prior[k]for k in keys};A='SOURCE_NEURAL_FLOW_FINAL';R='SOURCE_NEURAL_FLOW_REAL_CONTROL'
env=json.loads((ROOT.parent/'R33_RESIDUAL_OT_FLOW_001/remote_receipts/SERVER_ENVIRONMENT.json').read_text('utf8'))
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],absolute_deadline_UTC=reg['stage_deadline_UTC'],
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R42_REMOTE_20261009_231107_UTC',
    arms=[A],endpoints=[A,R],updates=2950,batch=32,training_wall_s=900,training_total_limit_s=900,science_budget=2700,numeric_wall_limit_s=2700,
    model_seed=20261010420,t_seed=20261010331,lr=.0002,betas=[.9,.999],gradient_clip=5.,generation_chunk=64,solver_steps=32,solver_check_steps=64,
    initial_noise_V=.001,preADC_charge_relative_error_limit=1e-5,parameter_initialization='new zero-final U-Net; no pretrained response weights',
    architecture='R36 32/64/128/128 two ResBlocks, GN8,60-point4headattention, nearest-up skip, continuous17+8 embedding25',
    objective='only full480 standardized tangent velocity MSE',selection='sole actual2950 endpoint; no validation checkpoint selection',
    HGB_max_fits=20,HGB_cap=20,HGB_plan=20,density_EM_max=16,width_slots=16,main_generation_rows=4000,repeated_rows_cap=1024,source_probe_rows=832,extra_guide_probe_rows=128,replay_rows=4000,
    constants_path=str(ROOT/'UNIT_CHARGE_CONSTANTS.npz'),endpoint_weight_paths={A:str(ROOT/A/'FINAL_FLOW.pt'),R:str(ROOT/A/'FINAL_FLOW.pt')},
    expected_R33_environment={k:env[k]for k in ['python','numpy','torch','scipy','device','GPU']},
    original_training_receipt_path=str(ROOT.parent/'R33_RESIDUAL_OT_FLOW_001/TRAINING_NUMERIC_RECEIPT.json'),
    execution={'training':'registered Linux CUDA standalone source flow','generation_evaluation_replay':'local CPU4 E/D','actual_PID_nonce_RSS_before_imports':True},
    old_G_reader_D_used=False,old_response_weights_loaded=False,z_style_NCE_unused=True,fresh_J20_not_applicable=True,
    source_support_policy='all rows processed; conditioning index nearest edge only; no Q/wave clipping or BASE fallback',
    first_actual_command={'tool':'d6bd4e','PID':39220,'native_exit':0,'UTC':'2026-10-09T23:11:39.3072518Z'})
files=['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py','model.py','legacy_model.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py','covariance_description.py','generate_all.py','evaluate_saved.py','replay_all.py','reorder_core.py','retrieve.py']
for n in files:shutil.copy2(OLD/n,ROOT/n)
shutil.copy2(ROOT.parent/'R35_PSD_ORDERED_FLOW_001/psd_order_core.py',ROOT/'psd_order_core.py')
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for n in payload:shutil.copy2(OLD/n,ROOT/n)
# Only the saved R36 architecture is reused, never its trained state.
body=(R36/'diffusion_core.py').read_text('utf8');body=body[body.index('class ResidualBlock'):body.index('def record_noise')]
body=body.replace('class Denoiser','class FlowNetwork').replace('nn.Linear(27,128)','nn.Linear(25,128)').replace('def forward(self,y,t,b,c):','def forward(self,y,t,b):').replace('time=t.to(y).reshape(-1,1)/100.;','time=t.to(y).reshape(-1,1);').replace('onehot,c.to(y)','onehot')
body=body.replace('        time=t.to(y)',"        y=y.to(self.final.weight);t=t.to(y)\n        time=t.to(y)")
(ROOT/'network.py').write_text('from __future__ import annotations\nimport math,torch\nfrom torch import nn\n'+body,'utf8')
for n in ['generate_all.py','evaluate_saved.py','replay_all.py','retrieve.py']:
    s=(ROOT/n).read_text('utf8').replace('NO_PATCH_NCE_FINAL',A).replace('NO_PATCH_NCE_REAL_CONTROL',R)
    if n=='generate_all.py':
        s=s.replace("    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');", "    check=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'));assert check['pass'], 'fixed32 solver failed; no DEV generation'\n    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');")
        s=s.replace("'G_applied_N':2000","'standalone_flow_applied_N':2000").replace("'G_file':CFG['endpoint_weight_paths']","'flow_file':CFG['endpoint_weight_paths']")
        s=s.replace("'post_G_calibration_or_projection':False","'old_G_used':False,'charge_tangent_projection_inside_velocity':True")
        s=s.replace("'compatibility_flow_fields_no_actual_post_G_transform':True","'functional_component_semantics':{'base_signal':'actual full SOURCE input','base_electronic':'projected initial Gaussian, not electronics','base_sum_rounding':'float32 initial sum residue','flow_increment':'learned full480 flow plus final cast residual; not independent signal'},'unused_random_inputs':['z','style']")
        s=s.replace("'source_support_outside_new_G_no_fallback'","'source_support_outside_edge_conditioning_no_fallback'")
    if n=='evaluate_saved.py':s=s.replace('R40 saved evaluation complete','R42 saved evaluation complete')
    if n=='replay_all.py':s=s.replace('complete_source_to_G_to_oneADC_replayed','complete_source_to_standalone_flow_to_oneADC_replayed').replace('source_support_outside_new_G_no_fallback_replayed','source_support_outside_edge_conditioning_no_fallback_replayed')
    if n=='retrieve.py':
        s=s.replace("'MATCHED_TRAINING_FLOW_RECEIPT.json','INITIAL_EQUIVALENCE.json'","'MATCHED_TRAINING_FLOW_RECEIPT.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json','FLOW_FORMULA_CHECK.json'")
    (ROOT/n).write_text(s,'utf8')
remote=dict(cfg);r36=json.loads((R36/'remote_config.json').read_text('utf8'));r41=json.loads((OLD/'remote_config.json').read_text('utf8'))
for k in ['training_input_paths','affine_path','original_training_receipt_path']:remote[k]=r36[k]
for k in ['consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt']:remote[k]=r41[k]
remote['constants_path']=cfg['remote_root']+'/UNIT_CHARGE_CONSTANTS.npz'
for n,c in [('config.json',cfg),('remote_config.json',remote)]:(ROOT/n).write_text(json.dumps(c,indent=2,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
(ROOT/'PROTOCOL.md').write_text('''# R42 source-initialized flow, fixed registered recipe

Authoritative immutable task and registration R42. Full480 input drives the ODE initial state; no previous G/reader/D or response checkpoint. New R36-body U-Net25-dimensional continuous-time/bin embedding, zero last layer, fixed seed20261010420; unique MSE2950 updates/newAdam(.9,.999)/clip5. No GAN/NCE/identity/PSD penalty/SW/J20 or checkpoint selection.

q=sum100:400-3sum0:100. Shared REAL TRAIN per-bin unit-q mean (normalized to q=1), pointwise sample SD with max(.05medianSD,1e-6) floor. P(v)=v-mu*q(v); initial unit waveform SOURCE/q+P(.001*originalwhite/q). Actual field P(s*f)/s; arithmetic coordinates, projection and Heun state float64; network/weights float32. Final physical preADC cast float32 then original single14bit ADC. This precision implementation applies the same formulas, no final renormalization. Record float64 and actual float32 charge errors.

Original94400 source/REAL/bin/ordinal/white/t stream reused; PSD stable ordered statistical one-to-one coupling within original Q subgroup (source raw unnoised keys), no particle or physical pair claim. z/style/NCE positions retained as unused audit. Source/target positive finite q mandatory. All2000 including417 outside support receive new flow with nearest-edge constant/bin only, no q/voltage clip/fallback.

Before any DEV, first64 supported TRAIN source+first64 supported TRAIN REAL with original first128 white, fixed32 vs64 Heun; median<=.25LSB/p95<=1LSB and max preADC relativeq<=1e-5. Failure closes fixed solver, no step rescue. Local generation/replay chunk64 identical. Functional decomposition source/projected initial Gaussian/learned flow/cast residues is not physical electronics/signal independence.

20HGB/16slots only for SOURCE2000 and newREAL2000; same frozen600/250/group/bootstrap definitions. Old BASE_REAL R39 and all references reused with0newfits, RAW2D paired unknown. First five common gates unchanged; gate6 new model/flow/formula/solver/q/replay/finite resources (oldG/J20/NCE/fresh checks not applicable). One recipe, no scans. Science2700/train900s,4threads/RSS+GPU4GiB/output512MiB, E/D local cache/temp. Stop00:41:07.309998UTC, root00:51:07. No EVAL/CAL/newREAL/G4/install/agents/Git/migration tests.
''','utf8')
deploy=(OLD/'deploy_training.py').read_text('utf8')
start=deploy.index('files=');end=deploy.index('\nbindings={}',start)
deploy=deploy[:start]+"files=['train_server.py','source_flow_core.py','network.py','psd_order_core.py','reorder_core.py','linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']"+deploy[end:]
deploy=deploy.replace("assert (old/'inputs/TRAIN_PRIVATE.npz').is_file()and(old/'inputs/INITIAL_TRANSLATOR.pt').is_file()","assert (old/'inputs/TRAIN_PRIVATE.npz').is_file()")
(ROOT/'deploy_training.py').write_text(deploy,'utf8')
bindings={}
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_AST':bindings,'dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),'science_started':False,'old_response_weights_used':False,'old_stage_writes':0,'migration_content_tests':0}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':rec['UTC'],'AST_count':len(bindings),'dependency_bytes':rec['dependency_payload_bytes']}),flush=True)
