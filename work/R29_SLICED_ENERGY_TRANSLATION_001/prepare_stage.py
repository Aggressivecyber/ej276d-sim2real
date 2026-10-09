from pathlib import Path
import os, ast, json, hashlib, shutil, datetime
ROOT=Path(__file__).resolve().parent;P=ROOT.parent.parent
R28=P/'work/R28_FRESH_PROJECTION_TRANSLATION_001'
for n in ['model.py','legacy_model.py','prior.py','fine_core.py','reorder_core.py','evaluation.py','assessment.py','diagnostic_core.py','energy_width.py','data_support.py','random_diagnostics.py','common_math.py','endpoint_common.py','temporal_common.py','supervise.py','owned_worker.py']:
    shutil.copyfile(R28/n,ROOT/n)
for n in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','EVALUATOR_SELECTIONS_PRIVATE.json','FROZEN_TILT_TABLE.json']:
    shutil.copyfile(R28/n,ROOT/n)
cfg=json.loads((R28/'config.json').read_text('utf8'))
cfg.update(stage=ROOT.name,registered_UTC='2026-10-09T14:52:30.236280+00:00',deadline_UTC='2026-10-09T16:12:30.236280+00:00',absolute_deadline_UTC='2026-10-09T16:22:30.236280+00:00',first_actual_command_UTC='2026-10-09T14:53:11.9328658+00:00',first_actual_PID=53192,first_actual_tool='02cd4f',
    arms=['SLICED_ENERGY'],endpoints=['ENERGY_SELECTED','ENERGY_FINAL'],R28_root=str(R28),
    critic_intervention='SLICED_ENERGY explicitly preserves STANDARDIZED input center and scale',
    scientific_package='only fine training quantile-L1 replaced by three-diagonal-excluded energy U; exact old row/random/fresh direction stream and other losses unchanged',
    old_control='saved R28 FRESH_SELECTED/FRESH_FINAL; no old generation, refit or re-evaluation',
    primary_intervention_comparison='ENERGY_FINAL versus R28 FRESH_FINAL at equal2950 updates if attained',
    expected_R28_projection_stream=str(R28/'FRESH_FINE/FRESH_PROJECTION_STREAM.json'),
    energy_formula='mean_k sum_i!=j (2abs(Ai-Bj)-abs(Ai-Aj)-abs(Bi-Bj))/(n(n-1))',
    energy_three_diagonals_excluded=True,energy_two_generated_arguments_differentiable=True,energy_negative_values_preserved=True,
    energy_natural_gradient_scale_changed=True,fine_validation_unchanged_W1=True,
    old_density_reference_location=str(R28/'metrics.json'))
cfg['endpoint_weight_paths']={'ENERGY_SELECTED':str(ROOT/'SLICED_ENERGY/TRANSLATOR.pt'),'ENERGY_FINAL':str(ROOT/'SLICED_ENERGY/FINAL_TRANSLATOR.pt')}
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
t=(R28/'online_training.py').read_text('utf8')
t=t.replace('from fine_core import FreshFineLoss as FineLoss,NAMES','from fine_core import NAMES\nfrom energy_core import EnergyFineLoss as FineLoss')
t=t.replace("'HOMOGENEOUS_REORDER','FRESH_FINE'", "'HOMOGENEOUS_REORDER','FRESH_FINE','SLICED_ENERGY'")
t=t.replace("seed=cfg['fresh_projection_seed'],root=root)", "seed=cfg['fresh_projection_seed'],root=root,expected_stream=cfg['expected_R28_projection_stream'])")
t=t.replace("fine_value,fine_groups=fine.training(y,real);", "fine_value,fine_groups=fine.training(y,real);fine.record_context(step,bn);")
t=t.replace("'fine_validation_consumes_fresh_RNG':False", "'fine_validation_consumes_fresh_RNG':False,'fine_training_distance':'energy_U','energy_training_summary':fine.summary()")
(ROOT/'online_training.py').write_text(t,'utf8')
train=(R28/'train_gpu.py').read_text('utf8').replace("'FRESH_FINE'","'SLICED_ENERGY'")
(ROOT/'train_gpu.py').write_text(train,'utf8')
# Exact frozen evaluator/core source, scientific adaptation is prepared separately before it runs.
for n in ['evaluate_endpoints.py','replay_endpoints.py','render_figures.py']:
    shutil.copyfile(R28/n,ROOT/n)
sources={}
for q in ROOT.glob('*.py'):
    b=q.read_bytes();ast.parse(b);sources[q.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_AST_pass':True,'sources':sources,'scientific_arrays_decoded_in_prepare':0,'old_stage_writes':0,'migration_content_tests':0,'first_actual_native_tool':'02cd4f','first_actual_PID':53192}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':receipt['UTC'],'prepared':str(ROOT),'no_response_fit_started':True}),flush=True)
