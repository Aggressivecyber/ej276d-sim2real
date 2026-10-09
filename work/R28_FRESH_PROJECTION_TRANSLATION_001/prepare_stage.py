from pathlib import Path
import ast, json, shutil, hashlib, datetime, os

ROOT = Path(__file__).resolve().parent
P = ROOT.parent.parent
R26 = P / 'work/R26_CONDITIONAL_BATCH_TRANSLATION_001'
R27 = P / 'work/R27_FIXED_PROJECTION_COVERAGE_DIAGNOSTIC_001'
for n in ['model.py','legacy_model.py','prior.py','evaluation.py','assessment.py','diagnostic_core.py',
          'energy_width.py','data_support.py','random_diagnostics.py','common_math.py','endpoint_common.py',
          'supervise.py','owned_worker.py','replay_endpoints.py']:
    shutil.copyfile(R26/n, ROOT/n)
for n in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz',
          'JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json',
          'VALLEY_DEFINITION.json','EVALUATOR_SELECTIONS_PRIVATE.json','FROZEN_TILT_TABLE.json']:
    shutil.copyfile(R26/n, ROOT/n)
cfg = json.loads((R26/'config.json').read_text('utf8'))
cfg.update(stage=ROOT.name, registered_UTC='2026-10-09T14:11:09.766899+00:00',
    deadline_UTC='2026-10-09T15:31:09.766899+00:00', absolute_deadline_UTC='2026-10-09T15:41:09.766899+00:00',
    first_actual_command_UTC='2026-10-09T14:11:50.633161+00:00', first_actual_PID=22864, first_actual_tool='312724',
    arms=['FRESH_FINE'], endpoints=['FRESH_SELECTED','FRESH_FINAL'], HGB_max_fits=20, HGB_plan=20, HGB_cap=20,
    width_slots=16, density_EM_max=16, repeated_rows_planned=2048, repeated_rows_cap=2048,
    source_probe_rows=1664, main_generation_rows=4000, replay_rows=4000,
    R26_root=str(R26), R27_root=str(R27), fresh_projection_seed=20261009281,
    fresh_projection_draw_order=['shape480','diff479','baseline100_V'],
    fresh_projection_generation_dtype='float64 Gaussian and unit normalization, cast float32 for Torch calculation',
    fresh_projection_stream_consumers='fine.training exactly once per actual G update; no validation/D/identity/evaluation',
    critic_intervention='FRESH_FINE explicitly preserves STANDARDIZED fixed input center and scale',
    scientific_package='only fresh independent fine training projections; row bindings/network/loss weights/optimizer/critic unchanged',
    old_control='saved R26 HOMO_SELECTED/HOMO_FINAL and corresponding R27 evaluator results, no old refit/generation',
    consumed_R26_stream=str(R26/'HOMOGENEOUS_REORDER/TRAINING_STREAMS_PRIVATE.npz'),
    consumed_R26_receipt=str(R26/'HOMOGENEOUS_REORDER/TRAINING_RECEIPT.json'),
    old_reorder_receipt=str(R26/'REORDER_CONSTRUCTION_RECEIPT.json'),
    primary_intervention_comparison='FRESH_FINAL versus R26 HOMO_FINAL at identical2950 updates if reached',
    source_rng_seed_and_pairing='read actual R26 reordered rows and original ordinal; reconstruct old noise and style from original recorded PCG64 states; NCE unchanged')
cfg['endpoint_weight_paths']={'FRESH_SELECTED':str(ROOT/'FRESH_FINE/TRANSLATOR.pt'),
                              'FRESH_FINAL':str(ROOT/'FRESH_FINE/FINAL_TRANSLATOR.pt')}
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
t=(R26/'online_training.py').read_text('utf8')
assert "['STANDARDIZED','HOMOGENEOUS_REORDER']" in t
t=t.replace("['STANDARDIZED','HOMOGENEOUS_REORDER']", "['STANDARDIZED','HOMOGENEOUS_REORDER','FRESH_FINE']")
t=t.replace('from fine_core import FineLoss,NAMES','from fine_core import FreshFineLoss as FineLoss,NAMES')
t=t.replace('fine=FineLoss(constants,device);', "fine=FineLoss(constants,device,seed=cfg['fresh_projection_seed'],root=root);")
t=t.replace("for n,obj in [('FINAL_TRANSLATOR.pt'", "fine.persist();assert fine.calls==updates\n        for n,obj in [('FINAL_TRANSLATOR.pt'")
t=t.replace("'initial_D':'R18 initial no R22pretraining'", "'initial_D':'R18 initial no R22pretraining','critic_STANDARDIZED_input':True,'fresh_projection_calls':fine.calls,'fine_validation_consumes_fresh_RNG':False")
t=t.replace("except BaseException as e:\n        (root/'FAILED_TRAINING.json')", "except BaseException as e:\n        fine.persist();\n        (root/'FAILED_TRAINING.json')")
(ROOT/'online_training.py').write_text(t,'utf8');ast.parse(t)
# Original fixed-projection validation and all feature definitions stay intact.
fine=(R26/'fine_core.py').read_text('utf8')
(ROOT/'fine_core.py').write_text(fine+'\n'+(ROOT/'fresh_fine_append.txt').read_text('utf8'),'utf8')
train=(R26/'train_gpu.py').read_text('utf8').replace("'HOMOGENEOUS_REORDER'", "'FRESH_FINE'").replace("ROOT/'HOMOGENEOUS_REORDER'", "ROOT/'FRESH_FINE'")
(ROOT/'train_gpu.py').write_text(train,'utf8')
sources={}
for p in ROOT.glob('*.py'):
    data=p.read_bytes();ast.parse(data);sources[p.name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_AST_pass':True,'sources':sources,'scientific_arrays_decoded_in_prepare':0,'source_old_stages_immutable':True,
    'migration_content_tests':0,'current_critic_branch_STANDARDIZED':True,'same_R26_rows_no_new_permutation':True},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_prepared':str(ROOT),'no_science_started':True}))
