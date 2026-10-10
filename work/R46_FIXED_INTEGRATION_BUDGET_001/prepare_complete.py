"""Prepare the authorized post-numeric pipeline; normal copies, no migration tests."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os, shutil, tarfile

R = Path(__file__).resolve().parent
P = R.parent
OLD = P/'R45_EXTENDED_BUDGET_NEURAL_FLOW_001'
check = json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
assert check['both_pass'] and all(x['pass'] for x in check['models'].values())
def text(path):
    return path.read_bytes().decode('utf8').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
def put(name, value):
    (R/name).write_bytes(value.encode('utf8'))
def save(name, value):
    put(name, json.dumps(value, indent=2, allow_nan=False)+'\n')

modules = ['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','common_math.py',
           'model.py','legacy_model.py','endpoint_common.py','temporal_common.py','mean_description.py',
           'covariance_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py']
module_bindings = {}
for name in modules:
    original = (OLD/name).read_bytes()
    code = text(OLD/name)
    assert ast.dump(ast.parse(original.decode())) == ast.dump(ast.parse(code))
    put(name, code)
    module_bindings[name] = {'historical_path':str(OLD/name),'AST_unchanged':True,
                             'new_LF_sha256':hashlib.sha256(code.encode()).hexdigest()}
runtime = text(OLD/'flow_runtime.py').replace('self.details=None;', "self.steps=cfg['solver_steps'];assert self.steps==64;self.details=None;")
runtime = runtime.replace('model.edges,32,chunk)', 'model.edges,model.steps,chunk)')
assert 'model.edges,32,chunk)' not in runtime
put('flow_runtime.py', runtime)

replay = text(OLD/'replay_all.py')
replay = replay.replace("    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');", "    assert json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']\n    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');")
replay = replay.replace("for name in ['EXTENDED_FLOW_FINAL','EXTENDED_FLOW_REAL_CONTROL']:", "for name in CFG['endpoints']:")
replay = replay.replace("name=='EXTENDED_FLOW_FINAL'", "name in CFG['arms']")
replay = replay.replace('assert COUNT==4000', 'assert COUNT==8000')
put('replay_all.py', replay)

cfg = json.loads((R/'config_remote.json').read_text('utf8'))
local = json.loads((R/'config.json').read_text('utf8'))
remote = cfg['remote_root']
roots = {n:P/d for n,d in {
 'R7':'R7_RESPONSE_RESIDUAL_DIAGNOSIS_001','R26':'R26_CONDITIONAL_BATCH_TRANSLATION_001',
 'R27':'R27_FIXED_PROJECTION_COVERAGE_DIAGNOSTIC_001','R32':'R32_CONDITIONAL_MEAN_TRANSLATION_001',
 'R33':'R33_RESIDUAL_OT_FLOW_001','R35':'R35_PSD_ORDERED_FLOW_001','R36':'R36_CONDITIONAL_DIFFUSION_EDIT_001',
 'R37':'R37_FULL_COVARIANCE_CALIBRATION_001','R38':'R38_SLICED_MONOTONE_TRANSPORT_001',
 'R39':'R39_FULL_JOINT_KERNEL_TRANSLATION_001','R43':'R43_SCALAR_TIME_NEURAL_FLOW_001'}.items()}
for n in roots:cfg[n+'_root']=remote+'/references/'+n
cfg.update(new_updates=0,total_updates=0,training_t_rule='not used',training_white_seed=None,
           main_generation_rows=8000,repeated_rows_cap=2048,source_probe_rows=1664,extra_guide_probe_rows=256,
           selection='SHORT fixed control and LONG sole primary; no DEV selection',
           first_actual_command=json.loads((R/'FIRST_CONTROL_RECEIPT.json').read_text('utf-8-sig')),
           historical_training={'SHORT':{'updates':2950,'exposure_rows_each_domain':94400},
                                'LONG':{'updates':50000,'exposure_rows_each_domain':1600000,'R45_new_updates':47050}})
cfg['local_input_paths']['width_definition']=remote+'/references/DIAGNOSTIC_DEFINITION.json'
cfg['local_training_input_paths']['FINE_CONSTANTS.npz']=remote+'/references/FINE_CONSTANTS.npz'
cfg['old_BASE_REAL_generated']=remote+'/references/R39/BASE_REAL_CONTROL/generated.npz'
cfg['old_BASE_REAL_metrics']=remote+'/references/R39/metrics.json'
cfg['execution']['generation_evaluation_replay']='same server CPU12, same NumPy/Torch backend, chunk64; GPU jobs0'
save('config_cpu.json',cfg)

records=[]
def add(path, dest, kind):
    p=Path(path)
    records.append({'source':str(p),'destination':dest,'bytes':p.stat().st_size,'kind':kind})
for name in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz',
             'PROBE_INPUTS_CORRECTED.npz','JOINT8_INPUTS_PRIVATE.npz','VALLEY_DEFINITION.json','CAL_REFERENCE_METADATA.json','DATA_ROLES.json']:
    add(roots['R43']/name,name,'frozen input copied only after both numerical gates passed')
for name in ['DIAGNOSTIC_SELECTIONS.json','C2ST_HOLDOUT_PREDICTIONS.npz']:
    add(roots['R7']/'results'/name,'references/R7/results/'+name,'old fixed reference')
add(roots['R7']/'metrics.json','references/R7/metrics.json','old aggregate')
add(roots['R26']/'results/C2ST_PRIVATE.npz','references/R26/results/C2ST_PRIVATE.npz','old fixed row selection')
for name in ['PROJECTION_SPACE_PRIVATE.npz','NEW_PROJECTIONS_PRIVATE.npz']:
    add(roots['R27']/'results'/name,'references/R27/results/'+name,'old projection definition')
pure=['normalized_u480','seen_reconstruction480','complement_reconstruction480']
for view in pure:
    for name in ['TRAIN_REFERENCE__'+view+'_PREDICTIONS_PRIVATE.npz','C_TRAIN_REFERENCE__'+view+'.json']:
        add(roots['R27']/'results'/name,'references/R27/results/'+name,'old pure480 reference')
for view in cfg['C2ST_views']+pure:
    for stage,arm in [('R32','CONTROL_FINAL'),('R39','BASE_REAL_CONTROL')]:
        name=arm+'__'+view+'_PREDICTIONS_PRIVATE.npz'
        add(roots[stage]/'results'/name,'references/'+stage+'/results/'+name,'old reference prediction; no refit')
add(roots['R32']/'MEAN_TARGETS_PRIVATE.npz','references/R32/MEAN_TARGETS_PRIVATE.npz','old TRAIN mean target')
for stage,arm in [('R32','CONTROL_FINAL'),('R33','INDEPENDENT_CFM'),('R33','OT_CFM'),('R35','PSD_ORDERED_CFM'),
                  ('R36','CONDITIONAL_DIFFUSION_EDIT'),('R37','FULL_COVARIANCE_CALIBRATION'),
                  ('R38','SLICED_MONOTONE_TRANSPORT'),('R39','BASE_REAL_CONTROL')]:
    add(roots[stage]/arm/'generated.npz','references/'+stage+'/'+arm+'/generated.npz','old output reuse')
for name in ['metrics.json','GENERATION_RECEIPT.json','REPLAY_RECEIPT.json']:
    add(roots['R39']/name,'references/R39/'+name,'old aggregate result')
add(roots['R43']/'metrics.json','references/R43/metrics.json','historical32 budget context only')
add(local['local_input_paths']['width_definition'],'references/DIAGNOSTIC_DEFINITION.json','old width definition')
add(local['local_training_input_paths']['FINE_CONSTANTS.npz'],'references/FINE_CONSTANTS.npz','old fixed projections')

code_names=[*modules,'generate_all.py','evaluate_saved.py','replay_all.py','flow_runtime.py','budget_comparison.py','config_cpu.json']
for name in code_names:
    if name.endswith('.py'):ast.parse(text(R/name))
    add(R/name,name,'R46 implementation/configuration')
total=sum(x['bytes']for x in records)
assert total<2*1024**3
with tarfile.open(R/'UPLOAD_CPU_EVALUATION.tar','w')as t:
    for row in records:t.add(row['source'],arcname=row['destination'],recursive=False)
save('CPU_PAYLOAD_PACKAGE_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'both_NUMERIC_pass':True,'records':records,'total_payload_bytes':total,'scientific_array_decodes':0,
 'response_inference_or_fit':0,'normal_copy_only':True,'migration_content_consistency_test':False,'module_AST_bindings':module_bindings})
save('FULL_PIPELINE_FREEZE.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'both_NUMERIC_pass':True,'solver_steps':64,'generation_replay_chunk':64,
 'files':{n:{'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest()}for n in code_names},
 'new_training_updates':0,'candidate_selection':'LONG only; SHORT fixed control','no_migration_tests':True})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'files':len(records),'payload_bytes':total,'both_pass':True}))
