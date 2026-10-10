from pathlib import Path
from datetime import datetime, timezone
import ast, json, os, sys, tarfile
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'R43_SCALAR_TIME_NEURAL_FLOW_001'
cfg=json.loads((ROOT/'config_remote.json').read_text('utf8'))
local=json.loads((ROOT/'config.json').read_text('utf8'))
remote=cfg['remote_root']
records=[]
def add(path,destination,kind):
    p=Path(path); n=p.stat().st_size
    records.append({'source':str(p),'destination':destination,'bytes':n,'kind':kind})

# This entry initially records only stat metadata; archive body copying is gated below.
for name in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz',
             'PROBE_INPUTS_CORRECTED.npz','JOINT8_INPUTS_PRIVATE.npz','VALLEY_DEFINITION.json',
             'CAL_REFERENCE_METADATA.json','DATA_ROLES.json']:
    add(OLD/name,name,'frozen scientific input' if name.endswith('.npz') else 'frozen metadata')
roots={name:Path(local[name+'_root']) for name in ['R26','R27','R32','R33','R35','R36','R37','R38','R39','R43']}
roots['R7']=ROOT.parent/'R7_RESPONSE_RESIDUAL_DIAGNOSIS_001'
for name in roots:cfg[name+'_root']=remote+'/references/'+name
add(roots['R7']/'results/DIAGNOSTIC_SELECTIONS.json','references/R7/results/DIAGNOSTIC_SELECTIONS.json','old fixed selection')
add(roots['R7']/'results/C2ST_HOLDOUT_PREDICTIONS.npz','references/R7/results/C2ST_HOLDOUT_PREDICTIONS.npz','old predictions')
add(roots['R7']/'metrics.json','references/R7/metrics.json','old aggregate metrics')
add(roots['R26']/'results/C2ST_PRIVATE.npz','references/R26/results/C2ST_PRIVATE.npz','old fixed selection')
add(roots['R27']/'results/PROJECTION_SPACE_PRIVATE.npz','references/R27/results/PROJECTION_SPACE_PRIVATE.npz','old fixed projection')
pure=['normalized_u480','seen_reconstruction480','complement_reconstruction480']
views=cfg['C2ST_views']+pure
for view in pure:
    for filename in ['TRAIN_REFERENCE__'+view+'_PREDICTIONS_PRIVATE.npz','C_TRAIN_REFERENCE__'+view+'.json']:
        add(roots['R27']/'results'/filename,'references/R27/results/'+filename,'old reference prediction/metrics')
for view in views:
    add(roots['R32']/'results'/('CONTROL_FINAL__'+view+'_PREDICTIONS_PRIVATE.npz'),
        'references/R32/results/CONTROL_FINAL__'+view+'_PREDICTIONS_PRIVATE.npz','old BASE prediction')
    add(roots['R39']/'results'/('BASE_REAL_CONTROL__'+view+'_PREDICTIONS_PRIVATE.npz'),
        'references/R39/results/BASE_REAL_CONTROL__'+view+'_PREDICTIONS_PRIVATE.npz','old BASE REAL prediction')
add(roots['R32']/'MEAN_TARGETS_PRIVATE.npz','references/R32/MEAN_TARGETS_PRIVATE.npz','old TRAIN target means')
old_domains=[('R32','CONTROL_FINAL'),('R33','INDEPENDENT_CFM'),('R33','OT_CFM'),
             ('R35','PSD_ORDERED_CFM'),('R36','CONDITIONAL_DIFFUSION_EDIT'),
             ('R37','FULL_COVARIANCE_CALIBRATION'),('R38','SLICED_MONOTONE_TRANSPORT'),('R39','BASE_REAL_CONTROL')]
for name,arm in old_domains:
    add(roots[name]/arm/'generated.npz','references/'+name+'/'+arm+'/generated.npz','old saved output; no model recomputation')
for name in ['metrics.json','GENERATION_RECEIPT.json','REPLAY_RECEIPT.json']:
    add(roots['R39']/name,'references/R39/'+name,'old aggregate result')
add(roots['R43']/'metrics.json','references/R43/metrics.json','short budget aggregate result')
for view in ['full_historical_shape','normalized_u480']:
    filename='SCALAR_TIME_FLOW_FINAL__'+view+'_PREDICTIONS_PRIVATE.npz'
    add(roots['R43']/'results'/filename,'references/R43/results/'+filename,'short budget existing predictions')
add(local['local_input_paths']['width_definition'],'references/DIAGNOSTIC_DEFINITION.json','frozen width definition')
add(local['local_training_input_paths']['FINE_CONSTANTS.npz'],'references/FINE_CONSTANTS.npz','old fixed projections')
cfg['local_input_paths']['width_definition']=remote+'/references/DIAGNOSTIC_DEFINITION.json'
cfg['local_training_input_paths']['FINE_CONSTANTS.npz']=remote+'/references/FINE_CONSTANTS.npz'
cfg['old_BASE_REAL_generated']=remote+'/references/R39/BASE_REAL_CONTROL/generated.npz'
cfg['old_BASE_REAL_metrics']=remote+'/references/R39/metrics.json'
cfg['threads']=cfg['CPU_threads']=12
cfg['execution']['generation_evaluation_replay']='same server CPU12, same NumPy/Torch backend, chunk64'
(ROOT/'config_cpu.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
for name in ['flow_core.py']:
    b=(OLD/name).read_bytes();ast.parse(b.decode());(ROOT/name).write_bytes(b)
code_names=['generate_all.py','evaluate_saved.py','replay_all.py','flow_runtime.py','flow_diagnostics.py',
            'evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','common_math.py',
            'model.py','legacy_model.py','endpoint_common.py','temporal_common.py','mean_description.py',
            'covariance_description.py','random_diagnostics.py','flow_core.py','budget_comparison.py','config_cpu.json']
for name in code_names:
    add(ROOT/name,name,'R45 CPU implementation' if name.endswith('.py') else 'R45 CPU configuration')
total=sum(r['bytes'] for r in records)
assert total<2*1024**3
pack='--pack' in sys.argv
if pack:
    check=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
    assert check['pass'], 'no DEV body copy before TRAIN numerical gate'
    assert json.loads((ROOT/'train_extended.py.LOCAL_NATIVE_RECEIPT.json').read_text('utf8'))['SSH_native_exit']==0
    with tarfile.open(ROOT/'UPLOAD_CPU_EVALUATION.tar','w') as t:
        for item in records:t.add(item['source'],arcname=item['destination'],recursive=False)
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'mode':'archive_after_numeric_pass' if pack else 'stat_plan_only',
     'files':records,'total_payload_bytes':total,'raw_file_body_copies':len(records) if pack else 0,'scientific_array_decodes':0,
     'new_response_forward_fit':0,'migration_content_consistency_test':False,
     'old_outputs_and_evaluators_only_reused':True,'threads':12,'generation_replay_chunk':64}
(ROOT/('CPU_PAYLOAD_PACKAGE_RECEIPT.json' if pack else 'CPU_PAYLOAD_PLAN.json')).write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps({k:v for k,v in rec.items() if k!='files'}))
