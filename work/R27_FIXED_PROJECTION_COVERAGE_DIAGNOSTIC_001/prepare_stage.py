from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os, shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OLD = PROJECT/'work/R26_CONDITIONAL_BATCH_TRANSLATION_001'
R23 = PROJECT/'work/R23_ONLINE_STANDARDIZED_CRITIC_001'
oldcfg = json.loads((OLD/'config.json').read_bytes())
for name in ('supervise.py', 'owned_worker.py'):
    shutil.copyfile(OLD/name, ROOT/name)

# Retain the exact existing fixed-classifier bootstrap and JSON conversion.
pieces = ['import numpy as np\n']
for file, names in [('evaluation.py', ['safe', 'summary']), ('diagnostic_core.py', ['bootstrap'])]:
    source = (OLD/file).read_text('utf8')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            pieces.append(ast.get_source_segment(source, node)+'\n')
(ROOT/'support.py').write_bytes(('\n'.join(pieces)).encode('utf8'))

names = ['R23_D1_SELECTED', 'HOMO_SELECTED', 'HOMO_FINAL', 'R23_D1_FINAL2950']
generated = {name:str((R23/'STANDARDIZED' if name=='R23_D1_SELECTED' else OLD/name)/'generated.npz') for name in names}
joint = {name:str(R23/'results/STANDARDIZED_joint8_REPEAT_PRIVATE.npz' if name=='R23_D1_SELECTED' else OLD/'results'/f'{name}_joint8_REPEAT_PRIVATE.npz') for name in names}
edges = json.loads(Path(oldcfg['local_input_paths']['width_definition']).read_bytes())['edges_logQ']
cfg = {
    'stage': ROOT.name, 'registered_UTC':'2026-10-09T13:30:26.408171+00:00',
    'deadline_UTC':'2026-10-09T14:20:26.408171+00:00',
    'absolute_deadline_UTC':'2026-10-09T14:30:26.408171+00:00',
    'science_budget':600, 'threads':4, 'RSS_limit_bytes':4*1024**3,
    'GPU_allocated_limit_bytes':4*1024**3, 'output_limit_bytes':128*1024**2,
    'C2ST':oldcfg['C2ST'], 'HGB_max_fits':15, 'bootstrap_repeats':500,
    'bootstrap_seed':oldcfg['bootstrap_seed'], 'C2ST_views':['normalized_u480','seen_reconstruction480','complement_reconstruction480'],
    'candidates':names, 'train_each_domain':600, 'holdout_each_domain':250,
    'minimum_classification_support':[150,75], 'condition_support_min':40,
    'edges_logQ':edges, 'last_edge_inclusive':True,
    'rank_relative_threshold':1e-10, 'normalization_peak_floor_V':.0025,
    'new_projection_seeds':[20261009271,20261009272,20261009273,20261009274],
    'projection_draw_order':['shape480','diff479','baseline100_V'], 'projection_columns':128,
    'quantile_levels':{'first':.005,'last':.995,'count':128,'method':'linear'},
    'new_projection_dtype':'float64', 'old_projection_dtype':'original saved float32 promoted to float64',
    'linear_span_columns':['scaled_shape128','scaled_diff128','pooled8_32','first100_coordinate_basis','charge_short_indicator','charge_total_indicator'],
    'joint_sample_variance_ddof':1, 'joint_repeats':8,
    'branch_gates':{'all':[0,480],'baseline':[0,100],'prompt':[100,150],'postpeak':[150,220],'tail':[220,400]},
    'inputs':{
        'generated':generated,'joint':joint,
        'real':str(OLD/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'),
        'source':str(OLD/'ACTUAL_GENERATION_INPUTS.npz'),
        'source_metadata':str(OLD/'DEV_SOURCE_METADATA.npz'),
        'fine_constants':str(R23/'FINE_CONSTANTS.npz'),
        'pooled_constants':str(R23/'VALIDATION_INPUTS.npz'),
        'classification_rows':str(OLD/'results/C2ST_PRIVATE.npz'),
        'old_metrics_compact':str(OLD/'METRICS_COMPACT.json'),
        'R26_manifest':str(OLD/'FINAL_MANIFEST.json'),
        'R23_manifest':str(R23/'FINAL_MANIFEST.json'),
    },
    'response_fit_updates_generation':0,'density_EM':0,'EVAL_read':0,'CAL_waveform_decode':0,'SSH':0,'installations':0,
}
(ROOT/'config.json').write_bytes((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode())
bindings = {}
for prefix, stage in [('R26', OLD), ('R23', R23)]:
    q = stage/'FINAL_MANIFEST.json'
    if not q.exists():
        bindings[prefix] = {'manifest':'not present; input schema and prior stage receipts retained'}
        continue
    b = q.read_bytes(); d = json.loads(b)
    entries = d.get('files',d.get('members',[]))
    selected = {}
    targets = list(generated.values())+list(joint.values())+list(cfg['inputs'].values())
    for item in entries:
        if not isinstance(item,dict): continue
        rel = item.get('path',item.get('relative_path',''))
        if str(stage/rel) in targets or Path(rel).name in ['FINE_CONSTANTS.npz','VALIDATION_INPUTS.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','METRICS_COMPACT.json']:
            selected[rel]=item
    bindings[prefix]={'manifest_sha256':hashlib.sha256(b).hexdigest(),'selected_historical_entries':selected,'no_new_source_hash_audit':True}
(ROOT/'INPUT_BINDINGS.json').write_bytes((json.dumps(bindings,indent=2)+'\n').encode())
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'config_written':True,'scientific_import_or_array_decode':0,'candidate_paths':generated}))
