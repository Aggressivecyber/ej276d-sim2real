"""Bind existing evaluation inputs without decoding arrays or starting science."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import subprocess
import sys

P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P/'path_preparation_runtime')


def read(path):
    return json.loads(path.read_text('utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old = read(R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/config_post.json')
train = read(R/'work/R77_FIXED25000_FULL_DATA_TRAIN_001/config.json')
frozen = old['frozen_eval_input_root']
refs = {
    'CAL_metadata': frozen+'/CAL_REFERENCE_METADATA.json',
    'valley': frozen+'/VALLEY_DEFINITION.json',
    'data_roles': frozen+'/DATA_ROLES.json',
    'width_definition': old['local_input_paths']['width_definition'],
    'frozen_width_metrics': old['R39_root']+'/metrics.json',
    'BASE_output': old['R32_root']+'/CONTROL_FINAL/generated.npz',
    'INDEPENDENT_CFM_output': old['R33_root']+'/INDEPENDENT_CFM/generated.npz',
    'BASE_REAL_output': old['old_BASE_REAL_generated'],
    'fixed_selection': old['R26_root']+'/results/C2ST_PRIVATE.npz',
    'frozen_C2ST_metrics': old['canonical_R51_root']+'/CANONICAL_C2ST.json',
    'frozen_C2ST_predictions': old['canonical_R51_root']+'/CANONICAL_PREDICTIONS_PRIVATE.npz',
}
inputs = {key: frozen+'/'+name for key, name in {
    'actual':'ACTUAL_GENERATION_INPUTS.npz',
    'source_metadata':'DEV_SOURCE_METADATA.npz',
    'REAL_packet':'REAL_DEV_AND_REFERENCE_PRIVATE.npz',
    'probes':'PROBE_INPUTS_CORRECTED.npz',
    'joint8':'JOINT8_INPUTS_PRIVATE.npz',
}.items()}
paths = {**{'reference.'+k:v for k,v in refs.items()},
         **{'inputs.'+k:v for k,v in inputs.items()},
         'model.original_Q_weights':train['input_paths']['Q_head_weights']}
assert len(paths) == 17
target = P/'FROZEN_REFERENCE_BINDINGS.json'
assert not target.exists(), 'Existing preparation is immutable; inspect it instead of repeating.'
code = ('from pathlib import Path\nimport os,json,hashlib,resource\n'
        'os.sched_setaffinity(0,{11})\n'
        'resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))\n'
        'paths='+repr(paths)+'\n'+r'''
rows={}
for key,name in paths.items():
 p=Path(name);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 rows[key]={'path':name,'sha256':h.hexdigest(),'bytes':p.stat().st_size}
print(json.dumps({'files':rows,'array_or_model_decode':0,'science_fits':0,
 'CPU_affinity':sorted(os.sched_getaffinity(0)),
 'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
''')
ast.parse(code)
proc = subprocess.run(['ssh','-T','-x','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'rylan@100.82.231.102','/usr/bin/python3 -B -'], input=code, text=True,
    capture_output=True, encoding='utf8', timeout=45)
for kind,value in [('stdout',proc.stdout),('stderr',proc.stderr)]:
    (P/('FROZEN_PATH_BINDING.'+kind+'.log')).write_text(value,'utf8')
assert proc.returncode == 0, proc.stderr
bound = json.loads(proc.stdout)
assert set(bound['files']) == set(paths)
old_bind = read(R/'work/R77_FIXED25000_FULL_DATA_TRAIN_001/INPUT_BINDING.json')['inputs']
assert bound['files']['model.original_Q_weights']['sha256'] == old_bind['Q_head_weights']['sha256']
assert bound['files']['reference.valley']['sha256'] == old_bind['valley_definition']['sha256']
bound.update(UTC=datetime.now(timezone.utc).isoformat(),status='STATIC_EXISTING_INPUT_BINDINGS_ONLY',
    native_metadata_query_exit=proc.returncode, local_resource={**guard,**memory()},
    scientific_job_started=False,training_completion_or_A_F_claim=False)
write(target,bound)
ac = read(R/'reviews/R74_EXACT_LOSS_FULL_TRAIN_001/evaluation_candidate/evaluation_config.example.json')
ac.update(stage=P.name,registered_stage=P.name,
    remote_root='/PENDING_R77_COMPLETION_AND_R78_REGISTRATION',
    output_root='/PENDING_R77_COMPLETION_AND_R78_REGISTRATION',
    SOURCE_arm='R77_FULL_RESPONSE',REAL_arm='R77_REAL_CONTROL',
    allowed_scientific_entries=['generate_full_response.py','replay_full_response.py',
        'evaluate_saved.py','summarize_and_plot.py'],
    binding_status='DISABLED: R77 actual terminal weights, numeric and native receipts still pending')
ac['inputs']={k:bound['files']['inputs.'+k] for k in inputs}
ac['model']['original_Q_weights']=bound['files']['model.original_Q_weights']
write(P/'config.disabled.json',ac)
post_keys=['C2ST','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed',
    'C2ST_views','local_input_paths','R32_root','R33_root','R39_root','old_BASE_REAL_generated',
    'R26_root','frozen_eval_input_root','canonical_R51_root']
post={k:old[k] for k in post_keys}
post.update(stage=P.name,acceptance_version=5,updates=25000,
    SOURCE_arm=ac['SOURCE_arm'],REAL_arm=ac['REAL_arm'],evaluation_adapter_config='config.json',
    HGB_max_fits=10,density_EM_max=16,width_slots=16,CPU_threads=12,
    RSS_limit_bytes=48*1024**3,GPU_limit_bytes=10*1024**3,
    recipe_description='R77 full-parent25000 full-waveform MLP; frozen R71 public Q head; same public SOURCE/REAL operator; unique final endpoint',
    historical_controls_frozen_no_new_fit=True,required_density_bins_1based=[2,3,4,5,6,7],
    reference_bindings={path:bound['files']['reference.'+key]['sha256'] for key,path in refs.items()},
    D_E_evaluation_not_authorized_by_this_config=True,
    execution_enabled=False)
write(P/'config_post.disabled.json',post)
write(P/'REFERENCE_FILES.json',{'paths':refs,'count':11,'configuration_key':'reference_bindings',
    'binding_receipt_SHA256':sha(target),'array_decode':0})
print(json.dumps({'status':'STATIC_BINDINGS_PREPARED','bound_files':len(paths),
    'total_input_bytes':sum(r['bytes']for r in bound['files'].values()),
    'remote_metadata_native_exit':proc.returncode,'array_decode':0,
    'scientific_job_started':False,'memory':memory()}))
