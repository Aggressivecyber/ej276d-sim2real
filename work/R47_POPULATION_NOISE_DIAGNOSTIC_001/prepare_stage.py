"""Small R47 controls; references only, no scientific array reads or migrations."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, ast, hashlib

P=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent
OLD=P/'work/R46_FIXED_INTEGRATION_BUDGET_001'
remote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R47_REMOTE_20261010_053712_UTC'
oldremote='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R46_REMOTE_20261010_014939_UTC'
reg=json.loads((P/'reviews/R47_REGISTRATION_001/RECEIPT.json').read_text('utf-8-sig'))
previous=json.loads((OLD/'config_cpu.json').read_text('utf-8-sig'))
def put(n,v): (R/n).write_bytes((json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
copies=['evaluation.py','diagnostic_core.py','common_math.py','linux_supervise.py','linux_owned_worker.py','launch_server.py','supervise.py','owned_worker.py']
bindings=[]
for n in copies:
    b=(OLD/n).read_bytes();text=b.decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
    if n in ['linux_supervise.py','launch_server.py']:
        text=text.replace("['check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']","['numeric_worker.py']")
    (R/n).write_bytes(text.encode())
    bindings.append({'source':str(OLD/n),'source_bytes':len(b),'source_sha256':hashlib.sha256(b).hexdigest(),'destination':n,'adaptation':'LF; registered entry only for supervisor/launcher' if n in ['linux_supervise.py','launch_server.py'] else 'LF only'})
inputs={'actual':oldremote+'/ACTUAL_GENERATION_INPUTS.npz','metadata':oldremote+'/DEV_SOURCE_METADATA.npz',
 'real':oldremote+'/REAL_DEV_AND_REFERENCE_PRIVATE.npz','constants':oldremote+'/UNIT_CHARGE_CONSTANTS.npz',
 'selection':oldremote+'/references/R26/results/C2ST_PRIVATE.npz',
 'width_definition':oldremote+'/references/DIAGNOSTIC_DEFINITION.json',
 'CAL_metadata':oldremote+'/CAL_REFERENCE_METADATA.json',
 'RAW_RR_prediction':oldremote+'/references/R7/results/C2ST_HOLDOUT_PREDICTIONS.npz',
 'RAW_RR_selection':oldremote+'/references/R7/results/DIAGNOSTIC_SELECTIONS.json',
 'R46_metrics':oldremote+'/metrics.json','BASE':oldremote+'/references/R32/CONTROL_FINAL/generated.npz',
 'BASE_REAL':oldremote+'/references/R39/BASE_REAL_CONTROL/generated.npz'}
arms=['SHORT_2950_HEUN64','LONG_50000_HEUN64','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']
inputs.update({n:oldremote+'/'+n+'/generated.npz' for n in arms})
c={'stage':reg['stage'],'remote_root':remote,'old_R46_remote_root':oldremote,'server_python':previous['server_python'],
 'CPU_python':previous['CPU_python'],'deadline_UTC':reg['executor_seal_deadline_UTC'],
 'science_budget':900,'CPU_threads':12,'RSS_limit_bytes':48*1024**3,'GPU_limit_bytes':0,
 'disk_free_reserve_bytes':8*1024**3,'output_limit_bytes':64*1024**2,'training_total_limit_s':0,
 'C2ST':previous['C2ST'],'train_each_domain':600,'holdout_each_domain':250,
 'bootstrap_repeats':500,'bootstrap_seed':2026100871,'HGB_max_fits':5,'inputs':inputs,
 'arms':arms,'no_response_training_ODE_EM':True,'new_response_candidate':False,
 'initial_ADC_rule':'cast S+G to float32; same frozen14bit ADC once; deterministic feature diagnostic only',
 'covariance_rule':'center each component within baseline100 row; ddof1 denominator99; all six2Cov terms',
 'population_rule':'all original rows retained; nonfinite summaries explicit; source original Qbin and observed Qbin separated',
 'input_key_contract':{'actual':['source_waveforms_V','source_ids'],'metadata':['source_ids','composition'],
   'real':['real_V','real_ids','real_groups','real_QC_valid','reference_V','reference_ids','reference_groups'],
   'endpoints':['pre_ADC_V','waveforms_V','base_signal_V','projected_initial_Gaussian_f64_V','learned_flow_increment_f64_V','final_cast_rounding_f64_V','initial_state_f64_V'],
   'selection':[f'phase{p}_{j}' for p in [0,1] for j in [0,1,2]]}}
put('config.json',c)
put('SOURCE_REUSE_BINDINGS.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'sources':bindings,
 'reference_paths':inputs,'R46_manifest_sha256':reg['input_bindings'][3]['sha256'],
 'old_input_bytes_not_rehashed_or_migrated':True,'no_scientific_arrays_read':True,'task_sha256':reg['task_sha256']})
put('FIRST_CONTROL_RECEIPT.json',{'PID':44412,'UTC':'2026-10-10T05:37:47.5032863+00:00','tool':'a8dbb4','native_exit':0})
put('ENGINEERING_FAILURES.json',{'scientific_failures':[],'control_failures':[]})
put('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'method_frozen_before_science':True,'HGB_max':5,'task_sha256':reg['task_sha256'],'science_array_decode':0,
 'implementation':[p.name for p in R.glob('*.py')],'all_python_LF':True})
for p in R.glob('*.py'):ast.parse(p.read_text('utf-8-sig'))
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'remote_root':remote,'AST_pass':True,'no_science_run':True}),flush=True)
