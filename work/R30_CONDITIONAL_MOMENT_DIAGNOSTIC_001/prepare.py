from pathlib import Path
import ast, datetime, hashlib, json, os, shutil

ROOT = Path(__file__).resolve().parent
P = ROOT.parent.parent
R28 = P/'work/R28_FRESH_PROJECTION_TRANSLATION_001'
R29 = P/'work/R29_SLICED_ENERGY_TRANSLATION_001'
R27 = P/'work/R27_FIXED_PROJECTION_COVERAGE_DIAGNOSTIC_001'
def write(name, v):
    (ROOT/name).write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')

for name in ['supervise.py','owned_worker.py','diagnostic_core.py','evaluation.py','common_math.py']:
    shutil.copyfile(R29/name, ROOT/name)
text = (R29/'assessment.py').read_text('utf8')
node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name=='paired_bootstrap')
(ROOT/'paired_bootstrap.py').write_text('import numpy as np\n\n'+ast.get_source_segment(text,node)+'\n','utf8')
definition = P/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/DIAGNOSTIC_DEFINITION.json'
edges = json.loads(definition.read_text('utf8'))['edges_logQ']
cfg = {
    'stage':ROOT.name, 'registered_UTC':'2026-10-09T16:09:11.291876+00:00',
    'deadline_UTC':'2026-10-09T16:59:11.291876+00:00',
    'root_deadline_UTC':'2026-10-09T17:09:11.291876+00:00',
    'first_actual_command':{'tool':'d87edf','PID':21408,'native_exit':0,'UTC':'2026-10-09T16:10:24.564528+00:00'},
    'science_budget':600, 'threads':4,'RSS_limit_bytes':4*1024**3,
    'GPU_allocated_limit_bytes':4*1024**3,'output_limit_bytes':128*1024**2,
    'HGB_max_fits':15,'train_each_domain':600,'holdout_each_domain':250,
    'bootstrap_repeats':500,'bootstrap_seed':2026100871,
    'C2ST':{'max_iter':100,'learning_rate':.06,'max_leaf_nodes':15,
            'min_samples_leaf':20,'l2_regularization':1.,'early_stopping':False,'random_state':20261008},
    'views':['GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'],
    'normalization_peak_floor_V':.0025,'conditional_train_min_each_all_six':20,
    'regional_holdout_min_each':10,'scale_floor_multiplier':.05,'scale_floor_absolute':1e-6,
    'edges_logQ':edges,'regions':{'baseline':[0,100],'prompt':[100,150],'postpeak':[150,220],'tail':[220,400]},
    'execution':{'location':'local','device':'CPU','python':'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe','scientific_worker_PID_RSS_guard':True},
    'inputs':{'real':str(R28/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'),
              'classification_rows':str(R27/'results/CLASSIFICATION_ROWS_PRIVATE.npz'),
              'source_ids':str(R28/'ACTUAL_GENERATION_INPUTS.npz'),
              'fine_constants':str(P/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/FINE_CONSTANTS.npz'),
              'width_definition':str(definition)},
    'candidates':['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL'],
    'generated':{k:str((R28 if k.startswith('FRESH') else R29)/k/'generated.npz') for k in ['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL']},
    'raw_predictions':{k:str((R28 if k.startswith('FRESH') else R29 if k.startswith('ENERGY') else R27)/'results'/f'{k}__normalized_u480_PREDICTIONS_PRIVATE.npz') for k in ['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL','TRAIN_REFERENCE']},
    'raw_results':{k:str((R28 if k.startswith('FRESH') else R29 if k.startswith('ENERGY') else R27)/'results'/f'C_{k}__normalized_u480.json') for k in ['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL','TRAIN_REFERENCE']},
    'prohibited_counts':{'response_training':0,'response_inference':0,'response_generation':0,'density_EM':0,'new_REAL':0,'EVAL':0,'CAL_waveform':0,'new_G4':0,'SSH':0,'installation':0},
    'selection':'exact old phase0/phase1 rows; no resampling or model choice',
    'purpose':'domain-identity moment transforms for diagnosis only; not deployable response or new PSD',
}
write('config.json',cfg)
protocol = '''# R30 frozen conditional moment diagnostic

Six saved domains: REAL DEV, REAL TRAIN reference, R28 FRESH_SELECTED2250 / FRESH_FINAL2950, R29 ENERGY_SELECTED2750 / ENERGY_FINAL2950. All four outputs retain the same 2000 source parents. No response computation, EM, new waveform source, EVAL or CAL decoding.

Float64 u=(V-mean(V[:100]))/max(max(V-mean(V[:100])),0.0025), all 480 columns. The actual old classifier phase0=600 and phase1=250 rows, groups and QC are reused. Original observed charge assigns the frozen eight logQ bins; the exact highest edge is included in bin8. Six-domain train-only counts >=20 define one uniform supported-bin set. Unsupported bins and outside/undefined Q fall back to each domain's global 600-row mean/SD. No event is discarded for conditional support.

GLOBAL_CENTER subtracts each domain's 600-row global mean. Q_CENTER subtracts its corresponding train-only bin mean, or global mean. Q_CENTER_SCALE additionally divides by its bin/global sample SD (ddof1), with coordinate floor=max(0.05*old REAL TRAIN shape480 scale,1e-6). No renormalization, peak finding, rebinning, ADC or PSD recomputation on transformed features. Parameters are fixed by classifier training rows, not the 250 holdout rows. Report arithmetic checks, counts, reasons, floor usage and maximum amplification.

Five domain comparisons x three views = exactly 15 HGB fits, with original hyperparameters, train-only imputation/scaling, >=150 train and >=75 holdout rows per domain. Count a fit before fitting and save predictions immediately, before 500 fixed-classifier artificial-group bootstrap replicates. Reuse old normalized_u480 predictions including REAL TRAIN-reference control. Pair transformed-minus-raw bootstrap only on identical labels/groups/holdout rows. No new classifier fits for regional summaries. Describe original-Q-bin holdout BA at >=10 per domain, plus a common-bin equal mean without choosing bins by performance.

For full valid populations and fixed holdout rows report raw/transformed means and pointwise SD, plus baseline0:100/prompt100:150/postpeak150:220/tail220:400 residual/variance summaries. Holdout covariance uses ddof1 for raw/Q_CENTER/Q_CENTER_SCALE, reports trace, trace^2/trace(C@C), and Frobenius difference relative to REAL. Zero denominators are undefined. No covariance whitening/inverse. Whole-population curves are descriptive, not extra independent test data.

These transforms use domain identity and are not deployable waveform corrections. Removing conditional means also removes composition and charge-mix effects; residual separability does not uniquely identify covariance, electronics, architecture or physical cause. Artificial acquisition groups and fixed-classifier confidence intervals do not establish independent runs or equivalence thresholds. Preserve old source/PSD/width results by reference; no new PSD claim from these transforms.

Local CPU4, actual science PID guarded before imports, RSS4GiB, science600s including failures and plotting, output128MiB. Executor stop/seal 2026-10-09T16:59:11.291876Z. No SSH, installations, new agents or Git. All writes and caches E/D; immutable prior stages are read-only.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
bound=[]
for f in ROOT.glob('*.py'):
    b=f.read_bytes();ast.parse(b.decode('utf8'));bound.append({'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_imports':0,'source_AST':bound,'actual_first_command':cfg['first_actual_command'],'prior_scientific_cost_s':0,'input_versions':'existing immutable R23/R27/R28/R29 saved artifacts; no source re-audit or migration test'})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':True,'source_count':len(bound)}),flush=True)
