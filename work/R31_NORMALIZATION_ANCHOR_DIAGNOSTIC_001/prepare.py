from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parent;R30=ROOT.parent/'R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001'
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
for n in ['supervise.py','owned_worker.py','diagnostic_core.py','evaluation.py','common_math.py','paired_bootstrap.py']:
    shutil.copyfile(R30/n,ROOT/n)
old=json.loads((R30/'config.json').read_text('utf8'))
cfg={k:old[k] for k in ['threads','RSS_limit_bytes','GPU_allocated_limit_bytes','C2ST','bootstrap_repeats','bootstrap_seed','train_each_domain','holdout_each_domain','normalization_peak_floor_V','regions','edges_logQ','execution']}
cfg.update(stage=ROOT.name,registered_UTC='2026-10-09T16:41:45.842690+00:00',deadline_UTC='2026-10-09T17:11:45.842690+00:00',root_deadline_UTC='2026-10-09T17:21:45.842690+00:00',science_budget=300,HGB_max_fits=12,output_limit_bytes=96*1024**2,
           candidates=['FRESH_FINAL','ENERGY_FINAL'],domains=['REAL','TRAIN_REFERENCE','FRESH_FINAL','ENERGY_FINAL'],views=['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'],
           inputs={'real':old['inputs']['real'],'rows':str(R30/'results/INPUT_ROWS_PRIVATE.npz'),'moments':str(R30/'results/MOMENT_PARAMETERS_PRIVATE.npz'),'full480_metrics':str(R30/'CLASSIFIER_METRICS.json')},
           generated={k:old['generated'][k] for k in ['FRESH_FINAL','ENERGY_FINAL']},
           raw480_predictions={k:old['raw_predictions'][k] for k in ['FRESH_FINAL','ENERGY_FINAL','TRAIN_REFERENCE']},
           transform480_predictions={k:{v:str(R30/'results'/f'{k}__{v}_PREDICTIONS_PRIVATE.npz') for v in ['GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE']} for k in ['FRESH_FINAL','ENERGY_FINAL','TRAIN_REFERENCE']},
           retained_indices=list(range(100))+list(range(150,480)),excluded_interval=[100,150],decoder_abstain_error_difference_atol=1e-12,
           control_raw_CI_contains=.5,control_paired_BA_tolerance=[-.10,.10],regional_holdout_min_each=10,
           conditional_support='Reuse original R30 six-domain bins1–7; no re-estimation',moment_reestimation=0,
           first_actual_command={'tool':'510cc9','PID':38468,'native_exit':0,'UTC':'2026-10-09T16:42:59.605264+00:00'},
           prohibited_counts={'response_training':0,'response_inference':0,'response_generation':0,'EM':0,'new_REAL':0,'EVAL':0,'CAL':0,'G4':0,'SSH':0,'installation':0})
write('config.json',cfg)
protocol='''# R31 frozen normalization anchor diagnostic

Four saved domains: REAL DEV, REAL TRAIN reference, R28 FRESH_FINAL2950, R29 ENERGY_FINAL2950. Exact old classifier train600/holdout250, groups, QC, original Q bins, R30 domain moment parameters and its original six-domain support bins1–7 are reused. No moment estimation, response computation, EM, new REAL, EVAL or CAL reads.

Reconstruct the frozen full480 u from saved V, then apply the original R30 GLOBAL_CENTER/Q_CENTER/Q_CENTER_SCALE parameters with unchanged global fallback/floor. The sole new mask keeps0:100 and150:480 (430 columns). Never rebaseline, renormalize, rebin or change the mask. Full480 old scores remain visible. All source parents/events remain.

For every domain describe original peaks, peak-floor activation, max(u)-1 errors and argmax inside/outside100:150 for all/train/holdout rows. The no-fit decoder operates on full480 GLOBAL_CENTER holdout features w: e_A=abs(max(w+mu_A)-1), e_B=abs(max(w+mu_B)-1), predict the lower-error domain, abstain if abs(e_A-e_B)<=1e-12. True labels only score predictions. Keep floor-active events in main classification and separately report applicability; save error quantiles[0,.5,.9,.99,1], class coverage, nonabstention BA and all-event worst/best abstention bounds. No tuned threshold or fit.

Three comparisons x four430 views =12 HGB fits, old parameters/seed and train-only imputer/StandardScaler. Increment before fit; persist predictions before500 fixed-classifier artificial-group bootstrap replicates. Pair every430 view against its same-row old480 classifier, and three transformed430 views against RAW430. Any row/label/group mismatch makes a pair undefined. Original-Q regional BA at both domain N>=10 is descriptive; no regional fit or bootstrap.

Control rule: RAW430 REAL–REAL BA95CI must include.5. A transformed430 view is usable only if its paired-minus-RAW430 BA95CI is entirely inside[-.10,+.10]. This operational diagnostic tolerance is not physical equivalence. One mask only; if control still fails, close this domain-moment-C2ST causal attribution route. Masked success cannot promote complete waveforms. No unique covariance/electronics attribution.

CPU4 actual PID/RSS guard, science300s including failures and plots, RSS4GiB, output96MiB. Executor stop/seal17:11:45.842690UTC. E/D only, no SSH/install/Git/agents. All old stages remain immutable.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
sources=[]
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));sources.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'first_actual_command':cfg['first_actual_command'],'science_started':False,'source_AST':sources,'old_stage_writes':0})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'ready':True}),flush=True)
