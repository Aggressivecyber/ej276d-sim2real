from pathlib import Path
from datetime import datetime, timezone
import json, shutil, ast, hashlib, os, tarfile

ROOT = Path(__file__).resolve().parent
P = ROOT.parent.parent
R28 = ROOT.parent / 'R28_FRESH_PROJECTION_TRANSLATION_001'
REMOTE = '/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R32_REMOTE_20261009_170201_UTC'
ROOT.mkdir(exist_ok=True)
if (ROOT / 'PREPARATION_RECEIPT.json').exists(): raise RuntimeError('preparation already completed')
cfg = json.loads((R28 / 'config.json').read_text('utf8'))
cfg.update(stage=ROOT.name, registered_UTC='2026-10-09T17:02:01.024475+00:00',
    deadline_UTC='2026-10-09T18:52:01.024475+00:00', absolute_deadline_UTC='2026-10-09T19:02:01.024475+00:00',
    science_budget=2400, training_wall_s=900, training_total_limit_s=1800, updates=2950,
    arms=['SERVER_FRESH_CONTROL','CONDITIONAL_MEAN'], endpoints=['CONTROL_SELECTED','CONTROL_FINAL','MEAN_SELECTED','MEAN_FINAL'],
    HGB_max_fits=40, HGB_plan=40, HGB_cap=40, density_EM_max=32, width_slots=32,
    main_generation_rows=8000, repeated_rows_cap=4096, source_probe_rows=3328, replay_rows=8000,
    output_limit_bytes=512*1024**2, mean_coefficient=1.0, mean_loss_condition='frozen actual source-Q bin',
    server_python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python', remote_root=REMOTE,
    training_device='cuda:0 if existing CUDA is available, else cpu', evaluation_device='cpu',
    first_actual_command={'tool':'dcc907','PID':59048,'native_exit':0,'UTC':'2026-10-09T17:02:35.687235+00:00'},
    execution={'training':'registered independent Linux server directory','evaluation_replay':'local CPU4 E/D',
               'CPU_python':cfg['CPU_python'],'actual_scientific_PID_nonce_RSS_before_imports':True},
    SSH='necessary own-stage transfer/training only', sole_training_intervention='fixed TRAIN conditional mean U coefficient1',
    endpoint_selection_uses_DEV=False, mean_term_not_in_J20=True, cross_platform_bitwise_training_claim=False,
    critic_intervention='both named arms explicitly STANDARDIZED; otherwise original D1',
    primary_intervention_comparison='MEAN_FINAL2950 minus same-server CONTROL_FINAL2950',
    local_training_input_paths=cfg['training_input_paths'].copy())
cfg['endpoint_weight_paths'] = {
    'CONTROL_SELECTED':str(ROOT/'SERVER_FRESH_CONTROL/TRANSLATOR.pt'),
    'CONTROL_FINAL':str(ROOT/'SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt'),
    'MEAN_SELECTED':str(ROOT/'CONDITIONAL_MEAN/TRANSLATOR.pt'),
    'MEAN_FINAL':str(ROOT/'CONDITIONAL_MEAN/FINAL_TRANSLATOR.pt')}
copies=['model.py','legacy_model.py','prior.py','fine_core.py','reorder_core.py','evaluation.py','diagnostic_core.py',
        'assessment.py','energy_width.py','data_support.py','endpoint_common.py','random_diagnostics.py',
        'temporal_common.py','common_math.py','supervise.py','owned_worker.py','replay_endpoints.py']
for name in copies: shutil.copy2(R28/name,ROOT/name)
for name in ['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz',
             'REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','FROZEN_TILT_TABLE.json']:
    shutil.copy2(R28/name,ROOT/name)
t=(R28/'online_training.py').read_text('utf8')
t=t.replace('from prior import weighted_quantile','from prior import weighted_quantile\nfrom conditional_mean import ConditionalMean')
t=t.replace("['STANDARDIZED','HOMOGENEOUS_REORDER','FRESH_FINE']", "['STANDARDIZED','HOMOGENEOUS_REORDER','FRESH_FINE','SERVER_FRESH_CONTROL','CONDITIONAL_MEAN']")
t=t.replace("counter['training_s']+time.monotonic()-clock>=900-10", "counter['training_s']+time.monotonic()-clock>=cfg['training_total_limit_s']-10")
t=t.replace("fine=FineLoss(constants,device,seed=cfg['fresh_projection_seed'],root=root);", "mean_term=ConditionalMean(data['mean_mu'],constants['shape480__scale'],device);mean_rows=[];mean_negative=0;mean_singletons=0;\n    fine=FineLoss(constants,device,seed=cfg['fresh_projection_seed'],root=root);")
needle="loss=base+fine_value+cfg['style_reconstruction_weight']*style_loss"
assert t.count(needle)==1
t=t.replace(needle, "mean_value,mean_square,mean_variance_correction,singletons=mean_term(y,bn);mean_negative+=int(float(mean_value)<0);mean_singletons+=singletons;mean_rows.append([float(mean_value.detach()),float(mean_square.detach()),float(mean_variance_correction.detach()),singletons]);loss=base+fine_value+cfg['style_reconstruction_weight']*style_loss+(cfg['mean_coefficient']*mean_value if arm=='CONDITIONAL_MEAN' else 0.0)")
t=t.replace("'fine_train':float(fine_value),", "'conditional_mean_U':float(mean_value.detach()),'uncorrected_batch_mean_square':float(mean_square.detach()),'variance_correction':float(mean_variance_correction.detach()),'negative_mean_batches_cumulative':mean_negative,'singleton_group_rows_cumulative':mean_singletons,'mean_coefficient_in_objective':cfg['mean_coefficient'] if arm=='CONDITIONAL_MEAN' else 0.0,'fine_train':float(fine_value),")
t=t.replace("np.savez_compressed(root/'GRADIENT_NORMS_PRIVATE.npz'", "np.savez_compressed(root/'MEAN_LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(mean_rows,dtype='f8'));np.savez_compressed(root/'GRADIENT_NORMS_PRIVATE.npz'")
t=t.replace("'fresh_projection_calls':fine.calls", "'mean_negative_batches':mean_negative,'mean_singleton_rows':mean_singletons,'mean_rows_columns':['mean_U','uncorrected_batch_mean_square','variance_correction','singleton_rows'],'mean_coefficient':cfg['mean_coefficient'] if arm=='CONDITIONAL_MEAN' else 0.0,'fresh_projection_calls':fine.calls")
assert "counter['training_s']+time.monotonic()-clock>=900-10" not in t
(ROOT/'online_training.py').write_text(t,'utf8');ast.parse(t)
payloads={
    'TRAIN_PRIVATE.npz':cfg['training_input_paths']['TRAIN_PRIVATE.npz'],
    'VALIDATION_INPUTS.npz':cfg['training_input_paths']['VALIDATION_INPUTS.npz'],
    'FINE_CONSTANTS.npz':cfg['training_input_paths']['FINE_CONSTANTS.npz'],
    'CRITIC_SCALER.npz':cfg['training_input_paths']['CRITIC_SCALER.npz'],
    'R26_TRAINING_STREAMS_PRIVATE.npz':cfg['consumed_R26_stream'],
    'R23_TRAINING_RANDOM_STATES.json':cfg['actual_training_random_states'],
    'R23_TRAINING_RECEIPT.json':cfg['actual_training_receipt'],
    'R26_TRAINING_RECEIPT.json':cfg['consumed_R26_receipt'],
    'INITIAL_TRANSLATOR.pt':cfg['G_initial'], 'INITIAL_STYLE_READER.pt':cfg['reader_initial'], 'INITIAL_CRITIC.pt':cfg['D_initial']}
cfg['remote_input_sources']={n:{'path':str(p),'bytes':Path(p).stat().st_size} for n,p in payloads.items()}
remote_cfg=json.loads(json.dumps(cfg))
remote_cfg['training_input_paths']={n:REMOTE+'/inputs/'+n for n in cfg['training_input_paths']}
for key,name in [('consumed_R26_stream','R26_TRAINING_STREAMS_PRIVATE.npz'),('actual_training_random_states','R23_TRAINING_RANDOM_STATES.json'),
                 ('actual_training_receipt','R23_TRAINING_RECEIPT.json'),('consumed_R26_receipt','R26_TRAINING_RECEIPT.json'),
                 ('G_initial','INITIAL_TRANSLATOR.pt'),('reader_initial','INITIAL_STYLE_READER.pt'),('D_initial','INITIAL_CRITIC.pt')]:remote_cfg[key]=REMOTE+'/inputs/'+name
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
write('config.json',cfg);write('remote_config.json',remote_cfg)
write('REMOTE_RESOURCE_PREFLIGHT.json',{'tool':'682266','native_exit':0,'remote_PID':21175,'UTC':'2026-10-09T17:03:19.667730+00:00',
    'owned_project_processes_argv_and_cwd':[],'MemAvailable_kB':61100468,'GPU':'RTX5070,12227MiB total,243MiB used,0percent utilization',
    'not_a_scientific_worker':True,'existing_others_not_modified':True})
protocol='''# R32 fixed protocol

Two sequential Linux same-platform fits SERVER_FRESH_CONTROL and CONDITIONAL_MEAN. Both names explicitly use the original STANDARDIZED critic. Same R21 selected2000 G/reader, R18 initial D, new Adam, complete R26 94400-row trajectory and corresponding z/white/style/NCE, 2946 homogeneous+4 mixed batches; PCG64(20261009281) fresh three-group128 directions in original order. All original losses/weights, fine quantile L1 and J20 validation are unchanged. No DEV selection.

Only added objective: fixed REAL TRAIN-fit per-Q-bin peak-normalized mean mu; shared existing shape480 TRAIN scale. For each actual source-bin subset n>=2, off-diagonal generated standardized error inner products divided by480*n*(n-1). Mixed groups weighted n/32; singletons zero without replacement. Negative values retained and all generated operands differentiate. Coefficient exactly1 on CONDITIONAL_MEAN,0 on control. Same-forward U/MSE/variance correction logged in both. TRAIN target row ordinal and source container/producer lineage identify the mean fit rows; no validation/DEV target moments.

Small float64 explicit pair-loop/algebra/gradient checks precede first update, including singleton and mixed32 weights. Constants have no gradients. Ideal IID expectation is a mean error, but actual finite sampled/update trajectory is not claimed unbiased.

Selection remains J20, step0/every250/final, strict improvement/early tie, same synchronous G/reader/D/optimizer. Max2950 each/900s perarm/1800s total; unequal actual update counts disclosed. Four selected/final endpoints, identities checked before same-arm alias reuse. Local CPU main generation and independent replay use same saved source/z/white/style, including signal/electronic/ADC/IDs.

Complete original evaluation:7 historical+3pure480 views, original600/250 rows/groups,500fixed-classifier bootstrap;40HGB/32EM maximum and immediate persistence. No domain-moment transformed classification. Joint8<=4096/probes<=3328; primary2000 perendpoint unchanged source composition. Original and four exposed new projection blocks retained. Means/SD described overall/output-Q/source-Q with N>=40 support and all gaps.

Predefined gate: both final fullshape/raw480 paired BA CI upper<0,common output-Q mean MSE reduction>=20percent; no>10percent worsening of PSDW1/widthMAE/baselineRMS/ACF1, no>2pp purehad sourcebin3/4 valley worsening. Undefined support cannot pass. Operational limits, not physical equivalence. Means close alone are not success.

Science2400s including Linux guard probe, setup/failures/2fits/evaluation/replay/render,4threads,actualRSS/GPUallocated4GiB,scientific output512MiB. Executor stop18:52:01.024475UTC. No EVAL/CAL/newREAL/G4/install/Git/agents. New independent server leaf, actual PID/nonce/RSS before imports. Normal transfer/load only, no cross-end consistency tests. Existing stage files immutable.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
bindings=[]
for p in sorted(ROOT.glob('*.py')):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'first_actual_command':cfg['first_actual_command'],'scientific_updates':0,'source_AST':bindings,
    'input_reuse_bytes':sum(x['bytes'] for x in cfg['remote_input_sources'].values()),'no_migration_content_testing':True,'old_stage_writes':0})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'prepared':True,'remote_reuse_bytes':sum(x['bytes'] for x in cfg['remote_input_sources'].values())}),flush=True)
