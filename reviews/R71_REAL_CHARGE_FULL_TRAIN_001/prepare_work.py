"""Prepare prospective source/config; no scientific arrays or execution."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, shutil, sys
P=Path(__file__).resolve().parent; R=P.parents[1]; S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'preparation_runtime')
S.mkdir(exist_ok=True)
old=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
cfg0=json.loads((old/'config.json').read_text('utf8'))
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
cfg={k:cfg0[k] for k in ['server_python','CPU_threads','RSS_limit_bytes','GPU_limit_bytes','MemAvailable_min_bytes','expected_R33_environment']}
cfg.update(stage=P.name, owner='/root', remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R71_ROOT_'+stamp+'_UTC',
 exact_CPU_affinity=list(range(12)), science_budget=None, deadline_UTC=None, output_limit_bytes=None, compute_budget_enforced=False,
 allowed_scientific_entries=['train_full.py'], GPU_scientific_entries=['train_full.py'],
 model_seed=20261011710, source_epsilon_seed=20261011711, real_epsilon_seed=20261011712,
 artificial_seed=20261011713, updates=50000, benchmark_warmup_updates=10, benchmark_timed_updates=40,
 cost_limit_seconds=14400, cost_safety_multiplier=1.25, postprocess_reserve_seconds=900,
 lr=2e-4, lr_initial=1e-6, lr_final=2e-5, warmup_updates=500, betas=[.9,.999], gradient_clip=5.,
 recovery_interval_updates=5000, progress_interval_updates=250, SOURCE_N=6593, REAL_N=21047,
 hidden=16, dtype='float64', input_scale_floor=1e-6, charge_limit=300., generation_chunk=256,
 loss_weights=dict(charge_distribution=1.,REAL_identity=1.,source_charge_order=1.),
 loss_scope='pre-ADC predicted charge Q* component only; actual canonical waveform metrics are postchecks',
 endpoint='sole completed update50000, no checkpoint selection', numeric_CPU_reference_rows_per_domain=64,
 numeric_median_LSB_max=.25,numeric_p95_LSB_max=1.,preADC_relative_Qstar_error_max=1e-5,
 model_goal_complete=False, current_acceptance='PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json',
 R65_root=cfg0['remote_root'])
binding0=json.loads((old/'INPUT_BINDING.json').read_text('utf8'))['inputs']
keys=['TRAIN_path','constants_path','noise_calibration_path','valley_definition']
cfg['input_paths']={k:binding0[k]['path'] for k in keys}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
assert not (P/'REGISTRATION.json').exists()
write(S/'config.json',cfg)
write(S/'INPUT_BINDING.json',{'inputs':{k:binding0[k] for k in keys},'scientific_arrays_decoded':False})
shutil.copy2(P/'implementation_candidate/public_charge_head.py',S/'public_charge_head.py')
shutil.copy2(old/'linux_owned_worker.py',S/'linux_owned_worker.py')
supervisor=(old/'linux_supervise.py').read_text('utf8')
supervisor=supervisor.replace("os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['CPU_threads']])", "assert sorted(os.sched_getaffinity(0))==cfg['exact_CPU_affinity'],'Actual12-CPU allocation required'")
supervisor=supervisor.replace('R61 requires explicit null budget fields','R71 has no wall-clock science cap; fixed endpoint and measured cost admission')
(S/'linux_supervise.py').write_text(supervisor,'utf8')
source=(old/'source_flow_core.py').read_text('utf8'); tree=ast.parse(source)
names={'q_torch','charge_vector','BoxChargeProjection','bounded_charge','canonical_adc_torch','canonical_REAL_torch'}
segments=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
assert len(segments)==6
(S/'physical_operators.py').write_text('"""Frozen R65 Box and ADC operators, verbatim AST extraction."""\nimport torch\n\n'+'\n\n'.join(segments)+'\n','utf8')
write(P/'PREPARATION_RECEIPT.json',{'UTC':datetime.now(timezone.utc).isoformat(),'science_arrays_read':False,'guard':{**guard,**memory()},'remote_root':cfg['remote_root'],'copied_operator_names':sorted(names)})
print(json.dumps({'prepared':str(S),'remote_root':cfg['remote_root'],'science_arrays_read':False}))
