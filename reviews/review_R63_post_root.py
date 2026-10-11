from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,difflib
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';O=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
C=json.loads((S/'config.json').read_text('utf8'));P=json.loads((S/'config_post.json').read_text('utf8'));old=json.loads((O/'config_post.json').read_text('utf8'))
prepared=json.loads((S/'POSTPROCESS_PREPARATION.json').read_text('utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in prepared['training_files_and_protocol_acceptance_untouched'].items():assert sha(Path(p))==h,p
files=list(S.glob('*.py'))
for p in files+list((R/'reviews').glob('*R63*root*.py')):ast.parse(p.read_text('utf8'),filename=str(p))
fixed=['C2ST','C2ST_views','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed','local_input_paths','R26_root','R32_root','R33_root','R39_root','old_BASE_REAL_generated','canonical_R51_root','frozen_eval_input_root','solver_steps','solver_check_steps','generation_chunk','canonical_ADC_features','raw_SOURCE_input_quantized_for_features']
assert all(P[k]==old[k]for k in fixed)
assert all(P[k]==v for k,v in C.items()),'post config changed frozen registration'
assert P['arms']==[C['arm']] and P['HGB_max_fits']==P['HGB_cap']==P['HGB_plan']==10
assert P['density_EM_max']==P['width_slots']==16 and P['main_generation_rows']==P['replay_rows']==4000
assert P['new_response_training_runs']==1 and P['updates']==50000
assert P['endpoint_weight_paths']=={n:C['remote_root']+'/'+C['arm']+'/FINAL_FLOW.pt'for n in P['endpoints']}
assert all(P[k]is None for k in ['science_budget','deadline_UTC','output_limit_bytes'])
def convert(t):
 t=t.replace('PRECISION_RESOLVED_RESPONSE','COVERAGE_RESPONSE').replace('PRECISION_RESOLVED_REAL_CONTROL','COVERAGE_REAL_CONTROL')
 return t.replace('R61','R63').replace('R59','R61').replace('r59','r61').replace('DIRECT_DISTRIBUTION_RESPONSE','PRECISION_RESOLVED_RESPONSE').replace('DIRECT_DISTRIBUTION_REAL_CONTROL','PRECISION_RESOLVED_REAL_CONTROL')
new=(S/'summarize_and_plot.py').read_text('utf8');previous=convert((O/'summarize_and_plot.py').read_text('utf8'))
for begin,end in [('for arm,ctrl in zip(A,RC):',' g6='),("views=['full_historical_shape'","gate['primary_A_C_pass']")]:
 left=previous[previous.index(begin):previous.index(end)];right=new[new.index(begin):new.index(end)]
 assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right)),('original_acceptance_changed',begin)
left=previous[previous.index(' g6='):previous.index(' gates[arm]')];right=new[new.index(' g6='):new.index(' gates[arm]')]
assert ast.dump(ast.parse('if True:\n'+left))==ast.dump(ast.parse('if True:\n'+right))
support=['canonical.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py','data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py','flow_runtime.py','generate_all.py','replay_all.py','infer_saved.py','numerical_check.py']
for name in support:assert convert((O/name).read_text('utf8'))==(S/name).read_text('utf8'),name
numerical=(S/'numerical_check.py').read_text('utf8');assert numerical.index("assert audit['pass']")<numerical.index('a,da,ra=solve(')
assert (S/'HISTORICAL_R61_TRAIN_SOLVER_CHECK.json').read_bytes()==(O/'TRAIN_SOLVER_CHECK.json').read_bytes()
evaluate=(S/'evaluate_saved.py').read_text('utf8')
assert "r61names={'R61_SOURCE':'PRECISION_RESOLVED_RESPONSE','R61_REAL_CONTROL':'PRECISION_RESOLVED_REAL_CONTROL'}"in evaluate
assert "'single_factor_ablation':False,'frozen_objective_extension_integrity':"in evaluate and 'PAIRED_R63_MINUS_R61.json'in evaluate
assert 'other six losses'not in new and 'precision-order ablation'not in evaluate
audit=(S/'post_audit.py').read_text('utf8');assert 'losses.shape==(50000,22)'in audit and "checks['nine_losses_sum']"in audit
assert 'coverage_extension_audit' in audit and "checks={k:bool(v)for k,v in checks.items()}"in audit
checks=json.loads((S/'POST_AUDIT_ARTIFICIAL_CHECK.json').read_text('utf8'));assert checks['pass']and checks['scientific_data_read']==0 and len(checks['independent_negative_perturbations_rejected'])==11
assert sha(S/'coverage_audit.py')==checks['coverage_audit_SHA256']
launch=(R/'reviews/launch_R63_entry_root.py').read_text('utf8')
assert "{k:train['stream_hashes'][k]for k in cfg['expected_R61_stream_hashes']}"in launch and "train['objective_extension_integrity']"in launch
assert "train['temporal_direction_calls']==50000"in launch
protected_names={r['name']for r in json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))['files']}
report={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','AST_files':len(files),'frozen_training_code_protocol_and_A_F_unchanged':True,
 'original_gates1_to5_AST_identical':True,'original_A_B_acceptance_AST_identical':True,'original_numerical_replay_gate_AST_identical':True,
 'original_evaluation_support_modules_identical_modulo_arm_labels':support,'fixed_evaluation_config_keys':fixed,'objective_extension_audit_before_new_TRAIN_forward_and_DEV':True,
 'historical_R61_predictions_reused_no_new_control_fit':True,'comparison_conditional_on_fixed_classifiers_and_single_training_seed':True,
 'coverage_precision_and_internal_projection_in_figures':True,'audit_artificial_checks':checks,'new_scientific_execution':0,'DEV_read':0,'EVAL_read':0,'model_goal_complete':False,
 'postprocess_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}for p in files if p.name not in protected_names],
 'reviewed_diffs':{n:''.join(difflib.unified_diff((O/n).read_text('utf8').splitlines(True),(S/n.replace('render_R61','render_R63')).read_text('utf8').splitlines(True),fromfile='R61/'+n,tofile='R63/'+n))for n in ['post_audit.py','numerical_check.py','evaluate_saved.py','summarize_and_plot.py','render_R61_figures.py']}}
(S/'POSTPROCESS_ROOT_REVIEW.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps({k:v for k,v in report.items()if k not in ['postprocess_files','reviewed_diffs','audit_artificial_checks','fixed_evaluation_config_keys','original_evaluation_support_modules_identical_modulo_arm_labels']},indent=2))
