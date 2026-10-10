from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,difflib

R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';O=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
C=json.loads((S/'config.json').read_text('utf8'));P=json.loads((S/'config_post.json').read_text('utf8'));old=json.loads((O/'config_post.json').read_text('utf8'))
prepared=json.loads((S/'POSTPROCESS_PREPARATION.json').read_text('utf8'))
for p,h in prepared['training_files_and_protocol_acceptance_untouched'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
files=list(S.glob('*.py'))
for p in files:ast.parse(p.read_text('utf8'),filename=str(p))
fixed=['C2ST','C2ST_views','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed',
    'local_input_paths','R26_root','R32_root','R33_root','R39_root','old_BASE_REAL_generated',
    'canonical_R51_root','frozen_eval_input_root','solver_steps','solver_check_steps','generation_chunk',
    'canonical_ADC_features','raw_SOURCE_input_quantized_for_features']
assert all(P[k]==old[k]for k in fixed)
assert all(P[k]==v for k,v in C.items()),'post config changed a frozen training or protocol field'
assert P['arms']==[C['arm']] and P['HGB_max_fits']==P['HGB_cap']==P['HGB_plan']==10
assert P['density_EM_max']==P['width_slots']==16 and P['main_generation_rows']==P['replay_rows']==4000
assert P['new_response_training_runs']==1 and P['updates']==50000
assert P['endpoint_weight_paths']=={n:C['remote_root']+'/'+C['arm']+'/FINAL_FLOW.pt'for n in P['endpoints']}
assert all(P[k]is None for k in ['science_budget','deadline_UTC','output_limit_bytes'])
new=(S/'summarize_and_plot.py').read_text('utf8');previous=(O/'summarize_and_plot.py').read_text('utf8')
normalized=new.replace('PRECISION_RESOLVED_RESPONSE','DIRECT_DISTRIBUTION_RESPONSE').replace('PRECISION_RESOLVED_REAL_CONTROL','DIRECT_DISTRIBUTION_REAL_CONTROL')
left=previous[previous.index('for arm,ctrl in zip(A,RC):'):previous.index(' g6=')]
right=normalized[normalized.index('for arm,ctrl in zip(A,RC):'):normalized.index(' g6=')]
assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right)),'original gates1-5 changed'
left=previous[previous.index("views=['full_historical_shape'"):previous.index("gate['primary_A_C_pass']")]
right=normalized[normalized.index("views=['full_historical_shape'"):normalized.index("gate['primary_A_C_pass']")]
assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right)),'original A/B acceptance changed'
left=previous[previous.index(' g6='):previous.index(' gates[arm]')].replace('new_R59_updates_per_arm','new_R61_updates_per_arm')
right=new[new.index(' g6='):new.index(' gates[arm]')]
assert ast.dump(ast.parse('if True:\n'+left))==ast.dump(ast.parse('if True:\n'+right)),'original numerical gate changed'
support=['canonical.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py',
    'data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py',
    'flow_runtime.py','generate_all.py','replay_all.py','infer_saved.py']
for name in support:
    a=(O/name).read_text('utf8').replace('DIRECT_DISTRIBUTION_RESPONSE',C['arm']).replace('DIRECT_DISTRIBUTION_REAL_CONTROL','PRECISION_RESOLVED_REAL_CONTROL')
    a=a.replace('R59','R61').replace('R54','R59').replace('r54','r59').replace('CONTINUOUS_Q_RESPONSE','DIRECT_DISTRIBUTION_RESPONSE').replace('CONTINUOUS_Q_REAL_CONTROL','DIRECT_DISTRIBUTION_REAL_CONTROL')
    assert a==(S/name).read_text('utf8'),name
numerical=(S/'numerical_check.py').read_text('utf8')
assert numerical.index("assert audit['pass']")<numerical.index('a,da,ra=solve(')
assert 'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json'in numerical
assert (S/'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json').read_bytes()==(O/'TRAIN_SOLVER_CHECK.json').read_bytes()
evaluate=(S/'evaluate_saved.py').read_text('utf8')
assert "r59names={'R59_SOURCE':'DIRECT_DISTRIBUTION_RESPONSE','R59_REAL_CONTROL':'DIRECT_DISTRIBUTION_REAL_CONTROL'}"in evaluate
assert "'single_factor_ablation':json.loads((ROOT/'TRAINING_FROZEN_AUDIT.json').read_text())['pass']"in evaluate
assert "'single_training_seed':True,'training_seed_variability_not_estimated':True"in evaluate
assert 'PAIRED_R61_MINUS_R59.json'in evaluate
assert 'new sampling streams'not in evaluate
precision=(S/'precision_audit.py').read_text('utf8')
assert all(key in precision for key in ['R59_initial_state_equal','R59_all_six_consumed_streams_equal','R59_parent_arrays_and_exposures_equal',
    'R59_final_PCG64_states_equal','R59_frozen_distribution_statistics_byte_identical','original_order_equals_resolved_plus_omitted',
    'precision_intervals_cover_original_source','precision_rule_frozen_original_denominator'])
checks=json.loads((S/'POST_AUDIT_ARTIFICIAL_CHECK.json').read_text('utf8'));assert checks['pass']and checks['scientific_data_read']==0
protected_names={row['name']for row in json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))['files']}
report={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','AST_files':len(files),
    'frozen_training_code_protocol_and_A_F_unchanged':True,
    'original_gates1_to5_AST_identical':True,'original_A_B_acceptance_AST_identical':True,'original_numerical_replay_gate_AST_identical':True,
    'original_evaluation_support_modules_identical_modulo_arm_labels':support,'fixed_evaluation_config_keys':fixed,
    'single_factor_audit_before_new_TRAIN_forward_and_DEV':True,
    'historical_R59_predicted_probabilities_reused_no_new_control_fit':True,
    'comparison_conditional_on_fixed_classifiers_and_single_training_seed':True,
    'precision_trace_and_internal_projection_in_figures':True,
    'audit_artificial_checks':checks,'new_scientific_execution':0,'DEV_read':0,'EVAL_read':0,'model_goal_complete':False,
    'postprocess_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files if p.name not in protected_names],
    'reviewed_diffs':{n:''.join(difflib.unified_diff((O/n).read_text('utf8').splitlines(True),(S/n.replace('render_R59','render_R61')).read_text('utf8').splitlines(True),fromfile='R59/'+n,tofile='R61/'+n))for n in ['post_audit.py','numerical_check.py','evaluate_saved.py','summarize_and_plot.py','render_R59_figures.py']}}
(S/'POSTPROCESS_ROOT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n','utf8')
print(json.dumps({k:v for k,v in report.items()if k not in ['postprocess_files','reviewed_diffs','audit_artificial_checks','fixed_evaluation_config_keys','original_evaluation_support_modules_identical_modulo_arm_labels']},indent=2))
