from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,difflib
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';O=R/'work/R54_CONTINUOUS_CHARGE_FLOW_001'
C=json.loads((S/'config.json').read_text('utf8'));P=json.loads((S/'config_post.json').read_text('utf8'));old=json.loads((O/'config_post.json').read_text('utf8'))
prepared=json.loads((S/'POSTPROCESS_PREPARATION.json').read_text('utf8'))
for p,h in prepared['training_files_and_protocol_acceptance_untouched'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
files=list(S.glob('*.py'))
for p in files:ast.parse(p.read_text('utf8'),filename=str(p))
fixed=['C2ST','C2ST_views','train_each_domain','holdout_each_domain','bootstrap_repeats','bootstrap_seed',
    'local_input_paths','R26_root','R32_root','R33_root','R39_root','old_BASE_REAL_generated',
    'canonical_R51_root','frozen_eval_input_root','solver_steps','solver_check_steps','generation_chunk']
assert all(P[k]==old[k] for k in fixed)
assert P['arms']==[C['arm']] and P['HGB_max_fits']==P['HGB_cap']==P['HGB_plan']==10
assert P['density_EM_max']==P['width_slots']==16 and P['main_generation_rows']==P['replay_rows']==4000
assert P['new_response_training_runs']==1 and P['updates']==50000
assert P['endpoint_weight_paths']=={n:C['remote_root']+'/'+C['arm']+'/FINAL_FLOW.pt' for n in P['endpoints']}
assert all(P[k] is None for k in ['science_budget','deadline_UTC','output_limit_bytes'])
new=(S/'summarize_and_plot.py').read_text('utf8');previous=(O/'summarize_and_plot.py').read_text('utf8')
normalized=new.replace('DIRECT_DISTRIBUTION_RESPONSE','CONTINUOUS_Q_RESPONSE').replace('DIRECT_DISTRIBUTION_REAL_CONTROL','CONTINUOUS_Q_REAL_CONTROL')
# The complete numerical definitions of gates1-5 and acceptance A/B stay identical.
left=previous[previous.index('for arm,ctrl in zip(A,RC):'):previous.index(' g6=')]
right=normalized[normalized.index('for arm,ctrl in zip(A,RC):'):normalized.index(' g6=')]
right=right.replace("'internal_constraint_projection':M['explicit_constraint_projection'][arm],",'')
assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right)), 'original stage gates1-5 changed'
left=previous[previous.index("views=['full_historical_shape'"):previous.index("gate['primary_A_C_pass']")]
right=normalized[normalized.index("views=['full_historical_shape'"):normalized.index("gate['primary_A_C_pass']")]
assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right)), 'original A/B acceptance changed'
support=['canonical.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py',
    'data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py']
for name in support:
    a=(O/name).read_text('utf8').replace('CONTINUOUS_Q_RESPONSE',C['arm']).replace('CONTINUOUS_Q_REAL_CONTROL','DIRECT_DISTRIBUTION_REAL_CONTROL')
    a=a.replace('R54','R59').replace('R53','R54').replace('r53','r54').replace('JOINT_PULSE_RESPONSE','CONTINUOUS_Q_RESPONSE').replace('JOINT_PULSE_REAL_CONTROL','CONTINUOUS_Q_REAL_CONTROL')
    if name=='flow_diagnostics.py':
        a=a.replace('learned_increment_plus_final_f32_cast','learned_and_constraint_increment_plus_final_f32_cast')
        a=a.replace('The last component includes the final float32 cast.', 'The last component includes the explicit box-charge projection and final float32 cast.')
        a=a.replace("'last_component_includes_final_float32_cast':True,", "'last_component_includes_final_float32_cast':True,'last_component_includes_explicit_constraint_projection':True,")
    assert a==(S/name).read_text('utf8'),name
assert 'same_R48_' not in new and 'TEACHER_VALIDATION' not in new
assert 'training_audit(R,C,training[L])' in new
assert 'constraint_description(detail)' in (S/'generate_all.py').read_text('utf8')
assert 'constraint_description(cd)' in (S/'generate_all.py').read_text('utf8')
assert 'actual_network_rows' in (S/'generate_all.py').read_text('utf8')
assert "historical_R54_root" in (S/'evaluate_saved.py').read_text('utf8')
assert "'single_factor_ablation':False" in (S/'evaluate_saved.py').read_text('utf8')
report={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','AST_files':len(files),
    'frozen_training_code_protocol_and_A_F_unchanged':True,
    'original_gates1_to5_AST_identical_except_projection_disclosure':True,
    'original_A_B_acceptance_AST_identical':True,
    'original_evaluation_support_modules_unchanged_except_constraint_component_label':support,'fixed_evaluation_config_keys':fixed,
    'new_model_receipt_checks_replace_inapplicable_teacher_stream_checks':True,
    'explicit_internal_projection_and_actual_neural_call_counts_reported':True,
    'historical_R54_is_frozen_descriptive_comparison_no_single_factor_claim':True,
    'new_scientific_execution':0,'DEV_read':0,'EVAL_read':0,'model_goal_complete':False,
    'postprocess_files':[{ 'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name not in ['train_final.py','network.py','source_flow_core.py','noise_core.py','distribution_loss.py','artificial_check.py','linux_supervise.py','linux_owned_worker.py']]}
(S/'POSTPROCESS_ROOT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n','utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ['postprocess_files','original_evaluation_support_modules_unchanged','fixed_evaluation_config_keys']},indent=2))
