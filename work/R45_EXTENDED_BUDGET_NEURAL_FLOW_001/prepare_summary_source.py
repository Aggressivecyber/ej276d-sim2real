from pathlib import Path
from datetime import datetime,timezone
import ast,json,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R43_SCALAR_TIME_NEURAL_FLOW_001'
code=(OLD/'summarize_and_plot.py').read_text('utf8')
code=code.replace('SCALAR_TIME_FLOW_FINAL','EXTENDED_FLOW_FINAL').replace('SCALAR_TIME_FLOW_REAL_CONTROL','EXTENDED_FLOW_REAL_CONTROL')
code=code.replace('Scalar-time flow','Extended50000 flow').replace('Scalar-time flow on REAL','Extended50000 flow on REAL')
code=code.replace("(ROOT/'config.json')","(ROOT/'config_cpu.json')")
start=code.index("flow=json.loads(")
end=code.index("write('TRAINING_COMPACT.json',stats)")+len("write('TRAINING_COMPACT.json',stats)")
replacement="""solver=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
participation=json.loads((ROOT/'MODEL_STRUCTURE.json').read_text('utf8'))
init=json.loads((ROOT/'INHERITANCE_RECEIPT.json').read_text('utf8'))
exposure=json.loads((ROOT/A/'EXPOSURE_SUMMARY.json').read_text('utf8'))
probes=json.loads((ROOT/'FIXED_PROBE_TRAJECTORY.json').read_text('utf8'))
stats={k:record[k]for k in ['global_updates','training_s','selected_endpoint','counter','optimizer_final_steps',
    'fixed_probe_MSE_initial','fixed_probe_MSE_final','probe_final_over_initial_MSE','descriptive_probe_fit_improved',
    'new_t_sha256','new_white_sha256','additional_Monte_Carlo_states_not_reuse_old94400_white',
    'GPU_peak_allocated_bytes','GPU_peak_reserved_bytes','max_path_q_residual','max_field_q_residual','historical_R43_training_s']}
stats.update(inheritance=init,parameter_participation=participation,
    loaded_SOURCE_REAL=exposure['loaded_SOURCE_REAL'],unique_SOURCE_REAL=exposure['unique_SOURCE_REAL'],
    new_rows_per_domain=exposure['new_rows_per_domain'],historical_rows_per_domain=exposure['historical_rows_per_domain'],
    total_rows_per_domain=exposure['total_rows_per_domain'],batch_count_min_max=exposure['batch_count_min_max'],
    exposure_not_independent_events=True,numerical_TRAIN_check=solver,only_fixed_final50000_candidate=True,
    intermediate_checkpoints_engineering_only=True,constants_refits=0,
    probe_trace=[{'global_step':p['global_step'],'overall':p['overall']}for p in probes])
write('TRAINING_COMPACT.json',stats)"""
code=code[:start]+replacement+code[end:]
start=code.index("g6={'unique_new_velocity_model_2950'")
end=code.index("\ngates=",start)
replacement="""g6={'inherited2950_new47050_total50000':init['inherited_optimizer_steps']==[2950]and record['counter']['new_updates']==47050 and record['global_updates']==50000 and record['optimizer_final_steps']==[50000],
    'inherited_all_weights_exact':init['model_all_state_tensor_equal'],
    'scalar_time2048_zero_grad_and_unchanged':participation['2048_zero_columns_gradient_checks']==47050 and participation['2048_zero_columns_value_checks']==47050,
    't_bin_other_gradient_participation':all(participation[k]>0 for k in ['t_nonzero_gradient_steps','bin_nonzero_gradient_steps','other_nonzero_gradient_steps'])and participation['all_parameter_finite_checks']==47050,
    'fixed_source_target_population_cycles':exposure['new_rows_per_domain']==1505600 and exposure['total_rows_per_domain']==1600000,
    'fixed_probe11_no_selection':[p['global_step']for p in probes]==[2950]+list(range(5000,50001,5000))and record['counter']['fixed_probe_forward_rows']==45056,
    'new_MC_t_white_allowed_not_original_states':record['additional_Monte_Carlo_states_not_reuse_old94400_white'],
    'TRAIN_fixed32_vs64_pass':solver['pass'],'TRAIN_no_DEV_read_or_generated':solver['DEV_read_or_generated']==0,
    'all_main_preADC_f64_f32_charge_error_le1e_5':all(max(v['actual_f32_preADC_max_relative_error'],v['f64_max_relative_error'])<=1e-5 for v in charge.values()),
    'all_main_rows_processed':all(v['all_rows_neural_flow']for v in charge.values()),
    'independent_same_server_CPU12_chunk64_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'inheritance':init,'parameter_participation':participation,'charge_error_summaries':charge,
    'TRAIN_probe_improvement_only_descriptive':record['descriptive_probe_fit_improved'],
    'no_old_G_D_reader_or_NCE':not participation['old_G_D_reader_used'],
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
required_g6=['inherited2950_new47050_total50000','inherited_all_weights_exact','scalar_time2048_zero_grad_and_unchanged',
    't_bin_other_gradient_participation','fixed_source_target_population_cycles','fixed_probe11_no_selection',
    'new_MC_t_white_allowed_not_original_states','TRAIN_fixed32_vs64_pass','TRAIN_no_DEV_read_or_generated',
    'all_main_preADC_f64_f32_charge_error_le1e_5','all_main_rows_processed','bitwise_pass','no_old_G_D_reader_or_NCE']
g6['pass']=all(g6[k]for k in required_g6)and replay['rows']==4000
"""
code=code[:start]+replacement+code[end:]
code=code.replace('fixed final2950/32Heun','fixed total50000/32Heun')
code=code.replace('close_this_fixed_scalar_time_neural_flow_recipe','close_this_fixed_extended_budget_neural_flow_recipe')
needle="write('METRICS_COMPACT.json',compact)"
code=code.replace(needle,"compact['short_budget_R43_comparison']=m['short_budget_R43_comparison']\n"+needle)
start=code.index("trace=json.loads(")
end=code.index("num=load(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz'",start)
code=code[:start]+"# Separate plot_training.py publishes the full fixed TRAIN probe and update trajectory.\n\n"+code[end:]
code=code.replace('R43 scalar-time','R45 extended50000')
ast.parse(code);(ROOT/'summarize_and_plot.py').write_text(code,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'AST_source_only':True,
    'scientific_arrays_read':0,'gate1_to5_inherited_unchanged':True,'gate6_only_registered_R45_training_numerical_replay_fields':True}))
