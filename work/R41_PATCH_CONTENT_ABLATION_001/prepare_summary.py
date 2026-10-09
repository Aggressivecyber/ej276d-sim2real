from pathlib import Path
import ast,json,hashlib,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R40_GLOBAL_TIME_MIXER_TRANSLATION_001'
s=(OLD/'summarize_and_plot.py').read_text('utf8').replace('GLOBAL_TIME_MIXER_FINAL','NO_PATCH_NCE_FINAL').replace('GLOBAL_TIME_MIXER_REAL_CONTROL','NO_PATCH_NCE_REAL_CONTROL').replace('Time-mixer G','No-PatchNCE G')
start=s.index('import torch\n');end=s.index("limits={'full_historical_shape'",start)
s=s[:start]+'''record=json.loads((ROOT/A/'TRAINING_RECEIPT.json').read_text('utf8'))
flow=json.loads((ROOT/'MATCHED_TRAINING_FLOW_RECEIPT.json').read_text('utf8'))
participation=json.loads((ROOT/A/'NCE_PARTICIPATION_RECEIPT.json').read_text('utf8'))
stats={k:record[k]for k in ['updates','D_updates','style_reader_updates','training_s','selected_step','validation_generation_rows','actually_sampled_unique_source','actually_sampled_unique_REAL','actually_consumed_rows','fresh_projection_calls','initial_equivalence','global_gradient_norms']}
stats.update(step0_J20=record['initial_validation']['J'],selected_J20=record['selected_validation']['J'],final_J20=record['final_validation']['J'],original_flow_match=flow,NCE_participation=participation,
    internal_selection_does_not_change_final_endpoint=True,extra_initial_forward_rows=0,
    initial_J20_check_execution_order='strict original state plus J20 assertion immediately after validate0 and before first D/G/reader update',
    original_weight_initialization_preserved=True,mean_or_MMD_extra_loss=False,new_parameters=0,
    objective_component_definition={'computed_basic_components':{k:v for k,v in CFG['loss_weights'].items()if k!='content_NCE'},'content_NCE':{'evaluated':False,'measured_loss':None,'objective_weight':0.,'dedicated_forward_calls':0},'fresh_fine_weight':1.,'style_L1_weight':CFG['style_reconstruction_weight']})
write('TRAINING_COMPACT.json',stats)

'''+s[end:]
start=s.index("g6={'unique_G_D_reader_");end=s.index('\ngates=',start)
s=s[:start]+'''g6={'unique_G_D_reader_trajectory_each_2950':stats['updates']==stats['D_updates']==stats['style_reader_updates']==2950,
    'initial_extra_forward_rows':0,'original_state_keys_and_values_equal':all(stats['initial_equivalence']['state_key_and_value_equal'].values()),
    'initial_J20_equal_before_updates':stats['initial_equivalence']['absolute_difference']<=stats['initial_equivalence']['tolerance']and stats['initial_equivalence']['checks_before_first_D_G_reader_update']and all(stats['initial_equivalence'][k]==0 for k in ['G_updates_before_check','D_updates_before_check','reader_updates_before_check']),
    'all_seven_original_streams_match':all(flow['all_original_update_stream_fields'].values()),
    'fresh_projection_calls':stats['fresh_projection_calls'],'fresh_direction_groups':flow['fresh_direction_groups'],
    'all_8850_groups_equal':flow['fresh_directions_all_8850_match_saved_BASE'],
    'actual_PatchNCE_not_computed':not participation['NCE_evaluated']and participation['NCE_loss_value']is None and participation['NCE_dedicated_forward_calls']==0 and participation['NCE_objective_weight']==0,
    'original_8850_NCE_lists_audit_only':participation['NCE_retained_position_lists']==8850 and participation['unused_NCE_positions_only'],
    'project_heads_grad_None_and_initial_final_equal':participation['project_grad_None_update_checks']==2950 and all(participation['project_initial_final_bitwise_equal'].values()),
    'other_original_parameters_trainable_and_finite':participation['all_original_G_parameters_requires_grad']and participation['all_other_G_and_reader_parameters_finite']and not participation['all_other_G_parameter_names_with_None_final_gradient'],
    'participating_G_parameters':participation['participating_G_parameters'],'unused_project_parameters':participation['project_parameter_count'],
    'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
g6['pass']=all(g6[k]for k in ['unique_G_D_reader_trajectory_each_2950','original_state_keys_and_values_equal','initial_J20_equal_before_updates','all_seven_original_streams_match','all_8850_groups_equal','actual_PatchNCE_not_computed','original_8850_NCE_lists_audit_only','project_heads_grad_None_and_initial_final_equal','other_original_parameters_trainable_and_finite','bitwise_pass'])and replay['rows']==4000 and stats['fresh_projection_calls']==2950 and g6['fresh_direction_groups']==8850
'''+s[end:]
s=s.replace('One shared-channel zero-output time MLP, restored original fresh fine loss, source G final2950','Original complete CNN, PatchNCE actual computation/objective removed only, source G final2950').replace('close_this_fixed_single_time_MLP_recipe','close_this_fixed_no_PatchNCE_recipe').replace('pre_registered_R40_same_treated_scope','pre_registered_R41_same_treated_scope')
s=s.replace("['BASE','COV','TIME'][j]","['BASE','COV','NO-NCE'][j]")
start=s.index("trace=json.loads((ROOT/A/'TRAINING_TRACE.json')");end=s.index("cov=load(ROOT/'results/DEV_COVARIANCE",start)
s=s[:start]+'''trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'));trace=[r for r in trace if 'step'in r]
xx=[0]+[r['step']for r in trace]+[2950];yy=[stats['step0_J20']]+[r['validation']['J']for r in trace]+[stats['final_J20']]
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
axs[0,0].plot(xx,yy,'o-',ms=3,color=colors[A],label='fixed J20');axs[0,0].axvline(stats['selected_step'],ls=':',color='gray',label=f"record-only selected{stats['selected_step']}")
axs[0,0].axvline(2950,ls='--',color=colors[A],label='sole DEV endpoint2950');axs[0,0].set(title='Internal TRAIN validation including actual final',xlabel='update',ylabel='J20');axs[0,0].legend(fontsize=7)
for group in ['shape480','diff479','baseline100_V']:axs[0,1].plot([r['step']for r in trace],[r['fine_train_groups'][group]for r in trace],'o-',ms=2,label=group)
axs[0,1].set(title='Unchanged fresh-direction quantile losses',xlabel='update',ylabel='sampled training component');axs[0,1].legend(fontsize=7)
gr=load(ROOT/A/'GRADIENT_NORMS_PRIVATE.npz')['values']
for j,tag in enumerate(['G+reader','D']):axs[1,0].plot(np.arange(1,len(gr)+1),gr[:,j],lw=.5,alpha=.8,label=tag)
axs[1,0].axhline(CFG['gradient_clip'],ls=':',color='gray');axs[1,0].set(title='Global preclip gradients; original clip5',xlabel='update',ylabel='L2 norm');axs[1,0].legend(fontsize=7)
for k in ['adversarial','identity','Q_relative','PSD_rank','domain_SW']:
    axs[1,1].plot([r['step']for r in trace],[CFG['loss_weights'][k]*r['components'][k]for r in trace],label=k,lw=1)
axs[1,1].set(title='Actual weighted components: NCE not evaluated',xlabel='update',ylabel='weighted loss');axs[1,1].legend(fontsize=6.5)
axs[1,1].text(.02,.95,'3 project heads: grad=None, weights unchanged',transform=axs[1,1].transAxes,va='top',fontsize=7)
for ax in axs.flat:ax.grid(alpha=.12)
fig.tight_layout();save(fig,'training_no_patch_NCE.png')

'''+s[end:]
start=s.index("zz=load(ROOT/'results/FINAL_MODULE_ZERO_DESCRIPTION_PRIVATE.npz')");end=s.index('files={p.name:',start);s=s[:start]+s[end:]
assert 'time_mixer'not in s and 'FINAL_MODULE_ZERO'not in s and 'module_parameter_change'not in s
ast.parse(s);(ROOT/'summarize_and_plot.py').write_text(s,'utf8')
sources={}
for n in ['generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','flow_runtime.py','model.py','legacy_model.py']:
    b=(ROOT/n).read_bytes();ast.parse(b.decode('utf8'));sources[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(ROOT/'EVALUATION_SOURCE_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'sources':sources,'new_endpoints':['NO_PATCH_NCE_FINAL','NO_PATCH_NCE_REAL_CONTROL'],'newHGB20_densityslots16':'no old reference fits','old_BASE_REAL_reused_from_R39':True},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'source_AST':len(sources),'R40_module_logic_removed':True}),flush=True)
