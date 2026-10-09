from pathlib import Path
from datetime import datetime,timezone
import json,ast,hashlib,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R39_FULL_JOINT_KERNEL_TRANSLATION_001'
A='GLOBAL_TIME_MIXER_FINAL';R='GLOBAL_TIME_MIXER_REAL_CONTROL'
s=(OLD/'summarize_and_plot_recover.py').read_text('utf8')
s=s.replace('JOINT_KERNEL_FINAL',A).replace('JOINT_KERNEL_REAL_CONTROL',R).replace('Joint-kernel G','Time-mixer G').replace("['BASE','COV','MMD']","['BASE','COV','TIME']")
start=s.index("cal=json.loads(");end=s.index("limits=",start)
s=s[:start]+s[end:]
start=s.index("g6={'unique_G");end=s.index("\ngates=",start)
g6='''g6={'unique_G_D_reader_trajectory_each_2950':stats['updates']==stats['D_updates']==stats['style_reader_updates']==2950,
    'initial_extra_forward_rows':256,'initial_components_bitwise':all(all(v['all_components_bitwise_equal'].values())and v['mixer_delta_exact_zero']for v in stats['initial_equivalence']['rows'].values()),
    'initial_J20_equal':stats['original_flow_match']['initial_J20_absolute_difference']<=stats['original_flow_match']['initial_J20_tolerance'],
    'all_seven_original_streams_match':all(stats['original_flow_match']['all_original_update_stream_fields'].values()),
    'fresh_projection_calls':stats['fresh_projection_calls'],'fresh_direction_groups':stats['original_flow_match']['fresh_direction_groups'],
    'all_8850_groups_equal':stats['original_flow_match']['fresh_directions_all_8850_match_saved_BASE'],
    'module_all_four_parameters_updated_and_finite':all(v['updated']and v['finite']for v in stats['module_parameter_change']['parameters'].values()),
    'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
g6['pass']=all(g6[k]for k in ['unique_G_D_reader_trajectory_each_2950','initial_components_bitwise','initial_J20_equal','all_seven_original_streams_match','all_8850_groups_equal','module_all_four_parameters_updated_and_finite','bitwise_pass'])and replay['rows']==4000 and stats['fresh_projection_calls']==2950 and g6['fresh_direction_groups']==8850
'''
s=s[:start]+g6+s[end:]
s=s.replace('Joint full-time Gaussian-mixture MMD, fixed initial grad-y coefficient, end-to-end source G final2950','One shared-channel zero-output time MLP, restored original fresh fine loss, source G final2950')
s=s.replace('close_this_fixed_joint_Gaussian_MMD_recipe','close_this_fixed_single_time_MLP_recipe')
s=s.replace("'startup_addendum_bound':True","'pre_registered_R40_same_treated_scope':True")
s=s.replace("'fixed_R37_plus0_01_limits'","'fixed_complete_response_point_limits'")
# Original figure fixes are applied before this first plot run.
s=s.replace("figsize=(14,7.6)","figsize=(14,8.3)")
s=s.replace("for ax in axs:ax.set_xticks(range(10),short);ax.legend(fontsize=8,ncol=3);ax.grid(axis='y',alpha=.12)","for ax in axs:ax.set_xticks(range(10),short);ax.grid(axis='y',alpha=.12)\naxs[0].legend(fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.12))\naxs[1].legend(fontsize=8,loc='upper left')")
start=s.index("trace=json.loads((ROOT/A/'TRAINING_TRACE.json')");end=s.index("cov=load",start)
trainplot='''trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'))
trace=[r for r in trace if 'step'in r]
xx=[0]+[r['step']for r in trace]+[2950];yy=[stats['step0_J20']]+[r['validation']['J']for r in trace]+[stats['final_J20']]
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
axs[0,0].plot(xx,yy,'o-',ms=3,color=colors[A],label='fixed J20');axs[0,0].axvline(stats['selected_step'],ls=':',color='gray',label=f"record-only selected{stats['selected_step']}")
axs[0,0].axvline(2950,ls='--',color=colors[A],label='sole DEV endpoint2950');axs[0,0].set(title='Internal TRAIN validation including actual final',xlabel='update',ylabel='J20');axs[0,0].legend(fontsize=7)
for group in ['shape480','diff479','baseline100_V']:axs[0,1].plot([r['step']for r in trace],[r['fine_train_groups'][group]for r in trace],'o-',ms=2,label=group)
axs[0,1].set(title='Restored fresh-direction quantile losses',xlabel='update',ylabel='sampled training component');axs[0,1].legend(fontsize=7)
gr=load(ROOT/A/'MIXER_GRADIENT_NORMS_PRIVATE.npz')['values']
for j,tag in enumerate(['W1 weight','W1 bias','W2 weight','W2 bias']):axs[1,0].plot(np.arange(1,2951),gr[:,j],lw=.5,alpha=.8,label=tag)
axs[1,0].set(title='Mixer preclip gradients; first W1=0 is expected',xlabel='update',ylabel='L2 norm');axs[1,0].legend(fontsize=7)
for j,(name,row)in enumerate(stats['module_parameter_change']['parameters'].items()):axs[1,1].bar(j,row['delta_L2_norm'])
axs[1,1].set_xticks(range(4),['W1 weight','W1 bias','W2 weight','W2 bias']);axs[1,1].set(title='Final minus saved initial parameters',ylabel='L2 norm')
for ax in axs.flat:ax.grid(alpha=.12)
fig.tight_layout();save(fig,'training_time_mixer.png')

'''
s=s[:start]+trainplot+s[end:]
s=s.replace("shortcv=['shape493','pure u480','baseline','V480']","shortcv=['shape493','pure u480','baseline','V480\\n(descriptive)']")
s=s.replace("axs[0,1].axhline(.01,ls=':',color='red',label='point limit +.01');axs[0,1].axhline(.10,ls='--',color='gray',label='95% upper limit +.10');axs[0,1].axhline(0,lw=.7,color='gray')", "axs[0,1].hlines(.01,-.4,2.4,linestyles=':',color='red',label='first3: point +.01');axs[0,1].hlines(.10,-.4,2.4,linestyles='--',color='gray',label='first3: CI upper +.10');axs[0,1].axvspan(2.6,3.4,color='#eeeeee',alpha=.4);axs[0,1].axhline(0,lw=.7,color='gray')")
s=s.replace('REAL controls are processing comparisons, not a stochastic identity test','Same-treated REAL controls; gates apply to shape/u/baseline, V480 descriptive')
start=s.index('files={p.name:')
moduleplot='''zz=load(ROOT/'results/FINAL_MODULE_ZERO_DESCRIPTION_PRIVATE.npz')
fig,axs=plt.subplots(1,3,figsize=(14,4.2))
axs[0].plot(tt,zz['full_module_preADC_V'][0]*1000,color=colors[A],label='final full module');axs[0].plot(tt,zz['zero_module_preADC_V'][0]*1000,color='gray',ls='--',label='final module disabled')
axs[0].set(title='Prespecified first source; same final network',xlabel='relative ns',ylabel='raw preADC mV');axs[0].legend(fontsize=7)
for key in ['preADC','signal','electronic']:
    a=zz[{'preADC':'full_module_preADC_V','signal':'full_module_signal_V','electronic':'full_module_electronic_V'}[key]].astype('f8')
    b=zz[{'preADC':'zero_module_preADC_V','signal':'zero_module_signal_V','electronic':'zero_module_electronic_V'}[key]].astype('f8')
    value=np.sort(np.sqrt(np.mean((a-b)**2,axis=1))*1000);axs[1].plot(value,np.arange(1,len(value)+1)/len(value),label=key)
axs[1].set(title='First128 full480 paired RMS differences',xlabel='RMS mV',ylabel='CDF');axs[1].legend(fontsize=7)
delta=zz['full_module_preADC_V'].astype('f8')-zz['zero_module_preADC_V'].astype('f8')
axs[2].plot(tt,np.median(delta,axis=0)*1000,color=colors[A],label='median');axs[2].plot(tt,delta.mean(0)*1000,color='#555',ls=':',label='mean')
axs[2].set(title='Shared mixer changes both output heads',xlabel='relative ns',ylabel='paired delta mV');axs[2].legend(fontsize=7)
for ax in axs:ax.grid(alpha=.12)
fig.suptitle('Module-zero is an action description, not a second candidate or a training ablation');fig.tight_layout();save(fig,'final_module_action.png')

'''
s=s[:start]+moduleplot+s[start:]
# Minimal training/model aggregate preamble, no new inference or fit.
needle="stats=json.loads((ROOT/'TRAINING_COMPACT.json').read_text('utf8'))"
pre='''import torch
record=json.loads((ROOT/A/'TRAINING_RECEIPT.json').read_text('utf8'))
flow=json.loads((ROOT/'MATCHED_TRAINING_FLOW_RECEIPT.json').read_text('utf8'))
firstgrad=json.loads((ROOT/A/'FIRST_UPDATE_MIXER_GRADIENTS.json').read_text('utf8'))
initial=torch.load(ROOT/A/'INITIAL_TIME_MIXER.pt',map_location='cpu',weights_only=True)
final=torch.load(ROOT/A/'FINAL_TRANSLATOR.pt',map_location='cpu',weights_only=True)
change={}
for name,old in initial.items():
    new=final['time_mixer.'+name];d=new.double()-old.double()
    change[name]={'initial_L2_norm':float(torch.linalg.vector_norm(old.double())),'final_L2_norm':float(torch.linalg.vector_norm(new.double())),
        'delta_L2_norm':float(torch.linalg.vector_norm(d)),'max_abs_delta':float(d.abs().max()),'updated':bool(torch.count_nonzero(d)>0),'finite':bool(torch.isfinite(new).all()),
        'initial_tensor_sha256':hashlib.sha256(old.numpy().tobytes()).hexdigest(),'final_tensor_sha256':hashlib.sha256(new.numpy().tobytes()).hexdigest()}
module={'parameters':change,'new_parameter_count':123488,'no_extra_fit_or_inference':True,'comparison':'final versus saved initial module, not final norm alone'}
write('MODULE_PARAMETER_CHANGE.json',module)
stats={k:record[k]for k in ['updates','D_updates','style_reader_updates','training_s','selected_step','validation_generation_rows','actually_sampled_unique_source','actually_sampled_unique_REAL','actually_consumed_rows','fresh_projection_calls','initial_equivalence','time_mixer_final']}
stats.update(step0_J20=record['initial_validation']['J'],selected_J20=record['selected_validation']['J'],final_J20=record['final_validation']['J'],original_flow_match=flow,first_update_mixer_gradients=firstgrad,module_parameter_change=module,
    internal_selection_does_not_change_final_endpoint=True,extra_initial_forward_rows=256,
    initial_J20_check_execution_order='comparison assertion made at completion; actual initial difference0 and no initial function mismatch',
    original_weight_initialization_preserved=True,mean_or_MMD_extra_loss=False)
write('TRAINING_COMPACT.json',stats)
'''
assert s.count(needle)==1;s=s.replace(needle,pre)
ast.parse(s)
(ROOT/'summarize_and_plot.py').write_text(s,'utf8')
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source':'summarize_and_plot.py','bytes':len(s.encode()),'sha256':hashlib.sha256(s.encode()).hexdigest(),'AST':True,'new_fits_or_generation':0}
(ROOT/'SUMMARY_SOURCE_BINDING.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps(r),flush=True)
