from __future__ import annotations
import pathlib, ast, hashlib, json
ROOT=pathlib.Path(__file__).resolve().parent
old=(ROOT.parent/'R38_SLICED_MONOTONE_TRANSPORT_001/summarize_and_plot.py').read_text('utf8')
header=old[:old.index('paired=m[')]
header=header.replace("A='SLICED_MONOTONE_TRANSPORT'", "A='JOINT_KERNEL_FINAL'")
header=header.replace("A:'Sliced monotone map'", "A:'Joint-kernel G'")
header=header.replace("'SOURCE_CLEAN':'#6e839e','SOURCE_INPUT':'#6e839e'}", "'SOURCE_CLEAN':'#6e839e','SOURCE_INPUT':'#6e839e','BASE_REAL_CONTROL':'#888888','JOINT_KERNEL_REAL_CONTROL':'#0867ad'}")
header=header.replace("styles={'BASE'", "labels.update(BASE_REAL_CONTROL='BASE G on REAL',JOINT_KERNEL_REAL_CONTROL='Joint-kernel G on REAL')\nstyles={'BASE'")
gates=r'''
paired=m['C2ST']['paired'][A];cs=m['C2ST']['results']
stats=json.loads((ROOT/'TRAINING_COMPACT.json').read_text('utf8'))
cal=json.loads((ROOT/'INITIAL_GRADIENT_SCALE_COMPACT.json').read_text('utf8'))
formula=json.loads((ROOT/'KERNEL_FORMULA_CHECK.json').read_text('utf8'))
limits={'full_historical_shape':.934,'normalized_u480':.928}
g1={'paired_minus_BASE':{v:paired[v]for v in full},'fixed_R37_plus0_01_limits':limits,
    'candidate_BA':{v:cs[A][v]['balanced_accuracy']for v in full},
    'both_BASE_CI_upper_negative':all(paired[v]['BA_delta_95CI'][1]<0 for v in full),
    'both_fixed_R37_point_limits':all(cs[A][v]['balanced_accuracy']<=limits[v]+1e-14 for v in full)}
g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_fixed_R37_point_limits']
common=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in ['REAL','BASE',A])]
pw={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in common]))if common else None for n in ['BASE',A]}
measures={k+'_overall_W1':measure(m['domain_vs_REAL'][A]['overall'][k]['W1'],m['domain_vs_REAL']['BASE']['overall'][k]['W1'])for k in ['PSD','baseline_RMS_V','ACF1']}
measures['PSD_common_Q_equal_mean_W1']={**measure(pw[A],pw['BASE']),'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40}
for row in width['common_support_errors']['JOINT_vs_BASE']:
    if row['field']in ['mean','sigma_PSD']:
        field='peak_position_MAE'if row['field']=='mean'else'raw_width_MAE'
        measures['component'+str(row['component'])+'_'+field]={**measure(row['MAE'][A],row['MAE']['BASE']),
            'common_bins_1based':row['common_bin_ids_1based'],'density_failure_not_hidden_by_intersection':True}
required=[i for i in range(1,7)if width['tables']['BASE']['slices'][i]['valid']]
newfail=[i+1 for i in required if not width['tables'][A]['slices'][i]['valid']]
zeros={n:int(np.sum(features[n+'__baseline_RMS_V']==0))for n in ['BASE',A]}
gaps={n:[{'bin_id_1based':s['bin']+1,'N':s['N'],'failure_reasons':s['failure_reasons']}for s in width['tables'][n]['slices']if not s['valid']]for n in width['tables']}
g2={'measures':measures,'BASE_valid_required_bins_1based':[i+1 for i in required],'new_required_density_failures':newfail,
    'all_density_support_gaps':gaps,'zero_baseline_counts':zeros,'zero_baseline_not_increased':zeros[A]<=zeros['BASE']}
g2['pass']=len(measures)==8 and all(v['pass']for v in measures.values())and not newfail and zeros[A]<=zeros['BASE']
primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4'];valleys=[];ranks=[]
for b in ['bin3','bin4']:
    x=primary[A][b]['valley_fraction'];base=primary['BASE'][b]['valley_fraction'];d=x-base if x is not None and base is not None else None
    valleys.append({'actual_source_bin':b,'N':primary[A][b]['source_count'],'candidate':x,'BASE':base,'delta':d,'pass':d is not None and d<=.02})
for field in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
    x=m['source_association_actual_preADC_input'][A]['composition_and_PDG']['all'][field]['Spearman']
    b=m['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][field]['Spearman'];d=b-x if x is not None and b is not None else None
    ranks.append({'measure':field,'candidate':x,'BASE':b,'BASE_minus_candidate':d,'pass':d is not None and d<=.05})
gen=m['generation'][A];sens=m['source_probes'][A]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
g3={'actual_source_pure_had_valleys':valleys,'whole_population_source_ranks':ranks,
    'source_ID_order_composition_weights_preserved':gen['all_source_ID_order_weights_and_composition_preserved']and m['data_roles']['DEV_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518}and not m['primary_DEV_population_reweighted'],
    'no_new_invalid_nonfinite_ADC_clip':m['domains'][A]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,
    'outside_TRAIN_support_rows':gen['TRAIN_support_outside_N'],'all_outside_rows_use_new_G':gen['BASE_fallback_N']==0 and gen['G_applied_N']==2000,
    'source_roll_RMS_V':sens,'sensitivity_not_correct_physical_mapping':True}
g3['pass']=all(x['pass']for x in valleys+ranks)and g3['source_ID_order_composition_weights_preserved']and g3['no_new_invalid_nonfinite_ADC_clip']and g3['all_outside_rows_use_new_G']and sens is not None and sens>0
bd=m['boundary_differences']['results'];bounds=[]
for key in bd[A]:
    x=bd[A][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd['INDEPENDENT_CFM'][key]['W1_V'];lim=max(b,o)+1/16383 if b is not None and o is not None else None
    bounds.append({'location':key,'candidate_W1_V':x,'BASE_W1_V':b,'INDEPENDENT_W1_V':o,'limit_V':lim,'pass':x is not None and lim is not None and x<=lim})
g4={'boundaries':bounds,'full_SOURCE_perturbation':m['guide_probe'],'pass':len(bounds)==9 and all(x['pass']for x in bounds)and m['guide_probe']['pass']}
control_pair=json.loads((ROOT/'SAME_TREATED_REAL_CONTROL_PAIRED.json').read_text('utf8'))
control_views=['full_historical_shape','normalized_u480','baseline_local'];control_rows={}
for v in control_views:
    r=control_pair[v];control_rows[v]={'BASE_G_REAL_BA':cs['BASE_REAL_CONTROL'][v]['balanced_accuracy'],
        'new_G_REAL_BA':cs['JOINT_KERNEL_REAL_CONTROL'][v]['balanced_accuracy'],**r,
        'point_delta_le0_01':r['BA_delta_first_minus_second']<=.01+1e-14,'CI_upper_le0_10':r['BA_delta_95CI'][1]<=.10}
    control_rows[v]['pass']=control_rows[v]['point_delta_le0_01']and control_rows[v]['CI_upper_le0_10']
g5={'startup_addendum_bound':True,'comparison':'new G REAL control minus BASE G REAL control',
    'three_views':control_rows,'RAW_RR_absolute_effects_descriptive':{n:m['C2ST']['paired_vs_RAW_RR'][n]for n in ['BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']},
    'RAW_identity_not_required':True,'pass':all(r['pass']for r in control_rows.values())}
g6={'unique_G_D_reader_trajectory_each_2950':stats['updates']==2950,
    'actual_D_compute_calls_including_failure':stats['actual_D_update_compute_calls_including_recovery'],
    'initial_gradient_calibration_once':stats['c_reevaluation_calls']==0 and cal['batches']==64 and stats['gradient_coefficient']>0,
    'kernel_formula_pass':formula['pass'],'all_seven_original_streams_match':all(stats['original_flow_match']['all_original_update_stream_fields'].values()),
    'MMD_calls':stats['MMD']['calls'],'Gram_groups':stats['MMD']['Gram_groups'],'kernel_terms':stats['MMD']['kernel_terms'],
    'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
g6['pass']=all(g6[k]for k in ['unique_G_D_reader_trajectory_each_2950','initial_gradient_calibration_once','kernel_formula_pass','all_seven_original_streams_match','bitwise_pass'])and replay['rows']==6000 and stats['MMD']['calls']==2950
gates={'gate1_complete_response':g1,'gate2_PSD_width_centers_noise':g2,'gate3_source_preservation':g3,
    'gate4_boundary_source':g4,'gate5_same_treated_REAL_control':g5,'gate6_numerical_replay':g6}
retain=all(x['pass']for x in gates.values());gate={'fixed_recipe':'Joint full-time Gaussian-mixture MMD, fixed initial grad-y coefficient, end-to-end source G final2950',
    'prespecified_rules_changed_after_data':False,**gates,'retain':retain,'decision':'retain_DEV_candidate_pending_actual_QA'if retain else'close_this_fixed_joint_Gaussian_MMD_recipe',
    'no_general_impossibility_claim':True,'no_new_fit_or_generation_in_summary':True}
write('GATE_METRICS.json',gate)
# This publisher explicitly retains only aggregate classifier statistics, never split indices or predictions.
allowed_cs=['results','paired','paired_vs_RAW_RR','same_treated_REAL_control_paired','REAL_control_description','prediction_reference_missing']
compact={'status':m['status'],'decision':gate['decision'],'gate':gate,'counters':m['counters'],'training':stats,
    'C2ST':{k:m['C2ST'][k]for k in allowed_cs if k in m['C2ST']},'same_treated_REAL_control_paired':control_pair,
    'overall_domain_vs_REAL':{n:v['overall']for n,v in m['domain_vs_REAL'].items()},'PSD_main_vs_REAL':psd['vs_REAL'],
    'width_common_support_errors':width['common_support_errors'],'density_support_gaps':gaps,'source_primary_actual_Q':primary,
    'source_historical_ADC_reference_Q':m['source_composition']['historical_postADC_reference_Q_bins3_4'],
    'source_ranks_actual_Q':{n:r['composition_and_PDG']['all']for n,r in m['source_association_actual_preADC_input'].items()},
    'full_SOURCE_probe':m['guide_probe'],'boundary_differences':m['boundary_differences'],'raw_voltage_moments':m['raw_voltage_moments'],
    'covariance_description':m['covariance_description'],'projection_common_support':m['projection_distances']['common_support_summary'],
    'replay':{'rows':replay['rows'],'pass':replay['pass']},'source_IDs_order_weights_preserved':True,
    'functional_components_not_physical_truth':True,'EVAL_read':0,'CAL_waveform_decode':0,
    'privacy':'aggregate only; identities, predictions, selected rows, RNG streams and waveforms stay local'}
write('METRICS_COMPACT.json',compact)
'''
plots=old[old.index('plt.rcParams'):]
plots=plots.replace('Sliced monotone map minus BASE','Joint-kernel G minus BASE')
plots=plots.replace("['BASE','COV','SLICE']","['BASE','COV','MMD']")
start=plots.index("fig,axs=plt.subplots(2,2,figsize=(13,7))")
end=plots.index("cov=load",start)
training=r'''
trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'))
grams=load(ROOT/A/'MMD_GRAMS_BY_UPDATE_PRIVATE.npz')['values']
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
x=[0]+[r['step']for r in trace];y=[stats['step0_J20']]+[r['validation']['J']for r in trace]
axs[0,0].plot(x,y,'o-',ms=3,color=colors[A],label='fixed J20');axs[0,0].axvline(stats['selected_step'],ls=':',color='gray',label='record-only selected2750');axs[0,0].axvline(2950,ls='--',color=colors[A],label='sole DEV endpoint2950')
axs[0,0].set(title='TRAIN internal validation; sole final endpoint',xlabel='update',ylabel='J20');axs[0,0].legend(fontsize=7)
for j,(title,c)in enumerate([('within G',2),('within REAL',3),('cross all',4)]):axs[0,1].plot(np.arange(1,2951),grams[:,c],lw=.35,alpha=.65,label=title)
axs[0,1].set(title='Saved Gram means; not separate feature losses',xlabel='update',ylabel='kernel mean');axs[0,1].legend(fontsize=7)
for col,title in [(0,'raw MMD U'),(1,'fixed c x MMD U')]:axs[1,0].plot(np.arange(1,2951),grams[:,col],lw=.4,label=title)
axs[1,0].axhline(0,color='gray',lw=.7);axs[1,0].set(title=f'Fixed c={stats["gradient_coefficient"]:.5f}; no truncation',xlabel='update',ylabel='loss');axs[1,0].legend(fontsize=7)
cr=cal['rows'];xx=np.arange(1,65)
axs[1,1].semilogy(xx,[r['old_grad_y_Frobenius_squared']for r in cr],lw=.9,label='old fresh fine grad-y squared')
axs[1,1].semilogy(xx,[r['new_grad_y_Frobenius_squared']for r in cr],lw=.9,label='raw joint MMD grad-y squared')
axs[1,1].semilogy(xx,[stats['gradient_coefficient']**2*r['new_grad_y_Frobenius_squared']for r in cr],lw=.9,ls=':',label='c-scaled joint grad-y squared')
axs[1,1].set(title='Once-only initial64 TRAIN batches; no parameter update',xlabel='original update ordinal',ylabel='Frobenius squared');axs[1,1].legend(fontsize=7)
for ax in axs.flat:ax.grid(alpha=.12)
fig.tight_layout();save(fig,'training_joint_kernel.png')

'''
plots=plots[:start]+training+plots[end:]
plots=plots.replace('actual guide delta','actual SOURCE delta').replace('128 fixed-bin guide-only changes','128 fixed-bin full-SOURCE changes')
plots=plots.replace('Had fraction0','Fast/slow r=0').replace('Had fraction.25','Fast/slow r=.25').replace('Had fraction.5','Fast/slow r=.5').replace('Had fraction.75','Fast/slow r=.75').replace('Had fraction1','Fast/slow r=1')
plots=plots.replace('Guide dependence and additive algebra; mapped physical branch split unknown','Full-SOURCE sensitivity and functional G components; not physical branch truth')
plots=plots.replace("save(fig,'guide_and_random_decomposition.png')","save(fig,'source_and_random_decomposition.png')")
end=plots.index('files={p.name:')
control=r'''
fig,axs=plt.subplots(2,2,figsize=(13,7.4))
cv=['full_historical_shape','normalized_u480','baseline_local','full_voltage']
shortcv=['shape493','pure u480','baseline','V480']
for j,n in enumerate(['RAW_REAL_REAL','BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']):
    color=colors['TRAIN_REFERENCE']if n=='RAW_REAL_REAL'else colors[n]
    yy=np.asarray([cs[n][v]['balanced_accuracy']for v in cv]);cc=np.asarray([cs[n][v]['heldout_CI']['balanced_accuracy_95CI']for v in cv])
    axs[0,0].errorbar(np.arange(4)+(j-1)*.13,yy,yerr=np.maximum(0,np.vstack([yy-cc[:,0],cc[:,1]-yy])),fmt='o',capsize=2,ms=4,color=color,label='RAW TRAIN-DEV'if n=='RAW_REAL_REAL'else labels[n])
axs[0,0].axhline(.5,color='gray',ls=':');axs[0,0].set(title='Same reference rows / stochastic processing',ylabel='fixed holdout BA',ylim=(.43,1.02));axs[0,0].set_xticks(range(4),shortcv);axs[0,0].legend(fontsize=7)
delta=np.asarray([control_pair[v]['BA_delta_first_minus_second']for v in cv]);cc=np.asarray([control_pair[v]['BA_delta_95CI']for v in cv])
axs[0,1].errorbar(range(4),delta,yerr=np.maximum(0,np.vstack([delta-cc[:,0],cc[:,1]-delta])),fmt='o',capsize=2,color=colors[A])
axs[0,1].axhline(.01,ls=':',color='red',label='point limit +.01');axs[0,1].axhline(.10,ls='--',color='gray',label='95% upper limit +.10');axs[0,1].axhline(0,lw=.7,color='gray')
axs[0,1].set(title='New minus BASE same-treated REAL',ylabel='paired BA difference');axs[0,1].set_xticks(range(4),shortcv);axs[0,1].legend(fontsize=7)
for n in ['REAL','TRAIN_REFERENCE','BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']:
    cdf(axs[1,0],n,features[n+'__PSD'][features[n+'__valid']])
    cdf(axs[1,1],n,features[n+'__baseline_RMS_V'][features[n+'__valid']],1000)
axs[1,0].set(title='Complete valid populations',xlabel='signed PSD',ylabel='CDF');axs[1,0].legend(fontsize=7)
axs[1,1].set(title='REAL may receive added stochastic response',xlabel='baseline RMS (mV)',ylabel='CDF')
for ax in axs.flat:ax.grid(alpha=.12)
fig.suptitle('REAL controls are processing comparisons, not a stochastic identity test');fig.tight_layout();save(fig,'same_treated_REAL_controls.png')

'''
plots=plots[:end]+control+plots[end:]
text=header+gates+plots
ast.parse(text)
(ROOT/'summarize_and_plot.py').write_text(text,'utf8')
receipt={'source':'summarize_and_plot.py','bytes':len(text.encode()),'sha256':hashlib.sha256(text.encode()).hexdigest(),'AST':True,
    'adapted_saved_result_plots_only':True,'new_training_fit_generation':0,'gate5_startup_addendum_applied':True}
(ROOT/'SUMMARY_SOURCE_BINDING.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt))
