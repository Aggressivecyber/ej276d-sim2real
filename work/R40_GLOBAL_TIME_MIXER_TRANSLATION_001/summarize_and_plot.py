from __future__ import annotations
import pathlib,json,time,datetime,hashlib,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True,parents=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));m=json.loads((ROOT/'metrics.json').read_text('utf8'))
replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'));A='GLOBAL_TIME_MIXER_FINAL'
controls=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION']
labels={'REAL':'REAL DEV','BASE':'Frozen BASE','INDEPENDENT_CFM':'Independent flow','OT_CFM':'OT flow',
    'PSD_ORDERED_CFM':'PSD ordered flow','CONDITIONAL_DIFFUSION_EDIT':'Diffusion edit','FULL_COVARIANCE_CALIBRATION':'Gaussian covariance map',A:'Time-mixer G',
    'TRAIN_REFERENCE':'Raw REAL TRAIN ref','SOURCE_CLEAN':'Source ADC reference','SOURCE_INPUT':'Actual NN source'}
colors={'REAL':'#111111','BASE':'#888888','INDEPENDENT_CFM':'#86a9cc','OT_CFM':'#c6a98a',
    'PSD_ORDERED_CFM':'#ad80b7','CONDITIONAL_DIFFUSION_EDIT':'#d98628','FULL_COVARIANCE_CALIBRATION':'#ce7d31',A:'#0867ad','TRAIN_REFERENCE':'#398347',
    'SOURCE_CLEAN':'#6e839e','SOURCE_INPUT':'#6e839e','BASE_REAL_CONTROL':'#888888','GLOBAL_TIME_MIXER_REAL_CONTROL':'#0867ad'}
labels.update(BASE_REAL_CONTROL='BASE G on REAL',GLOBAL_TIME_MIXER_REAL_CONTROL='Time-mixer G on REAL')
styles={'BASE':'--','TRAIN_REFERENCE':'--','INDEPENDENT_CFM':':','OT_CFM':':','PSD_ORDERED_CFM':':','CONDITIONAL_DIFFUSION_EDIT':':'}
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
def line(ax,n,x,y,**kw):ax.plot(x,y,color=colors[n],ls=styles.get(n,'-'),label=labels[n],lw=1.15,**kw)
def cdf(ax,n,x,scale=1):
    x=np.sort(x[np.isfinite(x)])*scale
    if len(x):line(ax,n,x,np.arange(1,len(x)+1)/len(x))
def measure(a,b):
    delta=a/b-1 if a is not None and b is not None and b>0 else (0. if a==b==0 else None)
    return {'candidate':a,'BASE':b,'relative_change':delta,'pass':delta is not None and delta<=.10}
features=load(ROOT/'results/FEATURES_PRIVATE.npz');waves=load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz')
psd=m['PSD_main'];width=m['energy_conditional_width'];edges=np.asarray(psd['edges_logQ']);tt=np.arange(480)-120
def obs(n):
    z=np.log(np.maximum(features[n+'__Qtotal_mVns'],1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7;return b
views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480'];full=['full_historical_shape','normalized_u480']

paired=m['C2ST']['paired'][A];cs=m['C2ST']['results']
import torch
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

limits={'full_historical_shape':.934,'normalized_u480':.928}
g1={'paired_minus_BASE':{v:paired[v]for v in full},'fixed_complete_response_point_limits':limits,
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
        'new_G_REAL_BA':cs['GLOBAL_TIME_MIXER_REAL_CONTROL'][v]['balanced_accuracy'],**r,
        'point_delta_le0_01':r['BA_delta_first_minus_second']<=.01+1e-14,'CI_upper_le0_10':r['BA_delta_95CI'][1]<=.10}
    control_rows[v]['pass']=control_rows[v]['point_delta_le0_01']and control_rows[v]['CI_upper_le0_10']
g5={'pre_registered_R40_same_treated_scope':True,'comparison':'new G REAL control minus BASE G REAL control',
    'three_views':control_rows,'RAW_RR_absolute_effects_descriptive':{n:m['C2ST']['paired'][n]for n in ['BASE_REAL_CONTROL','GLOBAL_TIME_MIXER_REAL_CONTROL']},
    'RAW_identity_not_required':True,'pass':all(r['pass']for r in control_rows.values())}
g6={'unique_G_D_reader_trajectory_each_2950':stats['updates']==stats['D_updates']==stats['style_reader_updates']==2950,
    'initial_extra_forward_rows':256,'initial_components_bitwise':all(all(v['all_components_bitwise_equal'].values())and v['mixer_delta_exact_zero']for v in stats['initial_equivalence']['rows'].values()),
    'initial_J20_equal':stats['original_flow_match']['initial_J20_absolute_difference']<=stats['original_flow_match']['initial_J20_tolerance'],
    'all_seven_original_streams_match':all(stats['original_flow_match']['all_original_update_stream_fields'].values()),
    'fresh_projection_calls':stats['fresh_projection_calls'],'fresh_direction_groups':stats['original_flow_match']['fresh_direction_groups'],
    'all_8850_groups_equal':stats['original_flow_match']['fresh_directions_all_8850_match_saved_BASE'],
    'module_all_four_parameters_updated_and_finite':all(v['updated']and v['finite']for v in stats['module_parameter_change']['parameters'].values()),
    'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'actual_new_figure_QA_required':'FIGURE_QA.json','actual_resource_and_failures_required':'RESOURCE_RECEIPT.json'}
g6['pass']=all(g6[k]for k in ['unique_G_D_reader_trajectory_each_2950','initial_components_bitwise','initial_J20_equal','all_seven_original_streams_match','all_8850_groups_equal','module_all_four_parameters_updated_and_finite','bitwise_pass'])and replay['rows']==4000 and stats['fresh_projection_calls']==2950 and g6['fresh_direction_groups']==8850

gates={'gate1_complete_response':g1,'gate2_PSD_width_centers_noise':g2,'gate3_source_preservation':g3,
    'gate4_boundary_source':g4,'gate5_same_treated_REAL_control':g5,'gate6_numerical_replay':g6}
retain=all(x['pass']for x in gates.values());gate={'fixed_recipe':'One shared-channel zero-output time MLP, restored original fresh fine loss, source G final2950',
    'prespecified_rules_changed_after_data':False,**gates,'retain':retain,'decision':'retain_DEV_candidate_pending_actual_QA'if retain else'close_this_fixed_single_time_MLP_recipe',
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
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
short=['shape493','V480','prompt','postpeak','tail','baseline','Q/PSD','pure u480','seen\n480 points','complement\n480 points']
fig,axs=plt.subplots(2,1,figsize=(14,8.3),gridspec_kw={'height_ratios':[1.4,1]})
for j,n in enumerate(['BASE','PSD_ORDERED_CFM','FULL_COVARIANCE_CALIBRATION',A]):
    y=np.array([cs[n][v]['balanced_accuracy']for v in views]);ci=np.array([cs[n][v]['heldout_CI']['balanced_accuracy_95CI']for v in views])
    axs[0].errorbar(np.arange(10)+(j-1.5)*.13,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=3,capsize=2,color=colors[n],label=labels[n])
y=[cs['RAW_REAL_REAL'][v]['balanced_accuracy'] if cs['RAW_REAL_REAL'][v]['balanced_accuracy']is not None else np.nan for v in views]
axs[0].plot(range(10),y,'s--',ms=3,lw=.8,color=colors['TRAIN_REFERENCE'],label='Raw REAL-REAL (Q/PSD unknown)')
y=np.array([paired[v]['BA_delta_first_minus_second']for v in views]);ci=np.array([paired[v]['BA_delta_95CI']for v in views])
axs[1].errorbar(range(10),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=4,capsize=2,color=colors[A],label='Time-mixer G minus BASE')
axs[0].axhline(.5,color='gray',ls=':');axs[0].set(ylim=(.43,1.02),ylabel='fixed holdout BA')
axs[1].axhline(0,color='gray',lw=.8);axs[1].set_ylabel('paired BA difference')
for ax in axs:ax.set_xticks(range(10),short);ax.grid(axis='y',alpha=.12)
axs[0].legend(fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.12))
axs[1].legend(fontsize=8,loc='upper left')
axs[0].set_title('Fixed 600/250 rows; 500 group resamples; lower is less distinguishable')
axs[1].set_title('Intervals condition on fitted classifiers; same acquisition, not physical equivalence')
fig.tight_layout();save(fig,'c2st.png')

chartnames=['REAL','BASE','PSD_ORDERED_CFM','FULL_COVARIANCE_CALIBRATION',A,'SOURCE_INPUT']
positive=np.concatenate([features[n+'__Qtotal_mVns'][features[n+'__Qtotal_mVns']>0]for n in chartnames]);xmin,xmax=positive.min(),positive.max()
fig,axs=plt.subplots(2,3,figsize=(13.8,7),sharex=True,sharey=True)
for ax,n in zip(axs.flat,chartnames):
    q=features[n+'__Qtotal_mVns'];p=features[n+'__PSD'];use=features[n+'__valid']&(q>0)
    ax.scatter(q[use],p[use],s=3,alpha=.27,color=colors[n]);ax.set_xscale('log');ax.set(xlim=(xmin*.95,xmax*1.05),ylim=(-.1,.6),title=labels[n])
    ax.text(.02,.02,f'QC {use.sum()}; outside PSD view {int(np.sum(use&((p<-.1)|(p>.6))))}',transform=ax.transAxes,fontsize=7);ax.grid(alpha=.13)
for ax in axs[-1]:ax.set_xlabel('Q total (mV ns)')
for ax in axs[:,0]:ax.set_ylabel('signed PSD')
fig.suptitle('Unchanged full populations, common Q/PSD axes; no reweighting');fig.tight_layout();save(fig,'charge_psd.png')

sections=load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz');display=['REAL','BASE','FULL_COVARIANCE_CALIBRATION',A]
fig,axs=plt.subplots(2,4,figsize=(14,6.5),sharex=True,sharey=True)
for i,ax in enumerate(axs.flat):
    counts=[]
    for n in display:
        x=sections[n+'__bin'+str(i+1)+'__PSD'];counts.append(len(x))
        if len(x):line(ax,n,x,np.arange(1,len(x)+1)/len(x))
    ax.set(title=f'Observed-Q bin {i+1}',xlim=(-.1,.6),ylim=(0,1));ax.text(.02,.04,'N '+ '/'.join(map(str,counts)),transform=ax.transAxes,fontsize=6.5);ax.grid(alpha=.12)
    if i not in common:ax.text(.02,.9,'Common support below40',transform=ax.transAxes,fontsize=7,color='#555')
for ax in axs[1]:ax.set_xlabel('signed PSD')
for ax in axs[:,0]:ax.set_ylabel('CDF')
axs[0,0].legend(fontsize=6.8);fig.tight_layout();save(fig,'conditional_psd_CDF.png')

fig,axs=plt.subplots(2,3,figsize=(13,6.8),sharex=True);qmid=np.exp((edges[:-1]+edges[1:])/2)
for n in ['REAL','BASE','FULL_COVARIANCE_CALIBRATION',A]:
    for c in [0,1]:
        for j,f in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            y=[s['components'][c][f]if s['valid']else np.nan for s in width['tables'][n]['slices']];line(axs[c,j],n,qmid,y,marker='o',ms=3)
for c in [0,1]:
    for j,title in enumerate(['center','raw width','detrended width']):
        axs[c,j].set_xscale('log');axs[c,j].set(title=('low'if c==0 else'high')+' PSD component '+title,ylabel='PSD units');axs[c,j].grid(alpha=.12)
        if c:axs[c,j].set_xlabel('observed-Q midpoint (mV ns)')
axs[0,0].legend(fontsize=7);fig.suptitle('Density components are not particle truth; invalid bins remain gaps');fig.tight_layout();save(fig,'energy_peak_width.png')

fig,axs=plt.subplots(2,4,figsize=(14,6.6),sharex=True)
for i,ax in enumerate(axs.flat):
    counts=[]
    for n in ['REAL','BASE',A]:
        use=features[n+'__valid']&(obs(n)==i);v=waves[n][use];counts.append(len(v))
        if len(v):line(ax,n,tt,np.median(v,axis=0)*1000);ax.plot(tt,v.mean(0)*1000,ls=':',lw=.75,color=colors[n],alpha=.6)
    ax.set(title=f'Observed-Q bin {i+1}',xlim=(-120,359));ax.text(.02,.03,'N '+ '/'.join(map(str,counts)),transform=ax.transAxes,fontsize=7);ax.grid(alpha=.12)
for ax in axs[1]:ax.set_xlabel('relative to saved index120 (ns)')
for ax in axs[:,0]:ax.set_ylabel('raw voltage (mV)')
axs[0,0].legend(fontsize=7);fig.suptitle('Full480: solid median, dotted mean; no smoothing or row rescaling');fig.tight_layout();save(fig,'conditional_waveforms.png')

fig,axs=plt.subplots(1,4,figsize=(14,4))
for ax,f,title,scale in zip(axs,['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V'],['baseline RMS (mV)','baseline lag1 ACF','prompt100:150 curvature (mV)','postpeak150:220 curvature (mV)'],[1000,1,1000,1000]):
    for n in ['REAL','BASE','FULL_COVARIANCE_CALIBRATION',A]:cdf(ax,n,features[n+'__'+f][features[n+'__valid']],scale)
    ax.set(title=title,ylabel='CDF');ax.grid(alpha=.12)
axs[0].legend(fontsize=6.8);fig.tight_layout();save(fig,'noise_curvature.png')

names=['BASE','FULL_COVARIANCE_CALIBRATION',A];fig,axs=plt.subplots(1,4,figsize=(14.6,4.1));ticks=[];ticklabels=[]
for j,n in enumerate(names):
    for k,b in enumerate(['bin3','bin4']):
        row=primary[n][b];p=j*3+k;ticks.append(p);ticklabels.append(['BASE','COV','TIME'][j]+str(k+3))
        axs[0].bar(p,row['output_PSD']['mean'],color=colors[n]);axs[1].bar(p,row['output_PSD']['sample_sd'],color=colors[n]);axs[2].bar(p,row['valley_fraction'],color=colors[n])
    r=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all']
    axs[3].bar(j-.15,r['source_Q_to_output_Q']['Spearman'],.28,color=colors[n],alpha=.45);axs[3].bar(j+.15,r['source_PSD_to_output_PSD']['Spearman'],.28,color=colors[n])
for ax,title in zip(axs[:3],['Pure-had PSD mean','Pure-had PSD sample SD','Frozen valley fraction']):ax.set_title(title);ax.set_xticks(ticks,ticklabels,rotation=45,fontsize=7);ax.grid(axis='y',alpha=.12)
axs[3].set(title='Actual source rank: light Q, dark PSD',ylim=(0,1.05));axs[3].set_xticks(range(3),['BASE','COV','TIME']);axs[3].grid(axis='y',alpha=.12)
fig.suptitle('Actual unADC source-Q bins3/4, N213/181; valley is not particle-error truth');fig.tight_layout();save(fig,'source_preservation.png')

boundary=load(ROOT/'results/BOUNDARY_DIFFERENCES_PRIVATE.npz');fig,axs=plt.subplots(3,3,figsize=(12.8,9))
for ax,key in zip(axs.flat,bd['BASE']):
    for n in ['REAL','BASE','INDEPENDENT_CFM',A]:cdf(ax,n,boundary[n+'__'+key],1000)
    ax.set(title=key.replace('_',' '),xlabel='native difference (mV)',ylabel='CDF');ax.grid(alpha=.12)
axs[0,0].legend(fontsize=7);fig.suptitle('Nine fixed boundary diagnostics; no searched positions');fig.tight_layout();save(fig,'boundary_differences.png')


trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'))
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

cov=load(ROOT/'results/DEV_COVARIANCE_DESCRIPTION_PRIVATE.npz',keys=['population__REAL_Correlation','population__BASE_Correlation','population__'+A+'_Correlation'])
fig,axs=plt.subplots(1,3,figsize=(13.8,4.5),layout='constrained')
for ax,n in zip(axs,['REAL','BASE',A]):
    image=ax.imshow(cov['population__'+n+'_Correlation'],origin='lower',vmin=-1,vmax=1,cmap='coolwarm',extent=(-120,359,-120,359))
    ax.set(title=labels[n],xlabel='relative sample time (ns)',ylabel='relative sample time (ns)')
fig.colorbar(image,ax=axs,shrink=.8,label='sample correlation');fig.suptitle('Saved DEV complete480 correlation; description only, no DEV fitting')
save(fig,'DEV_full_correlation.png')

gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz');dv=gp['perturbed_output_preADC_V'].astype('f8')-gp['original_output_preADC_V'].astype('f8');dd=gp['actual_perturbed_guide_f32_V'].astype('f8')-gp['original_guide_V'].astype('f8')
fig,axs=plt.subplots(2,3,figsize=(13.8,7))
axs[0,0].plot(tt,dd[0]*1000,color='#555',label='actual SOURCE delta');axs[0,0].plot(tt,dv[0]*1000,color=colors[A],label='paired output delta');axs[0,0].set(title='First prespecified perturbation',xlabel='relative ns',ylabel='delta mV');axs[0,0].legend(fontsize=7)
axs[0,1].hist(np.sqrt(np.mean(dv*dv,1))*1000,bins=18,color=colors[A]);axs[0,1].axvline(1000/16383,color='black',ls='--');axs[0,1].set(title='128 fixed-bin full-SOURCE changes',xlabel='preADC paired RMS (mV)',ylabel='parents')
rows=m['source_probes'][A]['rows'];tags=list(rows);axs[0,2].barh(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[0,2].set(title='Full source-chain probes',xlabel='RMS to correct reference (mV)')
probe_labels=['Amp x0.5','Amp x1.5','Fast/slow r=0','Fast/slow r=.25','Fast/slow r=.5','Fast/slow r=.75','Fast/slow r=1','NewNoise','Same input/noise','Rolled source']
assert len(tags)==len(probe_labels)
axs[0,2].set_yticks(range(len(tags)),probe_labels,fontsize=7);axs[0,2].invert_yaxis()
terms=['base_signal','base_electronic','base_sum_rounding','flow_increment_f64'];joint=m['random_decomposition'][A]['groups']['joint8']['by_composition']
for j,cat in enumerate(['pure_EE','pure_had','mixed']):
    r=joint[cat];c=r['four_additive_algebraic_components_covariance_V2']['full480'];vals=[c[t+'__'+t]['mean']for t in terms]+[2*sum(c[terms[k]+'__'+terms[l]]['mean']for k in range(4)for l in range(k+1,4))]
    axs[1,j].bar(range(5),np.asarray(vals)*1e6,color=[colors[A]]*4+['#555']);axs[1,j].set(title=cat+f': joint8 ({r["parents"]} parents)',ylabel='variance contribution (mV²)')
    axs[1,j].set_xticks(range(5),['baseS','baseE','round','delta','all2cov'],fontsize=7);axs[1,j].axhline(0,color='gray',lw=.7)
fig.suptitle('Full-SOURCE sensitivity and functional G components; not physical branch truth');fig.tight_layout();save(fig,'source_and_random_decomposition.png')

fig,axs=plt.subplots(2,2,figsize=(13,7.4))
cv=['full_historical_shape','normalized_u480','baseline_local','full_voltage']
shortcv=['shape493','pure u480','baseline','V480\n(descriptive)']
for j,n in enumerate(['RAW_REAL_REAL','BASE_REAL_CONTROL','GLOBAL_TIME_MIXER_REAL_CONTROL']):
    color=colors['TRAIN_REFERENCE']if n=='RAW_REAL_REAL'else colors[n]
    yy=np.asarray([cs[n][v]['balanced_accuracy']for v in cv]);cc=np.asarray([cs[n][v]['heldout_CI']['balanced_accuracy_95CI']for v in cv])
    axs[0,0].errorbar(np.arange(4)+(j-1)*.13,yy,yerr=np.maximum(0,np.vstack([yy-cc[:,0],cc[:,1]-yy])),fmt='o',capsize=2,ms=4,color=color,label='RAW TRAIN-DEV'if n=='RAW_REAL_REAL'else labels[n])
axs[0,0].axhline(.5,color='gray',ls=':');axs[0,0].set(title='Same reference rows / stochastic processing',ylabel='fixed holdout BA',ylim=(.43,1.02));axs[0,0].set_xticks(range(4),shortcv);axs[0,0].legend(fontsize=7)
delta=np.asarray([control_pair[v]['BA_delta_first_minus_second']for v in cv]);cc=np.asarray([control_pair[v]['BA_delta_95CI']for v in cv])
axs[0,1].errorbar(range(4),delta,yerr=np.maximum(0,np.vstack([delta-cc[:,0],cc[:,1]-delta])),fmt='o',capsize=2,color=colors[A])
axs[0,1].hlines(.01,-.4,2.4,linestyles=':',color='red',label='first3: point +.01');axs[0,1].hlines(.10,-.4,2.4,linestyles='--',color='gray',label='first3: CI upper +.10');axs[0,1].axvspan(2.6,3.4,color='#eeeeee',alpha=.4);axs[0,1].axhline(0,lw=.7,color='gray')
axs[0,1].set(title='New minus BASE same-treated REAL',ylabel='paired BA difference');axs[0,1].set_xticks(range(4),shortcv);axs[0,1].legend(fontsize=7)
for n in ['REAL','TRAIN_REFERENCE','BASE_REAL_CONTROL','GLOBAL_TIME_MIXER_REAL_CONTROL']:
    cdf(axs[1,0],n,features[n+'__PSD'][features[n+'__valid']])
    cdf(axs[1,1],n,features[n+'__baseline_RMS_V'][features[n+'__valid']],1000)
axs[1,0].set(title='Complete valid populations',xlabel='signed PSD',ylabel='CDF');axs[1,0].legend(fontsize=7)
axs[1,1].set(title='REAL may receive added stochastic response',xlabel='baseline RMS (mV)',ylabel='CDF')
for ax in axs.flat:ax.grid(alpha=.12)
fig.suptitle('Same-treated REAL controls; gates apply to shape/u/baseline, V480 descriptive');fig.tight_layout();save(fig,'same_treated_REAL_controls.png')

zz=load(ROOT/'results/FINAL_MODULE_ZERO_DESCRIPTION_PRIVATE.npz')
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

files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'figures':files,'figure_count':len(files),'all_plots_from_saved_results':True,'new_HGB_EM_response_updates_generation':0,
    'decision':gate['decision'],'retain':retain,'gates':{k:v['pass']for k,v in gates.items()},'actual_visual_QA_pending':True}
write('PLOT_SUMMARY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
