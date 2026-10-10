from __future__ import annotations
import os,json,time,pathlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
R=pathlib.Path(__file__).resolve().parent;F=R/'results/figures';F.mkdir(parents=True,exist_ok=True);START=time.monotonic()
C=json.loads((R/'config.json').read_text('utf8'));M=json.loads((R/'metrics.json').read_text('utf8'))
S,L=C['arms'];A=[S,L];RC=['SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']
num=json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text('utf8'));replay=json.loads((R/'REPLAY_RECEIPT.json').read_text('utf8'))
def write(n,x):(R/n).write_bytes((json.dumps(ev.safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def load(p,keys=None):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
feat=load(R/'results/FEATURES_PRIVATE.npz');waves=load(R/'results/PLOT_WAVEFORMS_PRIVATE.npz')
labels={'REAL':'REAL DEV','BASE':'Frozen BASE',S:'SHORT 2950 / 64 steps',L:'LONG 50000 / 64 steps',
 'BASE_REAL_CONTROL':'BASE on REAL',RC[0]:'SHORT on REAL',RC[1]:'LONG on REAL','SOURCE_INPUT':'Full source input'}
colors={'REAL':'#222222','BASE':'#909090',S:'#2877b5',L:'#d65f23','BASE_REAL_CONTROL':'#909090',RC[0]:'#2877b5',RC[1]:'#d65f23','SOURCE_INPUT':'#55806c'}
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
cs=M['C2ST']['results'];psd=M['PSD_main'];width=M['energy_conditional_width'];edges=np.asarray(psd['edges_logQ']);tt=np.arange(480)-120
def finite(x):return np.asarray(x)[np.isfinite(x)]
def cdf(ax,n,x,scale=1):
 x=np.sort(finite(x))*scale
 if len(x):ax.plot(x,np.arange(1,len(x)+1)/len(x),color=colors[n],label=labels[n],lw=1.4)
def save(fig,name):
 fig.savefig(F/name,bbox_inches='tight');plt.close(fig)
def measure(x,b):
 d=x/b-1 if x is not None and b is not None and b>0 else (0. if x==b==0 else None)
 return {'candidate':x,'BASE':b,'relative_change':d,'pass':d is not None and d<=.10}
charge={}
for n in C['endpoints']:
 d=load(R/n/'generated.npz',keys=['preADC_charge_relative_error_f64','preADC_charge_relative_error_f32','source_q_Vns','waveforms_V','neural_flow_applied_all_rows','inside_TRAIN_support'])
 w=d['waveforms_V'].astype('f8');pq=w[:,100:400].sum(1)-3*w[:,:100].sum(1)
 charge[n]={'rows':len(w),'f64_max_relative_error':float(d['preADC_charge_relative_error_f64'].max()),
  'actual_f32_preADC_max_relative_error':float(d['preADC_charge_relative_error_f32'].max()),
  'postADC_relative_charge_error':ev.summary((pq-d['source_q_Vns'])/d['source_q_Vns']),
  'postADC_not_claimed_exact':True,'all_rows_neural_flow':bool(d['neural_flow_applied_all_rows'].all()),
  'inside_TRAIN_support_rows':int(d['inside_TRAIN_support'].sum())}
write('CHARGE_CONSERVATION_SUMMARY.json',charge)
gaps={n:[{'bin_id_1based':s['bin']+1,'N':s['N'],'failure_reasons':s['failure_reasons']}for s in t['slices']if not s['valid']]for n,t in width['tables'].items()}
gates={};primary=M['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4'];bd=M['boundary_differences']['results']
for arm,ctrl in zip(A,RC):
 paired=M['C2ST']['paired'][arm];full=['full_historical_shape','normalized_u480'];limits={'full_historical_shape':.934,'normalized_u480':.928}
 g1={'paired_minus_BASE':{v:paired[v]for v in full},'fixed_point_limits':limits,
  'candidate_BA':{v:cs[arm][v]['balanced_accuracy']for v in full},
  'both_BASE_CI_upper_negative':all(paired[v]['BA_delta_95CI'][1]<0 for v in full),
  'both_fixed_R37_point_limits':all(cs[arm][v]['balanced_accuracy']<=limits[v]+1e-14 for v in full)}
 g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_fixed_R37_point_limits']
 common=list(range(7));missingQ={n:[i+1 for i in common if psd['tables'][n]['bins'][i]['N']<40]for n in ['REAL','BASE',arm]}
 pw={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in common]))if not missingQ['REAL']and not missingQ[n]else None for n in ['BASE',arm]}
 meas={k+'_overall_W1':measure(M['domain_vs_REAL'][arm]['overall'][k]['W1'],M['domain_vs_REAL']['BASE']['overall'][k]['W1'])for k in ['PSD','baseline_RMS_V','ACF1']}
 meas['PSD_common_Q_equal_mean_W1']={**measure(pw[arm],pw['BASE']),'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40,'fixed_required_bins':True,'unsupported_required_bins_by_domain':missingQ}
 for row in width['common_support_errors'][arm+'_vs_BASE']:
  if row['field']in ['mean','sigma_PSD']:
   key='peak_position_MAE'if row['field']=='mean'else'raw_width_MAE'
   meas['component'+str(row['component'])+'_'+key]={**measure(row['MAE'][arm],row['MAE']['BASE']),
    'common_bins_1based':row['common_bin_ids_1based'],'density_failure_not_hidden_by_intersection':True}
 required=list(range(1,7));assert all(width['tables']['BASE']['slices'][i]['valid']for i in required)
 newfail=[i+1 for i in required if not width['tables'][arm]['slices'][i]['valid']]
 zeros={n:int(np.sum(feat[n+'__baseline_RMS_V']==0))for n in ['BASE',arm]}
 g2={'measures':meas,'BASE_valid_required_bins_1based':[i+1 for i in required],'new_required_density_failures':newfail,
  'all_density_support_gaps':gaps,'zero_baseline_counts':zeros,'zero_baseline_not_increased':zeros[arm]<=zeros['BASE']}
 g2['pass']=len(meas)==8 and all(v['pass']for v in meas.values())and not newfail and zeros[arm]<=zeros['BASE']
 valleys=[];ranks=[]
 for b in ['bin3','bin4']:
  x=primary[arm][b]['valley_fraction'];base=primary['BASE'][b]['valley_fraction'];d=x-base if x is not None and base is not None else None
  valleys.append({'actual_source_bin':b,'N':primary[arm][b]['source_count'],'candidate':x,'BASE':base,'delta':d,'pass':d is not None and d<=.02})
 for f in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
  x=M['source_association_actual_preADC_input'][arm]['composition_and_PDG']['all'][f]['Spearman'];b=M['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][f]['Spearman'];d=b-x if x is not None and b is not None else None
  ranks.append({'measure':f,'candidate':x,'BASE':b,'BASE_minus_candidate':d,'pass':d is not None and d<=.05})
 gen=M['generation'][arm];sens=M['source_probes'][arm]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
 g3={'actual_source_pure_had_valleys':valleys,'whole_population_source_ranks':ranks,
  'source_ID_order_composition_weights_preserved':gen['all_source_ID_order_weights_and_composition_preserved']and M['data_roles']['DEV_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518}and not M['primary_DEV_population_reweighted'],
  'no_new_invalid_nonfinite_ADC_clip':M['domains'][arm]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,
  'outside_TRAIN_support_rows':gen['TRAIN_support_outside_N'],'all_outside_rows_use_new_flow':gen['BASE_fallback_N']==0 and gen['standalone_flow_applied_N']==2000,
  'source_roll_RMS_V':sens,'sensitivity_not_correct_physical_mapping':True}
 g3['pass']=all(x['pass']for x in valleys+ranks)and all(g3[k]for k in ['source_ID_order_composition_weights_preserved','no_new_invalid_nonfinite_ADC_clip','all_outside_rows_use_new_flow'])and sens is not None and sens>0
 bounds=[]
 for key in bd[arm]:
  x=bd[arm][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd['INDEPENDENT_CFM'][key]['W1_V'];lim=max(b,o)+1/16383 if b is not None and o is not None else None
  bounds.append({'location':key,'candidate_W1_V':x,'BASE_W1_V':b,'INDEPENDENT_W1_V':o,'limit_V':lim,'pass':x is not None and lim is not None and x<=lim})
 g4={'boundaries':bounds,'full_SOURCE_perturbation':M['guide_probe'][arm],'pass':len(bounds)==9 and all(x['pass']for x in bounds)and M['guide_probe'][arm]['pass']}
 cp=M['C2ST']['same_treated_REAL_control_paired'][ctrl];control_rows={}
 for v in ['full_historical_shape','normalized_u480','baseline_local']:
  r=cp[v];control_rows[v]={'BASE_G_REAL_BA':cs['BASE_REAL_CONTROL'][v]['balanced_accuracy'],'new_G_REAL_BA':cs[ctrl][v]['balanced_accuracy'],**r,
   'point_delta_le0_01':r['BA_delta_first_minus_second']<=.01+1e-14,'CI_upper_le0_10':r['BA_delta_95CI'][1]<=.10}
  control_rows[v]['pass']=control_rows[v]['point_delta_le0_01']and control_rows[v]['CI_upper_le0_10']
 g5={'pre_registered_same_treated_scope':True,'comparison':'new SOURCE flow on REAL minus frozen BASE G on REAL',
  'three_views':control_rows,'RAW_RR_absolute_effects_descriptive':{n:M['C2ST']['paired'][n]for n in ['BASE_REAL_CONTROL',ctrl]},
  'RAW_identity_not_required':True,'pass':all(r['pass']for r in control_rows.values())}
 g6={'frozen_model_loaded':True,'new_response_or_optimizer_updates':0,'no_new_TRAIN_constants_or_random_states':True,
  'TRAIN_fixed64_vs128_pass':num['models'][arm]['pass'],'both_models_pass_before_DEV':num['both_pass'],'TRAIN_check_DEV_read':num['DEV_read_or_generated'],
  'all_preADC_f32_charge_error_le1e_5':all(charge[n]['actual_f32_preADC_max_relative_error']<=1e-5 for n in [arm,ctrl]),
  'all_main_rows_processed':all(charge[n]['all_rows_neural_flow']for n in [arm,ctrl]),'charge_error_summaries':{n:charge[n]for n in [arm,ctrl]},
  'independent_CPU_full_chain_replay_rows':sum(replay['endpoints'][n]['rows']for n in [arm,ctrl]),
  'bitwise_pass':all(replay['endpoints'][n]['bitwise_pass']for n in [arm,ctrl]),'CPU_threads':12,'chunk':64,
  'functional_components_not_independent_physical_noise':True,'solver128_reference_not_exact_truth':True}
 g6['pass']=num['models'][arm]['pass']and num['both_pass']and num['DEV_read_or_generated']==0 and g6['all_preADC_f32_charge_error_le1e_5']and g6['all_main_rows_processed']and g6['bitwise_pass']and g6['independent_CPU_full_chain_replay_rows']==4000
 gates[arm]={'gate1_complete_response':g1,'gate2_PSD_width_centers_noise':g2,'gate3_source_preservation':g3,
  'gate4_boundary_source':g4,'gate5_same_treated_REAL_control':g5,'gate6_numerical_replay':g6,
  'retain':all(x['pass']for x in [g1,g2,g3,g4,g5,g6]),'role':'primary'if arm==L else'fixed control; cannot be chosen as fallback'}
gate={'recipe':'frozen SHORT2950/LONG50000 scalar-time flow at common fixed64 Heun steps','models':gates,
 'LONG_primary_retain':gates[L]['retain'],'decision':'retain_LONG_DEV_candidate_pending_final_acceptance'if gates[L]['retain']else'close_fixed_LONG50000_HEUN64_recipe',
 'SHORT_not_fallback':True,'prespecified_rules_changed_after_data':False,'LONG_SHORT_comparison':M['long_short_common64_comparison']}
write('GATE_METRICS.json',gate)
def compact_scores(x):
 keep=['balanced_accuracy','AUC','train_balanced_accuracy','train_AUC','BA_95CI','AUC_95CI','train_N','holdout_N','feature_dimension']
 return {k:x[k]for k in keep if k in x}
compact={'status':M['status'],'gate':gate,'counters':M['counters'],'training_history':M['training'],
 'C2ST':{'results':{n:{v:compact_scores(x)for v,x in rows.items()}for n,rows in cs.items()},
  'paired':M['C2ST']['paired'],'same_treated_REAL_control_paired':M['C2ST']['same_treated_REAL_control_paired']},
 'long_short_common64_comparison':M['long_short_common64_comparison'],
 'overall_domain_vs_REAL':{n:v['overall']for n,v in M['domain_vs_REAL'].items()},'PSD_main_vs_REAL':psd['vs_REAL'],
 'width_common_support_errors':width['common_support_errors'],'density_support_gaps':gaps,'source_primary_actual_Q':primary,
 'source_historical_ADC_reference_Q':M['source_composition']['historical_postADC_reference_Q_bins3_4'],
 'source_ranks_actual_Q':{n:r['composition_and_PDG']['all']for n,r in M['source_association_actual_preADC_input'].items()},
 'source_guide_summary':M['guide_probe'],'boundary_differences':M['boundary_differences'],'covariance_description':M['covariance_description'],
 'projection_common_support':M['projection_distances']['common_support_summary'],'replay':{'rows':replay['rows'],'pass':replay['pass']},
 'privacy':'aggregate only; no event identities, split rows, predictions, random arrays or waveform payloads',
 'EVAL_read':0,'CAL_waveform_decode':0,'new_training_updates':0}
write('METRICS_COMPACT.json',compact)

# 1. Numerical evidence, the128-step result is only a reference.
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for n in A:
 x=load(R/n/'TRAIN_SOLVER_CHECK_PRIVATE.npz',keys=['RMS_difference_V'])['RMS_difference_V']*16383
 cdf(axs[0],n,x);row=num['models'][n]
 axs[1].scatter([0 if n==S else 1],[row['preADC_relative_q_max']],color=colors[n],s=45)
axs[0].axvline(.25,color='gray',ls=':');axs[0].axvline(1,color='gray',ls='--');axs[0].set(xlabel='64 vs128 RMS difference (LSB)',ylabel='TRAIN128 CDF',title='Fixed candidate/reference check');axs[0].legend()
axs[1].set_xticks([0,1],['SHORT','LONG']);axs[1].axhline(1e-5,color='gray',ls='--');axs[1].set(yscale='log',ylabel='Maximum relative preADC charge error',title='Both64 and128; fixed tolerance1e-5')
fig.tight_layout();save(fig,'numeric_integration_budget.png')

# 2. Full and local classifiers, all fixed pools and fitted-classifier CIs.
views=C['C2ST_views']+['normalized_u480','seen_reconstruction480','complement_reconstruction480']
vlabels=['shape493','V480','prompt','postpeak','tail','baseline','Q-PSD2D','pure u480','seen\n480 points','complement\n480 points']
fig,ax=plt.subplots(figsize=(12,4.7));xx=np.arange(len(views))
for shift,n in zip([-.24,0,.24],['BASE',S,L]):ax.plot(xx+shift,[cs[n][v]['balanced_accuracy']for v in views],'o-',color=colors[n],label=labels[n],ms=5)
rr=[cs['RAW_REAL_REAL'][v].get('balanced_accuracy')for v in views]
ax.scatter([i for i,x in enumerate(rr)if x is not None],[x for x in rr if x is not None],color='black',marker='x',label='Saved RAW REAL-REAL')
ax.set_xticks(xx,vlabels);ax.set(ylabel='Holdout balanced accuracy',ylim=(.4,1.025),title='Fixed600/250 pools; lower is less separable, not model selection');ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=4);fig.subplots_adjust(top=.77,bottom=.18);save(fig,'c2st.png')

# 3. LONG-SHORT paired evidence and all8 prespecified side effects.
cmp=M['long_short_common64_comparison'];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for i,v in enumerate(['full_historical_shape','normalized_u480']):
 row=cmp['paired_BA_long_minus_short'][v];d=row['BA_delta_first_minus_second'];lo,hi=row['BA_delta_95CI'];axs[0].errorbar(i,d,yerr=[[d-lo],[hi-d]],fmt='o',color=colors[L],capsize=5)
axs[0].axhline(0,color='black',ls='--');axs[0].set_xticks([0,1],['shape493','pure u480']);axs[0].set(ylabel='LONG minus SHORT BA',title='Paired500 group bootstrap; fitted models fixed')
rows=cmp['eight_prespecified_distance_differences'];names=list(rows);vv=[rows[k]['relative_change_long_minus_short_over_short']for k in names]
axs[1].barh(np.arange(8),[100*x if x is not None else 0 for x in vv],color=[colors[L]if x is not None else'#cccccc'for x in vv]);axs[1].axvline(10,color='red',ls='--',lw=1);axs[1].axvline(0,color='black',lw=.7)
axs[1].set_yticks(np.arange(8),['PSD W1','RMS W1','ACF1 W1','Q-cond PSD W1','Low center MAE','Low width MAE','High center MAE','High width MAE'])
for i,x in enumerate(vv):
 if x is None:axs[1].text(0,i,' n/a',va='center')
axs[1].set(xlabel='Relative distance change (%)',title='Same qualified bins; n/a retained');fig.tight_layout();save(fig,'long_short_budget_comparison.png')

# 4. Common axes preserve the whole source charge population.
domains=['REAL','BASE',S,L];qall=np.concatenate([feat[n+'__Qtotal_mVns'][feat[n+'__valid']]for n in domains]);pall=np.concatenate([feat[n+'__PSD'][feat[n+'__valid']]for n in domains]);xr=[float(np.log10(qall[qall>0]).min()),float(np.log10(qall).max())]
for full_range,name in [(False,'charge_psd_main_body.png'),(True,'charge_psd_full_range.png')]:
 fig,axs=plt.subplots(1,4,figsize=(12,3.8),sharex=True,sharey=True);yr=[float(pall.min()),float(pall.max())]if full_range else[0,.6]
 for ax,n in zip(axs,domains):
  ok=feat[n+'__valid'];q=feat[n+'__Qtotal_mVns'][ok];p=feat[n+'__PSD'][ok];ax.hexbin(np.log10(q),p,gridsize=45,extent=(*xr,*yr),mincnt=1,cmap='viridis',bins='log');out=int(np.sum((p<yr[0])|(p>yr[1])))
  ax.set(title=('REAL'if n=='REAL'else'BASE'if n=='BASE'else'SHORT'if n==S else'LONG')+f'\nN={len(p)}, outside={out}',xlabel='log10 Qtotal (mV ns)',xlim=xr,ylim=yr)
 axs[0].set_ylabel('Signed PSD =1-Qfast/Qtotal');fig.suptitle('Identical axes; all2000 parents retained; zoom does not delete outliers');fig.tight_layout();save(fig,name)

# 5. Same observed-Q CDFs with support counts.
sections=load(R/'results/PSD_SECTIONS_PRIVATE.npz');fig,axs=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
for i,ax in enumerate(axs.flat):
 counts=[]
 for n in domains:
  x=sections[n+'__bin'+str(i+1)+'__PSD'];cdf(ax,n,x);counts.append(len(x))
 ax.set(title=f'bin{i+1}\nR/B/S/L: '+ '/'.join(map(str,counts)),xlim=(0,.6),ylim=(0,1));ax.grid(alpha=.12)
 if i>=4:ax.set_xlabel('Signed PSD')
 if i%4==0:ax.set_ylabel('CDF')
axs[0,0].legend(fontsize=7);fig.suptitle('Fixed8 observed-Q intervals; outside display kept in full-range plot');fig.tight_layout();save(fig,'conditional_psd_CDF.png')

# 6. Component support gaps are shown rather than bridged.
fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True);cx=np.exp((edges[:-1]+edges[1:])/2)
for k in [0,1]:
 for j,f in enumerate(['mean','sigma_PSD']):
  ax=axs[k,j]
  for n in domains:
   vals=[s['components'][k][f]if s['valid']and len(s['components'])==2 else np.nan for s in width['tables'][n]['slices']]
   ax.plot(cx,vals,'o-',color=colors[n],label=labels[n],lw=1.25,ms=4)
  ax.set(xscale='log',ylabel='PSD center'if j==0 else'Raw PSD SD',title=('Low'if k==0 else'High')+' density component; missing=undefined');ax.grid(alpha=.15)
  if k:ax.set_xlabel('FixedQ-bin geometric center (mV ns)')
axs[0,0].legend(fontsize=7);fig.suptitle('Deterministic two-density descriptions, not particle truth; no failed fit restart');fig.tight_layout();save(fig,'energy_peak_width.png')

# 7. Fixed observable definitions, no normalization to hide voltage effects.
fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,f,title,scale in zip(axs.flat,['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V'],['Baseline RMS (mV)','Baseline lag1 ACF','Prompt100:150 curvature (mV)','Postpeak150:220 curvature (mV)'],[1000,1,1000,1000]):
 for n in domains:cdf(ax,n,feat[n+'__'+f][feat[n+'__valid']],scale)
 ax.set(title=title,ylabel='CDF');ax.grid(alpha=.12)
axs[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'noise_curvature.png')

# 8. Source-Q conditioning is actual unquantized input, not output-Q or particle labels.
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,b in zip(axs[:2],['bin3','bin4']):
 for j,n in enumerate(['BASE',S,L]):ax.bar(j,100*primary[n][b]['valley_fraction'],color=colors[n])
 ax.set_xticks([0,1,2],['BASE','SHORT','LONG']);ax.set(title='pure-had actual source '+b+f"\nN={primary[S][b]['source_count']}",ylabel='Output frozen valley occupancy (%)')
for j,n in enumerate(['BASE',S,L]):
 r=M['source_association_actual_preADC_input'][n]['composition_and_PDG']['all'];axs[2].bar(j-.15,r['source_Q_to_output_Q']['Spearman'],.28,color=colors[n],alpha=.45);axs[2].bar(j+.15,r['source_PSD_to_output_PSD']['Spearman'],.28,color=colors[n])
axs[2].set_xticks([0,1,2],['BASE','SHORT','LONG']);axs[2].set(title='Source rank: lightQ / darkPSD',ylim=(0,1.05));fig.suptitle('Actual source bins3/4 N213/181; historical ADC source207/182 is separately retained');fig.tight_layout();save(fig,'source_preservation.png')

# 9. Same-treated controls have gate lines only in the three prespecified views.
vv=['full_historical_shape','normalized_u480','baseline_local','full_voltage'];fig,axs=plt.subplots(1,2,figsize=(11,4))
for n in ['BASE_REAL_CONTROL',*RC]:axs[0].plot(np.arange(4),[cs[n][v]['balanced_accuracy']for v in vv],'o-',color=colors[n],label=labels[n])
axs[0].set_xticks(range(4),['shape493','pure u480','baseline','V480\ndescriptive']);axs[0].set(ylabel='BA vsRAW DEV',ylim=(.45,1.02));axs[0].legend(fontsize=7)
for off,n in zip([-.08,.08],RC):
 for i,v in enumerate(vv):
  row=M['C2ST']['same_treated_REAL_control_paired'][n][v];d=row['BA_delta_first_minus_second'];lo,hi=row['BA_delta_95CI'];axs[1].errorbar(i+off,d,yerr=[[d-lo],[hi-d]],fmt='o',color=colors[n],capsize=3)
axs[1].hlines([.01,.10],-.3,2.3,colors=['gray','red'],linestyles=[':','--']);axs[1].set_xticks(range(4),['shape493','pure u480','baseline','V480\ndescriptive']);axs[1].set(ylabel='New minus BASE-REAL BA',title='Point<=.01 / CI upper<=.10 only first3');axs[1].axhline(0,color='black',lw=.6)
fig.suptitle('Stochastic processing on already noisy REAL is not assumed to be identity');fig.tight_layout();save(fig,'same_treated_REAL_controls.png')

# 10. Nine fixed boundary statistics, first and second differences.
fig,ax=plt.subplots(figsize=(11,4));keys=list(bd[S]);xx=np.arange(len(keys))
for n in ['BASE',S,L]:ax.plot(xx,[1000*bd[n][k]['W1_V']for k in keys],'o-',color=colors[n],label=labels[n],ms=4)
ax.plot(xx,[1000*max(bd['BASE'][k]['W1_V'],bd['INDEPENDENT_CFM'][k]['W1_V'])+1000/16383 for k in keys],'k--',lw=1,label='Fixed limit: max(BASE,Independent)+1LSB')
ax.set_xticks(xx,[k.replace('center_','c').replace('first_','d1 ').replace('second_','d2 ')for k in keys],rotation=25,ha='right');ax.set(ylabel='W1 of voltage difference (mV)',title='All9 predetermined boundary tests');ax.legend(fontsize=7);fig.tight_layout();save(fig,'boundary_differences.png')

# 11. Source-only selected examples, REAL is an unpaired illustration.
actual=load(R/'ACTUAL_GENERATION_INPUTS.npz',keys=['source_waveforms_V','source_ids']);meta=load(R/'DEV_SOURCE_METADATA.npz',keys=['composition'])
source=actual['source_waveforms_V'];q=source.astype('f8')[:,100:400].sum(1)-3*source.astype('f8')[:,:100].sum(1);rows=[]
fig,axs=plt.subplots(3,4,figsize=(13,7),sharex=True)
for r,cat in enumerate(['pure_EE','pure_had','mixed']):
 ix=np.flatnonzero(meta['composition']==cat);i=int(ix[np.argmin(np.abs(q[ix]-np.median(q[ix])))]);ri=int(np.argmin(abs(feat['REAL__Qtotal_mVns']-q[i]*1000)))
 vals=[source[i],waves[S][i],waves[L][i],waves['REAL'][ri]];lo=min(float(v.min())for v in vals)*1000;hi=max(float(v.max())for v in vals)*1000;pad=(hi-lo)*.08
 for j,(n,v)in enumerate(zip(['SOURCE_INPUT',S,L,'REAL'],vals)):
  axs[r,j].plot(tt,v*1000,color=colors[n],lw=1);axs[r,j].set_ylim(lo-pad,hi+pad);axs[r,j].grid(alpha=.12)
  if r==0:axs[r,j].set_title(['Full source input','SHORT64 output','LONG64 output','Unpaired REAL example'][j])
  if j==0:axs[r,j].set_ylabel(cat+'\nmV')
  if r==2:axs[r,j].set_xlabel('Time relative to saved index120 (ns)')
 rows.append({'composition':cat,'source_parent_index':i,'source_id':str(actual['source_ids'][i]),'source_Q_mVns':float(q[i]*1000),'unpaired_REAL_example_index':ri})
write('EXAMPLE_SELECTION_PRIVATE.json',{'rule':'closest source charge to composition median; first original index breaks ties; no output-based selection','rows':rows})
fig.suptitle('Complete480 points, same row voltage axes, no smoothing or independent normalization');fig.tight_layout();save(fig,'source_output_REAL_examples.png')

# 12. Functional variability and source perturbation, not independent physical noise.
fig,axs=plt.subplots(1,3,figsize=(13,4));cats=['pure_EE','pure_had','mixed'];joint=M['random_decomposition']
for off,n in zip([-.17,.17],A):
 by=joint[n]['groups']['joint8']['by_composition']
 axs[0].bar(np.arange(3)+off,[by[c]['within_parent_PSD_SD']['mean']for c in cats],.32,color=colors[n],label=labels[n])
 for j,c in enumerate(cats):
  cov=by[c]['four_additive_algebraic_components_covariance_V2']['full480'];e=cov['base_electronic__base_electronic']['mean'];f=cov['flow_increment_f64__flow_increment_f64']['mean'];cross=2*cov['base_electronic__flow_increment_f64']['mean']
  axs[1].plot([j-.2,j,j+.2],[e*1e6,f*1e6,cross*1e6],marker='o',color=colors[n],alpha=.75,label=labels[n]if j==0 else None)
 x=load(R/'results'/(n+'_GUIDE_PERTURBATION_PRIVATE.npz'),keys=['original_output_preADC_V','perturbed_output_preADC_V']);rms=np.sqrt(np.mean((x['perturbed_output_preADC_V'].astype('f8')-x['original_output_preADC_V'].astype('f8'))**2,1))*1000
 axs[2].hist(rms,bins=20,histtype='step',color=colors[n],label=labels[n],lw=1.5)
axs[0].set_xticks(range(3),['pureEE','pureHad','mixed']);axs[0].set(ylabel='Mean within-parent PSD SD',title='128parents x8 draws; same-source repeats');axs[0].legend(fontsize=7)
axs[1].set_xticks(range(3),['pureEE','pureHad','mixed']);axs[1].set(ylabel='Variance/cross contribution (mV²)',title='Within each group: Gaussian / flow / 2cross');axs[1].axhline(0,color='black',lw=.5)
axs[2].axvline(1000/16383,color='gray',ls='--');axs[2].set(xlabel='Full-source perturbation output RMS (mV)',ylabel='Parents',title='128same-bin zero-integral source perturbations');axs[2].legend(fontsize=7)
fig.suptitle('Shared-source functional components; covariance retained, physical split unknown');fig.tight_layout();save(fig,'source_and_random_decomposition.png')
write('SUMMARY_PLOT_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
 'figures':[p.name for p in F.glob('*.png')],'new_classifier_or_density_fit':0,'new_response_forward':0,'LONG_primary_retain':gates[L]['retain'],
 'gates_by_arm':{a:{k:v['pass']for k,v in g.items()if isinstance(v,dict)and'pass'in v}for a,g in gates.items()},'actual_visual_QA_pending':True})
print(json.dumps({'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'primary_retain':gates[L]['retain'],'figures':len(list(F.glob('*.png')))}),flush=True)
