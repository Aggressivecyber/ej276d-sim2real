from pathlib import Path
import ast, hashlib, json, os, datetime
ROOT = Path(__file__).resolve().parent
old = (ROOT.parent/'R34_PROTECTED_FLOW_CORRECTION_001'/'summarize_and_plot.py').read_text('utf8')
header = r'''from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True,parents=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));m=json.loads((ROOT/'metrics.json').read_text('utf8'));replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'))
old=json.loads((pathlib.Path(CFG['R33_root'])/'metrics.json').read_text('utf8'))
arm='PSD_ORDERED_CFM';arms=[arm];controls=['BASE','INDEPENDENT_CFM','OT_CFM'];names=controls+arms;allnames=['REAL',*names]
labels={'REAL':'REAL DEV','BASE':'Frozen base','INDEPENDENT_CFM':'Independent','OT_CFM':'OT','PSD_ORDERED_CFM':'PSD ordered','TRAIN_REFERENCE':'REAL TRAIN ref'}
colors={'REAL':'#111111','BASE':'#888888','INDEPENDENT_CFM':'#7eb1d8','OT_CFM':'#e9b887','PSD_ORDERED_CFM':'#1265a8','TRAIN_REFERENCE':'#528a57'}
styles={'REAL':'-','BASE':'--','INDEPENDENT_CFM':':','OT_CFM':':','PSD_ORDERED_CFM':'-'}
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def save(f,n):f.savefig(OUT/n,bbox_inches='tight');plt.close(f)
def guarded_measure(a,b):
    delta=a/b-1 if a is not None and b is not None and b>0 else(0. if a==b==0 else None)
    return {'candidate':a,'BASE':b,'relative_change':delta,'pass':delta is not None and delta<=.10}
psd=m['PSD_main'];width=m['energy_conditional_width'];primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
common_all=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in allnames)]
fullviews=['full_historical_shape','normalized_u480'];basepaired=m['C2ST']['paired']['ORDERED_minus_BASE'];ownpaired=m['C2ST']['paired']['ORDERED_minus_INDEPENDENT']
g1={'paired_minus_BASE':{v:basepaired[v]for v in fullviews},'paired_minus_INDEPENDENT':{v:ownpaired[v]for v in fullviews},'both_BASE_CI_upper_negative':all(basepaired[v]['BA_delta_95CI'][1]<0 for v in fullviews),'both_INDEPENDENT_BA_delta_le_0_01':all(ownpaired[v]['BA_delta_first_minus_second']<=.01+1e-14 for v in fullviews),'threshold_roundoff_allowance':1e-14}
g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_INDEPENDENT_BA_delta_le_0_01']
bins=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in ['REAL','BASE',arm])]
commonPSD={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in bins]))if bins else None for n in ['BASE',arm]}
measures={field+'_overall_W1':guarded_measure(m['domain_vs_REAL'][arm]['overall'][field]['W1'],m['domain_vs_REAL']['BASE']['overall'][field]['W1'])for field in ['PSD','baseline_RMS_V','ACF1']}
measures['PSD_common_Q_equal_mean_W1']={**guarded_measure(commonPSD[arm],commonPSD['BASE']),'common_bins_1based':[i+1 for i in bins],'minimum_N_each_domain':40}
for row in width['common_support_errors']['ORDERED_vs_BASE']:
    if row['field']=='sigma_PSD':measures['component'+str(row['component'])+'_raw_width_MAE']={**guarded_measure(row['MAE'][arm],row['MAE']['BASE']),'common_bin_ids_1based':row['common_bin_ids_1based']}
required=[i for i in range(1,7)if width['tables']['BASE']['slices'][i]['valid']]
failed=[i+1 for i in required if not width['tables'][arm]['slices'][i]['valid']]
density_gaps={n:[{'bin_id_1based':r['bin']+1,'N':r['N'],'failure_reasons':r['failure_reasons']}for r in width['tables'][n]['slices']if not r['valid']]for n in allnames}
g2={'measures':measures,'BASE_valid_required_density_bins_1based':[i+1 for i in required],'new_failures_in_required_bins':failed,'density_gaps':density_gaps[arm],'common_density_errors':width['common_support_errors']['ORDERED_vs_BASE'],'pass':all(r['pass']for r in measures.values())and not failed,'support_intersection_not_used_to_hide_failure':True}
valleys=[];ranks=[]
for b in ['bin3','bin4']:
    a=primary[arm][b]['valley_fraction'];br=primary['BASE'][b]['valley_fraction'];d=a-br if a is not None and br is not None else None
    valleys.append({'actual_source_bin':b,'fixed_source_count':primary[arm][b]['source_count'],'candidate':a,'BASE':br,'delta':d,'pass':d is not None and d<=.02})
for field in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
    a=m['source_association_actual_preADC_input'][arm]['composition_and_PDG']['all'][field]['Spearman'];br=m['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][field]['Spearman'];d=br-a if a is not None and br is not None else None
    ranks.append({'measure':field,'fixed_N':2000,'candidate':a,'BASE':br,'BASE_minus_candidate':d,'pass':d is not None and d<=.05})
gen=m['generation'][arm];probe=m['source_probes'][arm];sens=probe['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
g3={'source_valleys':valleys,'whole_source_ranks':ranks,'original_IDs_order_composition_weights_preserved':gen['all_original_source_ids_preserved']and m['data_roles']['DEV_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518}and not m['primary_DEV_population_reweighted'],'no_new_invalid_nonfinite_or_ADC_clip':m['domains'][arm]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,'nonzero_source_sensitivity_RMS_V':sens,'source_sensitivity_not_correct_physics':True}
g3['pass']=all(r['pass']for r in valleys+ranks)and g3['original_IDs_order_composition_weights_preserved']and g3['no_new_invalid_nonfinite_or_ADC_clip']and sens is not None and sens>0
bd=m['boundary_differences']['results'];boundary_rows=[]
for key in bd[arm]:
    a=bd[arm][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd['INDEPENDENT_CFM'][key]['W1_V'];limit=max(b,o)+1/16383 if b is not None and o is not None else None
    boundary_rows.append({'location':key,'candidate_W1_V':a,'BASE_W1_V':b,'INDEPENDENT_W1_V':o,'limit_V':limit,'pass':a is not None and limit is not None and a<=limit})
g4={'boundaries':boundary_rows,'ADC_LSB_V':1/16383,'solver_check':m['solver_check'][arm],'full_chain_replay_pass':replay['pass']and replay['rows']==2000 and replay['endpoints'][arm]['bitwise_pass'],'boundary_limit_not_physical_equivalence':True}
g4['pass']=all(r['pass']for r in boundary_rows)and g4['solver_check']['pass']and g4['full_chain_replay_pass']
retain=all(g['pass']for g in [g1,g2,g3,g4]);gate={'fixed_recipe':'within-original-Q stable preADC BASE/REAL PSD ordered CFM','prespecified_rules_changed':False,'gate1_complete_response':g1,'gate2_PSD_width_noise':g2,'gate3_source_preservation':g3,'gate4_boundary_solver_replay':g4,'retain':retain,'decision':'retain_fixed_DEV_candidate'if retain else'close_frozen_BASE_MLP_CFM_coupling_small_variants','all_controls_common_CDF_bins_1based':[i+1 for i in common_all],'same_training_input_flow':m['same_training_input_flow'],'no_new_fit_or_neural_generation_in_summary':True}
write('GATE_METRICS.json',gate)
compact={'status':m['status'],'gate':gate,'counters':m['counters'],'C2ST':m['C2ST'],'domain_vs_REAL':m['domain_vs_REAL'],'PSD_main':psd,'density_support_gaps':density_gaps,'width_common_support_errors':width['common_support_errors'],'source_primary_actual_Q':primary,'source_historical_ADC_reference_Q':m['source_composition']['historical_postADC_reference_Q_bins3_4'],'source_actual_rank_summaries':m['source_association_actual_preADC_input'],'source_composition_counts':m['data_roles']['DEV_composition'],'source_PDG_counts_inferred':m['data_roles']['DEV_PDG_counts_inferred'],'source_PDG_provenance':m['data_roles']['DEV_PDG_provenance'],'projection_common_support_summary':m['projection_distances']['common_support_summary'],'raw_voltage_moments':m['raw_voltage_moments'],'boundary_differences':m['boundary_differences'],'replay_pass':replay['pass'],'random_summary':{n:{c:{k:r[k]for k in ['parents','within_parent_PSD_SD','within_parent_Q_SD_mVns','parents_straddling_valley','within_parent_waveform_RMS_V']}for c,r in m['random_decomposition'][n]['groups']['joint8']['by_composition'].items()}for n in arms},'physical_final_signal_electronic_split':'unknown','all2000_source_parents_weights_preserved':True,'endpoint_selection_uses_DEV':False,'EVAL_read':0,'CAL_waveform_decode':0,'new_G_D_reader_updates':0,'new_G4':0,'privacy':'aggregate only; actual source IDs, row indices, predictions, target permutations and RNG states remain private'}
write('METRICS_COMPACT.json',compact)
# Reuse immutable R33 aggregate rank summaries for its two controls, without refitting.
for name in ['INDEPENDENT_CFM','OT_CFM']:m['source_association_actual_preADC_input'][name]=old['source_association_actual_preADC_input'][name]
'''
plots=old[old.index('plt.rcParams.update'):old.index("\nfig,ax=plt.subplots(2,3,figsize=(12.5,7))")]
plots=plots.replace("for k,ref in enumerate(['BASE',CFG['endpoint_origins'][n]]):", "for k,ref in enumerate(['BASE','INDEPENDENT_CFM']):")
plots=plots.replace("rows=m['C2ST']['paired'][n+'_minus_'+ref]", "rows=m['C2ST']['paired']['ORDERED_minus_'+('BASE'if ref=='BASE'else'INDEPENDENT')]")
plots=plots.replace("for a in ax[-1]:a.set_xlabel('Q total (mV ns)')", "ax.flat[-1].set_axis_off()\nfor a in ax[-1,:2]:a.set_xlabel('Q total (mV ns)')")
plots=plots.replace('Outside all-five common N>=40','Outside all-domain common N>=40')
start=plots.index("raw=load(ROOT/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz')")
end=plots.index("\nfig,ax=plt.subplots(1,4,figsize=(14,3.7))",start)
plots=plots[:start]+plots[start:end].replace('for n in allnames:',"for n in ['REAL','BASE','PSD_ORDERED_CFM']:")+plots[end:]
plots=plots.replace('Prespecified P-boundaries only; no searched time positions','Prespecified fixed boundaries; no searched time positions')
plots=plots.replace("range(5)for k in range(2)","range(4)for k in range(2)").replace("['B','I','OT','IP','OTP']","['B','I','OT','PSD']").replace("set_xticks(range(5)","set_xticks(range(4)")
tail=r'''
actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');qsrc=ev.features(actual['source_waveforms_V'])['features']['Qtotal_mVns'];qr=features['REAL__Qtotal_mVns'];fig,ax=plt.subplots(4,4,figsize=(16,12));realpick={}
columns=[(0,480,'whole480'),(94,107,'boundary99/100'),(143,157,'boundary149/150'),(393,408,'boundary399/400')]
for row,parent in enumerate([0,500,1000,1999]):
    k=np.searchsorted(edges,np.log(max(qsrc[parent],1e-30)),side='right')-1;candidates=np.flatnonzero(features['REAL__valid']&(obs('REAL')==k));candidates=candidates if len(candidates)else np.flatnonzero(features['REAL__valid']);ri=int(candidates[np.argmin(np.abs(qr[candidates]-qsrc[parent]))]);realpick[str(parent)]={'REAL_index':ri,'rule':'same frozen actual source-Q bin if supported, nearest REAL observedQ; unpaired reference only','source_Q_mVns':float(qsrc[parent]),'REAL_Q_mVns':float(qr[ri])}
    for col,(left,right,title)in enumerate(columns):
        a=ax[row,col]
        for n in allnames:line(a,n,tt[left:right],waves[n][ri if n=='REAL'else parent,left:right]*1000)
        for p in [100,150,400]:
            if left<=p<right:a.axvline(p-120,color='gray',ls='--',lw=.6)
        a.set(title=f'Parent {parent}: {title}',xlabel='relative ns',ylabel='raw mV');a.grid(alpha=.12)
ax[0,0].legend(fontsize=6.4,ncol=2);fig.suptitle('Prespecified source positions0/500/1000/1999; REAL is an unpaired same-Q reference',y=1.01);fig.tight_layout();save(fig,'fixed_parent_boundary_examples.png');write('FIXED_EXAMPLE_METADATA_PRIVATE.json',realpick)

fig,ax=plt.subplots(2,3,figsize=(14,7));terms=['base_signal','base_electronic','base_sum_rounding','flow_increment_f64']
for c,category in enumerate(['pure_EE','pure_had','mixed']):
    r=m['random_decomposition'][arm]['groups']['joint8']['by_composition'][category];cov=r['four_additive_algebraic_components_covariance_V2']['full480'];parts=[cov[t+'__'+t]['mean']for t in terms]+[2*sum(cov[terms[i]+'__'+terms[k]]['mean']for i in range(4)for k in range(i+1,4))]
    ax[0,c].bar(np.arange(5),np.asarray(parts)*1e6,color=[colors[arm]]*4+['#555555']);q=r['within_parent_PSD_SD'];ax[1,c].plot(q['quantiles'],q['quantile_levels'],'o-',ms=3,color=colors[arm]);ax[0,c].set(title=category+f' ({r["parents"]} parents)',ylabel='mean variance contribution (mV squared)');ax[0,c].set_xticks(range(5),['baseS','baseE','rounding','flow','all2cov'],fontsize=7);ax[0,c].axhline(0,color='gray',lw=.6);ax[0,c].grid(axis='y',alpha=.12);ax[1,c].set(title='joint8 PSD SD',xlabel='within-parent PSD SD',ylabel='type7 quantile level');ax[1,c].grid(alpha=.12)
fig.suptitle('Exact four-term additive algebra with cross terms; final physical split unknown',y=1.01);fig.tight_layout();save(fig,'joint_random_components.png')

tags=list(m['source_probes'][arm]['rows']);fig,ax=plt.subplots(1,2,figsize=(13,4));rows=m['source_probes'][arm]['rows'];x=np.arange(len(tags))
ax[0].bar(x,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],color=colors[arm]);ax[1].bar(x,[rows[t]['PSD_vs_reference']['sample_sd']for t in tags],color=colors[arm])
for a,title in zip(ax,['RMS to correct source/noise reference (mV)','PSD difference SD to correct reference']):a.set(title=title);a.set_xticks(x,[t.replace('_','\n')for t in tags],fontsize=7);a.grid(axis='y',alpha=.12)
fig.tight_layout();save(fig,'source_controls.png')

trace=json.loads((ROOT/arm/'TRAINING_TRACE.json').read_text('utf8'));coupling=json.loads((ROOT/arm/'COUPLING_BY_UPDATE_PRIVATE.json').read_text('utf8'));train=json.loads((ROOT/arm/'TRAINING_RECEIPT.json').read_text('utf8'));step=np.asarray([r['step']for r in trace]);groups=[g for r in coupling for g in r['Q_subgroups']]
coupling_summary={'updates':len(coupling),'rows':sum(g['N']for g in groups),'subgroups':len(groups),'mixed_updates':sum(len(r['Q_subgroups'])>1 for r in coupling),'singleton_subgroups':sum(g['N']==1 for g in groups),'source_ties':sum(g['source_tie_count']for g in groups),'target_ties':sum(g['target_tie_count']for g in groups),'every_update_one_to_one_no_cross_Q':all(r['one_to_one_no_cross_Q']for r in coupling),'every_update_source_order_restored':all(r['original_source_order_restored']for r in coupling),'every_update_row_marginals_preserved':all(r['all_source_target_row_marginals_preserved']for r in coupling),'every_subgroup_no_sorted_inversion':all(g['no_sorted_PSD_inversion']for g in groups),'weighted_mean_absolute_PSD_gap_original':sum(g['N']*g['mean_abs_PSD_gap_original']for g in groups)/train['actual_rows'],'weighted_mean_absolute_PSD_gap_ordered':sum(g['N']*g['mean_abs_PSD_gap_ordered']for g in groups)/train['actual_rows'],'not_a_validation_or_physical_pairing_metric':True,'original_training_private_files_retained':True};write('COUPLING_SUMMARY.json',coupling_summary)
fig,ax=plt.subplots(2,3,figsize=(12.8,7));ax[0,0].plot(step,[r['CFM_loss']for r in trace],'o-',color=colors[arm]);ax[0,0].set(title='Recorded velocity MSE',xlabel='actual update',ylabel='CFM loss');ax[0,1].plot(step,[r['gradient_norm_before_clip']for r in trace],'o-');ax[0,1].set(title='Gradient norm before clip',xlabel='actual update');ax[0,2].plot(step,[r['velocity_label_RMS']for r in trace],label='target velocity');ax[0,2].plot(step,[r['prediction_RMS']for r in trace],label='prediction');ax[0,2].set(title='Scaled velocity RMS',xlabel='actual update');ax[0,2].legend(fontsize=8)
before=np.asarray([sum(g['N']*g['mean_abs_PSD_gap_original']for g in r['Q_subgroups'])/32 for r in coupling]);after=np.asarray([sum(g['N']*g['mean_abs_PSD_gap_ordered']for g in r['Q_subgroups'])/32 for r in coupling]);xx=np.arange(1,len(coupling)+1)
ax[1,0].plot(xx,before,color='#888',alpha=.6,lw=.5,label='original');ax[1,0].plot(xx,after,color=colors[arm],alpha=.7,lw=.5,label='ordered');ax[1,0].set(title='Actual pairing PSD gap (not validation)',xlabel='actual update',ylabel='mean absolute PSD gap');ax[1,0].legend(fontsize=8)
bybin=[]
for b in range(1,9):
    gs=[g for g in groups if g['bin_id_1based']==b];n=sum(g['N']for g in gs);bybin.append({'bin_id_1based':b,'rows':n,'original':sum(g['N']*g['mean_abs_PSD_gap_original']for g in gs)/n if n else None,'ordered':sum(g['N']*g['mean_abs_PSD_gap_ordered']for g in gs)/n if n else None})
ax[1,1].bar(np.arange(8)-.18,[g['original']or 0 for g in bybin],.34,color='#aaa',label='original');ax[1,1].bar(np.arange(8)+.18,[g['ordered']or 0 for g in bybin],.34,color=colors[arm],label='ordered');ax[1,1].set_xticks(np.arange(8),[str(i)for i in range(1,9)]);ax[1,1].set(title='Row-weighted pairing gap by source Q',xlabel='frozen training Q bin');ax[1,1].legend(fontsize=8)
for i,g in enumerate(bybin):
    if not g['rows']:ax[1,1].text(i,.003,'n/a',ha='center',fontsize=7)
ax[1,2].set_axis_off();ax[1,2].text(.02,.96,f"{train['updates']} updates / {train['actual_rows']} rows\nTraining {train['training_s']:.3f}s\nSame initial state + all8 original streams\nAll marginals and source order retained\nMixed updates: {coupling_summary['mixed_updates']}\nSingleton subgroups: {coupling_summary['singleton_subgroups']}\nSource/target ties: {coupling_summary['source_ties']}/{coupling_summary['target_ties']}\nNo invalid PSD key\nFinal only; no checkpoint selection",va='top',fontsize=9)
for a in ax.flat[:5]:a.grid(alpha=.12)
fig.tight_layout();save(fig,'training_coupling.png');coupling_summary['by_bin']=bybin;write('COUPLING_SUMMARY.json',coupling_summary)

files={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'figure_count':len(files),'new_HGB_EM_response_updates_in_this_entry':0,'new_neural_generation':0,'all_plots_from_saved_results':True,'decision':gate['decision'],'retain':gate['retain'],'checks':{'all_saved_coupling_marginals':coupling_summary['every_update_row_marginals_preserved'],'all_8_streams_match':all(m['same_training_input_flow']['same_actual_source_REAL_original_bin_ordinal_z_white_style_t_base_hashes'].values())}}
write('PLOT_SUMMARY_RECEIPT.json',receipt);print(json.dumps({'receipt':receipt,'gates':[g1['pass'],g2['pass'],g3['pass'],g4['pass']],'common_PSD_bins':[i+1 for i in bins],'density_new_failures':failed}),flush=True)
'''
source=header+'\n'+plots+'\n'+tail
ast.parse(source)
(ROOT/'summarize_and_plot.py').write_text(source,'utf8')
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output':'summarize_and_plot.py','bytes':len(source.encode()),'sha256':hashlib.sha256(source.encode()).hexdigest(),'AST':True,'scientific_execution':False,'old_plot_source_read_only':str(ROOT.parent/'R34_PROTECTED_FLOW_CORRECTION_001'/'summarize_and_plot.py')}
(ROOT/'PLOT_SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
