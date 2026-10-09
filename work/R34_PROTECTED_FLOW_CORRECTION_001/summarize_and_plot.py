from __future__ import annotations
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
arms=CFG['endpoints'];controls=['BASE','INDEPENDENT_CFM','OT_CFM'];names=controls+arms;allnames=['REAL',*names]
labels={'REAL':'REAL DEV','BASE':'Frozen base','INDEPENDENT_CFM':'Independent','OT_CFM':'OT','INDEPENDENT_PROTECTED':'Independent + P','OT_PROTECTED':'OT + P','TRAIN_REFERENCE':'REAL TRAIN ref'}
colors={'REAL':'#111111','BASE':'#888888','INDEPENDENT_CFM':'#7eb1d8','OT_CFM':'#e9b887','INDEPENDENT_PROTECTED':'#1265a8','OT_PROTECTED':'#c55d10','TRAIN_REFERENCE':'#528a57'}
styles={'REAL':'-','BASE':'--','INDEPENDENT_CFM':':','OT_CFM':':','INDEPENDENT_PROTECTED':'-','OT_PROTECTED':'-'}
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def save(f,n):f.savefig(OUT/n,bbox_inches='tight');plt.close(f)
def change(a,b):return a/b-1 if a is not None and b is not None and b>0 else(0. if a==b==0 else None)
def guarded_measure(a,b):
    delta=change(a,b);return {'candidate':a,'BASE':b,'relative_change':delta,'pass':delta is not None and delta<=.10}

psd=m['PSD_main'];width=m['energy_conditional_width'];primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
common_all=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in allnames)]
gate={'fixed_recipe':'rank378 protected flow','new_fit_generation_in_summary':0,'prespecified_rules_changed':False,'endpoints':{}}
common_by_arm={};common_values={};density_gaps={n:[{'bin_id_1based':r['bin']+1,'N':r['N'],'failure_reasons':r['failure_reasons']}for r in width['tables'][n]['slices']if not r['valid']]for n in allnames}
for arm in arms:
    orig=CFG['endpoint_origins'][arm];basepaired=m['C2ST']['paired'][arm+'_minus_BASE'];ownpaired=m['C2ST']['paired'][arm+'_minus_'+orig]
    fullviews=['full_historical_shape','normalized_u480']
    g1={'paired_minus_BASE':{v:basepaired[v]for v in fullviews},'paired_minus_own_original':{v:ownpaired[v]for v in fullviews},
        'both_BASE_CI_upper_negative':all(basepaired[v]['BA_delta_95CI'][1]<0 for v in fullviews),
        'both_own_original_BA_delta_le_0_01':all(ownpaired[v]['BA_delta_first_minus_second']<=.01+1e-14 for v in fullviews),
        'threshold_roundoff_allowance':1e-14}
    g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_own_original_BA_delta_le_0_01']
    precision=m['precision_constraints'][arm]['actual_constraints'];gen=m['generation'][arm]
    valleys=[];ranks=[]
    for b in ['bin3','bin4']:
        a=primary[arm][b]['valley_fraction'];base=primary['BASE'][b]['valley_fraction'];delta=a-base if a is not None and base is not None else None
        valleys.append({'source_bin_id_1based':int(b[-1]),'fixed_source_N':primary[arm][b]['source_count'],'candidate':a,'BASE':base,'delta_fraction':delta,'pass':delta is not None and delta<=.02})
    for field in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
        a=m['source_association_actual_preADC_input'][arm]['composition_and_PDG']['all'][field]['Spearman'];base=m['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][field]['Spearman'];drop=base-a if a is not None and base is not None else None
        ranks.append({'measure':field,'fixed_N':2000,'candidate':a,'BASE':base,'BASE_minus_candidate':drop,'pass':drop is not None and drop<=.05})
    g2={'main_actual_precision_constraints':precision,'source_valleys':valleys,'whole_source_ranks':ranks,
        'all_source_IDs_composition_and_weights_preserved':gen['all_original_source_ids_preserved']and gen['source_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518},
        'no_new_invalid_nonfinite_or_ADC_clipping':m['domains'][arm]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,
        'auxiliary_exact_recovery_pass':all(r['exact_recovery_pass']for r in m['auxiliary_precision_recovery'][arm].values()),
        'independent_full_chain_replay_pass':replay['pass']and replay['rows']==4000 and replay['endpoints'][arm]['bitwise_pass']}
    g2['pass']=precision['constraint_pass']and all(r['pass']for r in valleys+ranks)and all(g2[k]for k in ['all_source_IDs_composition_and_weights_preserved','no_new_invalid_nonfinite_or_ADC_clipping','auxiliary_exact_recovery_pass','independent_full_chain_replay_pass'])
    bins=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in ['REAL','BASE',arm])];common_by_arm[arm]=[i+1 for i in bins]
    cv={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in bins]))if bins else None for n in ['BASE',arm]};common_values[arm]=cv
    measures={field+'_overall_W1':guarded_measure(m['domain_vs_REAL'][arm]['overall'][field]['W1'],m['domain_vs_REAL']['BASE']['overall'][field]['W1'])for field in ['PSD','baseline_RMS_V','ACF1']}
    measures['PSD_common_Q_equal_mean_W1']={**guarded_measure(cv[arm],cv['BASE']),'common_bins_1based':common_by_arm[arm],'minimum_N_each_domain':40}
    for row in width['common_support_errors'][arm+'_vs_BASE']:
        if row['field']=='sigma_PSD':measures['component'+str(row['component'])+'_raw_width_MAE']={**guarded_measure(row['MAE'][arm],row['MAE']['BASE']),'common_bin_ids_1based':row['common_bin_ids_1based']}
    required=[i for i in range(1,7)if width['tables']['BASE']['slices'][i]['valid']]
    failed=[i+1 for i in required if not width['tables'][arm]['slices'][i]['valid']]
    g3={'measures':measures,'BASE_valid_required_density_bins_1based':[i+1 for i in required],'new_failures_in_required_bins':failed,
        'all_candidate_density_gaps':density_gaps[arm],'per_arm_common_density_errors':width['common_support_errors'][arm+'_vs_BASE'],
        'pass':all(r['pass']for r in measures.values())and not failed,'smaller_common_intersection_not_used_to_hide_failures':True}
    bd=m['boundary_differences']['results'];br=[]
    for key in bd[arm]:
        a=bd[arm][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd[orig][key]['W1_V'];limit=max(b,o)+1/16383 if b is not None and o is not None else None
        br.append({'location':key,'candidate_W1_V':a,'BASE_W1_V':b,'own_original_W1_V':o,'limit_V':limit,'pass':a is not None and limit is not None and a<=limit})
    g4={'boundaries':br,'ADC_LSB_V':1/16383,'engineering_limit_not_physical_equivalence':True,'pass':all(r['pass']for r in br)}
    retain=all(r['pass']for r in [g1,g2,g3,g4]);gate['endpoints'][arm]={'retain':retain,'gate1_complete_response':g1,'gate2_precision_source_and_replay':g2,'gate3_PSD_width_noise':g3,'gate4_fixed_boundary_structure':g4}
gate['retain_any']=any(g['retain']for g in gate['endpoints'].values());gate['decision']='retain_only_passing_fixed_DEV_candidate'if gate['retain_any']else'close_unique_fixed_protection_projection_recipe'
gate['all_five_common_CDF_bins_1based']=[i+1 for i in common_all]
write('GATE_METRICS.json',gate)
compact={'status':m['status'],'gate':gate,'counters':m['counters'],'C2ST':m['C2ST'],'domain_vs_REAL':m['domain_vs_REAL'],'PSD_main':m['PSD_main'],
    'density_support_gaps':density_gaps,'width_common_support_errors':width['common_support_errors'],'precision_constraints':m['precision_constraints'],
    'boundary_differences':m['boundary_differences'],'source_primary_actual_Q':primary,'source_historical_ADC_reference_Q':m['source_composition']['historical_postADC_reference_Q_bins3_4'],
    'source_composition_counts':m['data_roles']['DEV_composition'],'projection_common_support_summary':m['projection_distances']['common_support_summary'],
    'raw_voltage_moments':m['raw_voltage_moments'],'replay_pass':replay['pass'],'actual_auxiliary_recovery_rows':3712,
    'random_summary':{n:{c:{k:r[k]for k in ['parents','within_parent_PSD_SD','within_parent_Q_SD_mVns','parents_straddling_valley','within_parent_waveform_RMS_V']}for c,r in m['random_decomposition'][n]['groups']['joint8']['by_composition'].items()}for n in arms},
    'physical_final_signal_electronic_split':'unknown','all2000_source_parents_weights_preserved':True,'endpoint_selection_uses_DEV':False,
    'EVAL_read':0,'CAL_waveform_decode':0,'new_response_updates':0,'new_G4':0,'privacy':'aggregate only; no event ID/index list, prediction or RNG state'}
write('METRICS_COMPACT.json',compact)

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
features=load(ROOT/'results/FEATURES_PRIVATE.npz');waves=load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz');tt=np.arange(480)-120;edges=np.asarray(psd['edges_logQ'])
def obs(n):
    q=features[n+'__Qtotal_mVns'];z=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7;return b
def line(a,n,x,y,**kw):a.plot(x,y,color=colors[n],ls=styles.get(n,'-'),label=labels[n],lw=1.1,**kw)

views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480'];short=['shape493','V480','prompt','postpeak','tail','baseline','Q/PSD','pure u480','seen\n480 points','complement\n480 points']
fig,ax=plt.subplots(2,1,figsize=(14,7),gridspec_kw={'height_ratios':[2,1]})
for j,n in enumerate(names):
    x=np.arange(10)+(j-2)*.12;row=m['C2ST']['results'][n];value=np.array([row[v]['balanced_accuracy']for v in views]);ci=np.array([row[v]['heldout_CI']['balanced_accuracy_95CI']for v in views])
    ax[0].errorbar(x,value,yerr=np.maximum(0,np.vstack([value-ci[:,0],ci[:,1]-value])),fmt='o',markersize=3,capsize=1,color=colors[n],label=labels[n])
ax[0].set(ylim=(.45,1.02),ylabel='fixed holdout BA',title='Original 600/250 rows per domain; 500 fixed-classifier cluster resamples');ax[0].axhline(.5,color='gray',ls=':');ax[0].set_xticks(range(10),short);ax[0].legend(ncol=3,fontsize=8)
for j,n in enumerate(arms):
    for k,ref in enumerate(['BASE',CFG['endpoint_origins'][n]]):
        rows=m['C2ST']['paired'][n+'_minus_'+ref];v=np.array([rows[x]['BA_delta_first_minus_second']for x in views]);ci=np.array([rows[x]['BA_delta_95CI']for x in views]);x=np.arange(10)+(2*j+k-1.5)*.16
        ax[1].errorbar(x,v,yerr=np.maximum(0,np.vstack([v-ci[:,0],ci[:,1]-v])),fmt='o'if ref=='BASE'else's',markersize=3,capsize=1,color=colors[n],alpha=1 if ref=='BASE'else.55,label=labels[n]+' minus '+labels[ref])
ax[1].axhline(0,color='gray',lw=.7);ax[1].set_xticks(range(10),short);ax[1].set_ylabel('paired BA difference');ax[1].legend(fontsize=7,ncol=2);fig.tight_layout();save(fig,'c2st.png')

fig,ax=plt.subplots(2,3,figsize=(13.5,7),sharex=True,sharey=True)
positive=np.concatenate([features[n+'__Qtotal_mVns'][features[n+'__Qtotal_mVns']>0]for n in allnames]);xmin,xmax=positive.min(),positive.max()
for a,n in zip(ax.flat,allnames):
    q=features[n+'__Qtotal_mVns'];p=features[n+'__PSD'];ok=features[n+'__valid']&(q>0);a.scatter(q[ok],p[ok],s=3,alpha=.25,color=colors[n]);a.set_xscale('log');a.set(xlim=(xmin*.95,xmax*1.05),ylim=(-.1,.6),title=labels[n]);a.grid(alpha=.12);a.text(.02,.02,f'QC {ok.sum()}; outside PSD view {np.sum(ok&((p<-.1)|(p>.6)))}',transform=a.transAxes,fontsize=7)
for a in ax[-1]:a.set_xlabel('Q total (mV ns)')
for a in ax[:,0]:a.set_ylabel('signed PSD')
fig.suptitle('Full unchanged parent populations; identical raw Q/PSD axes',y=1.01);fig.tight_layout();save(fig,'charge_psd.png')

sections=load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz');fig,ax=plt.subplots(2,4,figsize=(14,6.2),sharex=True,sharey=True)
for i,a in enumerate(ax.flat):
    counts=[]
    for n in allnames:
        x=sections[n+'__bin'+str(i+1)+'__PSD'];counts.append(len(x))
        if len(x):line(a,n,x,np.arange(1,len(x)+1)/len(x))
    a.set(title=f'Observed-Q bin {i+1}',xlim=(-.1,.6),ylim=(0,1));a.grid(alpha=.12);a.text(.02,.03,'N '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=6.7)
    if i not in common_all:a.text(.02,.92,'Outside all-five common N>=40',transform=a.transAxes,fontsize=7,color='#555')
for a in ax[1]:a.set_xlabel('signed PSD')
for a in ax[:,0]:a.set_ylabel('CDF')
ax[0,0].legend(fontsize=6.7,ncol=2);fig.tight_layout();save(fig,'conditional_psd_CDF.png')

qmid=np.exp((edges[:-1]+edges[1:])/2);fig,ax=plt.subplots(2,3,figsize=(13,6),sharex=True)
for n in allnames:
    for c in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            y=[r['components'][c][field]if r['valid']else np.nan for r in width['tables'][n]['slices']];line(ax[c,j],n,qmid,y,marker='o',ms=3)
for c in [0,1]:
    for j,title in enumerate(['component center','raw width','detrended width']):
        ax[c,j].set_xscale('log');ax[c,j].set(title=('low'if c==0 else'high')+' PSD '+title,ylabel='PSD units');ax[c,j].grid(alpha=.12)
        if c:ax[c,j].set_xlabel('observed-Q midpoint (mV ns)')
ax[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Failed or unsupported bins remain gaps; BASE-valid bins2–7 are independently required',y=1.02);fig.tight_layout();save(fig,'energy_peak_width.png')

fig,ax=plt.subplots(2,4,figsize=(14,6.2),sharex=True)
for i,a in enumerate(ax.flat):
    counts=[]
    for n in allnames:
        use=features[n+'__valid']&(obs(n)==i);v=waves[n][use];counts.append(len(v))
        if len(v):line(a,n,tt,np.median(v,axis=0)*1000)
    a.set(title=f'Observed-Q bin {i+1}',xlim=(-120,359));a.text(.02,.03,'N '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=6.7);a.grid(alpha=.12)
for a in ax[1]:a.set_xlabel('time relative to saved index120 (ns)')
for a in ax[:,0]:a.set_ylabel('median raw voltage (mV)')
ax[0,0].legend(fontsize=6.7,ncol=2);fig.suptitle('Complete 480 points; no smoothing or per-row rescaling',y=1.02);fig.tight_layout();save(fig,'conditional_waveforms.png')

raw=load(ROOT/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz');fig,ax=plt.subplots(2,4,figsize=(14,6.2),sharex=True)
for j,(b,title)in enumerate([(None,'population'),(1,'observed Q bin2'),(2,'observed Q bin3'),(5,'observed Q bin6')]):
    for n in allnames:
        for i,key in enumerate(['mean_V','SD_V']):line(ax[i,j],n,tt,raw[n+'__'+str(b)+'__'+key]*1000)
    ax[0,j].set_title(title)
    for a in ax[:,j]:a.set_xlim(-120,359);a.grid(alpha=.12)
    ax[1,j].set_xlabel('time relative to index120 (ns)')
ax[0,0].set_ylabel('mean voltage (mV)');ax[1,0].set_ylabel('pointwise SD (mV; ddof1)');ax[0,0].legend(fontsize=7,ncol=2);fig.tight_layout();save(fig,'raw_mean_SD.png')

fig,ax=plt.subplots(1,4,figsize=(14,3.7))
for a,field,title,factor in zip(ax,['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V'],['baseline RMS (mV)','baseline lag1 ACF','prompt100:150 curvature (mV)','postpeak150:220 curvature (mV)'],[1000,1,1000,1000]):
    for n in allnames:
        x=features[n+'__'+field];x=np.sort(x[np.isfinite(x)])*factor;line(a,n,x,np.arange(1,len(x)+1)/len(x))
    a.set(title=title,ylabel='CDF');a.grid(alpha=.12)
ax[0].legend(fontsize=6.8,ncol=2);fig.tight_layout();save(fig,'noise_curvature.png')

boundary=load(ROOT/'results/BOUNDARY_DIFFERENCES_PRIVATE.npz');keys=list(m['boundary_differences']['results']['BASE']);fig,ax=plt.subplots(3,3,figsize=(12.5,9))
for a,key in zip(ax.flat,keys):
    for n in allnames:
        x=np.sort(boundary[n+'__'+key])*1000;line(a,n,x,np.arange(1,len(x)+1)/len(x))
    a.set(title=key.replace('_',' '),xlabel='native difference (mV)',ylabel='CDF');a.grid(alpha=.12)
ax[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Prespecified P-boundaries only; no searched time positions',y=1.01);fig.tight_layout();save(fig,'boundary_differences.png')

fig,ax=plt.subplots(1,4,figsize=(15,3.8))
for j,n in enumerate(names):
    for k,b in enumerate(['bin3','bin4']):
        r=primary[n][b];pos=j*3+k
        for a,key in zip(ax[:3],['mean','sample_sd','valley_fraction']):a.bar(pos,r['output_PSD'][key]if key!='valley_fraction'else r[key],color=colors[n])
    rr=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all'];ax[3].bar(j-.14,rr['source_Q_to_output_Q']['Spearman'],.28,color=colors[n],alpha=.45);ax[3].bar(j+.14,rr['source_PSD_to_output_PSD']['Spearman'],.28,color=colors[n])
ticks=[j*3+k for j in range(5)for k in range(2)];ticktext=[a+str(b)for a in ['B','I','OT','IP','OTP']for b in [3,4]]
for a,title in zip(ax[:3],['Pure-had PSD center','Pure-had PSD sample SD','Frozen valley fraction']):a.set(title=title);a.set_xticks(ticks,ticktext,rotation=45,fontsize=7);a.grid(axis='y',alpha=.12)
ax[3].set(title='Actual source rank: light Q, dark PSD',ylim=(0,1.05));ax[3].set_xticks(range(5),['B','I','OT','IP','OTP']);ax[3].grid(axis='y',alpha=.12);fig.suptitle('Actual unADC NN-source Q bins3/4, N213/181; valley is not particle truth',y=1.02);fig.tight_layout();save(fig,'source_preservation.png')

fig,ax=plt.subplots(2,3,figsize=(12.5,7))
for j,n in enumerate(arms):
    p=load(ROOT/'results'/(n+'_PROJECTION_PRECISION_PRIVATE.npz'));g=m['precision_constraints'][n]
    for a,key in zip(ax[0],['full_original_RMS_V','full_kept_RMS_V','full_removed_RMS_V']):
        kind=key.split('_')[1];payload=load(ROOT/n/'generated.npz');x=np.sqrt(np.mean(payload[{'original':'original_increment_f64_V','kept':'kept_increment_f64_V','removed':'removed_increment_f64_V'}[kind]]**2,1))*1000;x=np.sort(x);line(a,n,x,np.arange(1,len(x)+1)/len(x));a.set(title=kind+' full480 increment RMS',xlabel='mV',ylabel='CDF');del payload
    for k,(region,r)in enumerate(g['projection_regions'].items()):
        ax[1,0].bar(k+(j-.5)*.32,r['kept_RMS_V']['mean']*1000,.3,color=colors[n],label=labels[n]if k==0 else None);ax[1,1].bar(k+(j-.5)*.32,r['removed_RMS_V']['mean']*1000,.3,color=colors[n])
    ratio=g['actual_constraints']['retained_energy_fraction_nonzero'];ax[1,2].plot(ratio['quantiles'],ratio['quantile_levels'],'o-',color=colors[n],label=labels[n]);ax[1,2].text(.05,.1+j*.1,f'{labels[n]} zero increment:417',transform=ax[1,2].transAxes,fontsize=7)
for a,title in zip(ax[1,:2],['kept regional RMS','removed regional RMS']):a.set(title=title,ylabel='mean RMS (mV)');a.set_xticks(range(4),['0:100','100:150','150:400','400:480'],rotation=15);a.grid(axis='y',alpha=.12)
ax[1,2].set(title='Retained energy, nonzero increments',xlabel='kept/original squared-norm ratio',ylabel='type7 quantile level');ax[0,0].legend(fontsize=7);ax[1,0].legend(fontsize=7);fig.suptitle('Rank378 P; zero increments reported separately',y=1.01);fig.tight_layout();save(fig,'kept_removed_increments.png')

actual=load(pathlib.Path(CFG['R33_root'])/'ACTUAL_GENERATION_INPUTS.npz');sf=ev.features(actual['source_waveforms_V']);qsrc=sf['features']['Qtotal_mVns'];qr=features['REAL__Qtotal_mVns'];fig,ax=plt.subplots(4,4,figsize=(16,12))
realpick={};columns=[(0,480,'whole480'),(94,107,'boundary99/100'),(143,157,'boundary149/150'),(393,408,'boundary399/400')]
for row,parent in enumerate([0,500,1000,1999]):
    k=np.searchsorted(edges,np.log(max(qsrc[parent],1e-30)),side='right')-1;candidate=np.flatnonzero(features['REAL__valid']&(obs('REAL')==k));candidate=candidate if len(candidate)else np.flatnonzero(features['REAL__valid']);ri=int(candidate[np.argmin(np.abs(qr[candidate]-qsrc[parent]))]);realpick[str(parent)]={'REAL_index':ri,'rule':'same frozen source-Q bin if supported, nearest REAL observedQ; unpaired reference only','source_Q_mVns':float(qsrc[parent]),'REAL_Q_mVns':float(qr[ri])}
    for col,(left,right,title)in enumerate(columns):
        a=ax[row,col]
        for n in allnames:line(a,n,tt[left:right],waves[n][ri if n=='REAL'else parent,left:right]*1000)
        for boundarypos in [100,150,400]:
            if left<=boundarypos<right:a.axvline(boundarypos-120,color='gray',ls='--',lw=.6)
        a.set(title=f'Parent {parent}: {title}',xlabel='relative ns',ylabel='raw mV');a.grid(alpha=.12)
ax[0,0].legend(fontsize=6.4,ncol=2);fig.suptitle('Fixed source positions0/500/1000/1999; REAL curve is an unpaired same-Q reference',y=1.01);fig.tight_layout();save(fig,'fixed_parent_boundary_examples.png');write('FIXED_EXAMPLE_METADATA.json',realpick)

fig,ax=plt.subplots(2,3,figsize=(14,7))
terms=['base_signal','base_electronic','base_sum_rounding','kept_increment','removed_increment']
for c,category in enumerate(['pure_EE','pure_had','mixed']):
    for j,n in enumerate(arms):
        r=m['random_decomposition'][n]['groups']['joint8']['by_composition'][category];cov=r['five_algebraic_components_covariance_V2']['full480'];summary=[cov[t+'__'+t]['mean']for t in terms[:4]]+[2*sum(cov[terms[i]+'__'+terms[k]]['mean']for i in range(4)for k in range(i+1,4))]
        ax[0,c].bar(np.arange(5)+(j-.5)*.32,np.asarray(summary)*1e6,.3,color=colors[n],label=labels[n]);rps=r['within_parent_PSD_SD'];ax[1,c].plot(rps['quantiles'],rps['quantile_levels'],'o-',ms=3,color=colors[n],label=labels[n])
    ax[0,c].set(title=category+' protected total variance',ylabel='mean contribution (mV squared)');ax[0,c].set_xticks(range(5),['baseS','baseE','rounding','kept','all2cov'],fontsize=7);ax[0,c].axhline(0,color='gray',lw=.6);ax[0,c].grid(axis='y',alpha=.12);ax[1,c].set(title=category+' joint8 PSD SD',xlabel='within-parent PSD SD',ylabel='type7 quantile level');ax[1,c].grid(alpha=.12)
ax[0,0].legend(fontsize=7);fig.suptitle('Algebraic covariance including cross terms; final physical split unknown',y=1.01);fig.tight_layout();save(fig,'joint_random_components.png')

tags=list(m['source_probes'][arms[0]]['rows']);fig,ax=plt.subplots(1,2,figsize=(13,4))
for j,n in enumerate(arms):
    rr=m['source_probes'][n]['rows'];x=np.arange(len(tags))+(j-.5)*.34
    ax[0].bar(x,[rr[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],.32,color=colors[n],label=labels[n]);ax[1].bar(x,[rr[t]['PSD_vs_reference']['sample_sd']for t in tags],.32,color=colors[n],label=labels[n])
for a,title in zip(ax,['RMS to correct source/noise reference (mV)','PSD difference SD to correct reference']):a.set(title=title);a.set_xticks(range(len(tags)),[t.replace('_','\n')for t in tags],fontsize=7);a.grid(axis='y',alpha=.12)
ax[0].legend(fontsize=7);fig.tight_layout();save(fig,'source_controls.png')

files={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'figure_count':len(files),'new_HGB_EM_response_updates_in_this_entry':0,'new_neural_generation':0,'all_plots_from_saved_results':True,'decision':gate['decision'],'retain_any':gate['retain_any']}
write('PLOT_SUMMARY_RECEIPT.json',receipt);print(json.dumps({'receipt':receipt,'gate_passes':{n:[v['pass']for k,v in g.items()if k.startswith('gate')]for n,g in gate['endpoints'].items()},'common_CDF_bins':common_by_arm}),flush=True)
