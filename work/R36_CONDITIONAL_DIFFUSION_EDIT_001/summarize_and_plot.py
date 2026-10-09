from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True,parents=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));m=json.loads((ROOT/'metrics.json').read_text('utf8'))
replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'))
A='CONDITIONAL_DIFFUSION_EDIT';R='REAL_GUIDE_EDIT';controls=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM']
labels={'REAL':'REAL DEV','BASE':'Frozen BASE','INDEPENDENT_CFM':'Independent flow','OT_CFM':'OT flow','PSD_ORDERED_CFM':'PSD ordered flow',A:'Source diffusion edit',R:'REAL guide edit','TRAIN_REFERENCE':'Raw REAL TRAIN ref','SOURCE_CLEAN':'Source (ADC reference)','SOURCE_INPUT':'Actual NN source'}
colors={'REAL':'#111111','BASE':'#888888','INDEPENDENT_CFM':'#86a9cc','OT_CFM':'#c6a98a','PSD_ORDERED_CFM':'#ae8bba',A:'#1565c0',R:'#e67820','TRAIN_REFERENCE':'#398347','SOURCE_CLEAN':'#74849c','SOURCE_INPUT':'#74849c'}
styles={'BASE':'--','TRAIN_REFERENCE':'--','INDEPENDENT_CFM':':','OT_CFM':':','PSD_ORDERED_CFM':':' }
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
features=load(ROOT/'results/FEATURES_PRIVATE.npz');waves=load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz');psd=m['PSD_main'];width=m['energy_conditional_width']
edges=np.asarray(psd['edges_logQ']);tt=np.arange(480)-120
def obs(n):
    q=features[n+'__Qtotal_mVns'];z=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7;return b
def line(ax,n,x,y,**kw):ax.plot(x,y,color=colors[n],ls=styles.get(n,'-'),label=labels[n],lw=1.2,**kw)
def cdf(ax,n,values,factor=1):
    x=np.sort(values[np.isfinite(values)])*factor
    if len(x):line(ax,n,x,np.arange(1,len(x)+1)/len(x))
def measure(a,b):
    d=a/b-1 if a is not None and b is not None and b>0 else(0. if a==b==0 else None)
    return{'candidate':a,'BASE':b,'relative_change':d,'pass':d is not None and d<=.10}
views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
full=['full_historical_shape','normalized_u480'];paired=m['C2ST']['paired'][A]
vs35={v:m['C2ST']['results'][A][v]['balanced_accuracy']-m['C2ST']['results']['PSD_ORDERED_CFM'][v]['balanced_accuracy']for v in full}
g1={'paired_minus_BASE':{v:paired[v]for v in full},'BA_point_delta_minus_R35':vs35,
    'both_BASE_CI_upper_negative':all(paired[v]['BA_delta_95CI'][1]<0 for v in full),
    'both_R35_point_delta_le_0_01':all(v<=.01+1e-14 for v in vs35.values()),'roundoff_allowance':1e-14}
g1['pass']=g1['both_BASE_CI_upper_negative']and g1['both_R35_point_delta_le_0_01']
common=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in ['REAL','BASE',A])]
pw={n:float(np.mean([psd['vs_REAL'][n]['bins'][i]['W1']for i in common]))if common else None for n in ['BASE',A]}
measures={k+'_overall_W1':measure(m['domain_vs_REAL'][A]['overall'][k]['W1'],m['domain_vs_REAL']['BASE']['overall'][k]['W1'])for k in ['PSD','baseline_RMS_V','ACF1']}
measures['PSD_common_Q_equal_mean_W1']={**measure(pw[A],pw['BASE']),'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40}
for row in width['common_support_errors']['DIFFUSION_vs_BASE']:
    if row['field']=='sigma_PSD':measures['component'+str(row['component'])+'_raw_width_MAE']={**measure(row['MAE'][A],row['MAE']['BASE']),'common_bins_1based':row['common_bin_ids_1based']}
required=[i for i in range(1,7)if width['tables']['BASE']['slices'][i]['valid']];newfail=[i+1 for i in required if not width['tables'][A]['slices'][i]['valid']]
zeros={n:int(np.sum(features[n+'__baseline_RMS_V']==0))for n in ['BASE',A]}
gaps={n:[{'bin_id_1based':s['bin']+1,'N':s['N'],'failure_reasons':s['failure_reasons']}for s in width['tables'][n]['slices']if not s['valid']]for n in ['REAL',*controls,A,R]}
g2={'measures':measures,'BASE_valid_required_bins_1based':[i+1 for i in required],'new_required_density_failures':newfail,
    'all_density_support_gaps':gaps,'zero_baseline_counts':zeros,'zero_baseline_not_increased':zeros[A]<=zeros['BASE'],
    'pass':all(v['pass']for v in measures.values())and not newfail and zeros[A]<=zeros['BASE']}
primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4'];valleys=[];ranks=[]
for b in ['bin3','bin4']:
    x=primary[A][b]['valley_fraction'];base=primary['BASE'][b]['valley_fraction'];d=x-base if x is not None and base is not None else None
    valleys.append({'actual_source_bin':b,'N':primary[A][b]['source_count'],'candidate':x,'BASE':base,'delta':d,'pass':d is not None and d<=.02})
for field in ['source_Q_to_output_Q','source_PSD_to_output_PSD']:
    x=m['source_association_actual_preADC_input'][A]['composition_and_PDG']['all'][field]['Spearman'];b=m['source_association_actual_preADC_input']['BASE']['composition_and_PDG']['all'][field]['Spearman'];d=b-x if x is not None and b is not None else None
    ranks.append({'measure':field,'candidate':x,'BASE':b,'BASE_minus_candidate':d,'pass':d is not None and d<=.05})
gen=m['generation'][A];sens=m['source_probes'][A]['rows']['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']
g3={'actual_source_pure_had_valleys':valleys,'whole_population_source_ranks':ranks,
    'source_ID_order_composition_weights_preserved':gen['all_original_source_ids_preserved']and m['data_roles']['DEV_composition']=={'mixed':65,'pure_EE':417,'pure_had':1518}and not m['primary_DEV_population_reweighted'],
    'no_new_invalid_nonfinite_ADC_clip':m['domains'][A]['invalid']==0 and gen['ADC']['pre_ADC_finite']and gen['ADC']['clip_count']==0,
    'source_roll_RMS_V':sens,'nonzero_sensitivity_not_correct_physical_mapping':True}
g3['pass']=all(x['pass']for x in valleys+ranks)and g3['source_ID_order_composition_weights_preserved']and g3['no_new_invalid_nonfinite_ADC_clip']and sens is not None and sens>0
bd=m['boundary_differences']['results'];bounds=[]
for key in bd[A]:
    x=bd[A][key]['W1_V'];b=bd['BASE'][key]['W1_V'];o=bd['INDEPENDENT_CFM'][key]['W1_V'];lim=max(b,o)+1/16383 if b is not None and o is not None else None
    bounds.append({'location':key,'candidate_W1_V':x,'BASE_W1_V':b,'INDEPENDENT_W1_V':o,'limit_V':lim,'pass':x is not None and lim is not None and x<=lim})
g4={'boundaries':bounds,'guide_probe':m['guide_probe'],'pass':all(x['pass']for x in bounds)and m['guide_probe']['pass'],
    'guide_response_and_Q_PSD_numerical_preservation_are_separate':True}
controlviews=['full_historical_shape','normalized_u480','baseline_local']
g5={'REAL_control_is_seen_TRAIN_reference':True,'paired_minus_RAW_REAL_REAL':{v:m['C2ST']['paired'][R][v]for v in controlviews},
    'no_old_Q_PSD_2D_RAW_RR_prediction_available':'unknown; no new reference fit',
    'BA_CI_upper_tolerance':.10,'W1_descriptive_not_equivalence_threshold':True,
    'pass':all(m['C2ST']['paired'][R][v]['BA_delta_95CI'][1]<=.10 for v in controlviews)}
g6={'independent_CPU_full_chain_replay_rows':replay['rows'],'bitwise_pass':replay['pass'],
    'pass':replay['rows']==4000 and replay['pass'],'all_new_figures_actual_QA_required':'recorded separately in FIGURE_QA.json'}
gates={'gate1_complete_response':g1,'gate2_PSD_width_noise':g2,'gate3_source_preservation':g3,'gate4_boundary_guide':g4,'gate5_REAL_control':g5,'gate6_replay':g6}
retain=all(x['pass']for x in gates.values())
gate={'fixed_recipe':'conditional finite-noise 50-step DDPM edit','prespecified_rules_changed':False,**gates,
    'retain':retain,'decision':'retain_fixed_DEV_candidate_pending_actual_figure_QA'if retain else'close_this_fixed_diffusion_edit_recipe',
    'no_general_impossibility_claim':True,'no_new_fit_or_generation_in_summary':True}
write('GATE_METRICS.json',gate)
compact={'status':m['status'],'decision':gate['decision'],'gate':gate,'counters':m['counters'],'training':m['training'],
    'C2ST':m['C2ST'],'domain_vs_REAL':m['domain_vs_REAL'],'PSD_main':psd,'width_common_support_errors':width['common_support_errors'],
    'density_support_gaps':gaps,'source_primary_actual_Q':primary,'source_historical_ADC_reference_Q':m['source_composition']['historical_postADC_reference_Q_bins3_4'],
    'source_rank_summary':m['source_association_actual_preADC_input'],'source_probe_summaries':m['source_probes'],
    'guide_probe':m['guide_probe'],'boundary_differences':m['boundary_differences'],'raw_voltage_moments':m['raw_voltage_moments'],
    'projection_common_support':m['projection_distances']['common_support_summary'],'all_source_IDs_order_weights_preserved':True,
    'final_physical_signal_electronic_split':'unknown','EVAL_read':0,'CAL_waveform_decode':0,
    'privacy':'aggregate only; identities, classifier predictions, row selection, seeds and actual RNG states stay local'}
write('METRICS_COMPACT.json',compact)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
short=['shape493','V480','prompt','postpeak','tail','baseline','Q/PSD','pure u480','seen\n480 points','complement\n480 points']
fig,axs=plt.subplots(3,1,figsize=(14,10),gridspec_kw={'height_ratios':[1.2,1,1]})
for j,n in enumerate(['BASE','PSD_ORDERED_CFM',A]):
    y=np.array([m['C2ST']['results'][n][v]['balanced_accuracy']for v in views]);ci=np.array([m['C2ST']['results'][n][v]['heldout_CI']['balanced_accuracy_95CI']for v in views])
    axs[0].errorbar(np.arange(10)+(j-1)*.15,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=3,capsize=2,color=colors[n],label=labels[n])
for n in ['RAW_REAL_REAL',R]:
    y=np.array([m['C2ST']['results'][n][v]['balanced_accuracy']if m['C2ST']['results'][n][v]['balanced_accuracy']is not None else np.nan for v in views]);axs[1].plot(range(10),y,'o-',color='#398347'if n=='RAW_REAL_REAL'else colors[n],label='Raw REAL-REAL'if n=='RAW_REAL_REAL'else labels[n],ms=3)
for j,(n,reference)in enumerate([(A,'BASE'),(R,'RAW REAL-REAL')]):
    pp=m['C2ST']['paired'][n];y=np.array([pp[v].get('BA_delta_first_minus_second',np.nan)for v in views]);ci=np.array([pp[v].get('BA_delta_95CI',[np.nan,np.nan])for v in views]);axs[2].errorbar(np.arange(10)+(j-.5)*.12,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=3,capsize=2,color=colors[n],label=labels[n]+' minus '+reference)
for ax in axs[:2]:ax.axhline(.5,color='gray',ls=':');ax.set_ylim(.43,1.02);ax.set_ylabel('fixed holdout BA')
axs[2].axhline(0,color='gray',lw=.8);axs[2].axhline(.1,color=colors[R],ls=':',lw=.8);axs[2].set_ylabel('paired BA difference')
for ax in axs:ax.set_xticks(range(10),short);ax.legend(fontsize=8,ncol=3);ax.grid(axis='y',alpha=.12)
axs[0].set_title('Source conversion: fixed 600/250 rows, 500 group resamples; lower is less distinguishable')
axs[1].set_title('Seen TRAIN-reference guide control; raw Q/PSD reference classifier unavailable (gap)')
axs[2].set_title('Intervals condition on the fitted classifiers; not training uncertainty or physical equivalence')
fig.tight_layout();save(fig,'c2st.png')

chartnames=['REAL','BASE',A,'TRAIN_REFERENCE',R,'SOURCE_CLEAN'];positive=np.concatenate([features[n+'__Qtotal_mVns'][features[n+'__Qtotal_mVns']>0]for n in chartnames]);xmin,xmax=positive.min(),positive.max()
fig,axs=plt.subplots(2,3,figsize=(13.8,7),sharex=True,sharey=True)
for ax,n in zip(axs.flat,chartnames):
    q=features[n+'__Qtotal_mVns'];p=features[n+'__PSD'];use=features[n+'__valid']&(q>0);ax.scatter(q[use],p[use],s=3,alpha=.27,color=colors[n]);ax.set_xscale('log');ax.set(xlim=(xmin*.95,xmax*1.05),ylim=(-.1,.6),title=labels[n]);ax.text(.02,.02,f'QC {use.sum()}; outside PSD view {int(np.sum(use&((p<-.1)|(p>.6))))}',transform=ax.transAxes,fontsize=7);ax.grid(alpha=.13)
for ax in axs[-1]:ax.set_xlabel('Q total (mV ns)')
for ax in axs[:,0]:ax.set_ylabel('signed PSD')
fig.suptitle('Unchanged full populations, common Q/PSD axes; no energy reweighting');fig.tight_layout();save(fig,'charge_psd.png')

sections=load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz');display=['REAL','BASE',A,'TRAIN_REFERENCE',R]
fig,axs=plt.subplots(2,4,figsize=(14,6.5),sharex=True,sharey=True)
for i,ax in enumerate(axs.flat):
    counts=[]
    for n in display:
        x=sections[n+'__bin'+str(i+1)+'__PSD'];counts.append(len(x))
        if len(x):line(ax,n,x,np.arange(1,len(x)+1)/len(x))
    ax.set(title=f'Observed-Q bin {i+1}',xlim=(-.1,.6),ylim=(0,1));ax.text(.02,.04,'N '+ '/'.join(map(str,counts)),transform=ax.transAxes,fontsize=6.5);ax.grid(alpha=.12)
    if i not in common:ax.text(.02,.9,'Source/BASE/REAL support below40',transform=ax.transAxes,fontsize=7,color='#555')
for ax in axs[1]:ax.set_xlabel('signed PSD')
for ax in axs[:,0]:ax.set_ylabel('CDF')
axs[0,0].legend(fontsize=6.8);fig.tight_layout();save(fig,'conditional_psd_CDF.png')

fig,axs=plt.subplots(2,3,figsize=(13,6.8),sharex=True);qmid=np.exp((edges[:-1]+edges[1:])/2)
for n in ['REAL','BASE',A,R]:
    for c in [0,1]:
        for j,f in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            y=[s['components'][c][f]if s['valid']else np.nan for s in width['tables'][n]['slices']];line(axs[c,j],n,qmid,y,marker='o',ms=3)
for c in [0,1]:
    for j,title in enumerate(['center','raw width','detrended width']):
        axs[c,j].set_xscale('log');axs[c,j].set(title=('low'if c==0 else'high')+' PSD component '+title,ylabel='PSD units');axs[c,j].grid(alpha=.12)
        if c:axs[c,j].set_xlabel('observed-Q midpoint (mV ns)')
axs[0,0].legend(fontsize=7);fig.suptitle('Two density components are not particle truth; invalid bins remain gaps');fig.tight_layout();save(fig,'energy_peak_width.png')

fig,axs=plt.subplots(2,4,figsize=(14,6.6),sharex=True)
for i,ax in enumerate(axs.flat):
    counts=[]
    for n in ['REAL','BASE',A]:
        use=features[n+'__valid']&(obs(n)==i);v=waves[n][use];counts.append(len(v))
        if len(v):line(ax,n,tt,np.median(v,axis=0)*1000);ax.plot(tt,np.mean(v,axis=0)*1000,ls=':',lw=.75,color=colors[n],alpha=.6)
    ax.set(title=f'Observed-Q bin {i+1}',xlim=(-120,359));ax.text(.02,.03,'N '+ '/'.join(map(str,counts)),transform=ax.transAxes,fontsize=7);ax.grid(alpha=.12)
for ax in axs[1]:ax.set_xlabel('relative to saved index120 (ns)')
for ax in axs[:,0]:ax.set_ylabel('raw voltage (mV)')
axs[0,0].legend(fontsize=7);fig.suptitle('Full 480 points: solid median, dotted mean; no smoothing or per-row rescaling');fig.tight_layout();save(fig,'conditional_waveforms.png')

fig,axs=plt.subplots(1,4,figsize=(14,4))
for ax,f,title,scale in zip(axs,['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V'],['baseline RMS (mV)','baseline lag1 ACF','prompt100:150 curvature (mV)','postpeak150:220 curvature (mV)'],[1000,1,1000,1000]):
    for n in display:cdf(ax,n,features[n+'__'+f][features[n+'__valid']],scale)
    ax.set(title=title,ylabel='CDF');ax.grid(alpha=.12)
axs[0].legend(fontsize=6.8);fig.tight_layout();save(fig,'noise_curvature.png')

names=['BASE','PSD_ORDERED_CFM',A];fig,axs=plt.subplots(1,4,figsize=(14.6,4.1));ticks=[];ticklabels=[]
for j,n in enumerate(names):
    for k,b in enumerate(['bin3','bin4']):
        row=primary[n][b];p=j*3+k;ticks.append(p);ticklabels.append(['BASE','ORDER','EDIT'][j]+str(k+3))
        axs[0].bar(p,row['output_PSD']['mean'],color=colors[n]);axs[1].bar(p,row['output_PSD']['sample_sd'],color=colors[n]);axs[2].bar(p,row['valley_fraction'],color=colors[n])
    r=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all'];axs[3].bar(j-.15,r['source_Q_to_output_Q']['Spearman'],.28,color=colors[n],alpha=.45);axs[3].bar(j+.15,r['source_PSD_to_output_PSD']['Spearman'],.28,color=colors[n])
for ax,title in zip(axs[:3],['Pure-had PSD mean','Pure-had PSD sample SD','Frozen valley fraction']):ax.set_title(title);ax.set_xticks(ticks,ticklabels,rotation=45,fontsize=7);ax.grid(axis='y',alpha=.12)
axs[3].set(title='Actual source rank: light Q, dark PSD',ylim=(0,1.05));axs[3].set_xticks(range(3),['BASE','ORDER','EDIT']);axs[3].grid(axis='y',alpha=.12)
fig.suptitle('Actual unADC source-Q bins3/4, N213/181; valley is not particle-error truth');fig.tight_layout();save(fig,'source_preservation.png')

boundary=load(ROOT/'results/BOUNDARY_DIFFERENCES_PRIVATE.npz');fig,axs=plt.subplots(3,3,figsize=(12.8,9));keys=list(bd['BASE'])
for ax,key in zip(axs.flat,keys):
    for n in ['REAL','BASE','INDEPENDENT_CFM',A]:cdf(ax,n,boundary[n+'__'+key],1000)
    ax.set(title=key.replace('_',' '),xlabel='native difference (mV)',ylabel='CDF');ax.grid(alpha=.12)
axs[0,0].legend(fontsize=7);fig.suptitle('Nine fixed boundaries; no searched positions');fig.tight_layout();save(fig,'boundary_differences.png')

actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');qsrc=features['SOURCE_INPUT__Qtotal_mVns'];qr=features['REAL__Qtotal_mVns'];fig,axs=plt.subplots(4,4,figsize=(15.5,10.8));pick={}
for row,parent in enumerate([0,500,1000,1999]):
    b=obs('SOURCE_INPUT')[parent];r=np.flatnonzero(features['REAL__valid']&(obs('REAL')==b));r=r if len(r)else np.flatnonzero(features['REAL__valid']);ri=int(r[np.argmin(abs(qr[r]-qsrc[parent]))]);pick[str(parent)]={'REAL_index':ri,'source_Q_mVns':float(qsrc[parent]),'REAL_Q_mVns':float(qr[ri]),'rule':'same source-Q bin then nearest REAL observedQ; unpaired reference'}
    for col,n in enumerate(['SOURCE_INPUT','BASE',A,'REAL']):
        idx=ri if n=='REAL'else parent;ax=axs[row,col];ax.plot(tt,waves[n][idx]*1000,color=colors[n],lw=.9);ax.set(title=f'Parent {parent}: '+labels[n],xlabel='relative ns',ylabel='raw mV');ax.grid(alpha=.13)
    limits=(min(ax.get_ylim()[0]for ax in axs[row]),max(ax.get_ylim()[1]for ax in axs[row]));[ax.set_ylim(limits)for ax in axs[row]]
fig.suptitle('Fixed source positions0/500/1000/1999, matched axes; REAL is not an event pair');fig.tight_layout();save(fig,'fixed_parent_examples.png');write('FIXED_EXAMPLES_PRIVATE.json',pick)

fig,axs=plt.subplots(3,3,figsize=(13.8,10.2));raw=load(ROOT/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz')
for j,(key,title)in enumerate([('mean_V','population mean'),('SD_V','population pointwise SD')]):
    for n in ['REAL','TRAIN_REFERENCE',R]:line(axs[0,j],n,tt,raw[n+'__None__'+key]*1000)
    axs[0,j].set(title='REAL guide control: '+title,xlabel='relative ns',ylabel='mV');axs[0,j].grid(alpha=.13)
for n in ['REAL','TRAIN_REFERENCE',R]:cdf(axs[0,2],n,features[n+'__PSD'][features[n+'__valid']])
axs[0,2].set(title='REAL guide control: full PSD',xlabel='signed PSD',ylabel='CDF');axs[0,0].legend(fontsize=7)
gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz');dv=gp['perturbed_output_preADC_V'].astype('f8')-gp['original_output_preADC_V'].astype('f8');dd=gp['actual_perturbed_guide_f32_V'].astype('f8')-gp['original_guide_V'].astype('f8')
axs[1,0].plot(tt,dd[0]*1000,color='#555',label='actual guide delta');axs[1,0].plot(tt,dv[0]*1000,color=colors[A],label='paired output delta');axs[1,0].set(title='First prespecified guide perturbation',xlabel='relative ns',ylabel='delta mV');axs[1,0].legend(fontsize=7)
rs=np.sqrt(np.mean(dv*dv,1));axs[1,1].hist(rs*1000,bins=18,color=colors[A]);axs[1,1].axvline(1000/16383,color='black',ls='--',label='1 ADC LSB');axs[1,1].set(title='Same c/bin/record noise, guide-only change',xlabel='preADC paired RMS (mV)',ylabel='parents');axs[1,1].legend(fontsize=7)
rows=m['source_probes'][A]['rows'];tags=list(rows);axs[1,2].bar(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[1,2].set(title='Full source-chain probes',ylabel='RMS to correct reference (mV)');axs[1,2].set_xticks(range(len(tags)),[t.replace('_','\n')for t in tags],fontsize=6.5)
terms=['base_signal','base_electronic','base_sum_rounding','flow_increment_f64'];joint=m['random_decomposition'][A]['groups']['joint8']['by_composition']
for j,cat in enumerate(['pure_EE','pure_had','mixed']):
    r=joint[cat];cov=r['four_additive_algebraic_components_covariance_V2']['full480'];vals=[cov[t+'__'+t]['mean']for t in terms]+[2*sum(cov[terms[k]+'__'+terms[l]]['mean']for k in range(4)for l in range(k+1,4))];axs[2,j].bar(range(5),np.asarray(vals)*1e6,color=[colors[A]]*4+['#555']);axs[2,j].set(title=cat+f': joint8 ({r["parents"]} parents)',ylabel='variance contribution (mV²)');axs[2,j].set_xticks(range(5),['baseS','baseE','round','edit','all2cov'],fontsize=7);axs[2,j].axhline(0,color='gray',lw=.7)
fig.suptitle('REAL control, guide dependence and additive algebra; physical edit/noise split unknown');fig.tight_layout();save(fig,'real_control_and_guide.png')

files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'figures':files,'figure_count':len(files),'all_plots_from_saved_results':True,'new_HGB_EM_response_updates_generation':0,
    'decision':gate['decision'],'retain':retain,'gates':{k:v['pass']for k,v in gates.items()},'actual_visual_QA_pending':True}
write('PLOT_SUMMARY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
