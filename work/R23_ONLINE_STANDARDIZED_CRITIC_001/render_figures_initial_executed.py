from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullLocator
import evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True);START=time.monotonic()
M=json.loads((ROOT/'metrics.json').read_text('utf8'));NAMES=['REAL','SOURCE_CLEAN','R21_GLOBAL_STYLE','RAW','STANDARDIZED'];COL={'REAL':'black','SOURCE_CLEAN':'#8b929a','R21_GLOBAL_STYLE':'#9467bd','RAW':'#1f77b4','STANDARDIZED':'#e66101'}
with np.load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz',allow_pickle=False)as z:waves={k:z[k].copy()for k in z.files}
with np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False)as z:feat={k:z[k].copy()for k in z.files}
with np.load(ROOT/'DEV_SOURCE_METADATA.npz',allow_pickle=False)as z:comp=z['composition'].copy()
edges=np.asarray(M['PSD_main']['edges_logQ']);x=np.exp((edges[:-1]+edges[1:])/2);files=[]
def f(name,key):return feat[name+'__'+key]
def save(fig,name):fig.savefig(OUT/name,dpi=150,bbox_inches='tight');plt.close(fig);files.append(name)
def cdf(ax,v,name,color=None,scale=1):
    a=np.asarray(v)*scale;a=np.sort(a[np.isfinite(a)])
    if len(a):ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',color=color or COL[name],label=name)

fig,axs=plt.subplots(1,5,figsize=(18,4),sharex=True,sharey=True,constrained_layout=True)
qs=np.concatenate([f(n,'Qtotal_mVns')[f(n,'valid')]for n in NAMES]);ps=np.concatenate([f(n,'PSD')[f(n,'valid')]for n in NAMES]);qlim=(max(.1,float(qs.min())*.9),float(qs.max())*1.1)
for ax,n in zip(axs,NAMES):
    good=f(n,'valid');q=f(n,'Qtotal_mVns');p=f(n,'PSD')
    if n=='REAL':ax.scatter(q[good],p[good],s=3,c='black',alpha=.4)
    else:
        for c,col in [('pure_EE','#377eb8'),('pure_had','#ff7f00'),('mixed','#4daf4a')]:
            use=good&(comp==c);ax.scatter(q[use],p[use],s=4,c=col,alpha=.4,label=c+': '+str(int(use.sum())))
        ax.legend(fontsize=7)
    ax.set_xscale('log');ax.set_xlim(qlim);ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_locator(NullLocator());ax.set_title(n+'\nQC '+str(int(good.sum()))+'/2000',fontsize=10);ax.set_xlabel('Qtotal [mV ns]');ax.grid(alpha=.15)
axs[0].set_ylim(min(-.1,float(ps.min())-.02),max(.6,float(ps.max())+.02));axs[0].set_ylabel('signed PSD');save(fig,'charge_psd.png')

fig,axs=plt.subplots(2,4,figsize=(14,8),sharex=True,sharey=True,constrained_layout=True)
for i,ax in enumerate(axs.flat):
    cnt=[]
    for n in NAMES:
        q=f(n,'Qtotal_mVns');b=np.searchsorted(edges,np.log(np.maximum(q,1e-30)),side='right')-1;use=f(n,'valid')&(b==i);cdf(ax,f(n,'PSD')[use],n);cnt.append(str(int(use.sum())))
    ax.set_title('bin '+str(i+1)+'\nN '+('/'.join(cnt)),fontsize=9);ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.supxlabel('signed PSD');fig.supylabel('CDF, observed output Q bins; original population');save(fig,'conditional_psd_CDF.png')

fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
tables=M['energy_conditional_width']['tables']
for k in [0,1]:
    for j,key in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
        ax=axs[k,j]
        for n in NAMES:
            v=[r['components'][k][key]if r['valid']and len(r['components'])==2 else np.nan for r in tables[n]['slices']];ax.plot(x,v,'o-',c=COL[n],label=n)
        ax.set_xscale('log');ax.set_title(('low'if k==0 else'high')+' density component: '+key,fontsize=10);ax.set_xlabel('observed Q [mV ns]');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.suptitle('Only qualified descriptions; gaps retain failure, no particle labels');save(fig,'energy_peak_width.png')

fig,axs=plt.subplots(2,4,figsize=(14,7),constrained_layout=True,sharex=True)
sb=np.searchsorted(edges,np.log(np.maximum(f('SOURCE_CLEAN','Qtotal_mVns'),1e-30)),side='right')-1
for i,ax in enumerate(axs.flat):
    for n in NAMES:
        b=np.searchsorted(edges,np.log(np.maximum(f(n,'Qtotal_mVns'),1e-30)),side='right')-1 if n=='REAL'else sb;use=f(n,'valid')&(b==i)
        if use.any():
            lo,med,hi=np.quantile(waves[n][use]*1000,[.1,.5,.9],axis=0);t=np.arange(480);ax.plot(t,med,c=COL[n],label=n);ax.fill_between(t,lo,hi,color=COL[n],alpha=.10)
    ax.set_xlim(100,300);ax.set_title('source bin '+str(i+1)+'; REAL own bin',fontsize=9);ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.supxlabel('sample [ns]');fig.supylabel('mV; median and 10–90% band');save(fig,'conditional_waveforms.png')

fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for n in ['RAW','STANDARDIZED']:
    tr=M['training'][n];v=[r['validation']for r in tr['validation_trace']];s=[r['step']for r in v]
    for ax,key in [(axs[0,0],'J'),(axs[0,1],'fine_val'),(axs[0,2],'style_reconstruction_L1'),(axs[1,0],'online_D_weighted_MSE'),(axs[1,1],'online_D_weighted_BA')]:ax.plot(s,[r[key]for r in v],'o-',c=COL[n],label=n)
    axs[0,0].scatter([tr['selected_step']],[tr['selected_validation']['J']],marker='*',s=140,c=COL[n]);auc=M['online_D_validation_AUC'][n];axs[1,2].plot([r['step']for r in auc],[r['weighted_AUC']for r in auc],'o-',c=COL[n],label=n)
titles=['common J20 (star=selected)','fine validation term','style L1, not selection','online D weighted MSE','online D weighted BA','online D weighted AUC, REAL=1']
for ax,title in zip(axs.flat,titles):ax.set_title(title,fontsize=10);ax.set_xlabel('actual G/D updates');ax.grid(alpha=.15);ax.legend(fontsize=8)
axs[1,0].axhline(.25,c='gray',ls='--');axs[1,1].axhline(.5,c='gray',ls='--');axs[1,2].axhline(.5,c='gray',ls='--');save(fig,'training_online_D.png')

fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True);views=list(M['C2ST']['results']['RAW']);xx=np.arange(len(views))
for off,n in [(-.24,'R21_GLOBAL_STYLE'),(0,'RAW'),(.24,'STANDARDIZED')]:
    r=M['C2ST']['reference_results_reused'][n]if n=='R21_GLOBAL_STYLE'else M['C2ST']['results'][n]
    axs[0].bar(xx+off,[r[v]['balanced_accuracy']for v in views],width=.24,color=COL[n],label=n)
axs[0].axhline(.5,c='gray',ls='--');axs[0].set_ylim(.45,1.02);axs[0].set_xticks(xx,[v.replace('full_historical_shape','full shape').replace('_local','').replace('full_voltage','full V')for v in views],rotation=40,ha='right');axs[0].set_ylabel('independent HGB holdout BA');axs[0].legend(fontsize=8)
ci=M['C2ST']['STANDARDIZED_minus_RAW_paired_CI'];d=np.array([ci[v]['BA_delta_first_minus_second']for v in views]);lo=np.array([ci[v]['BA_delta_95CI'][0]for v in views]);hi=np.array([ci[v]['BA_delta_95CI'][1]for v in views]);axs[1].errorbar(xx,d,yerr=np.maximum(0,np.vstack([d-lo,hi-d])),fmt='o',c=COL['STANDARDIZED']);axs[1].axhline(0,c='gray',ls='--');axs[1].set_xticks(xx,[v.replace('full_historical_shape','full shape').replace('_local','')for v in views],rotation=40,ha='right');axs[1].set_ylabel('STANDARDIZED minus RAW BA\n500 paired group bootstrap, fixed classifiers');save(fig,'c2st.png')

fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,key,label,mul in [(axs[0,0],'baseline_RMS_V','baseline RMS [mV]',1000),(axs[0,1],'ACF1','baseline lag1 ACF',1),(axs[0,2],'rise_10_90_ns','10–90% rise [ns]',1),(axs[1,0],'prompt_curvature_RMS_V','prompt 100:150 d2 RMS [mV]',1000),(axs[1,1],'postpeak_curvature_RMS_V','postpeak 150:220 d2 RMS [mV]',1000),(axs[1,2],'Qtotal_mVns','Qtotal [mV ns]',1)]:
    for n in NAMES:cdf(ax,f(n,key),n,scale=mul)
    ax.set_xlabel(label);ax.set_ylabel('CDF');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.suptitle('Prompt includes peak and early fall, not solely rising edge');save(fig,'noise_curvature.png')

fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
tags=list(M['source_probes']['RAW']['rows']);xx=np.arange(len(tags))
for off,n in [(-.18,'RAW'),(.18,'STANDARDIZED')]:
    rows=M['source_probes'][n]['rows'];axs[0,0].barh(xx+off,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],height=.36,color=COL[n],label=n)
axs[0,0].set_yticks(xx,[t.replace('_',' ')for t in tags],fontsize=8);axs[0,0].set_xlabel('RMS against correct source/noise reference [mV]');axs[0,0].legend(fontsize=8)
for n in ['RAW','STANDARDIZED']:
    r=M['random_decomposition'][n]['groups']['joint8']['rows']
    for kind,ls in [('pure_EE','-'),('pure_had','--'),('mixed',':')]:
        a=[v['output_PSD']['sample_sd']for v in r if v['composition']==kind and v['output_PSD']['sample_sd']is not None];a=np.sort(a)
        if len(a):axs[0,1].step(a,np.arange(1,len(a)+1)/len(a),c=COL[n],ls=ls,label=n+' '+kind)
axs[0,1].set_xlabel('same-parent PSD SD, original joint8');axs[0,1].set_ylabel('CDF of parents');axs[0,1].legend(fontsize=7)
xx=np.arange(3)
for off,n in [(-.18,'RAW'),(.18,'STANDARDIZED')]:
    de=M['random_decomposition'][n]['groups']['joint8']['by_composition'];kinds=['pure_EE','pure_had','mixed']
    val=[de[k]['within_parent_waveform_RMS_V']['signal_pre_ADC']['full480']['mean']*1000 if k in de and de[k]['within_parent_waveform_RMS_V']['signal_pre_ADC']['full480']['mean']is not None else np.nan for k in kinds];axs[1,0].bar(xx+off,val,width=.36,color=COL[n],label=n)
axs[1,0].set_xticks(xx,kinds);axs[1,0].set_ylabel('signal branch parent RMS SD [mV]');axs[1,0].set_title('Functional signal is not conditional expectation')
for n in ['R21_GLOBAL_STYLE','RAW','STANDARDIZED']:
    p=M['source_composition']['primary_pure_had_source_bins3_4'][n];axs[1,1].plot([3,4],[p['bin3']['output_PSD']['mean'],p['bin4']['output_PSD']['mean']],'o-',c=COL[n],label=n)
    for b in [3,4]:
        frac=p['bin'+str(b)]['valley_fraction'];mean=p['bin'+str(b)]['output_PSD']['mean']
        if mean is not None:axs[1,1].annotate('valley '+(format(frac,'.1%')if frac is not None else'undefined'),(b,mean),xytext=(4,8 if n=='RAW'else -12),textcoords='offset points',fontsize=7,c=COL[n])
axs[1,1].set_xticks([3,4]);axs[1,1].set_xlabel('prespecified pure-had source bins; N=207/182');axs[1,1].set_ylabel('mean output PSD');axs[1,1].legend(fontsize=7);save(fig,'source_controls_random.png')

fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);ims=[]
for ax,n in zip(axs,['RAW','STANDARDIZED']):
    a=np.array(M['source_association'][n]['migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid']);im=ax.imshow(a,origin='lower',cmap='viridis',vmin=0,vmax=max(np.max(np.array(M['source_association'][k]['migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid']))for k in ['RAW','STANDARDIZED']));ims.append(im)
    for i in range(10):
        for j in range(10):
            if a[i,j]:ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=7,color='white'if a[i,j]<im.norm.vmax*.5 else'black')
    labs=['low',*map(str,range(1,9)),'hi/invalid'];ax.set_xticks(range(10),labs,rotation=35);ax.set_yticks(range(10),labs);ax.set_xlabel('output Q bin');ax.set_ylabel('source Q bin');ax.set_title(n)
fig.colorbar(ims[-1],ax=axs,shrink=.8,label='all original parent counts');save(fig,'parent_migration.png')

receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'new_fit_generation_or_update':0,'only_saved_results_used':True}
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
