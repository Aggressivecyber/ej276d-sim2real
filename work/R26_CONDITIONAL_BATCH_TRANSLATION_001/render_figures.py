from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullLocator
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True);START=time.monotonic()
M=json.loads((ROOT/'metrics.json').read_text('utf8'));NAMES=['REAL','SOURCE_CLEAN','STANDARDIZED','HOMO_SELECTED','HOMO_FINAL','R23_D1_FINAL2950'];CAND=NAMES[2:]
COL=dict(zip(NAMES,['black','#999999','#e66101','#1f77b4','#2ca02c','#9467bd']));COL['HOMOGENEOUS_REORDER']=COL['HOMO_SELECTED']
LABEL=dict(zip(NAMES,['REAL','Source','D1 selected(0)','Homo selected','Homo final','D1 final(2950)']));LABEL['HOMOGENEOUS_REORDER']='Homogeneous new'
with np.load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz',allow_pickle=False)as z:waves={k:z[k].copy()for k in z.files}
with np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False)as z:feat={k:z[k].copy()for k in z.files}
with np.load(ROOT/'DEV_SOURCE_METADATA.npz',allow_pickle=False)as z:comp=z['composition'].copy()
edges=np.asarray(M['PSD_main']['edges_logQ']);centers=np.exp((edges[:-1]+edges[1:])/2);files=[]
def f(n,k):return feat[n+'__'+k]
def save(fig,n):fig.savefig(OUT/n,dpi=150,bbox_inches='tight');plt.close(fig);files.append(n)
def cdf(ax,a,n,mul=1,ls='-'):
    a=np.asarray(a)*mul;a=np.sort(a[np.isfinite(a)])
    if len(a):ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',color=COL[n],ls=ls,label=LABEL[n])

fig,axs=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True,constrained_layout=True)
qs=np.concatenate([f(n,'Qtotal_mVns')[f(n,'valid')]for n in NAMES]);ps=np.concatenate([f(n,'PSD')[f(n,'valid')]for n in NAMES]);qlim=(max(.1,float(qs.min())*.9),float(qs.max())*1.1)
for ax,n in zip(axs.flat,NAMES):
    good=f(n,'valid');q=f(n,'Qtotal_mVns');p=f(n,'PSD')
    if n=='REAL':ax.scatter(q[good],p[good],s=4,c='black',alpha=.4)
    else:
        for c,col in [('pure_EE','#377eb8'),('pure_had','#ff7f00'),('mixed','#4daf4a')]:
            use=good&(comp==c);ax.scatter(q[use],p[use],s=4,c=col,alpha=.4,label=c+': '+str(int(use.sum())))
        ax.legend(fontsize=7)
    ax.set_xscale('log');ax.set_xlim(qlim);ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_locator(NullLocator());ax.set_title(LABEL[n]+'; QC '+str(int(good.sum()))+'/2000');ax.set_xlabel('signed Qtotal [mV ns]');ax.set_ylabel('signed PSD');ax.grid(alpha=.15)
axs[0,0].set_ylim(min(-.1,float(ps.min())-.02),max(.6,float(ps.max())+.02));fig.suptitle('Complete original parents; common coordinates; no population reweighting');save(fig,'charge_psd.png')

fig,axs=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True,constrained_layout=True)
for i,ax in enumerate(axs.flat):
    cnt=[]
    for n in NAMES:
        q=f(n,'Qtotal_mVns');b=np.searchsorted(edges,np.log(np.maximum(q,1e-30)),side='right')-1;use=f(n,'valid')&(b==i);cdf(ax,f(n,'PSD')[use],n);cnt.append(str(int(use.sum())))
    ax.set_title('bin '+str(i+1)+'\nN '+('/'.join(cnt)),fontsize=9);ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.supxlabel('signed PSD');fig.supylabel('CDF, observed output Q bins');save(fig,'conditional_psd_CDF.png')

fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True);tables=M['energy_conditional_width']['tables']
for k in[0,1]:
    for j,key in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
        ax=axs[k,j]
        for n in NAMES:
            v=[r['components'][k][key]if r['valid']and len(r['components'])==2 else np.nan for r in tables[n]['slices']];ax.plot(centers,v,'o-',c=COL[n],label=LABEL[n])
        ax.set_xscale('log');ax.set_title(('low'if k==0 else'high')+' component: '+key,fontsize=10);ax.set_xlabel('observed Q [mV ns]');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7)
if M['energy_conditional_width'].get('delivery_unavailable'):
    fig.suptitle('New fit summaries lost before persistence: support UNKNOWN, no refit; old qualified tables only')
    axs[0,1].text(.05,.85,'New endpoint curves unavailable due to engineering failure',transform=axs[0,1].transAxes,fontsize=8)
else:fig.suptitle('Qualified descriptions only; gaps are failures, components are not particle truth')
save(fig,'energy_peak_width.png')

fig,axs=plt.subplots(2,4,figsize=(15,8),constrained_layout=True,sharex=True);sb=np.searchsorted(edges,np.log(np.maximum(f('SOURCE_CLEAN','Qtotal_mVns'),1e-30)),side='right')-1
for i,ax in enumerate(axs.flat):
    for n in NAMES:
        b=np.searchsorted(edges,np.log(np.maximum(f(n,'Qtotal_mVns'),1e-30)),side='right')-1 if n=='REAL'else sb;use=f(n,'valid')&(b==i)
        if use.any():
            lo,med,hi=np.quantile(waves[n][use]*1000,[.1,.5,.9],axis=0);t=np.arange(480);ax.plot(t,med,c=COL[n],label=LABEL[n]);ax.fill_between(t,lo,hi,color=COL[n],alpha=.08)
    ax.set_xlim(100,300);ax.set_title('source bin '+str(i+1)+'; REAL own bin',fontsize=9);ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.supxlabel('sample [ns]');fig.supylabel('mV; median and 10–90% bands');save(fig,'conditional_waveforms.png')

fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for n,tr in [('STANDARDIZED',M['control_training_reused']),('HOMOGENEOUS_REORDER',M['training']['HOMOGENEOUS_REORDER'])]:
    v=[r['validation']for r in tr['validation_trace']];s=[r['step']for r in v]
    for ax,key in [(axs[0,0],'J'),(axs[0,1],'fine_val'),(axs[0,2],'style_reconstruction_L1'),(axs[1,0],'online_D_weighted_MSE'),(axs[1,1],'online_D_weighted_BA')]:ax.plot(s,[r[key]for r in v],'o-',c=COL[n],label=LABEL[n])
    axs[0,0].scatter([tr['selected_step']],[tr['selected_validation']['J']],marker='*',s=150,c=COL[n]);auc=M['online_D_validation_AUC'][n];axs[1,2].plot([r['step']for r in auc],[r['weighted_AUC']for r in auc],'o-',c=COL[n],label=LABEL[n])
for ax,title in zip(axs.flat,['J20 (star=TRAIN-val choice)','fine validation term','style L1, not selection','online D weighted MSE','online D weighted BA','online D weighted AUC']):ax.set_title(title);ax.set_xlabel('actual G updates; both D1');ax.grid(alpha=.15);ax.legend(fontsize=8)
axs[1,0].axhline(.25,c='gray',ls='--');axs[1,1].axhline(.5,c='gray',ls='--');axs[1,2].axhline(.5,c='gray',ls='--');save(fig,'training_online_D.png')

fig,axs=plt.subplots(1,3,figsize=(19,5),constrained_layout=True);views=list(M['C2ST']['results']['HOMO_SELECTED']);xx=np.arange(len(views))
for off,n in zip([-.3,-.1,.1,.3],CAND):axs[0].bar(xx+off,[M['C2ST']['results'][n][v]['balanced_accuracy']for v in views],width=.2,color=COL[n],label=LABEL[n])
labs=[v.replace('full_historical_shape','full shape').replace('_local','').replace('full_voltage','full V')for v in views]
axs[0].axhline(.5,c='gray',ls='--');axs[0].set_ylim(.45,1.02);axs[0].set_xticks(xx,labs,rotation=40,ha='right');axs[0].set_ylabel('independent HGB holdout BA');axs[0].legend(fontsize=7)
for ax,ci,title,col in [(axs[1],M['C2ST']['endpoint_minus_R23_D1_SELECTED_paired_CI']['HOMO_SELECTED'],'Homo selected − D1 selected',COL['HOMO_SELECTED']),(axs[2],M['C2ST']['HOMO_FINAL_minus_R23_D1_FINAL_paired_CI'],'Homo final − D1 final',COL['HOMO_FINAL'])]:
    d=np.array([ci[v].get('BA_delta_first_minus_second',np.nan)for v in views]);lo=np.array([ci[v].get('BA_delta_95CI',[np.nan,np.nan])[0]for v in views]);hi=np.array([ci[v].get('BA_delta_95CI',[np.nan,np.nan])[1]for v in views]);ax.errorbar(xx,d,yerr=np.maximum(0,np.vstack([d-lo,hi-d])),fmt='o',c=col,capsize=3);ax.axhline(0,c='gray',ls='--');ax.set_xticks(xx,labs,rotation=40,ha='right');ax.set_title(title,fontsize=10);ax.set_ylabel('paired ΔBA; 500 group bootstrap')
fig.suptitle('Fixed classifiers and fixed heldout groups; no training uncertainty claim');save(fig,'c2st.png')

fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,key,label,mul in [(axs[0,0],'baseline_RMS_V','baseline RMS [mV]',1000),(axs[0,1],'ACF1','baseline lag1 ACF',1),(axs[0,2],'rise_10_90_ns','10–90% rise [ns]',1),(axs[1,0],'prompt_curvature_RMS_V','prompt 100:150 d2 RMS [mV]',1000),(axs[1,1],'postpeak_curvature_RMS_V','postpeak 150:220 d2 RMS [mV]',1000),(axs[1,2],'Qtotal_mVns','Qtotal [mV ns]',1)]:
    for n in NAMES:cdf(ax,f(n,key),n,mul)
    ax.set_xlabel(label);ax.set_ylabel('CDF');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);fig.suptitle('Prompt includes peak and early falling edge');save(fig,'noise_curvature.png')

fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True);ims=[];field='migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid';vmax=max(np.max(np.array(M['source_association'][n][field]))for n in CAND)
for ax,n in zip(axs.flat,CAND):
    a=np.array(M['source_association'][n][field]);im=ax.imshow(a,origin='lower',cmap='viridis',vmin=0,vmax=vmax);ims.append(im)
    for i in range(10):
        for j in range(10):
            if a[i,j]:ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=7,color='white'if a[i,j]<vmax*.5 else'black')
    labs=['low',*map(str,range(1,9)),'hi/invalid'];ax.set_xticks(range(10),labs,rotation=35);ax.set_yticks(range(10),labs);ax.set_xlabel('output Q bin');ax.set_ylabel('source Q bin');ax.set_title(LABEL[n])
fig.colorbar(ims[-1],ax=list(axs.flat),shrink=.8,label='all original parent counts');save(fig,'parent_migration.png')

fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True);tags=list(M['source_probes']['HOMO_SELECTED']['rows']);xx=np.arange(len(tags))
for off,n in zip([-.3,-.1,.1,.3],CAND):
    rows=M['source_probes'][n]['rows'];axs[0,0].barh(xx+off,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],height=.2,color=COL[n],label=LABEL[n])
axs[0,0].set_yticks(xx,[t.replace('_',' ')for t in tags],fontsize=8);axs[0,0].set_xlabel('RMS against correct reference [mV]');axs[0,0].legend(fontsize=7)
for n in CAND:
    r=M['random_decomposition'][n]['groups']['joint8']['rows'];a=np.sort([v['output_PSD']['sample_sd']for v in r if v['output_PSD']['sample_sd']is not None]);axs[0,1].step(a,np.arange(1,len(a)+1)/len(a),c=COL[n],label=LABEL[n])
axs[0,1].set_xlabel('same-parent PSD SD, original joint8');axs[0,1].set_ylabel('CDF of parents');axs[0,1].legend(fontsize=7)
xx=np.arange(2)
for off,n in zip([-.3,-.1,.1,.3],CAND):
    p=M['source_composition']['primary_pure_had_source_bins3_4'][n];axs[1,0].bar(xx+off,[p['bin3']['output_PSD']['mean'],p['bin4']['output_PSD']['mean']],width=.2,color=COL[n],label=LABEL[n]);axs[1,1].bar(xx+off,[p['bin3']['valley_fraction'],p['bin4']['valley_fraction']],width=.2,color=COL[n],label=LABEL[n])
for ax in axs[1]:ax.set_xticks(xx,['source bin3 N207','source bin4 N182']);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=7)
axs[1,0].set_ylabel('pure-had output PSD center');axs[1,0].set_ylim(.15,.35);axs[1,1].set_ylabel('TRAIN-frozen valley fraction');axs[1,1].set_ylim(0,1);save(fig,'source_controls_random.png')

fig,axs=plt.subplots(2,2,figsize=(14,8),constrained_layout=True);r=M['reorder_construction'];rows=r['bin_construction'];axs[0,0].bar(np.arange(1,9),[v['homogeneous_blocks']for v in rows]);axs[0,0].set_title(str(r['homogeneous_blocks'])+' homogeneous + '+str(r['mixed_blocks'])+' mixed blocks');axs[0,0].set_xlabel('source/REAL common Q bin');axs[0,0].set_ylabel('32-row homogeneous blocks');axs[0,1].bar(np.arange(1,9),[v['remainder']for v in rows]);axs[0,1].set_title('All remainders retained, then block shuffle');axs[0,1].set_xlabel('Q bin');axs[0,1].set_ylabel('remainder rows')
with np.load(ROOT/'HOMOGENEOUS_REORDER/GRADIENT_NORMS_PRIVATE.npz',allow_pickle=False)as z:a=z['values'].copy()
for j,label in[(0,'G+reader'),(1,'D')]:axs[1,0].plot(np.arange(1,len(a)+1),a[:,j],lw=.5,alpha=.6,label=label);axs[1,1].plot(np.arange(1,len(a)+1),a[:,j+2],lw=.5,alpha=.6,label=label)
axs[1,0].set_yscale('log');axs[1,0].axhline(5,c='gray',ls='--');axs[1,0].set_ylabel('pre-clip global norm');axs[1,1].set_ylabel('actual clip scale');axs[1,1].set_ylim(0,1.05)
for ax in axs[1]:ax.set_xlabel('actual G update');ax.legend();ax.grid(alpha=.15)
fig.suptitle('Complete original bound-row multiset; changed order and batch context');save(fig,'batch_gradient_context.png')

receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'new_fit_generation_or_update':0,'only_saved_results_used':True};(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
