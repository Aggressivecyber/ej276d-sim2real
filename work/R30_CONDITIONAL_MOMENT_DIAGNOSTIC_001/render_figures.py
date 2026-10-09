from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='4'
from pathlib import Path
from datetime import datetime,timezone
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
START=time.monotonic()
M=json.loads((ROOT/'METRICS_COMPACT.json').read_text('utf8'));C=M['classification'];P=M['moment_parameters'];D=M['descriptive_moments'];COV=M['holdout_covariance']
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
VIEWS=['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE']
VIEW_LABELS=['Raw u','Global center','Q-bin center','Q-bin center + scale']
NAMES=['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL','TRAIN_REFERENCE']
LABELS=['Fresh selected2250','Fresh final2950','Energy selected2750','Energy final2950','REAL TRAIN–DEV']
COLORS=['#0072B2','#56B4E9','#D55E00','#E69F00','#009E73']
DOMAIN_NAMES=['REAL','TRAIN_REFERENCE','FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL']
DOMAIN_LABELS=['REAL DEV','REAL TRAIN reference',*LABELS[:4]]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
SAVED=[]
def finish(fig,name,title):
    fig.suptitle(title,fontsize=14)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig);SAVED.append(name+'.png')
def err(ax,x,r,color):
    p=r['balanced_accuracy'];ci=r['heldout_CI']['balanced_accuracy_95CI']
    ax.errorbar(x,p,yerr=[[max(0,p-ci[0])],[max(0,ci[1]-p)]],color=color,fmt='o',capsize=3,ms=5)

fig,axes=plt.subplots(1,5,figsize=(17,4.3),layout='constrained',sharey=True)
for i,(n,ax) in enumerate(zip(NAMES,axes)):
    r=C['comparisons'][n]
    for j,v in enumerate(VIEWS):err(ax,j,r[v],COLORS[i])
    ax.plot(range(4),[r[v]['balanced_accuracy'] for v in VIEWS],color=COLORS[i],lw=1)
    ax.axhline(.5,color='.5',ls=':',lw=1);ax.set_ylim(.4,1.035);ax.set_xticks(range(4),['Raw','Global','Q-center','Q+scale'],rotation=45,ha='right')
    ax.set_title(LABELS[i]);ax.grid(axis='y',alpha=.18)
    if i==4:ax.text(.03,.06,'Transform control\nraw BA = 0.510',transform=ax.transAxes,color=COLORS[i])
axes[0].set_ylabel('Fixed holdout balanced accuracy (95% group bootstrap CI)')
finish(fig,'c2st_moment_views','Subtracting domain moments does not remove separability; REAL–REAL control also shifts')

fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained',sharey=True)
for j,v in enumerate(VIEWS[1:]):
    ax=axes[j]
    for i,n in enumerate(NAMES):
        p=C['comparisons'][n][v]['paired_vs_raw'];val=p['BA_delta_first_minus_second'];ci=p['BA_delta_95CI']
        ax.errorbar(val,i,xerr=[[max(0,val-ci[0])],[max(0,ci[1]-val)]],fmt='o',color=COLORS[i],capsize=4)
    ax.axvline(0,color='.5',ls=':',lw=1);ax.set_title(VIEW_LABELS[j+1]);ax.set_xlabel('Transformed − raw BA, paired 95% CI');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(range(5),LABELS);axes[0].invert_yaxis()
finish(fig,'paired_BA_changes','Paired fixed-classifier changes on the same 250 + 250 holdout events')

counts=np.asarray([P['domains'][n]['train_Qbin_counts'] for n in DOMAIN_NAMES])
fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');im=ax.imshow(counts,cmap='Blues',aspect='auto',vmin=0)
for i in range(6):
    for j in range(8):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>counts.max()*.58 else 'black')
support=P['six_domains_common_supported_bins_1based']
ax.set_xticks(range(8),[str(i+1)+(' ✓' if i+1 in support else ' fallback') for i in range(8)]);ax.set_yticks(range(6),DOMAIN_LABELS)
ax.set_xlabel('Frozen observed-charge bin (six-domain train N ≥ 20)');fig.colorbar(im,ax=ax,label='Classifier training events')
finish(fig,'train_conditional_support','One common conditional support set; unsupported/outside events use global moments')

with np.load(ROOT/'results/FIGURE_DATA_PRIVATE.npz',allow_pickle=False) as z:curves={k:z[k].copy() for k in z.files}
t=np.arange(480)
fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
for ax,v,lab in zip(axes.flat,VIEWS,VIEW_LABELS):
    ref=curves['REAL__holdout__'+v+'__mean']
    for i,n in enumerate(NAMES):
        mean=curves[n+'__holdout__'+v+'__mean'];ax.plot(t,mean-ref,color=COLORS[i],lw=1,ls='--' if n=='TRAIN_REFERENCE' else '-',label=LABELS[i])
    ax.axhline(0,color='.5',lw=.7);ax.set_title(lab);ax.set_xlabel('Saved sample index (480 samples; reference index120)');ax.set_ylabel('Holdout domain mean − REAL mean (u units)' if v!='Q_CENTER_SCALE' else 'Mean residual in domain SD units');ax.grid(alpha=.15)
axes[0,0].legend(ncol=2,fontsize=8,loc='best')
finish(fig,'holdout_mean_residuals','Descriptive holdout means: transformation parameters were fitted only on 600 training rows')

fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
for ax,v,lab in zip(axes.flat,VIEWS,VIEW_LABELS):
    ax.plot(t,curves['REAL__holdout__'+v+'__SD'],color='black',lw=1.5,label='REAL DEV')
    for i,n in enumerate(NAMES):ax.plot(t,curves[n+'__holdout__'+v+'__SD'],color=COLORS[i],lw=1,ls='--' if n=='TRAIN_REFERENCE' else '-',label=LABELS[i])
    ax.set_title(lab);ax.set_xlabel('Saved sample index');ax.set_ylabel('Pointwise sample SD (u)' if v!='Q_CENTER_SCALE' else 'Pointwise sample SD (scaled units)');ax.grid(alpha=.15)
axes[0,0].legend(ncol=2,fontsize=8,loc='best')
finish(fig,'holdout_pointwise_SD','Pointwise SD is not joint covariance; zero-centered / scaled training does not fix holdout geometry')

fig,axes=plt.subplots(2,2,figsize=(13,7.8),layout='constrained')
for ax,v,lab in zip(axes.flat,VIEWS,VIEW_LABELS):
    values=np.array([[np.nan if not r['qualified'] else r['balanced_accuracy'] for r in C['by_original_Qbin'][n][v]] for n in NAMES])
    cm=plt.get_cmap('viridis').copy();cm.set_bad('#dddddd');im=ax.imshow(np.ma.masked_invalid(values),cmap=cm,vmin=.4,vmax=1,aspect='auto')
    for i in range(5):
        for j in range(8):
            rr=C['by_original_Qbin'][NAMES[i]][v][j]
            s=f'{values[i,j]:.3f}' if np.isfinite(values[i,j]) else 'n/a'
            ax.text(j,i,s,ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[i,j]) and values[i,j]<.8 else 'black')
    ax.set_title(lab);ax.set_xticks(range(8),range(1,9));ax.set_xlabel('Original observed-charge bin');ax.set_yticks(range(5),['Fresh sel','Fresh final','Energy sel','Energy final','REAL–REAL'])
fig.colorbar(im,ax=list(axes.flat),label='Holdout BA, both domain N ≥ 10; no regional fit',shrink=.8)
finish(fig,'conditional_holdout_BA','All comparisons retain the same rows; common reporting bins1–6 are determined by counts')

fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
covviews=['RAW','Q_CENTER','Q_CENTER_SCALE'];covlabels=['Raw','Q-center','Q+scale'];x=np.arange(5);w=.22
for ax,key,title in zip(axes,['trace','effective_rank','relative_Frobenius_difference_to_REAL'],['Covariance trace','Effective rank','Relative Frobenius difference to REAL']):
    for j,v in enumerate(covviews):ax.bar(x+(j-1)*w,[COV[n][v][key] for n in NAMES],w,label=covlabels[j],color=['#8c8c8c','#0072B2','#D55E00'][j])
    ax.set_xticks(x,['Fresh\nselected','Fresh\nfinal','Energy\nselected','Energy\nfinal','REAL–\nREAL']);ax.set_title(title);ax.grid(axis='y',alpha=.15)
    if key=='trace':ax.set_yscale('log');ax.set_ylabel('Trace, representation-dependent units')
axes[0].legend()
finish(fig,'holdout_covariance','250-row sample covariance; high-dimensional descriptive summaries, not a unique cause')

fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
regions=list(CFG['regions'])
for j,v in enumerate(['RAW','Q_CENTER','Q_CENTER_SCALE']):
    raw=np.array([[D[n]['holdout'][v]['regions'][r]['mean_residual_RMS_u'] for r in regions] for n in NAMES])
    variance=np.array([[D[n]['holdout'][v]['regions'][r]['pointwise_variance_mean_u2']/D['REAL']['holdout'][v]['regions'][r]['pointwise_variance_mean_u2'] for r in regions] for n in NAMES])
    for row,values,title in [(0,raw,'Mean residual RMS'),(1,variance,'Pointwise variance ratio to REAL')]:
        ax=axes[row,j];im=ax.imshow(values,cmap='magma' if row==0 else 'cividis',aspect='auto')
        for i in range(5):
            for k in range(4):ax.text(k,i,f'{values[i,k]:.3g}',ha='center',va='center',fontsize=8,color='white' if values[i,k]<values.max()*.55 else 'black')
        ax.set_xticks(range(4),['base\n0:100','prompt\n100:150','postpeak\n150:220','tail\n220:400']);ax.set_yticks(range(5),['Fresh sel','Fresh final','Energy sel','Energy final','REAL–REAL']);ax.set_title(v.replace('_',' ')+'\n'+title);fig.colorbar(im,ax=ax,shrink=.75)
finish(fig,'regional_moment_summary','Holdout descriptive regions; prompt contains the peak and early descent')

receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':SAVED,
         'saved_aggregate_arrays_only':True,'response_generation':0,'new_classifier_fits':0,'new_EM':0,'scientific_figures_not_AIGeneration':True}
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
