import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
from datetime import datetime,timezone
import time,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True);START=time.monotonic()
M=json.loads((ROOT/'METRICS_COMPACT.json').read_text('utf8'));C=M['classification'];V=['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE']
VL=['Raw','Global center','Q-bin center','Q-bin center + scale'];N=['FRESH_FINAL','ENERGY_FINAL','TRAIN_REFERENCE'];NL=['R28 Fresh final2950','R29 Energy final2950','REAL TRAIN–DEV'];COL=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
saved=[]
def end(fig,name,title):
    fig.suptitle(title,fontsize=14);fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig);saved.append(name+'.png')
def error(ax,x,p,ci,color):ax.errorbar(x,p,yerr=[[max(0,p-ci[0])],[max(0,ci[1]-p)]],fmt='o',color=color,capsize=3)
fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharey=True,layout='constrained')
for i,n in enumerate(N):
    ax=axes[i]
    for j,v in enumerate(V):
        r=C['comparisons'][n][v];old=r['full480_same_view_reused'];error(ax,j-.04,r['balanced_accuracy'],r['heldout_CI']['balanced_accuracy_95CI'],COL[i]);ax.plot(j+.08,old['balanced_accuracy'],'x',color='.35',ms=7)
    ax.plot(range(4),[C['comparisons'][n][v]['balanced_accuracy'] for v in V],color=COL[i],lw=1,label='430 retained points')
    ax.plot(range(4),[C['comparisons'][n][v]['full480_same_view_reused']['balanced_accuracy'] for v in V],color='.5',ls='--',lw=1,label='Saved complete480')
    ax.axhline(.5,color='.5',ls=':',lw=1);ax.set_ylim(.4,1.035);ax.set_xticks(range(4),['Raw','Global','Q-center','Q+scale'],rotation=35,ha='right');ax.set_title(NL[i]);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Fixed holdout BA (430-point 95% group CI)');axes[2].legend(loc='lower right',fontsize=9)
end(fig,'complete_and_masked_C2ST','One fixed prompt exclusion: full480 scores retained; no complete-waveform acceptance')

fig,axes=plt.subplots(1,4,figsize=(15,4.2),sharey=True,layout='constrained')
for j,v in enumerate(V):
    ax=axes[j]
    for i,n in enumerate(N):
        r=C['comparisons'][n][v]['paired430_minus480'];d=r['BA_delta_first_minus_second'];ci=r['BA_delta_95CI']
        ax.errorbar(d,i,xerr=[[max(0,d-ci[0])],[max(0,ci[1]-d)]],fmt='o',color=COL[i],capsize=3)
    ax.axvline(0,color='.5',ls=':');ax.set_title(VL[j]);ax.set_xlabel('430 − saved480 BA, paired 95% CI');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(3),['Fresh final','Energy final','REAL–REAL']);axes[0].invert_yaxis()
end(fig,'paired_prompt_exclusion','Pairing uses identical original holdout events and artificial groups')

fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharey=True,layout='constrained')
for j,v in enumerate(V[1:]):
    ax=axes[j];ax.axvspan(-.10,.10,color='#dddddd',alpha=.55,label='Prespecified diagnostic tolerance')
    for i,n in enumerate(N):
        p=C['comparisons'][n][v]['paired430_transformed_minus_RAW430'];d=p['BA_delta_first_minus_second'];ci=p['BA_delta_95CI']
        ax.errorbar(d,i,xerr=[[max(0,d-ci[0])],[max(0,ci[1]-d)]],fmt='o',color=COL[i],capsize=3)
    ax.axvline(0,color='.5',ls=':');ax.set_title(VL[j+1]+'\nREAL–REAL control: FAIL');ax.set_xlabel('Transformed430 − raw430 BA (paired 95% CI)');ax.set_xlim(-.14,.25);ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(range(3),['Fresh final','Energy final','REAL–REAL']);axes[0].invert_yaxis()
end(fig,'control_tolerance','Raw430 REAL–REAL CI includes0.5, but all transformed control intervals exceed ±0.10')

with np.load(ROOT/'results/DECODER_PRIVATE.npz',allow_pickle=False) as z:dec={k:z[k].copy() for k in z.files}
fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True,layout='constrained')
for i,n in enumerate(N):
    ax=axes[i];a=np.maximum(dec[n+'__error_candidate'],1e-16);b=np.maximum(dec[n+'__error_REAL'],1e-16);y=dec[n+'__label'];ab=dec[n+'__abstain']
    for value,color,lab in [(1,COL[i],'Candidate'),(0,'#444444','REAL DEV')]:
        ix=(y==value)&~ab;ax.scatter(a[ix],b[ix],s=14,alpha=.45,color=color,label=lab)
    if ab.any():ax.scatter(a[ab],b[ab],s=14,color='.7',marker='x',label='Abstain')
    ax.plot([1e-16,.2],[1e-16,.2],color='.6',ls=':');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(3e-17,.2);ax.set_ylim(3e-17,.2);ax.set_title(NL[i]+'\nNo-fit BA1.000, coverage100%');ax.set_xlabel('|max(w + candidate mean) − 1|');ax.legend(fontsize=8,loc='best');ax.grid(alpha=.15)
axes[0].set_ylabel('|max(w + REAL mean) − 1|')
end(fig,'inverse_mean_decoder','Known normalization geometry recovers domain identity; zero errors displayed at1e−16')

with np.load(ROOT/'results/GEOMETRY_PRIVATE.npz',allow_pickle=False) as z:hist={n:z[n+'__argmax_histogram'].copy() for n in ['REAL','TRAIN_REFERENCE','FRESH_FINAL','ENERGY_FINAL']}
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,limits in zip(axes,[(0,479),(80,180)]):
    ax.axvspan(100,150,color='.85',label='Fixed excluded prompt100:150')
    for n,col,lab in zip(hist,['black','#009E73','#0072B2','#D55E00'],['REAL DEV','REAL TRAIN reference','Fresh final','Energy final']):ax.plot(np.arange(480),hist[n],lw=1.2,color=col,label=lab)
    ax.set_xlim(*limits);ax.set_xlabel('Original full480 argmax sample index');ax.set_ylabel('All saved2000 events');ax.grid(alpha=.15)
axes[0].set_title('Complete index range; no events filtered');axes[1].set_title('Display zoom only, not a second feature mask');axes[1].legend(fontsize=8)
end(fig,'normalization_peak_geometry','Normalization constraint applicability and peak location use the original complete waveforms')

fig,axes=plt.subplots(2,2,figsize=(12,6.3),layout='constrained')
for ax,v,lab in zip(axes.flat,V,VL):
    mat=np.array([[r['balanced_accuracy'] if r['qualified'] else np.nan for r in C['by_original_Qbin'][n][v]] for n in N]);cm=plt.get_cmap('viridis').copy();cm.set_bad('#dddddd')
    im=ax.imshow(np.ma.masked_invalid(mat),vmin=.4,vmax=1,cmap=cm,aspect='auto')
    for i in range(3):
        for j in range(8):ax.text(j,i,f'{mat[i,j]:.3f}' if np.isfinite(mat[i,j]) else 'n/a',ha='center',va='center',fontsize=8,color='white' if np.isfinite(mat[i,j]) and mat[i,j]<.8 else 'black')
    ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(3),['Fresh final','Energy final','REAL–REAL']);ax.set_xlabel('Original observed-charge bin');ax.set_title(lab+'430')
fig.colorbar(im,ax=list(axes.flat),label='BA only when both holdout N ≥10; no extra regional fit',shrink=.8)
end(fig,'conditional_masked_C2ST','Counts determine regional support; all500 holdout events remain in overall scores')

receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':saved,'new_classifier_fits':0,'new_generation':0,'saved_result_only':True}
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
