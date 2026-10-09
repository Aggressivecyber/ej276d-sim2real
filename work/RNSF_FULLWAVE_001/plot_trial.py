"""Plots of frozen results; no fitting or model selection."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'REAL':'#202b38','R3B':'#c17518','NSF':'#137d93','REAL_REAL':'#687487'}


def plots(domains,m,root):
    out=root/'results/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=155,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,name in zip(axes,['source','real','transport']):
        tr=json.loads((root/'results'/f'{name}_training.json').read_text())
        ax.plot([p['step'] for p in tr['trace']],[p['validation_nats_per_dim'] for p in tr['trace']],color=COLORS['NSF'])
        ax.axvline(tr['selected_update'],ls='--',color=COLORS['R3B'],label=f"Selected {tr['selected_update']}")
        ax.set(title=name.capitalize(),xlabel='Update',ylabel='Validation nats / dimension');ax.legend()
    fig.suptitle('Training selections: TRAIN validation only',y=1.04)
    save(fig,'01_training')
    if m['C2ST']['supported']:
        cs=m['C2ST']['results'];views=list(cs['NSF']);labels=['Whole shape','Whole voltage','Prompt','Post-peak','Tail','Baseline']
        fig,ax=plt.subplots(figsize=(11,4.4));xs=np.arange(len(views))
        for i,n in enumerate(['R3B','NSF','REAL_REAL']):
            y=np.array([cs[n][v]['balanced_accuracy'] for v in views]);ci=np.array([cs[n][v]['heldout_CI']['balanced_accuracy_95CI'] for v in views])
            ax.errorbar(xs+(i-1)*.18,y,yerr=np.maximum(np.array([y-ci[:,0],ci[:,1]-y]),0),fmt='o',capsize=3,color=COLORS[n],label=n)
        ax.axhline(.5,color='gray',ls='--');ax.set(xticks=xs,xticklabels=labels,ylim=(.43,1.04),ylabel='Held-out balanced accuracy',title='Frozen DEV classifier comparison: lower is less distinguishable')
        ax.legend(ncol=3,loc='lower right');save(fig,'02_classifier')
    valid={n:d['valid'] for n,d in domains.items()}
    q=np.concatenate([d['features']['Qtotal_mVns'][valid[n]] for n,d in domains.items()]);p=np.concatenate([d['features']['PSD'][valid[n]] for n,d in domains.items()])
    xmin,xmax=np.log10(np.quantile(q,[.001,.999]));ymin,ymax=np.quantile(p,[.001,.999]);pad=max(.01,(ymax-ymin)*.05)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True,layout='constrained')
    for ax,(n,d) in zip(axes,domains.items()):
        use=valid[n];x=np.log10(d['features']['Qtotal_mVns'][use]);y=d['features']['PSD'][use]
        off=int(((x<xmin)|(x>xmax)|(y<ymin-pad)|(y>ymax+pad)).sum())
        hb=ax.hexbin(x,y,gridsize=(38,28),extent=(xmin,xmax,ymin-pad,ymax+pad),mincnt=1,cmap='viridis',vmin=1,vmax=50)
        ax.set(title=f'{n}: QC {use.sum()}/2000; off-axis {off}',xlabel='log10 Q (mV ns)',xlim=(xmin,xmax),ylim=(ymin-pad,ymax+pad))
    axes[0].set_ylabel('PSD = Qtail / Qtotal');fig.colorbar(hb,ax=axes,label='Events per hexagon',shrink=.8)
    fig.suptitle('Observed-charge / PSD distributions; identical axes')
    fig.savefig(out/'03_charge_psd.png',dpi=155,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.4));wd=m['width'];x=np.arange(1,9)
    for c in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[c,j]
            for n,t in wd['tables'].items():
                y=[s['components'][c][field] if s['valid'] and len(s['components'])==2 and s['components'][c][field] is not None else np.nan for s in t['slices']]
                ax.plot(x,y,'o-',ms=4,color=COLORS[n],label=n)
            ax.set(title=f'Density component {c+1}: {field}',xlabel='Frozen charge bin',xticks=x)
    axes[0,0].legend();fig.suptitle('PSD density peaks and widths; missing points are unqualified fits',y=1.02)
    save(fig,'04_conditional_width')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for n,d in domains.items():
        a=d['V'][valid[n]];lo,mid,hi=np.quantile(a,[.1,.5,.9],axis=0)
        axes[0,0].plot(mid,color=COLORS[n],label=n);axes[0,0].fill_between(np.arange(480),lo,hi,color=COLORS[n],alpha=.13)
        for ax,key,label in [(axes[0,1],'rise_10_90_ns','Rise 10-90 (ns)'),(axes[1,0],'baseline_RMS_V','Baseline RMS (V)'),(axes[1,1],'PSD','PSD')]:
            a=d['features'][key][valid[n]];a=np.sort(a[np.isfinite(a)]);ax.plot(a,(np.arange(len(a))+.5)/len(a),label=n,color=COLORS[n]);ax.set(xlabel=label,ylabel='Empirical CDF')
    axes[0,0].set(xlabel='Waveform sample (1 ns)',ylabel='Voltage (V)',title='Median and 10-90% pointwise bands');axes[0,0].legend()
    save(fig,'05_waveforms_features')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,view in zip(axes,['V','own_shape']):
        with np.load(root/'results/NN_PRIVATE.npz',allow_pickle=False) as z:
            for n in domains:
                a=z[n+'__'+view+'__distance'];a=np.sort(a[np.isfinite(a)]);ax.plot(a,(np.arange(len(a))+.5)/len(a),color=COLORS[n],label=n)
        ax.set(xlabel='Nearest TRAIN RMS ('+('V' if view=='V' else 'normalized shape')+')',ylabel='Empirical CDF',title=view);ax.legend()
    fig.suptitle('Distance to 2,592 REAL-fit records; descriptive, not proof against memorization',y=1.04)
    save(fig,'06_nearest_train')
