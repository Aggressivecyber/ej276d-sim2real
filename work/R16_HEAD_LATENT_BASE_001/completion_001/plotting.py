from __future__ import annotations
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
COLORS={'REAL':'black','STANDARD':'#6c4c9b','GAUSSIAN':'#0072b2','GMM8':'#009e73','REAL_REAL':'#777777'}

def plots(domains,m,ref,root):
    cfg=json.loads((root/'config.json').read_text());figdir=root/'results/figures';plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});names=list(domains)
    def save(fig,n):fig.savefig(figdir/(n+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    def cdf(ax,x,n,ls='-'):
        x=np.sort(x[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x) else [],label=n,color=COLORS[n],ls=ls)
    allq=np.concatenate([d['features']['Qtotal_mVns'][d['valid']] for d in domains.values()]);qlim=(allq.min()*.92,allq.max()*1.08);outside={}
    for view in ['full','main_body']:
        fig,axes=plt.subplots(1,len(names),figsize=(4.2*len(names),5),sharex=True,sharey=True,layout='constrained')
        for n,ax in zip(names,axes):
            d=domains[n];valid=d['valid'];q=d['features']['Qtotal_mVns'][valid];ps=d['features']['PSD'][valid];lo=int((ps<0).sum());hi=int((ps>.6).sum());outside[n]={'below0':lo,'above0_6':hi,'QC_n':len(ps)}
            ax.scatter(q,ps,s=4,alpha=.3,color=COLORS[n]);ax.set_xscale('log');ax.set_xlim(qlim);ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.set_xlabel('Signed observed charge [mV ns]');ax.set_ylabel('Signed PSD');ax.set_title(n+'\nQC='+str(len(ps))+(' outside='+str(lo+hi) if view=='main_body' else ''))
            if view=='main_body':ax.set_ylim(0,.6)
        fig.suptitle('All QC-valid populations; '+('shared PSD[0,0.6], original full range retained' if view=='main_body' else 'shared full range preserves outliers'));save(fig,'charge_psd_'+view)
    (root/'CHARGE_PLOT_METADATA.json').write_text(json.dumps({'shared_Q_range':qlim,'main_body_PSD_range':[0,.6],'outside_counts':outside},indent=2)+'\n')
    cs=m['C2ST'];views=cfg['C2ST_views'];nv=len(views);fig,axes=plt.subplots(1,2,figsize=(19,6),layout='constrained')
    if cs['supported']:
        for j,(n,res) in enumerate(cs['results'].items()):
            y=np.array([res[v]['balanced_accuracy'] for v in views]);ci=np.array([res[v]['heldout_CI']['balanced_accuracy_95CI'] for v in views]);axes[0].errorbar(np.arange(nv)+.08*(j-1.5),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,label=n,color=COLORS[n])
        for j,(n,res) in enumerate(cs['paired_contrasts'].items()):
            y=np.array([res[v]['BA_delta_first_minus_second'] for v in views]);ci=np.array([res[v]['BA_delta_95CI'] for v in views]);axes[1].errorbar(np.arange(nv)+.08*(j-1),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=n.replace('_minus_',' - '),color=['#0072b2','#009e73','#cc79a7'][j])
    for ax in axes:ax.set_xticks(np.arange(nv),views,rotation=30,ha='right');ax.legend(fontsize=7)
    axes[0].set_ylim(.4,1.04);axes[0].axhline(.5,color='gray',ls='--');axes[0].set_ylabel('Heldout balanced accuracy');axes[1].axhline(0,color='gray',ls='--');axes[1].set_ylabel('Paired BA difference;500 cluster 95%CI');fig.suptitle('Same fixed-position common QC; new head34 / Q-PSD views explicitly global');save(fig,'c2st_paired')
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for n,color in [('TRAIN','#0072b2'),('validation','#cc79a7')]:
        axes[0,0].plot(np.arange(34),m['latent'][n]['mean'],label=n,color=color)
        axes[0,1].plot(np.arange(34),m['latent'][n]['covariance_eigenvalues'],label=n,color=color)
    axes[0,0].axhline(0,color='gray',ls='--');axes[0,0].set_title('Fixed head latent means');axes[0,0].set_xlabel('Coordinate');axes[0,0].legend();axes[0,1].axhline(1,color='gray',ls='--');axes[0,1].set_title('Fixed head latent covariance eigenvalues');axes[0,1].set_xlabel('Eigenvalue order');axes[0,1].legend()
    ns=list(m['train_validation_gap']);x=np.arange(len(ns));axes[0,2].bar(x-.18,[m['train_validation_gap'][n]['TRAIN_NLL'] for n in ns],.36,label='TRAIN21047');axes[0,2].bar(x+.18,[m['train_validation_gap'][n]['val_NLL'] for n in ns],.36,label='val599');axes[0,2].set_xticks(x,ns,rotation=15);axes[0,2].set_ylabel('Complete480 NLL[nat/event]');axes[0,2].set_title('No neural network updated');axes[0,2].legend()
    with np.load(root/'results/HEAD_PSD_PRIVATE.npz',allow_pickle=False) as d:
        for n in names:
            use=d[n+'__defined'];cdf(axes[1,0],d[n+'__head_PSD'][use],n);cdf(axes[1,1],d[n+'__full_PSD'][use],n);cdf(axes[1,2],d[n+'__delta'][use],n)
    for ax,title in zip(axes[1],['Head-only reconstructed PSD','Full reconstructed PSD','Full minus head-only PSD']):ax.set_xlabel('Signed PSD');ax.set_ylabel('CDF');ax.set_title(title);ax.legend(fontsize=7)
    fig.suptitle('Head-only is a projection diagnostic; complete response includes the frozen conditional kernel');save(fig,'latent_density_and_PSD_contribution')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');freq=np.fft.rfftfreq(100,d=1e-9)/1e6
    for n,d in domains.items():
        valid=d['valid'];x=d['x'][valid,:100];ps=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);ps[:,1:-1]*=2;axes[0,0].loglog(freq[1:],ps.mean(axis=0)[1:],label=n,color=COLORS[n]);axes[0,1].plot(np.arange(1,21),np.nanmedian(d['acf'][valid],axis=0),label=n,color=COLORS[n]);cdf(axes[0,2],d['features']['baseline_RMS_V'][valid]*1000,n)
        for j,key in enumerate(['prompt_curvature_RMS_V','postpeak_curvature_RMS_V']):
            y=[np.median(d['extra'][key][valid&(d['region']==r)])*1000 if np.any(valid&(d['region']==r)) else np.nan for r in range(4)];axes[1,j].plot(np.arange(1,5),y,marker='o',label=n,color=COLORS[n])
        cdf(axes[1,2],d['features']['rise_10_90_ns'][valid],n)
    labels=[('Baseline frequency [MHz]','Mean PSD[V²/Hz]'),('Lag[ns]','Median ACF'),('Baseline RMS[mV]','CDF'),('Output charge quartile','Prompt2nd-difference median[mV]'),('Output charge quartile','Postpeak2nd-difference median[mV]'),('Rise10–90[ns]','CDF')]
    for ax,(x,y) in zip(axes.ravel(),labels):ax.set_xlabel(x);ax.set_ylabel(y)
    axes[0,0].legend(fontsize=7);fig.suptitle('Frozen conditional kernel, changed head distribution; prompt100:150 includes peak/early fall');save(fig,'noise_and_curvature')
    fig,axes=plt.subplots(4,2,figsize=(14,15),layout='constrained');t=np.arange(480)-120
    for r in range(4):
        for j,k in enumerate(['x','shape']):
            ax=axes[r,j]
            for n,d in domains.items():
                use=d['valid']&(d['region']==r);a=d[k][use]
                if not len(a):continue
                q=np.quantile(a,[.1,.5,.9],axis=0);ax.plot(t,q[1],color=COLORS[n],label=n+' n='+str(len(a)));ax.fill_between(t,q[0],q[2],color=COLORS[n],alpha=.04)
            ax.set_xlabel('Saved alignment time[ns]');ax.set_ylabel('Baseline-subtracted V' if j==0 else 'Own normalized shape');ax.set_title('Output q'+str(r+1));ax.legend(fontsize=6)
    save(fig,'conditional_waveforms')
    w=m['energy_conditional_width'];fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for k in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for n,tab in w['tables'].items():
                y=[r['components'][k][field] if r['valid'] and len(r['components'])==2 and r['components'][k][field] is not None else np.nan for r in tab['slices']];ax.plot(np.arange(1,9),y,marker='o',label=n,color=COLORS[n]);bad=[i+1 for i,r in enumerate(tab['slices']) if not r['valid']];ax.plot(bad,[-.01-.008*list(w['tables']).index(n)]*len(bad),marker='x',ls='',ms=4,color=COLORS[n])
            ax.set_xticks(np.arange(1,9));ax.set_xlabel('Frozen charge bin ID');ax.set_ylabel(field);ax.set_title(('Low' if k==0 else 'High')+' PSD component; crosses unsupported')
    axes[0,0].legend(fontsize=6);save(fig,'energy_peak_width')
    with np.load(root/'results/NN_ORIGINAL_PRIVATE.npz',allow_pickle=False) as original,np.load(root/'results/NN_EXPANDED_PRIVATE.npz',allow_pickle=False) as expanded,np.load(pathlib.Path(cfg['remote_R15_root'])/'results/NN_ORIGINAL_PRIVATE.npz',allow_pickle=False) as oldorig,np.load(pathlib.Path(cfg['remote_R15_root'])/'results/NN_EXPANDED_PRIVATE.npz',allow_pickle=False) as oldexpanded:
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for row,(data,olddata) in enumerate([(original,oldorig),(expanded,oldexpanded)]):
            for col,view in enumerate(['V','own_shape']):
                ax=axes[row,col]
                for n in names:
                    if n=='REAL':continue
                    src=olddata if n=='STANDARD' else data;key=('RESIDUAL_FLOW' if n=='STANDARD' else n)+'__'+view+'__distance';cdf(ax,src[key],n)
                ax.set_xscale('log');ax.set_xlabel('Nearest TRAIN RMS distance '+('[V]' if view=='V' else '[dimensionless]'));ax.set_ylabel('CDF');ax.set_title('Reference original2592' if row==0 else 'Reference expanded21047');ax.legend(fontsize=8)
        fig.suptitle('All2000 queries per candidate; no exact copy is not proof of no memory');save(fig,'nearest_neighbours')
