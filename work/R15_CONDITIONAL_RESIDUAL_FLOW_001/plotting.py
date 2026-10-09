from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import evaluation as ev
COLORS={'REAL':'black','GAUSSIAN_BASE':'#6c4c9b','RESIDUAL_FLOW':'#009e73','REAL_REAL':'#777777'}

def plots(domains,m,ref,root):
    cfg=__import__('json').loads((root/'config.json').read_text());figdir=root/'results/figures';plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});names=list(domains)
    def save(fig,n):fig.savefig(figdir/(n+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    def cdf(ax,x,n):
        x=np.sort(x[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x) else [],label=n,color=COLORS[n])
    allq=np.concatenate([d['features']['Qtotal_mVns'][d['valid']] for d in domains.values()]);qlim=(allq.min()*.92,allq.max()*1.08);outside={}
    for view in ['full','main_body']:
        fig,axes=plt.subplots(1,len(names),figsize=(4.2*len(names),5),sharex=True,sharey=True,layout='constrained')
        for n,ax in zip(names,axes):
            d=domains[n];valid=d['valid'];q=d['features']['Qtotal_mVns'][valid];ps=d['features']['PSD'][valid];lo=int((ps<0).sum());hi=int((ps>.6).sum());outside[n]={'below0':lo,'above0_6':hi,'QC_n':len(ps)}
            ax.scatter(q,ps,s=4,alpha=.3,color=COLORS[n]);ax.set_xscale('log');ax.set_xlim(qlim);ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.set_xlabel('Signed observed charge [mV ns]');ax.set_ylabel('Signed PSD');ax.set_title(n+'\nQC='+str(len(ps))+(' outside='+str(lo+hi) if view=='main_body' else ''))
            if view=='main_body':ax.set_ylim(0,.6)
        fig.suptitle('All QC-valid populations; '+('sharedPSD[0,0.6], full-range original retained' if view=='main_body' else 'shared full range preserves outliers'));save(fig,'charge_psd_'+view)
    (root/'CHARGE_PLOT_METADATA.json').write_text(__import__('json').dumps({'shared_Q_range':qlim,'main_body_PSD_range':[0,.6],'outside_counts':outside},indent=2)+'\n')
    cs=m['C2ST'];views=cfg['C2ST_views'];fig,axes=plt.subplots(1,2,figsize=(17,6),layout='constrained')
    if cs['supported']:
        for j,(n,res) in enumerate(cs['results'].items()):
            y=np.array([res[v]['balanced_accuracy'] for v in views]);ci=np.array([res[v]['heldout_CI']['balanced_accuracy_95CI'] for v in views]);axes[0].errorbar(np.arange(6)+.08*(j-2),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,label=n,color=COLORS[n])
        for j,(n,res) in enumerate(cs['paired_contrasts'].items()):
            y=np.array([res[v]['BA_delta_first_minus_second'] for v in views]);ci=np.array([res[v]['BA_delta_95CI'] for v in views]);axes[1].errorbar(np.arange(6)+.08*(j-2.5),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=['U global−small','U cond−small','E global−U','E cond−U','U cond−global','E cond−global'][j],color=['#0072b2','#e69f00','#009e73','#6c4c9b','#aa5511','#cc79a7'][j])
    for ax in axes:ax.set_xticks(np.arange(6),views,rotation=25,ha='right');ax.legend(fontsize=7)
    axes[0].set_ylim(.4,1.04);axes[0].axhline(.5,color='gray',ls='--');axes[0].set_ylabel('Heldout balanced accuracy');axes[1].axhline(0,color='gray',ls='--');axes[1].set_ylabel('Paired new minus old BA;500 cluster 95%CI');fig.suptitle('All comparisons on same fixed-position common QC; RR is not equivalence cutoff');save(fig,'c2st_paired')
    fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    row=m['training'];trace=row['trace'];steps=[r['step'] for r in trace];values=[r['conditional_NLL_nat_event'] for r in trace]
    axes[0].plot(steps,values);chosen=next(r for r in trace if r['step']==row['selected_step']);axes[0].scatter([chosen['step']],[chosen['conditional_NLL_nat_event']],marker='*',s=110,color=COLORS['RESIDUAL_FLOW'],label='selected '+str(row['selected_step']));axes[0].legend();axes[0].set_xlabel('Update');axes[0].set_ylabel('All599 conditional val NLL [nat/event]');axes[0].set_title('Single residual flow; upstream frozen')
    ns=list(m['validation'])
    for ax,field in zip(axes[1:],['val_NLL','val_minus_TRAIN_NLL']):
        ax.bar(np.arange(2),[m['train_validation_gap'][n][field] for n in ns],color=[COLORS[n] for n in ns]);ax.set_xticks(np.arange(2),['Gaussian base','Residual flow']);ax.set_ylabel('NLL [nat/event]');ax.set_title('Complete480 val NLL' if field=='val_NLL' else 'Validation minus TRAIN; N21047')
    save(fig,'training_likelihood')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');freq=np.fft.rfftfreq(100,d=1e-9)/1e6
    for n,d in domains.items():
        valid=d['valid'];x=d['x'][valid,:100];ps=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);ps[:,1:-1]*=2;axes[0,0].loglog(freq[1:],ps.mean(axis=0)[1:],label=n,color=COLORS[n]);axes[0,1].plot(np.arange(1,21),np.nanmedian(d['acf'][valid],axis=0),label=n,color=COLORS[n]);cdf(axes[0,2],d['features']['baseline_RMS_V'][valid]*1000,n)
        for j,key in enumerate(['prompt_curvature_RMS_V','postpeak_curvature_RMS_V']):
            y=[np.median(d['extra'][key][valid&(d['region']==r)])*1000 if np.any(valid&(d['region']==r)) else np.nan for r in range(4)];axes[1,j].plot(np.arange(1,5),y,marker='o',label=n,color=COLORS[n])
        cdf(axes[1,2],d['features']['rise_10_90_ns'][valid],n)
    labels=[('Baseline frequency [MHz]','Mean PSD[V²/Hz]'),('Lag[ns]','Median ACF'),('Baseline RMS[mV]','CDF'),('Output charge quartile','Prompt2nd-difference median[mV]'),('Output charge quartile','Postpeak2nd-difference median[mV]'),('Rise10–90[ns]','CDF')]
    for ax,(x,y) in zip(axes.ravel(),labels):ax.set_xlabel(x);ax.set_ylabel(y)
    axes[0,0].legend(fontsize=7);fig.suptitle('Tail-coordinate covariance acts on full480; prompt100:150 includes peak/early fall');save(fig,'noise_and_curvature')
    fig,axes=plt.subplots(4,2,figsize=(14,15),layout='constrained');t=np.arange(480)-120
    for r in range(4):
        for j,k in enumerate(['x','shape']):
            ax=axes[r,j]
            for n,d in domains.items():
                valid=d['valid']&(d['region']==r);a=d[k][valid]
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
    with np.load(root/'results/NN_ORIGINAL_PRIVATE.npz',allow_pickle=False) as original,np.load(root/'results/NN_EXPANDED_PRIVATE.npz',allow_pickle=False) as expanded,np.load(__import__('pathlib').Path(cfg['remote_R14_root'])/'results/NN_ORIGINAL_PRIVATE.npz',allow_pickle=False) as oldorig,np.load(__import__('pathlib').Path(cfg['remote_R14_root'])/'results/NN_EXPANDED_PRIVATE.npz',allow_pickle=False) as oldexpanded:
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for row,(data,olddata) in enumerate([(original,oldorig),(expanded,oldexpanded)]):
            for col,view in enumerate(['V','own_shape']):
                ax=axes[row,col];cdf(ax,data['RESIDUAL_FLOW__'+view+'__distance'],'RESIDUAL_FLOW');cdf(ax,olddata['EXPOSURES_CONDITIONAL__'+view+'__distance'],'GAUSSIAN_BASE')
                ax.set_xscale('log');ax.set_xlabel('Nearest TRAIN RMS distance '+('[V]' if view=='V' else '[dimensionless]'));ax.set_ylabel('CDF');ax.set_title('Reference original2592' if row==0 else 'Reference expanded21047');ax.legend(fontsize=8)
        fig.suptitle('All2000 new queries; no exact copy is not proof of no memory');save(fig,'nearest_neighbours')
