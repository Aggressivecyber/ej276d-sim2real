from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
COLORS={'REAL':'black','R3B':'#777777','R10_FIXED':'#b55c8a','RAW_PCA_GMM':'#0072b2','FACTOR_PCA_GMM':'#d55e00','REAL_REAL':'#009e73'}
def plots(domains,models,c2st,width,nn,ref,root):
    figdir=root/'results/figures';plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(figdir/(name+'.png'),dpi=145,bbox_inches='tight');plt.close(fig)
    def cdf(ax,x,label,color):
        x=np.sort(np.asarray(x)[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x) else [],label=label+' n='+str(len(x)),color=color)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name,row in models.items():
        for part,style in [('TRAIN_fit','-'),('TRAIN_val','--')]:
            rs=row['reconstruction'][part];axes[0].plot(np.arange(len(rs)),[max(r['RMS_mean_V'],1e-18) for r in rs],style,marker='o',label=name+' '+part,color=COLORS[name]);axes[1].plot(np.arange(len(rs)),[r['PSD_abs_error_mean'] for r in rs],style,marker='o',label=name+' '+part,color=COLORS[name])
    for ax in axes:ax.set_xticks(np.arange(6),['8','16','32','64','128','full']);ax.set_xlabel('Diagnostic rank; generation head fixed32');ax.set_yscale('log');ax.legend(fontsize=7)
    axes[0].set_ylabel('Mean per-row waveform RMS error [V]');axes[1].set_ylabel('Mean absolute signed PSD error');fig.suptitle('Reconstruction from fixed TRAIN bases; full inverse error is numerical');save(fig,'reconstruction_vs_rank')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,(name,row) in zip(axes,models.items()):
        for rec in row['fits']:
            val=rec.get('complete_validation_NLL');val=np.nan if val is None or not np.isfinite(val) else val;ax.scatter(rec['components'],val,color=COLORS[name],marker='o' if rec['eligible_for_selection'] else 'x',s=60);ax.annotate('selected' if rec['components']==row['selected_components'] else ('ok' if rec['eligible_for_selection'] else 'excluded'),(rec['components'],val),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([1,4,8]);ax.set_xlabel('Fixed number of head mixture components');ax.set_ylabel('Complete continuous validation NLL [nat/event]');ax.set_title(name)
    fig.suptitle('All599 internal-val rows, tail density and exact coordinate Jacobian included');save(fig,'validation_complete_NLL')
    names=list(domains);fig,axes=plt.subplots(1,len(names),figsize=(4.3*len(names),4.8),layout='constrained',sharex=True,sharey=True)
    allq=np.concatenate([d['features']['Qtotal_mVns'][d['valid']] for d in domains.values()]);lims=(allq.min()*.92,allq.max()*1.08)
    for name,ax in zip(names,np.atleast_1d(axes)):
        d=domains[name];m=d['valid'];ax.scatter(d['features']['Qtotal_mVns'][m],d['features']['PSD'][m],s=4,alpha=.28,color=COLORS[name]);ax.set_xscale('log');ax.set_xlim(lims);ax.set_title(name+'\nQC '+str(m.sum())+'/'+str(len(m)));ax.set_xlabel('Observed signed charge [mV ns]');ax.set_ylabel('Signed PSD')
        for e in ref['CAL_charge_quantile_edges_mV_ns']:ax.axvline(e,color='gray',lw=.6,ls='--')
    fig.suptitle('Full QC-valid populations on shared coordinates; no reweighting or rejection');save(fig,'charge_psd')
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    for key,ax in zip(['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','baseline_RMS_over_peak','ACF1'],axes.ravel()):
        for name,d in domains.items():cdf(ax,d['features'][key][d['valid']],name,COLORS[name])
        ax.set_xlabel(key);ax.set_ylabel('Full QC empirical CDF')
        if key in ['Qtotal_mVns','peak_V','baseline_RMS_V','baseline_RMS_over_peak']:ax.set_xscale('log')
    axes[0,0].legend(fontsize=7);save(fig,'response_features')
    fig,axes=plt.subplots(4,2,figsize=(14,15),layout='constrained');t=np.arange(480)-120;wave={}
    for r in range(4):
        for j,key in enumerate(['x','shape']):
            ax=axes[r,j]
            for name,d in domains.items():
                use=d['valid']&(d['region']==r);x=d[key][use]
                if not len(x):continue
                qq=np.quantile(x,[.1,.5,.9],axis=0);ax.plot(t,qq[1],color=COLORS[name],label=name+' n='+str(len(x)));ax.fill_between(t,qq[0],qq[2],color=COLORS[name],alpha=.055);ax.plot(t,x.mean(axis=0),ls=':',color=COLORS[name],lw=.8);wave[name+f'__q{r+1}__'+key]=np.vstack([x.mean(axis=0),qq])
            ax.set_xlabel('Saved alignment time [ns]');ax.set_ylabel('Baseline-subtracted V' if key=='x' else 'Own peak normalized shape');ax.set_title(f'Output charge q{r+1}');ax.legend(fontsize=7)
    fig.suptitle('Median, dotted mean, 10–90% event bands; new density draws are not G4 parents');save(fig,'conditional_waveforms');np.savez_compressed(root/'results/WAVE_SUMMARY.npz',**wave)
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for r,ax in enumerate(axes):
        for name,d in domains.items():
            use=d['valid']&(d['region']==r);x=d['shape'][use,95:125]
            if not len(x):continue
            qq=np.quantile(x,[.1,.5,.9],axis=0);ax.plot(np.arange(95,125)-120,qq[1],color=COLORS[name],label=name);ax.fill_between(np.arange(95,125)-120,qq[0],qq[2],color=COLORS[name],alpha=.06)
        ax.set_title(f'q{r+1}; full sample bands');ax.set_xlabel('Saved alignment time [ns]');ax.set_ylabel('Own normalized amplitude')
    axes[0].legend(fontsize=7);save(fig,'pure_front_population')
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,region in enumerate(['prompt','postpeak']):
        for name,d in domains.items():
            cdf(axes[row,0],d['extra'][region+'_curvature_RMS_V'][d['valid']]*1000,name,COLORS[name]);m=[]
            for r in range(4):
                use=d['valid']&(d['region']==r);x=d['extra'][region+'_curvature_RMS_V'][use];m.append(np.median(x)*1000 if len(x) else np.nan)
            axes[row,1].plot(np.arange(1,5),m,marker='o',label=name,color=COLORS[name])
        axes[row,0].set_xlabel(region+' second difference RMS [mV]');axes[row,0].set_ylabel('CDF');axes[row,1].set_xlabel('Output charge quartile');axes[row,1].set_ylabel(region+' median curvature [mV]');axes[row,0].legend(fontsize=7)
    fig.suptitle('Prompt100:150 includes peak and early fall; postpeak150:220');save(fig,'conditional_curvature')
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained');freq=np.fft.rfftfreq(100,d=1e-9)/1e6;noise={}
    for name,d in domains.items():
        use=d['valid'];x=d['x'][use,:100];p=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);p[:,1:-1]*=2;mean=p.mean(axis=0);axes[0].loglog(freq[1:],mean[1:],label=name,color=COLORS[name]);axes[1].plot(np.arange(1,21),np.nanmedian(d['acf'][use],axis=0),color=COLORS[name],label=name);cdf(axes[2],d['features']['baseline_RMS_V'][use]*1000,name,COLORS[name]);noise[name+'__PSD']=mean
    axes[0].set_xlabel('Frequency [MHz]');axes[0].set_ylabel('Mean baseline spectrum [V²/Hz]');axes[1].set_xlabel('Lag [ns]');axes[1].set_ylabel('Median baseline ACF');axes[2].set_xlabel('Baseline RMS [mV]');axes[2].set_ylabel('CDF');axes[0].legend(fontsize=7);save(fig,'baseline_noise');np.savez_compressed(root/'results/NOISE_SUMMARY.npz',frequency_MHz=freq,**noise)
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');views=list(c2st['results'].values())[0] if c2st.get('supported') else []
    if views:
        for j,(name,res) in enumerate(c2st['results'].items()):
            ys=np.array([res[v]['balanced_accuracy'] for v in views]);ci=np.array([res[v]['heldout_CI']['balanced_accuracy_95CI'] for v in views]);ax.errorbar(np.arange(len(views))+.1*(j-1),ys,yerr=np.maximum(0,np.vstack([ys-ci[:,0],ci[:,1]-ys])),fmt='o',label=name,color=COLORS[name],capsize=3)
        ax.set_xticks(np.arange(len(views)),views,rotation=20,ha='right');ax.legend()
    else:ax.text(.1,.5,'Unsupported: '+str(c2st))
    ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.4,1.04);ax.set_ylabel('Heldout balanced accuracy; 95% conditional cluster CI');ax.set_title('Common fixed R7 positions; REAL–REAL is sampling reference, not acceptance threshold');save(fig,'c2st')
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained');xx=np.arange(1,9)
    for c in range(2):
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[c,j]
            for name,table in width['tables'].items():
                y=[r['components'][c][field] if r['valid'] and len(r['components'])==2 and r['components'][c][field] is not None else np.nan for r in table['slices']];ax.plot(xx,y,marker='o',label=name,color=COLORS[name])
                invalid=[i+1 for i,r in enumerate(table['slices']) if not r['valid']];ax.plot(invalid,[-.012-.009*list(width['tables']).index(name)]*len(invalid),ls='',marker='x',color=COLORS[name],ms=4)
            ax.set_xlabel('Frozen charge bin ID');ax.set_ylabel(field);ax.set_title(('Low' if c==0 else 'High')+' PSD density component; crosses=unsupported');ax.set_xticks(xx)
    axes[0,0].legend(fontsize=7);save(fig,'energy_peak_width')
    with np.load(root/'results/NN_PRIVATE.npz',allow_pickle=False) as z:
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        for view,ax in zip(['V','own_shape'],axes):
            for name in nn[view]:cdf(ax,z[name+'__'+view+'__distance'],name,COLORS[name])
            ax.set_xscale('log');ax.set_xlabel('Nearest TRAIN waveform RMS '+('[V]' if view=='V' else '[dimensionless]'));ax.set_ylabel('Defined-query CDF');ax.legend(fontsize=8)
        fig.suptitle('All2000 queries retained; undefined counts in metrics; no exact copy is not a memory proof');save(fig,'nearest_neighbours')
    with np.load(root/'results/NN_EXAMPLES.npz',allow_pickle=False) as z:
        nnames=[n for n in models if models[n]['selected_components'] is not None];fig,axes=plt.subplots(len(nnames),3,figsize=(15,4*len(nnames)),layout='constrained',squeeze=False)
        for row,name in enumerate(nnames):
            q=z[name+'__V__query'];n=z[name+'__V__neighbour']
            for j,ax in enumerate(axes[row]):ax.plot(np.arange(480)-120,q[j],label='Independent density draw',color=COLORS[name]);ax.plot(np.arange(480)-120,n[j],label='Nearest TRAIN event',color='black',lw=.8);ax.set_title(name+' hash example '+str(j+1));ax.set_xlabel('Saved alignment time [ns]');ax.set_ylabel('V');ax.legend(fontsize=7)
        save(fig,'nearest_examples')
