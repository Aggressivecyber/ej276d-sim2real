from __future__ import annotations
import pathlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
COLORS={'REAL':'black','STEP0':'#777777','NLL_CONT':'#0072b2','NLL_PSD':'#d55e00','REAL_REAL':'#009e73'}

def plots(domains,m,ref,root):
    cfg=json.loads((root/'config.json').read_text());figdir=root/'results/figures';plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});names=list(domains)
    def save(fig,n):fig.savefig(figdir/(n+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    def cdf(ax,x,n):
        x=np.sort(x[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x)else [],label=n,color=COLORS[n])
    allq=np.concatenate([d['features']['Qtotal_mVns'][d['valid']] for d in domains.values()]);qlim=(allq.min()*.92,allq.max()*1.08);outside={}
    for view in ['full','main_body']:
        fig,axes=plt.subplots(1,len(names),figsize=(4.1*len(names),5),sharex=True,sharey=True,layout='constrained')
        for n,ax in zip(names,axes):
            d=domains[n];q=d['features']['Qtotal_mVns'][d['valid']];ps=d['features']['PSD'][d['valid']];lo=int((ps<0).sum());hi=int((ps>.6).sum());outside[n]={'below0':lo,'above0_6':hi,'QC_N':len(ps)};ax.scatter(q,ps,s=4,alpha=.3,color=COLORS[n]);ax.set_xscale('log');ax.set_xlim(qlim);ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.set_xlabel('Signed observed charge [mV ns]');ax.set_ylabel('Signed PSD');ax.set_title(n+'\nQC='+str(len(ps))+(' outside='+str(lo+hi)if view=='main_body'else ''))
            if view=='main_body':ax.set_ylim(0,.6)
        fig.suptitle('Common axes, complete populations; '+('body zoom, outside points counted'if view=='main_body'else 'full range retains all outliers'));save(fig,'charge_psd_'+view)
    (root/'CHARGE_PLOT_METADATA.json').write_text(json.dumps({'shared_Q_range':qlim,'main_body_PSD_range':[0,.6],'outside_counts':outside},indent=2)+'\n')
    with np.load(root/'results/PSD_SECTIONS_PRIVATE.npz',allow_pickle=False)as p:
        fig,axes=plt.subplots(2,4,figsize=(20,10),sharex=True,sharey=True,layout='constrained')
        for i,ax in enumerate(axes.ravel()):
            for n in names:
                a=p[n+'__bin'+str(i+1)+'__PSD_sorted'];cdf(ax,a,n)
            low=m['PSD_main']['tables']['REAL']['bins'][i]['valley_definition']['low'];high=m['PSD_main']['tables']['REAL']['bins'][i]['valley_definition']['high'];ax.axvspan(low,high,color='gray',alpha=.12);ax.set_title('Frozen bin '+str(i+1)+'; '+' / '.join(n+':'+str(m['PSD_main']['tables'][n]['bins'][i]['N'])for n in names));ax.set_xlabel('Full finite signed PSD');ax.set_ylabel('Empirical CDF');ax.legend(fontsize=6)
        fig.suptitle('All finite post-ADC PSD retained; shaded valley fixed by old TRAIN fit, no new boundary fit');save(fig,'conditional_PSD_CDF')
    fig,axes=plt.subplots(1,3,figsize=(19,5),layout='constrained')
    for n in names:
        rows=m['PSD_main']['tables'][n]['bins'];axes[0].plot(np.arange(1,9),[r['valley_fraction'] if r['valley_fraction']is not None else np.nan for r in rows],marker='o',label=n,color=COLORS[n]);axes[1].plot(np.arange(1,9),[r['fraction_of_QC_finite']for r in rows],marker='o',label=n,color=COLORS[n])
        if n!='REAL':axes[2].plot(np.arange(1,9),[r['W1'] if r['W1']is not None else np.nan for r in m['PSD_main']['vs_REAL'][n]['bins']],marker='o',label=n,color=COLORS[n])
    for ax in axes:ax.set_xlabel('Frozen log-charge bin ID');ax.set_xticks(np.arange(1,9));ax.legend(fontsize=7)
    axes[0].set_ylabel('Within-bin fraction in fixed middle-third valley');axes[1].set_ylabel('Fraction of complete QC-finite population');axes[2].set_ylabel('PSD W1 vs REAL');fig.suptitle('Descriptive valley occupancy and charge coverage, not particle labels');save(fig,'valley_and_conditional_distances')
    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained')
    for n,r in m['training'].items():
        tr=r['trace'];x=[t['step']for t in tr]
        for ax,key,label in zip(axes[0],['head_NLL_div34','conditional_CDF_L1','window_occupancy_L1'],['Validation head NLL/34','Validation conditional CDF L1','Validation occupancy L1']):ax.plot(x,[t[key]for t in tr],label=n,color=COLORS[n]);ax.set_xlabel('Updates');ax.set_ylabel(label)
        axes[1,0].plot(x,[t['J']for t in tr],label=n,color=COLORS[n]);axes[1,0].scatter([r['selected_step']],[r['selected_J']],s=75,facecolors='none',edgecolors=COLORS[n]);axes[1,1].plot(x,[t['training_elapsed_s']for t in tr],label=n,color=COLORS[n]);axes[1,1].set_xlabel('Updates');axes[1,1].set_ylabel('Per-arm elapsed seconds')
    axes[1,0].set_xlabel('Updates');axes[1,0].set_ylabel('Unified validation J; circle selected');ns=list(m['complete_density_scores']);x=np.arange(len(ns));axes[1,2].bar(x-.18,[m['complete_density_scores'][n]['TRAIN']['complete_NLL']for n in ns],.36,label='TRAIN21047');axes[1,2].bar(x+.18,[m['complete_density_scores'][n]['VAL']['complete_NLL']for n in ns],.36,label='val599');axes[1,2].set_xticks(x,ns,rotation=15);axes[1,2].set_ylabel('Complete480 NLL [nat/event]')
    for ax in axes.ravel():ax.legend(fontsize=7)
    fig.suptitle('Both arms choose by same J including PSD; only TRAIN optimization differs');save(fig,'training_validation')
    cs=m['C2ST'];views=cfg['C2ST_views'];nv=len(views);fig,axes=plt.subplots(1,3,figsize=(24,6),layout='constrained')
    if cs['supported']:
        for j,(n,res)in enumerate(cs['results'].items()):
            y=np.array([res[v]['balanced_accuracy']for v in views]);ci=np.array([res[v]['heldout_CI']['balanced_accuracy_95CI']for v in views]);axes[0].errorbar(np.arange(nv)+.08*(j-1.5),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,label=n,color=COLORS[n]);axes[2].plot(np.arange(nv),[res[v].get('training_balanced_accuracy',np.nan)for v in views],marker='o',ls='--',label=n+' train',color=COLORS[n]);axes[2].plot(np.arange(nv),y,marker='.',label=n+' holdout',color=COLORS[n])
        for j,(n,res)in enumerate(cs['paired_contrasts'].items()):
            y=np.array([res[v]['BA_delta_first_minus_second']for v in views]);ci=np.array([res[v]['BA_delta_95CI']for v in views]);axes[1].errorbar(np.arange(nv)+.08*(j-1),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=n.replace('_minus_',' - '),color=['#d55e00','#0072b2','#777777'][j])
    for ax in axes:ax.set_xticks(np.arange(nv),views,rotation=35,ha='right');ax.legend(fontsize=6)
    axes[0].set_ylim(.4,1.04);axes[0].axhline(.5,color='gray',ls='--');axes[0].set_ylabel('Heldout BA');axes[1].axhline(0,color='gray',ls='--');axes[1].set_ylabel('Paired BA delta;500 cluster95%CI');axes[2].set_ylabel('BA training vs heldout');fig.suptitle('Fixed HGB, common600/250 per-domain positions; not the sole PSD acceptance measure');save(fig,'c2st_paired')
    w=m['energy_conditional_width'];fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for k in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for n,tab in w['tables'].items():
                y=[r['components'][k][field] if r['valid'] and len(r['components'])==2 and r['components'][k][field] is not None else np.nan for r in tab['slices']];ax.plot(np.arange(1,9),y,marker='o',label=n,color=COLORS[n]);bad=[i+1 for i,r in enumerate(tab['slices'])if not r['valid']];ax.plot(bad,[-.01-.008*list(w['tables']).index(n)]*len(bad),marker='x',ls='',ms=4,color=COLORS[n])
            ax.set_xticks(np.arange(1,9));ax.set_xlabel('Frozen charge bin ID');ax.set_ylabel(field);ax.set_title(('Low'if k==0 else 'High')+' PSD density component; crosses unsupported')
    axes[0,0].legend(fontsize=7);save(fig,'energy_peak_width')
    fig,axes=plt.subplots(3,4,figsize=(21,15),layout='constrained');freq=np.fft.rfftfreq(100,d=1e-9)/1e6;t=np.arange(480)-120
    for n,d in domains.items():
        use=d['valid'];x=d['x'][use,:100];power=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);power[:,1:-1]*=2;axes[0,0].loglog(freq[1:],power.mean(axis=0)[1:],label=n,color=COLORS[n]);axes[0,1].plot(np.arange(1,21),np.nanmedian(d['acf'][use],axis=0),label=n,color=COLORS[n]);cdf(axes[0,2],d['features']['baseline_RMS_V'][use]*1000,n);cdf(axes[0,3],d['features']['rise_10_90_ns'][use],n)
        for r in range(4):
            a=d['x'][use&(d['region']==r)]
            if len(a):
                qq=np.quantile(a,[.1,.5,.9],axis=0);axes[1,r].plot(t,qq[1],label=n+' n='+str(len(a)),color=COLORS[n]);axes[1,r].fill_between(t,qq[0],qq[2],color=COLORS[n],alpha=.04)
        for j,key in enumerate(['prompt_curvature_RMS_V','postpeak_curvature_RMS_V']):axes[2,j].plot(np.arange(1,5),[np.median(d['extra'][key][use&(d['region']==r)])*1000 if np.any(use&(d['region']==r))else np.nan for r in range(4)],marker='o',label=n,color=COLORS[n])
        for j,key in enumerate(['cross50_index','peak_index']):
            if key in d['extra']:cdf(axes[2,j+2],d['extra'][key][use],n)
            elif key in d['features']:cdf(axes[2,j+2],d['features'][key][use],n)
    labels=[('Baseline frequency [MHz]','Mean PSD [V2/Hz]'),('Lag [ns]','Median ACF'),('Baseline RMS [mV]','CDF'),('Rise10-90 [ns]','CDF')]
    for ax,(xx,yy)in zip(axes[0],labels):ax.set_xlabel(xx);ax.set_ylabel(yy);ax.legend(fontsize=6)
    for r,ax in enumerate(axes[1]):ax.set_title('Output charge q'+str(r+1)+' waveform10/50/90%');ax.set_xlabel('Saved alignment [ns]');ax.set_ylabel('Baseline-subtracted V');ax.legend(fontsize=5)
    for ax,title in zip(axes[2],['Prompt100:150 second-difference [mV]','Postpeak150:220 second-difference [mV]','cross50 [ns]','Saved peak index']):ax.set_title(title);ax.set_xlabel('Output q'if ax in list(axes[2,:2])else title);ax.legend(fontsize=6)
    fig.suptitle('Waveform side effects: frozen residual parameters still depend on changed head');save(fig,'waveform_noise_side_effects')
