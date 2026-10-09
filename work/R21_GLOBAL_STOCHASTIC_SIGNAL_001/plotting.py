from __future__ import annotations
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
NAMES=['REAL','SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']
COLOR={'REAL':'black','SOURCE_CLEAN':'#999fa6','R19_TILT':'tab:blue','R20_FINE':'tab:red','GLOBAL_STYLE':'tab:purple'}
def plots(domains,m,root):
    out=root/'results/figures';edges=np.asarray(m['PSD_main']['edges_logQ']);ref=json.loads((root/'REFERENCE_METRICS.json').read_text());valley=json.loads((root/'VALLEY_DEFINITION.json').read_text())
    with np.load(root/'DEV_SOURCE_METADATA.npz',allow_pickle=False)as a:comp=a['composition'].copy()
    def save(fig,n):fig.savefig(out/n,dpi=160,bbox_inches='tight');plt.close(fig)
    def cdf(ax,x,label,color):
        x=np.sort(np.asarray(x)[np.isfinite(x)])
        if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color)
    fig,axes=plt.subplots(1,5,figsize=(20,4),sharex=True,sharey=True,constrained_layout=True)
    qrange=np.concatenate([d['features']['Qtotal_mVns'][d['valid']]for d in domains.values()]);lims=(max(.1,qrange.min()*.9),qrange.max()*1.1)
    for ax,name in zip(axes,NAMES):
        d=domains[name];good=d['valid']
        if name=='REAL':ax.scatter(d['features']['Qtotal_mVns'][good],d['features']['PSD'][good],s=3,c='black',alpha=.35)
        else:
            for c,col in [('pure_EE','tab:blue'),('pure_had','tab:orange'),('mixed','tab:green')]:
                use=good&(comp==c);ax.scatter(d['features']['Qtotal_mVns'][use],d['features']['PSD'][use],s=4,color=col,alpha=.35,label=c+' '+str(int(use.sum())))
            ax.legend(fontsize=7)
        ax.set_title(name+'\nQC '+str(int(good.sum()))+'/2000');ax.set_xscale('log');ax.set_xlim(lims);ax.set_xlabel('Qtotal [mV ns]');ax.grid(alpha=.15)
    pvals=np.concatenate([d['features']['PSD'][d['valid']]for d in domains.values()]);axes[0].set_ylim(min(-.1,pvals.min()-.02),max(.7,pvals.max()+.02));axes[0].set_ylabel('signed PSD')
    save(fig,'charge_psd_source_composition.png')
    fig,axes=plt.subplots(2,4,figsize=(14,8),sharex=True,sharey=True,constrained_layout=True)
    for b,ax in enumerate(axes.flat):
        counts=[]
        for name in NAMES:
            d=domains[name];q=d['features']['Qtotal_mVns'];use=d['valid']&(np.log(np.maximum(q,1e-30))>=edges[b])&((np.log(np.maximum(q,1e-30))<edges[b+1])if b<7 else(np.log(np.maximum(q,1e-30))<=edges[b+1]))
            cdf(ax,d['features']['PSD'][use],name,COLOR[name]);counts.append(name+':'+str(int(use.sum())))
        ax.set_title('bin '+str(b+1)+'\n'+' / '.join(counts[:3])+'\n'+' / '.join(counts[3:]),fontsize=9);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supxlabel('signed PSD');fig.supylabel('CDF, observed own charge bins; unweighted main population');save(fig,'conditional_psd_CDF.png')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    tabs=m['energy_conditional_width']['tables'];x=np.exp((edges[:-1]+edges[1:])/2)
    for k in range(2):
        for j,field in enumerate(['mean','sigma_PSD']):
            ax=axes[k,j]
            for name in NAMES:
                v=[r['components'][k][field]if r['valid']and len(r['components'])==2 else np.nan for r in tabs[name]['slices']]
                ax.plot(x,v,'o-',color=COLOR[name],label=name)
            ax.set_xscale('log');ax.set_title(('lower'if k==0 else'upper')+' density component '+field);ax.set_xlabel('observed Qtotal [mV ns]');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle('Qualified descriptions only; gaps retain failure; no particle truth');save(fig,'energy_peak_width.png')
    fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True,sharex=True)
    sourceq=domains['SOURCE_CLEAN']['features']['Qtotal_mVns'];sourcebin=np.searchsorted(edges,np.log(sourceq),side='right')-1
    for b,ax in enumerate(axes.flat):
        for name in NAMES:
            d=domains[name];bb=np.searchsorted(edges,np.log(np.maximum(d['features']['Qtotal_mVns'],1e-30)),side='right')-1 if name=='REAL'else sourcebin;use=d['valid']&(bb==b)
            if use.any():
                lo,med,hi=np.quantile(d['V'][use]*1000,[.1,.5,.9],axis=0);t=np.arange(480);ax.plot(t,med,color=COLOR[name],label=name);ax.fill_between(t,lo,hi,color=COLOR[name],alpha=.10)
        ax.set_xlim(100,300);ax.set_title('source bin '+str(b+1)+'; REAL own bin');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supxlabel('sample [ns]');fig.supylabel('mV, median and 10–90% band');save(fig,'conditional_waveforms.png')
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    tr=m['training']['GLOBAL_STYLE'];v=[r['validation']for r in tr['validation_trace']];rv=[r['validation']for r in ref['training']['FINE']['validation_trace']]
    axes[0,0].plot([r['step']for r in rv],[r['J']for r in rv],color=COLOR['R20_FINE'],label='R20_FINE J20');axes[0,0].plot([r['step']for r in v],[r['J']for r in v],color=COLOR['GLOBAL_STYLE'],label='GLOBAL_STYLE J20')
    axes[0,0].scatter([tr['selected_step']],[tr['selected_validation']['J']],color=COLOR['GLOBAL_STYLE'],marker='*',s=110);axes[0,0].set_title('Common TRAIN-val J20, lower is better');axes[0,0].legend();axes[0,0].set_xlabel('G/D updates')
    axes[0,1].plot([r['step']for r in v],[r['style_reconstruction_L1']for r in v],label='val latent L1',color='tab:purple');rr=[r for r in tr['validation_trace']if'style_L1'in r]
    axes[0,1].plot([r['step']for r in rr],[r['style_L1']for r in rr],label='last train batch latent L1',color='tab:orange');axes[0,1].set_title('Latent recovery is not a physical success metric');axes[0,1].set_xlabel('G/D updates');axes[0,1].legend()
    views=list(m['C2ST']['results']['GLOBAL_STYLE']);xx=np.arange(len(views))
    for off,name in [(-.18,'R20_FINE'),(.18,'GLOBAL_STYLE')]:
        d=m['C2ST']['results'][name]if name=='GLOBAL_STYLE'else m['C2ST']['reference_results_reused'][name]
        axes[1,0].bar(xx+off,[d[z]['balanced_accuracy']for z in views],width=.36,color=COLOR[name],label=name)
    axes[1,0].axhline(.5,color='gray',ls='--');axes[1,0].set_ylim(.45,1.03);axes[1,0].set_xticks(xx,views,rotation=45,ha='right',fontsize=8);axes[1,0].set_ylabel('fixed HGB holdout BA');axes[1,0].legend(fontsize=8)
    for name in ['R19_TILT','R20_FINE','GLOBAL_STYLE']:
        axes[1,1].plot(np.arange(1,9),[r['W1']for r in m['PSD_main']['vs_REAL'][name]['bins']],'o-',color=COLOR[name],label=name)
    axes[1,1].set_xlabel('observed charge bin');axes[1,1].set_ylabel('PSD W1 vs REAL');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.15)
    save(fig,'training_c2st.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,unit in [(axes[0,0],'baseline_RMS_V',1000),(axes[0,1],'ACF1',1)]:
        for name in NAMES:
            if key in domains[name]['features']:a=domains[name]['features'][key]
            else:a=domains[name]['extra'][key]
            cdf(ax,a*unit,name,COLOR[name])
        ax.set_title('baseline RMS [mV]'if key=='baseline_RMS_V'else'baseline lag1 ACF');ax.set_ylabel('CDF');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    rows=m['source_probes']['rows'];tags=list(rows);axes[1,0].barh(tags,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],color=COLOR['GLOBAL_STYLE']);axes[1,0].set_xlabel('RMS vs correct reference [mV]');axes[1,0].set_title('Sensitivity is not calibrated physical correspondence')
    for mode,ls in [('joint8','-'),('style_only32','--'),('electronic_only32',':')]:
        r=m['random_decomposition']['groups'][mode]['rows']
        for kind,col in [('pure_EE','tab:blue'),('pure_had','tab:orange')]:
            values=[a['output_PSD']['sample_sd']for a in r if a['composition']==kind and a['output_PSD']['sample_sd']is not None];values=np.sort(values)
            if len(values):axes[1,1].step(values,np.arange(1,len(values)+1)/len(values),color=col,ls=ls,label=mode+' '+kind)
    axes[1,1].set_xlabel('same-parent PSD sample SD');axes[1,1].set_ylabel('CDF of parents');axes[1,1].legend(fontsize=7);save(fig,'noise_source_controls.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row,kind in enumerate(['pure_EE','pure_had']):
        for col,component in enumerate(['output_post_ADC','signal_pre_ADC','electronic_pre_ADC']):
            ax=axes[row,col];modes=['joint8','style_only32','electronic_only32'];stats=[m['random_decomposition']['groups'][n]['by_composition'][kind]['within_parent_waveform_RMS_V'][component]['full480']for n in modes]
            means=np.asarray([s['mean']for s in stats])*1000;ax.bar(np.arange(3),means,color=['tab:purple','tab:green','tab:orange']);ax.set_xticks(np.arange(3),modes,rotation=20);ax.set_title(kind+' '+component);ax.set_ylabel('mean parent RMS SD [mV]');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Signal/old electronic components are functional branches, not statistical means or independent physics');save(fig,'random_component_variation.png')
    with np.load(root/'results/style_only32_REPEAT_PRIVATE.npz',allow_pickle=False)as a:p=a['PSD'].reshape(32,32);valid=a['valid'].reshape(32,32);cat=a['parent_composition']
    rows=m['random_decomposition']['groups']['style_only32']['rows'];fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    for ax,kind in zip(axes,['pure_EE','pure_had']):
        inds=np.flatnonzero(cat==kind)
        for j,i in enumerate(inds):
            ax.scatter(np.full(np.sum(valid[i]),j),p[i][valid[i]],s=8,alpha=.5,color='tab:purple')
            b=rows[i]['source_Qbin_1based'];de=valley['bins'][b-1]if b is not None else None
            if de and de['valid']:ax.plot([j-.3,j+.3],[de['low'],de['low']],color='gray',alpha=.5);ax.plot([j-.3,j+.3],[de['high'],de['high']],color='gray',alpha=.5)
        ax.set_title(kind+': 16 fixed sources x32 styles\nold z/white fixed; gray segments = own-Q frozen valley');ax.set_xlabel('prespecified parent order (not a particle label)');ax.grid(alpha=.15)
    axes[0].set_ylabel('signed PSD');save(fig,'style_only_parent_PSD.png')
