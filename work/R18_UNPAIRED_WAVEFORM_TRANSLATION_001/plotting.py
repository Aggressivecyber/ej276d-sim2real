from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev

def plots(domains,m,source,output,comp,root):
    figdir=root/'results/figures';colors={'REAL':'black','SOURCE_CLEAN':'#4c78a8','TRANSLATED':'#e45756'};cats=['pure_EE','pure_had','mixed'];catcolors=['#4c78a8','#f58518','#54a24b'];edges=np.array(m['PSD_main']['edges_logQ']);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,n):fig.savefig(figdir/(n+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    for ax,n in zip(axs,['SOURCE_CLEAN','TRANSLATED','REAL']):
        d=domains[n];q=d['features']['Qtotal_mVns'];p=d['features']['PSD'];good=d['valid']
        if n=='REAL':ax.scatter(q[good],p[good],s=5,alpha=.35,c='black')
        else:
            for c,col in zip(cats,catcolors):ix=good&(comp==c);ax.scatter(q[ix],p[ix],s=6,alpha=.5,c=col,label=c+' n='+str(ix.sum()))
            ax.legend(fontsize=8)
        ax.set_xscale('log');ax.set_title(n+' QC '+str(good.sum())+'/2000');ax.set_xlabel('Qtotal [mV ns]');ax.set_ylim(-.1,.7);ax.grid(alpha=.15)
    qmin=min(np.min(d['features']['Qtotal_mVns'][d['valid']])for d in domains.values());qmax=max(np.max(d['features']['Qtotal_mVns'][d['valid']])for d in domains.values());axs[0].set_xlim(qmin*.85,qmax*1.15);axs[0].set_ylabel('signed PSD = 1-Qfast/Qtotal');save(fig,'charge_psd_source_composition')
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for i,ax in enumerate(axs.flat):
        ns=[]
        for n,d in domains.items():
            q=d['features']['Qtotal_mVns'];ix=d['valid']&(np.log(np.maximum(q,1e-30))>=edges[i])&(np.log(np.maximum(q,1e-30))<(edges[i+1]+(1e-12 if i==7 else 0)));p=np.sort(d['features']['PSD'][ix]);ns.append(n+':'+str(len(p)))
            if len(p):ax.step(p,np.arange(1,len(p)+1)/len(p),where='post',color=colors[n],label=n)
        ax.set_title('bin '+str(i+1)+'\n'+' / '.join(ns),fontsize=8);ax.set_xlim(-.1,.65);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=7);fig.supxlabel('signed PSD');fig.supylabel('empirical CDF (all QC in each observed charge bin)');fig.tight_layout();save(fig,'conditional_psd_CDF')
    fig,axs=plt.subplots(3,8,figsize=(24,9),sharex=True,sharey=True)
    src=domains['SOURCE_CLEAN'];sq=src['features']['Qtotal_mVns'];sb=np.searchsorted(edges,np.log(sq),side='right')-1
    for row,c in enumerate(cats):
        for i in range(8):
            ax=axs[row,i];use=(comp==c)&(sb==i)
            for n in ['SOURCE_CLEAN','TRANSLATED']:
                d=domains[n];p=np.sort(d['features']['PSD'][use&d['valid']]);
                if len(p):ax.step(p,np.arange(1,len(p)+1)/len(p),where='post',color=colors[n],label=n)
            ax.set_title('source bin '+str(i+1)+' n='+str(use.sum()),fontsize=8);ax.set_xlim(-.1,.65);ax.grid(alpha=.15)
            if i==0:ax.set_ylabel(c+'\nCDF')
    axs[0,0].legend(fontsize=7);fig.supxlabel('PSD; source-composition and SOURCE charge conditioning, not particle labels');fig.tight_layout();save(fig,'conditional_psd_by_composition')
    fig,axs=plt.subplots(2,2,figsize=(12,9));tables=m['energy_conditional_width']['tables'];qmid=np.exp((edges[:-1]+edges[1:])/2)
    for k in [0,1]:
        for col,field in enumerate(['mean','sigma_PSD']):
            ax=axs[k,col]
            for n,table in tables.items():
                vals=[r['components'][k][field]if r['valid']and len(r['components'])==2 else np.nan for r in table['slices']];ax.plot(qmid,vals,'o-',color=colors[n],label=n)
            ax.set_xscale('log');ax.set_title(('lower'if k==0 else'upper')+' density component '+field);ax.set_xlabel('observed Qtotal [mV ns]');ax.grid(alpha=.15)
    axs[0,0].legend();fig.suptitle('Qualified two-density descriptions only; gaps retain failures, not particle truth');fig.tight_layout();save(fig,'energy_peak_width')
    d=domains['REAL'];realbin=np.searchsorted(edges,np.log(np.maximum(d['features']['Qtotal_mVns'],1e-30)),side='right')-1
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for i,ax in enumerate(axs.flat):
        for n,dd in domains.items():
            b=sb if n!='REAL'else realbin;ix=dd['valid']&(b==i)
            if ix.any():
                y=dd['V'][ix]*1000;lo,med,hi=np.quantile(y,[.1,.5,.9],axis=0);ax.fill_between(np.arange(480),lo,hi,color=colors[n],alpha=.12);ax.plot(med,color=colors[n],label=n)
        ax.set_title('source bin '+str(i+1)+'; REAL own Q bin');ax.set_xlim(85,300);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=7);fig.supxlabel('saved sample [ns]');fig.supylabel('V [mV]; median and 10–90%');fig.tight_layout();save(fig,'conditional_waveforms')
    z=np.load(root/'results/SOURCE_PROBES_PRIVATE.npz',allow_pickle=False);tags=z['probe_kind'];fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,field,label in zip(axs[:2],['PSD','Q_mVns'],['PSD','Qtotal [mV ns]']):
        xx=[0,.25,.5,.75,1];vals=[z[field][tags=='analytic_fast_slow_'+str(t)]for t in xx];med=[np.median(a)for a in vals];lo=[np.quantile(a,.1)for a in vals];hi=[np.quantile(a,.9)for a in vals];ax.errorbar(xx,med,yerr=[np.array(med)-lo,np.array(hi)-med],fmt='o-');ax.set_xlabel('analytic slow-shape fraction');ax.set_ylabel(label);ax.grid(alpha=.2)
    rows=m['source_probes']['rows'];names=['same_input_same_noise','source_roll_no_fixed_points','fixed_source_changed_noise','amplitude_x0.5','amplitude_x1.5'];axs[2].barh(names,[rows[n]['waveform_RMS_vs_main_parent_V']['mean']*1000 for n in names]);axs[2].set_xlabel('mean output RMS change [mV]');fig.suptitle('Fixed-noise source probes: nonzero response is not correct physical correspondence');fig.tight_layout();save(fig,'source_controls')
    r=np.load(root/'results/REPEAT_NOISE_PRIVATE.npz',allow_pickle=False);p=r['PSD'];rc=r['parent_composition'];fig,axs=plt.subplots(1,3,figsize=(16,5))
    for c,col in zip(cats,catcolors):
        use=rc==c
        if use.any():axs[0].hist(np.std(p[use],axis=1,ddof=1),bins=20,histtype='step',label=c,color=col);axs[1].scatter([x['source_PSD']for x in m['same_parent_noise']['rows']if x['composition']==c],np.mean(p[use],axis=1),s=15,color=col,label=c);axs[2].scatter(np.mean(p[use],axis=1),np.ptp(p[use],axis=1),s=15,color=col,label=c)
    axs[0].set_xlabel('within-parent PSD SD across 8 noises');axs[0].set_ylabel('parent count');axs[1].set_xlabel('source PSD');axs[1].set_ylabel('mean translated PSD');axs[2].set_xlabel('mean translated PSD');axs[2].set_ylabel('within-parent PSD range');axs[0].legend();fig.suptitle('128 fixed parents, eight stochastic views; no per-parent mixture fitting');fig.tight_layout();save(fig,'repeated_noise')
    fig,axs=plt.subplots(2,2,figsize=(14,9));trace=m['training']['validation_trace'];vv=[r['validation']for r in trace];axs[0,0].plot([r['step']for r in vv],[r['J']for r in vv],'o-');axs[0,0].axvline(m['training']['selected_step'],color='red',ls='--',label='selected');axs[0,0].set_xlabel('G updates');axs[0,0].set_ylabel('internal validation J');axs[0,0].legend()
    tr=[r for r in trace if 'components'in r]
    for key in ['adversarial','content_NCE','identity','domain_SW','PSD_rank','Q_relative']:axs[0,1].plot([r['step']for r in tr],[r['components'][key]for r in tr],label=key)
    axs[0,1].set_xlabel('G updates');axs[0,1].set_ylabel('unweighted TRAIN loss');axs[0,1].legend(fontsize=7)
    views=list(m['C2ST']['results'].get('TRANSLATED',{}));x=np.arange(len(views))
    for j,n in enumerate(['SOURCE_CLEAN','TRANSLATED','REAL_REAL']):
        rows=m['C2ST']['results'].get(n,{})
        if rows:
            ba=np.array([rows[v]['balanced_accuracy']for v in views]);train=np.array([rows[v]['training_balanced_accuracy']for v in views]);axs[1,0].plot(x,ba,'o-',label=n);axs[1,1].plot(x,train,'o-',label=n)
    for ax in axs[1]:ax.set_xticks(x,views,rotation=25,ha='right');ax.axhline(.5,c='gray',ls='--');ax.set_ylim(.45,1.02);ax.legend(fontsize=7)
    axs[1,0].set_ylabel('holdout C2ST balanced accuracy');axs[1,1].set_ylabel('TRAIN C2ST balanced accuracy');fig.tight_layout();save(fig,'training_and_C2ST')
