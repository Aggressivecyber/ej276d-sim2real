from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plots(domains,m,source,comp,root):
    out=root/'results/figures';colors={'REAL':'black','SOURCE_CLEAN':'#9aa1a8','R18':'#4c78a8','CONT':'#f58518','TILT':'#e45756'};names=list(colors);cats=['pure_EE','pure_had','mixed'];edges=np.asarray(m['PSD_main']['edges_logQ']);centers=np.exp((edges[:-1]+edges[1:])/2)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,n):fig.savefig(out/(n+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,4.5));t=m['tilt']['tables'];b=np.arange(1,9)
    for field,lab,style in [('target_affinity','REAL TRAIN soft high-band','ko-'),('h_mean','source original h','s--'),('achieved_affinity','source tilted h','o-')]:axs[0].plot(b,[a.get(field,np.nan)for a in t],style,label=lab)
    axs[0].set_ylim(-.03,1.03);axs[0].set_ylabel('affinity (assumed correspondence)');axs[0].legend(fontsize=8)
    axs[1].plot(b,[a.get('lambda_value',np.nan)for a in t],'o-');axs[1].axhline(-np.log(10),ls='--',c='grey');axs[1].axhline(np.log(10),ls='--',c='grey');axs[1].set_ylabel('lambda; bounded +/- ln(10)')
    axs[2].bar(b-.18,[a['source_N']for a in t],.36,label='source N');axs[2].bar(b+.18,[a.get('Kish_ESS',0)for a in t],.36,label='tilted ESS');axs[2].legend()
    for ax in axs:ax.set_xlabel('frozen charge bin');ax.set_xticks(b);ax.grid(alpha=.15)
    fig.suptitle('TRAIN-only sampling intervention; boundary targets remain unachieved');fig.tight_layout();save(fig,'sampling_prior')
    fig,axs=plt.subplots(1,5,figsize=(22,4.5),sharex=True,sharey=True)
    qmin=min(d['features']['Qtotal_mVns'][d['valid']].min()for d in domains.values());qmax=max(d['features']['Qtotal_mVns'][d['valid']].max()for d in domains.values())
    for ax,n in zip(axs,names):
        d=domains[n];good=d['valid'];q=d['features']['Qtotal_mVns'];p=d['features']['PSD']
        if n=='REAL':ax.scatter(q[good],p[good],s=4,alpha=.35,c='black')
        else:
            for c,col in zip(cats,['#4c78a8','#f58518','#54a24b']):u=good&(comp==c);ax.scatter(q[u],p[u],s=4,alpha=.4,c=col,label=c+' '+str(u.sum()))
            ax.legend(fontsize=7)
        ax.set_xscale('log');ax.set_xlim(qmin*.85,qmax*1.15);ax.set_ylim(-.1,.7);ax.set_xlabel('Qtotal [mV ns]');ax.set_title(n+'\nQC '+str(good.sum())+'/2000');ax.grid(alpha=.1)
    axs[0].set_ylabel('signed PSD');fig.tight_layout();save(fig,'charge_psd_source_composition')
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for b,ax in enumerate(axs.flat):
        counts=[]
        for n,d in domains.items():
            q=d['features']['Qtotal_mVns'];l=np.log(np.maximum(q,1e-30));u=d['valid']&(l>=edges[b])&(l<=edges[b+1]if b==7 else l<edges[b+1]);x=np.sort(d['features']['PSD'][u]);counts.append(n+':'+str(len(x)))
            if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[n],label=n)
        ax.set_title('bin '+str(b+1)+'\n'+' / '.join(counts[:3])+'\n'+' / '.join(counts[3:]),fontsize=8);ax.set_xlim(-.1,.65);ax.grid(alpha=.1)
    axs[0,0].legend(fontsize=7);fig.supxlabel('signed PSD');fig.supylabel('CDF; observed output charge conditioning, unweighted main population');fig.tight_layout();save(fig,'conditional_psd_CDF')
    sb=np.searchsorted(edges,np.log(np.maximum(domains['SOURCE_CLEAN']['features']['Qtotal_mVns'],1e-30)),side='right')-1
    fig,axs=plt.subplots(3,8,figsize=(24,9),sharex=True,sharey=True)
    for row,c in enumerate(cats):
        for b in range(8):
            ax=axs[row,b];u=(comp==c)&(sb==b)
            for n in ['SOURCE_CLEAN','R18','CONT','TILT']:
                d=domains[n];x=np.sort(d['features']['PSD'][u&d['valid']])
                if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[n],label=n)
            de=m['PSD_main']['tables']['REAL']['bins'][b]
            if de['valley_low']is not None:ax.axvspan(de['valley_low'],de['valley_high'],color='grey',alpha=.15)
            ax.set_title('source bin '+str(b+1)+' n='+str(u.sum()),fontsize=8);ax.set_xlim(-.1,.65);ax.grid(alpha=.1)
            if b==0:ax.set_ylabel(c+'\nCDF')
    axs[0,0].legend(fontsize=7);fig.supxlabel('PSD; fixed source composition and source charge; shaded old TRAIN valley');fig.tight_layout();save(fig,'conditional_psd_by_composition')
    fig,axs=plt.subplots(2,2,figsize=(12,8));tabs=m['energy_conditional_width']['tables']
    for k in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD']):
            ax=axs[k,j]
            for n,t in tabs.items():v=[r['components'][k][field]if r['valid']and len(r['components'])==2 else np.nan for r in t['slices']];ax.plot(centers,v,'o-',color=colors[n],label=n)
            ax.set_xscale('log');ax.set_xlabel('observed Qtotal [mV ns]');ax.set_title(('lower'if k==0 else'upper')+' density component '+field);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8);fig.suptitle('Qualified density descriptions; gaps retain failure, no particle labels');fig.tight_layout();save(fig,'energy_peak_width')
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True)
    for b,ax in enumerate(axs.flat):
        for n,d in domains.items():
            bb=sb if n!='REAL'else np.searchsorted(edges,np.log(np.maximum(d['features']['Qtotal_mVns'],1e-30)),side='right')-1;u=d['valid']&(bb==b)
            if u.any():lo,med,hi=np.quantile(d['V'][u]*1000,[.1,.5,.9],axis=0);ax.fill_between(np.arange(480),lo,hi,color=colors[n],alpha=.09);ax.plot(med,color=colors[n],label=n)
        ax.set_xlim(85,300);ax.set_title('source bin '+str(b+1)+'; REAL own bin');ax.grid(alpha=.1)
    axs[0,0].legend(fontsize=7);fig.supxlabel('sample [ns]');fig.supylabel('V [mV]; median and 10–90%');fig.tight_layout();save(fig,'conditional_waveforms')
    fig,axs=plt.subplots(2,2,figsize=(14,9));tags=['same_input_same_noise','source_roll_no_fixed_points','fixed_source_changed_noise','amplitude_x0.5','amplitude_x1.5'];yy=np.arange(len(tags))
    for j,arm in enumerate(['CONT','TILT']):
        rows=m['source_probes'][arm]['rows'];axs[0,0].barh(yy+(j-.5)*.3,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],.3,color=colors[arm],label=arm)
        rr=np.load(root/'results'/(arm+'_REPEAT_NOISE_PRIVATE.npz'),allow_pickle=False);p=rr['PSD'];cc=rr['parent_composition'];sd=np.std(p,axis=1,ddof=1)
        for c in cats:xs=np.sort(sd[cc==c]);axs[0,1].step(xs,np.arange(1,len(xs)+1)/len(xs),where='post',label=arm+' '+c,color=colors[arm],ls={'pure_EE':'-','pure_had':'--','mixed':':'}[c])
    axs[0,0].set_yticks(yy,tags);axs[0,0].set_xlabel('mean RMS vs correct reference [mV]');axs[0,0].legend();axs[0,1].set_xlabel('same-parent PSD SD (8 noises)');axs[0,1].set_ylabel('CDF of parents');axs[0,1].legend(fontsize=7)
    for ax,field,lab in zip(axs[1],['baseline_RMS_V','ACF1'],['baseline RMS [mV]','baseline lag1 ACF']):
        for n,d in domains.items():
            if field not in d['features']and field not in d['extra']:continue
            a=(d['features'].get(field)if field in d['features']else d['extra'][field]);x=np.sort(a[d['valid']&np.isfinite(a)]);x=x*1000 if field=='baseline_RMS_V'else x
            if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[n],label=n)
        ax.set_xlabel(lab);ax.set_ylabel('CDF');ax.legend(fontsize=7);ax.grid(alpha=.1)
    fig.suptitle('Source controls are sensitivity tests; repeat noise is not independent physics');fig.tight_layout();save(fig,'noise_source_controls')
    fig,axs=plt.subplots(1,3,figsize=(19,5))
    for arm,tr in m['training'].items():
        rows=[a['validation']for a in tr['validation_trace']if 'validation'in a];steps=[a['step']for a in rows]
        axs[0].plot(steps,[a['J']for a in rows],'-o',color=colors[arm],label=arm+' weighted J');axs[0].plot(steps,[a['original_unweighted_J']for a in rows],'--',color=colors[arm],label=arm+' original J');axs[0].axvline(tr['selected_step'],color=colors[arm],alpha=.3)
    axs[0].set_xlabel('G/D updates');axs[0].set_ylabel('internal validation J (lower better)');axs[0].legend(fontsize=8)
    views=list(m['C2ST']['results']['CONT']);xx=np.arange(len(views));combined={'R18':m['C2ST']['reference_results_reused']['TRANSLATED'],**m['C2ST']['results']}
    for j,(n,v)in enumerate(combined.items()):axs[1].bar(xx+(j-1)*.22,[v[k]['balanced_accuracy']for k in views],.22,label=n,color=colors[n])
    axs[1].axhline(.5,color='grey',ls='--');axs[1].set_xticks(xx,views,rotation=55,ha='right',fontsize=8);axs[1].set_ylim(.45,1.03);axs[1].set_ylabel('fixed HGB holdout BA');axs[1].legend()
    for n in ['R18','CONT','TILT']:axs[2].plot(np.arange(1,9),[r['W1']for r in m['PSD_main']['vs_REAL'][n]['bins']],'o-',label=n,color=colors[n])
    axs[2].set_xlabel('observed charge bin');axs[2].set_ylabel('PSD W1 vs REAL');axs[2].legend();fig.tight_layout();save(fig,'training_c2st')
