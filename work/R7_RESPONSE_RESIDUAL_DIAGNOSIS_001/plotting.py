from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
from diagnostic_core import REGIONS

COLORS={'REAL':'#171717','GEN':'#e75b27','TRAIN_REFERENCE':'#7777aa','POSITIVE':'#999999','ADC_off':'#168b82','signal_block_mean20':'#aa4aaa'}

def plots(domains,transformed,m,cfg,figdir):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,n):fig.savefig(figdir/(n+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    c=m['C2ST'];contrasts=['GEN_vs_REAL','REAL_TRAIN_vs_DEV','positive_vs_REAL'];views=[r['view'] for r in cfg['matrix'] if r['operator']=='native' and r['contrast']=='GEN_vs_REAL']
    a=np.array([[c['native|'+v+'|'+co]['balanced_accuracy'] for co in contrasts] for v in views])
    fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(a,vmin=.45,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(np.arange(3),['R3B vs REAL DEV','REAL TRAIN vs DEV','Tail + noise positive']);ax.set_yticks(np.arange(len(views)),views)
    for i in range(len(views)):
        for j in range(3):
            ci=c['native|'+views[i]+'|'+contrasts[j]]['heldout_CI']['balanced_accuracy_95CI'];ax.text(j,i,f'{a[i,j]:.3f}\n[{ci[0]:.3f},{ci[1]:.3f}]',ha='center',va='center',fontsize=8,color='white' if a[i,j]<.85 else 'black')
    fig.colorbar(im,ax=ax,label='Heldout balanced accuracy');ax.set_title('Fixed regions: raw voltage or within-region centering / RMS only\n500 group/parent bootstrap intervals; no global features in regional views');fig.tight_layout();save(fig,'regional_c2st')
    ops=['native','fractional_align','coarse_ADC4','smooth_sigma1','align_then_smooth']
    fig,ax=plt.subplots(figsize=(12,6));xx=np.arange(len(ops))
    for j,(co,color) in enumerate(zip(contrasts,['#e75b27','#7777aa','#666666'])):
        y=np.array([c[o+'|full_historical_shape|'+co]['balanced_accuracy'] for o in ops]);ci=np.array([c[o+'|full_historical_shape|'+co]['heldout_CI']['balanced_accuracy_95CI'] for o in ops]);ax.errorbar(xx+(j-1)*.13,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o-',capsize=3,label=co,color=color)
    ax.axhline(.5,color='gray',ls='--');ax.set_xticks(xx,ops,rotation=15);ax.set_ylim(.4,1.04);ax.set_ylabel('Heldout balanced accuracy');ax.set_title('Same operator for both domains and controls; complete historical shape view');ax.legend();fig.tight_layout();save(fig,'common_transform_c2st')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,v in zip(axes,['full_voltage','full_historical_shape']):
        names=['native','ADC_off','signal_block_mean20'];y=np.array([c[o+'|'+v+'|GEN_vs_REAL']['balanced_accuracy'] for o in names]);ci=np.array([c[o+'|'+v+'|GEN_vs_REAL']['heldout_CI']['balanced_accuracy_95CI'] for o in names]);ax.errorbar(np.arange(3),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=4,color='#e75b27');ax.axhline(c['native|'+v+'|REAL_TRAIN_vs_DEV']['balanced_accuracy'],ls='--',color='#7777aa',label='same-view REAL TRAIN / DEV');ax.axhline(c['native|'+v+'|positive_vs_REAL']['balanced_accuracy'],ls=':',color='gray',label='same-view tail+noise positive');ax.set_xticks(np.arange(3),names,rotation=15);ax.set_ylim(.4,1.04);ax.set_title(v);ax.set_ylabel('Heldout balanced accuracy');ax.legend(fontsize=8)
    fig.suptitle('Fixed trained weights and same parent / z / white; signal-only block mean is not full acquisition');fig.tight_layout();save(fig,'fixed_intervention_c2st')
    fig,axes=plt.subplots(2,3,figsize=(15,8));keys=['cross50_phase','peak_index','ADC_lattice_RMS_steps','prompt_local_maxima','prompt_relative_curvature','prompt_local_ACF1']
    for ax,k in zip(axes.ravel(),keys):
        for n,d in domains.items():
            z=d['extra'][k][d['valid']];z=np.sort(z[np.isfinite(z)]);ax.plot(z,np.arange(1,len(z)+1)/len(z),label=n,color=COLORS[n])
        ax.set_xlabel(k);ax.set_ylabel('All QC event CDF')
        if k=='ADC_lattice_RMS_steps':ax.set_xscale('log')
    axes[0,0].legend(fontsize=8);fig.suptitle('Processing and local joint shape diagnostics; strict-sign extrema exclude plateaus');fig.tight_layout();save(fig,'processing_local_shape')
    fig,axes=plt.subplots(4,2,figsize=(14,15));t=np.arange(480)-120
    extra={'ADC_off':transformed['ADC_off']['GEN'],'signal_block_mean20':transformed['signal_block_mean20']['GEN']}
    for r in range(4):
        for j,k in enumerate(['x','shape']):
            ax=axes[r,j]
            for n,d in {**{k:domains[k] for k in ['REAL','GEN','TRAIN_REFERENCE']},**extra}.items():
                take=d['valid']&(d['region']==r);z=d[k][take,100:220]
                if len(z)<20:continue
                qq=np.quantile(z,[.1,.5,.9],axis=0);ax.plot(t[100:220],qq[1],label=n+f' n={len(z)}',color=COLORS[n]);
                if n in ['REAL','GEN']:ax.fill_between(t[100:220],qq[0],qq[2],color=COLORS[n],alpha=.08)
            ax.set_title(f'q{r+1}; '+('voltage, first100 baseline removed' if j==0 else 'descriptive own full peak normalization'));ax.set_xlabel('Time from saved integer CFD [ns]');ax.set_ylabel('V' if j==0 else 'Amplitude / full peak');ax.legend(fontsize=7)
    fig.suptitle('Local front / shoulder: median and10-90% bands; own Qt regions, not region-C2ST inputs');fig.tight_layout();save(fig,'conditional_fronts')
    fig,axes=plt.subplots(4,3,figsize=(16,13));keys=['prompt_local_maxima','prompt_relative_curvature','prompt_local_ACF1']
    for r in range(4):
        for j,k in enumerate(keys):
            ax=axes[r,j]
            for n,d in domains.items():
                take=d['valid']&(d['region']==r);a=d['extra'][k][take];a=np.sort(a[np.isfinite(a)])
                if len(a):ax.plot(a,np.arange(1,len(a)+1)/len(a),label=f'{n} n={len(a)}',color=COLORS[n])
            ax.set_title(f'q{r+1}: {k}');ax.set_ylabel('Conditional CDF');ax.legend(fontsize=7)
    fig.suptitle('Strict local extrema / unsmoothed curvature / joint adjacent-sample correlation');fig.tight_layout();save(fig,'conditional_curvature')
    fig,axes=plt.subplots(1,3,figsize=(16,5));freq=np.fft.rfftfreq(100,d=1e-9)/1e6
    for n,d in domains.items():
        a=d['x'][d['valid'],:100];power=np.mean(np.abs(np.fft.rfft(a,axis=1))**2,axis=0)/(1e9*100);power[1:-1]*=2
        axes[0].loglog(freq[1:],power[1:],label=n,color=COLORS[n]);axes[1].plot(np.arange(1,21),np.nanmedian(d['acf'][d['valid']],axis=0),label=n,color=COLORS[n]);a=np.sort(d['features']['baseline_RMS_V'][d['valid']]);axes[2].plot(a,np.arange(1,len(a)+1)/len(a),color=COLORS[n],label=n)
    axes[0].set_xlabel('Nonzero frequency [MHz]');axes[0].set_ylabel('Mean100ns baseline periodogram [V²/Hz]');axes[0].legend(fontsize=8);axes[1].set_xlabel('Lag [ns]');axes[1].set_ylabel('Median ACF, full100 denominator');axes[2].set_xlabel('Baseline RMS [V]');axes[2].set_ylabel('CDF');axes[2].set_xscale('log');fig.suptitle('Stationary saved-window baseline residuals; DC excluded from spectrum');fig.tight_layout();save(fig,'baseline_noise')
    fig,axes=plt.subplots(2,2,figsize=(13,9));keys=['PSD','rise_10_90_ns','Qtotal_mVns','peak_V']
    for ax,k in zip(axes.ravel(),keys):
        for name,color in [('REAL','#171717'),('GEN','#e75b27')]:
            vals=[m['after_operator_statistics'][o][name]['features'][k]['quantiles'][3] for o in ops];native=vals[0];ax.plot(np.arange(5),np.asarray(vals)-native,'o-',color=color,label=name)
        ax.set_xticks(np.arange(5),ops,rotation=20,ha='right');ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Median change from native: '+k);ax.legend()
    fig.suptitle('Both domains altered by common transforms; changes are information loss / redistribution');fig.tight_layout();save(fig,'common_observable_changes')
    fig,axes=plt.subplots(2,2,figsize=(14,9));opcolors=['#171717','#2878b5','#aa4aaa','#168b82','#e75b27']
    for col,name in enumerate(['REAL','GEN']):
        for component in [0,1]:
            ax=axes[component,col]
            for op,color in zip(ops,opcolors):
                rows=m['PSD_density_widths'][op][name]['slices'];x=[np.sqrt(np.prod(r['charge_edges_mVns'])) for r in rows];y=[r['components'][component]['sigma_PSD'] if len(r['components'])==2 else None for r in rows];ax.plot(x,[np.nan if v is None else v for v in y],'o-',color=color,label=op)
            ax.set_xscale('log');ax.set_xlabel('Signed Qt [mV ns]');ax.set_ylabel('Qualified density sigma PSD');ax.set_title(name+': component '+str(component));ax.legend(fontsize=8)
    fig.suptitle('Frozen TRAIN-defined narrow intervals / fixed single-init EM; failures remain gaps');fig.tight_layout();save(fig,'common_width_changes')
