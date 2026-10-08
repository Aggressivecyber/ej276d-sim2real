from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
import evaluation as ev

A='FULL_WAVE_STOCHASTIC';B='PULSE_STOCHASTIC_OFF'
SHORT={'REAL':'REAL','SIM':'SIM','GEN_W0':'old W0','R3B':'R3B','R8_JOINT':'R8 JOINT','R8_JOINT_SIGNAL':'R8 SIGNAL',A:'FULL',B:'OFF','TRAIN_REFERENCE':'REAL train'}
COLORS={'REAL':'#151515','SIM':'#9a9a9a','GEN_W0':'#a87b50','R3B':'#dc8c21','R8_JOINT':'#76b0cb','R8_JOINT_SIGNAL':'#82ac78',A:'#2871b6',B:'#b33059','TRAIN_REFERENCE':'#888888'}
NAMES=['REAL','SIM','GEN_W0','R3B','R8_JOINT','R8_JOINT_SIGNAL',A,B]

def save(fig,name,out):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
def color(n):return COLORS[n]
def label(n):return SHORT[n]

def plots(domains,waves,metadata,m,source,ref,out):
    plt.rcParams.update({'font.size':9,'axes.grid':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for n in [A,B]:
        rows=m['training'][n]['validation_trace'];step=[r['step'] for r in rows]
        axes[0].plot(step,[r['score'] for r in rows],'o-',color=color(n),label=label(n))
        for field,style in [('SW','-'),('CDF','--'),('baseline',':')]:axes[1].plot(step,[r['parts'][field] for r in rows],style,color=color(n),label=label(n)+' '+field)
    axes[0].set_ylabel('TRAIN internal validation score');axes[1].set_ylabel('Unweighted validation term')
    for ax in axes:ax.set_xlabel('Common D/G update round');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Checkpoint selection uses TRAIN validation only; all finite arms retained');save(fig,'training_validation',out)
    views=['full_historical_shape','full_voltage','prompt_local','postpeak_local','tail_local','baseline_local']
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    ns=['R3B','R8_JOINT','R8_JOINT_SIGNAL',A,B,'REAL_TRAIN_vs_DEV','positive_vs_REAL']
    for view,ax in zip(views,axes.ravel()):
        for i,n in enumerate(ns):
            r=m['C2ST'][n+'|'+view]
            if 'balanced_accuracy' not in r:ax.text(i,.55,'unsupported',rotation=90,ha='center',fontsize=7);continue
            y=r['balanced_accuracy'];ci=r['heldout_CI']['balanced_accuracy_95CI']
            ax.errorbar(i,y,yerr=[[max(0,y-ci[0])],[max(0,ci[1]-y)]],fmt='o',capsize=3,color=COLORS.get(n,'#555555'))
        ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.35,1.035);ax.set_xticks(range(7),['R3B','R8J','R8S','FULL','OFF','RR','positive'],rotation=25,ha='right')
        ax.set_title(view);ax.set_ylabel('Heldout balanced accuracy');ax.grid(alpha=.15)
    fig.suptitle('Same frozen 600/250 split; group CI conditional on fitted evaluator; no response-training uncertainty');save(fig,'c2st',out)
    fig,axes=plt.subplots(2,4,figsize=(18,9));edges=ref['CAL_charge_quantile_edges_mV_ns']
    for n,ax in zip(NAMES,axes.ravel()):
        d=domains[n];good=d['valid'];x=d['features']['Qtotal_mVns'][good];y=d['features']['PSD'][good]
        if len(x):ax.hexbin(x,y,gridsize=50,mincnt=1,xscale='log',bins='log',cmap='viridis')
        for e in edges:ax.axvline(e,color='orange',lw=.6)
        ax.set_xscale('log');ax.set_ylim(-.12,.7);ax.set_xlim(600,45000)
        ax.set_title(label(n)+f' QC={good.sum()} invalid={(~good).sum()}');ax.set_xlabel('Signed Qtotal [mV ns]');ax.set_ylabel('Signed PSD')
    fig.suptitle('Complete QC-valid population, including overlap/unknown; outside display range remains in metrics');save(fig,'charge_psd',out)
    fig,axes=plt.subplots(4,2,figsize=(15,15));t=np.arange(480)-120
    ns=['REAL','SIM','R3B','R8_JOINT_SIGNAL',A,B]
    for r in range(4):
        for j,field in enumerate(['V','shape']):
            ax=axes[r,j]
            for n in ns:
                d=domains[n];v=d[field][d['valid']&(d['region']==r)]
                if not len(v):continue
                lo,med,hi=np.quantile(v,[.1,.5,.9],axis=0)
                scale=1000 if field=='V' else 1;ax.plot(t,med*scale,color=color(n),label=f'{label(n)} n={len(v)}')
                if n in ['REAL',A,B]:ax.fill_between(t,lo*scale,hi*scale,color=color(n),alpha=.1)
            ax.set_xlim(-25,240);ax.set_xlabel('Time from CFD [ns]');ax.set_ylabel('Voltage [mV]' if field=='V' else 'Own-peak shape')
            ax.set_title(f'Output charge q{r+1}');ax.legend(fontsize=7);ax.grid(alpha=.12)
    fig.suptitle('Observed output charge bins; unconditional parent allocation also reported separately');save(fig,'conditional_waveforms',out)
    fields=['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','ACF1','baseline_RMS_over_peak']
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for k,ax in zip(fields,axes.ravel()):
        vals=np.concatenate([domains[n]['features'][k][domains[n]['valid']&np.isfinite(domains[n]['features'][k])] for n in ns])
        if not len(vals):continue
        lo,hi=np.quantile(vals,[.001,.999]);bins=np.linspace(lo,hi,60) if hi>lo else np.linspace(lo-1,hi+1,60)
        for n in ns:
            d=domains[n];v=d['features'][k][d['valid']];v=v[np.isfinite(v)];hh,_=np.histogram(v,bins)
            ax.step((bins[:-1]+bins[1:])/2,hh/max(1,len(v))/np.diff(bins),where='mid',color=color(n),label=label(n))
        ax.set_xlabel(k);ax.set_ylabel('Density (full valid denominator)');ax.grid(alpha=.1)
    axes[0,0].legend(fontsize=7);fig.suptitle('Full-population feature marginals; display tails clipped only visually');save(fig,'response_features',out)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for n in ['REAL','R3B','R8_JOINT_SIGNAL',A,B]:
        d=domains[n];v=d['x'][d['valid'],:100]
        if not len(v):continue
        ac=np.nanmean(d['acf'][d['valid']],axis=0);axes[0].plot(np.arange(1,21),ac,'o-',ms=3,color=color(n),label=label(n))
        pow=np.mean(np.abs(np.fft.rfft(v,axis=1))**2,axis=0);pow=pow[1:]/max(1e-30,pow[1:].sum())
        axes[1].plot(np.fft.rfftfreq(100,1e-9)[1:]/1e6,pow,color=color(n),label=label(n))
        axes[2].hist(d['features']['baseline_RMS_V'][d['valid']]*1000,bins=40,density=True,histtype='step',color=color(n),label=label(n))
    axes[0].set_xlabel('Baseline ACF lag [ns]');axes[0].set_ylabel('Mean ACF');axes[1].set_xlabel('Frequency [MHz]');axes[1].set_ylabel('Non-DC normalized power');axes[2].set_xlabel('Baseline RMS [mV]');axes[2].set_ylabel('Density')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'fluctuations',out)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,region in enumerate(['prompt','postpeak']):
        for j,key in enumerate([region+'_curvature_RMS_V',region+'_relative_curvature']):
            ax=axes[i,j]
            for n in ['REAL','R3B','R8_JOINT_SIGNAL',A,B]:
                points=[]
                for r in range(4):
                    d=domains[n];v=d['extra'][key][d['valid']&(d['region']==r)];v=v[np.isfinite(v)]
                    points.append(np.median(v) if len(v) else np.nan)
                ax.plot(range(1,5),np.array(points)*(1000 if j==0 else 1),'o-',color=color(n),label=label(n))
            ax.set_xlabel('Observed output charge quartile');ax.set_ylabel('d2 RMS [mV]' if j==0 else 'd2 RMS / local RMS');ax.set_title(region);ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('Prompt 100:150 includes peak and early fall; it is not pure rising-edge curvature');save(fig,'conditional_curvature',out)
    tables=m['energy_conditional_width']['tables'];centers=np.exp((np.asarray(m['energy_conditional_width']['edges_logQ'])[:-1]+np.asarray(m['energy_conditional_width']['edges_logQ'])[1:])/2)
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for k in range(2):
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for n in ['REAL','R3B','R8_JOINT_SIGNAL',A,B]:
                ss=tables[n]['slices'];v=[s['components'][k][field] if s['valid'] and len(s['components'])==2 else np.nan for s in ss]
                ax.plot(centers,v,'o-',ms=4,color=color(n),label=label(n))
            ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator(centers[::2]));ax.xaxis.set_major_formatter(FixedFormatter([f'{v:.0f}' for v in centers[::2]]));ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel('Output Qtotal [mV ns]');ax.set_ylabel(field);ax.set_title(f'Mean-ordered density component {k}');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('Gaps denote unqualified density fits; components are not particle truth');save(fig,'energy_peak_width',out)
    fig=plt.figure(figsize=(15,5));gs=fig.add_gridspec(1,3,width_ratios=[1,1,.045],wspace=.25);axes=[fig.add_subplot(gs[0,i]) for i in range(2)];cax=fig.add_subplot(gs[0,2]);mx=max(np.max(m['source_nominal_condition'][n]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']) for n in [A,B])
    for n,ax in zip([A,B],axes):
        table=np.array(m['source_nominal_condition'][n]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']);im=ax.imshow(table,vmin=0,vmax=mx,cmap='Blues')
        for (i,j),v in np.ndenumerate(table):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>mx*.55 else 'black')
        ax.set_xticks(range(5),['q1','q2','q3','q4','invalid']);ax.set_yticks(range(4),['q1','q2','q3','q4']);ax.set_title(label(n));ax.set_xlabel('Observed output charge');ax.set_ylabel('Nominal parent charge')
    fig.colorbar(im,cax=cax,label='All parent records');fig.suptitle('Entire parent groups retained; output-conditioned and parent-conditioned conclusions differ');fig.savefig(out/'parent_migration.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for i,n in enumerate([A,B]):
        for j,field in enumerate(['PSD','Qtotal_mVns','peak_V']):
            r=m['three_view_within_parent_variation'][n]['features'][field]['within_parent_SD'];q=r['quantiles'];axes[i,j].bar(range(5),np.array(q)[[1,2,3,4,5]] if q else np.full(5,np.nan),color=color(n));axes[i,j].set_xticks(range(5),['5%','25%','50%','75%','95%']);axes[i,j].set_ylabel(field+' within-parent SD');axes[i,j].set_title(label(n)+f' n={r["defined_n"]}')
    fig.suptitle('Three random views per same parent; baseline remains stochastic in OFF');save(fig,'parent_variation',out)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for j,n in enumerate([A,B]):
        for i,(field,unit) in enumerate([('RMS_delta_V','RMS waveform change [V]'),('Qtotal_mVns','Signed charge change [mV ns]')]):
            rows=[]
            for control in ['shuffle','replace_first_TRAIN_fit']:
                r=m['source_input_controls'][n][control];rows.append(r[field] if i==0 else r['feature_delta'][field])
            for k,r in enumerate(rows):
                q=r['quantiles']
                if q:axes[i,j].errorbar(k,q[3],yerr=[[max(0,q[3]-q[1])],[max(0,q[5]-q[3])]],fmt='o',capsize=4,color=color(n))
            axes[i,j].set_xticks([0,1],['shuffle bundle','replace bundle']);axes[i,j].set_title(label(n));axes[i,j].set_ylabel(unit);axes[i,j].grid(alpha=.15)
    fig.suptitle('Same noises,64 fixed parents; sensitivity is not validated physical correspondence');save(fig,'source_controls',out)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for col,n in enumerate(['REAL',A,B]):
        d=domains[n];good=np.flatnonzero(d['valid']);good=good[ev.order(d['ids'][good],'R8C-FRONT-EXAMPLES')[:8]]
        for idx in good:
            pi=d['peak_index'][idx];a=max(100,pi-30);tt=np.arange(a,pi+1)-pi
            axes[0,col].plot(tt,d['x'][idx,a:pi+1]*1000,lw=.8,alpha=.8)
            axes[1,col].plot(tt,d['shape'][idx,a:pi+1],lw=.8,alpha=.8)
        axes[0,col].set_title(label(n)+' fixed-hash examples');axes[0,col].set_ylabel('Pre-peak voltage [mV]');axes[1,col].set_ylabel('Own-peak pre-peak shape')
        for i in [0,1]:axes[i,col].set_xlabel('Time before actual first peak [ns]');axes[i,col].grid(alpha=.15)
    fig.suptitle('Actual pre-peak branches; example selection independent of visual smoothness');save(fig,'pure_front_examples',out)
