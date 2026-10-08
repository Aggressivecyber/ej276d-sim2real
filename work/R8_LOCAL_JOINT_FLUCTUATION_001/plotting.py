from __future__ import annotations
import pathlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
NAMES=['REAL','R3B','JOINT','JOINT_SIGNAL'];COLORS={'REAL':'#202020','R3B':'#e57e22','JOINT':'#227eb5','JOINT_SIGNAL':'#179449','TRAIN_REFERENCE':'#8a8a8a'}
ROOT=pathlib.Path(__file__).resolve().parent
def plots(domains,waves,metadata,m,source,ref,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for n in ['JOINT','JOINT_SIGNAL']:
        trace=json.loads((ROOT/'arms'/n/'TRAINING_TRACE.json').read_text());steps=[r['update'] for r in trace]
        axes[0,0].plot(steps,[r['validation'] for r in trace],'o-',label=n,color=COLORS[n])
        axes[0,1].plot(steps,[r['validation_parts']['new_joint'] for r in trace],'o-',label=n,color=COLORS[n])
        axes[1,0].plot(steps,[r['validation_parts']['conditional_CDF'] for r in trace],'o-',label=n,color=COLORS[n])
        axes[1,1].plot(steps,[r['validation_parts']['balanced_time_marginal'] for r in trace],'o-',label=n,color=COLORS[n])
        best=m['training'][n]['selected_update'];axes[0,0].axvline(best,color=COLORS[n],ls=':',lw=1)
    for ax,title in zip(axes.ravel(),['Frozen TRAIN validation selection loss','New unsmoothed local joint SW','Conditional CDF discrepancy','Old time-point marginal discrepancy']):ax.set_title(title);ax.set_xlabel('Optimizer update');ax.set_ylabel('Loss');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Same initial weights and draws; no DEV checkpoint selection');save(fig,'training_trace')
    views=['full_historical_shape','full_voltage','prompt_local','postpeak_local','tail_local','baseline_local'];fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ax,v in zip(axes.ravel(),views):
        for i,n in enumerate(['R3B','JOINT','JOINT_SIGNAL','REAL_TRAIN_vs_DEV','positive_vs_REAL']):
            r=m['C2ST'][n+'|'+v];y=r['balanced_accuracy'];lo,hi=r['heldout_CI']['balanced_accuracy_95CI'];col=COLORS.get(n,'#888888' if n.startswith('REAL') else '#ba4db1')
            ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color=col)
        ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.35,1.035);ax.set_xticks(range(5),['R3B','JOINT','JOINT_SIGNAL','REAL train/dev','positive'],rotation=25,ha='right');ax.set_title(v);ax.set_ylabel('Heldout balanced accuracy')
    fig.suptitle('Same 600/250 parent/group split; 500 cluster-bootstrap CI; old controls reused');save(fig,'regional_c2st')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for j,field in enumerate(['BA','AUC']):
        for i,v in enumerate(views):
            r=m['paired_C2ST_difference']['JOINT_SIGNAL_minus_JOINT|'+v];y=r[field+'_delta_first_minus_second'];lo,hi=r[field+'_delta_95CI']
            axes[j].errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color=COLORS['JOINT_SIGNAL'])
        axes[j].axhline(0,color='gray',ls='--');axes[j].set_xticks(range(len(views)),views,rotation=30,ha='right');axes[j].set_ylabel(field+' JOINT_SIGNAL minus JOINT');axes[j].grid(alpha=.15)
    fig.suptitle('Paired test parent/REAL groups; evaluator-conditional CI, not training uncertainty');save(fig,'paired_c2st_delta')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for n,ax in zip(NAMES,axes.ravel()):
        d=domains[n];v=d['valid'];ax.scatter(d['features']['Qtotal_mVns'][v],d['features']['PSD'][v],s=4,alpha=.3,color=COLORS[n]);ax.set_xscale('log')
        for edge in ref['CAL_charge_quantile_edges_mV_ns']:ax.axvline(edge,color='gray',lw=.7,ls='--')
        ax.set_title(n+f' QC={v.sum()}/2000');ax.set_xlabel('Signed total charge [mV ns]');ax.set_ylabel('Signed PSD = 1-Qfast/Qtotal');ax.set_ylim(-.08,.65)
    fig.suptitle('Complete QC distributions, including overlap/unknown; no energy reweighting');save(fig,'charge_psd')
    fig,axes=plt.subplots(2,4,figsize=(17,8));keys=['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','baseline_RMS_over_peak','ACF1']
    for ax,key in zip(axes.ravel(),keys):
        for n,d in domains.items():
            v=d['features'][key][d['valid']];v=np.sort(v[np.isfinite(v)]);ax.plot(v,np.arange(1,len(v)+1)/len(v),color=COLORS[n],label=n)
        ax.set_xlabel(key);ax.set_ylabel('Empirical CDF')
        if key in ['Qtotal_mVns','peak_V','baseline_RMS_V','baseline_RMS_over_peak']:ax.set_xscale('log')
    axes[0,0].legend(fontsize=8);fig.suptitle('Common saved-window observables; all QC/unknown retained');save(fig,'response_features')
    fig,axes=plt.subplots(2,3,figsize=(16,9));freq=np.fft.rfftfreq(100,d=1e-9)/1e6
    for n,d in domains.items():
        x=d['x'][d['valid'],:100];power=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);power[:,1:-1]*=2
        axes[0,0].loglog(freq[1:],power.mean(axis=0)[1:],color=COLORS[n],label=n)
        axes[0,1].plot(np.arange(1,21),np.nanmedian(d['acf'][d['valid']],axis=0),color=COLORS[n],label=n)
        for ax,key in zip(axes[1],['baseline_RMS_V','baseline_RMS_over_peak','ACF1']):
            z=d['features'][key][d['valid']];z=np.sort(z[np.isfinite(z)]);ax.plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS[n],label=n)
        axes[0,2].scatter(d['features']['Qtotal_mVns'][d['valid']],d['features']['baseline_RMS_V'][d['valid']],s=3,alpha=.16,color=COLORS[n])
    axes[0,0].set_xlabel('Frequency [MHz]');axes[0,0].set_ylabel('Baseline periodogram [V²/Hz]');axes[0,0].legend(fontsize=8);axes[0,1].set_xlabel('Lag [ns]');axes[0,1].set_ylabel('Median baseline ACF');axes[0,2].set_xscale('log');axes[0,2].set_xlabel('Charge [mV ns]');axes[0,2].set_ylabel('Baseline RMS [V]')
    for ax,key in zip(axes[1],['baseline_RMS_V','baseline_RMS_over_peak','ACF1']):ax.set_xlabel(key);ax.set_ylabel('CDF');ax.legend(fontsize=8)
    axes[1,0].set_xscale('log');axes[1,1].set_xscale('log');fig.suptitle('Pretrigger fluctuations remain a separate response test; not electronics calibration');save(fig,'fluctuations')
    fig,axes=plt.subplots(4,2,figsize=(14,16));t=np.arange(480)-120
    for r in range(4):
        for j,kind in enumerate(['x','shape']):
            ax=axes[r,j]
            for n,d in domains.items():
                v=d[kind][d['valid']&(d['region']==r)]
                if not len(v):continue
                qq=np.quantile(v,[.1,.5,.9],axis=0);ax.fill_between(t,qq[0],qq[2],color=COLORS[n],alpha=.07);ax.plot(t,qq[1],color=COLORS[n],label=n+f' n={len(v)}')
            ax.set_xlim(-20,200);ax.set_xlabel('Time from saved CFD [ns]');ax.set_ylabel('Baseline-subtracted V' if j==0 else 'Own-peak normalized');ax.set_title(f'Own-charge q{r+1}, '+('V' if j==0 else 'shape'));ax.legend(fontsize=8)
    fig.suptitle('Solid median / 10–90% event bands; output-defined charge regions');save(fig,'conditional_waveforms')
    fig,axes=plt.subplots(2,2,figsize=(13,9));same=m['same_N_conditional_reference']
    for j,region in enumerate(['prompt','postpeak']):
        key=region+'_curvature_RMS_V'
        for n in ['R3B','JOINT','JOINT_SIGNAL','TRAIN_REFERENCE']:
            rows=[same['q'+str(r+1)]['comparison'][n][key] for r in range(4)];q=[x['candidate']['quantiles'] for x in rows]
            med=np.array([x[3] if x else np.nan for x in q])*1000
            axes[0,j].plot(range(1,5),med,'o-',color=COLORS[n],label=n)
            axes[1,j].plot(range(1,5),[1000*x['distance']['W1'] if x['distance']['W1'] is not None else np.nan for x in rows],'o-',color=COLORS[n],label=n)
        vals=[same['q'+str(r+1)]['comparison']['R3B'][key]['REAL']['quantiles'][3]*1000 for r in range(4)];axes[0,j].plot(range(1,5),vals,'o-',color=COLORS['REAL'],label='REAL')
        axes[0,j].set_title(region+' second difference RMS median');axes[0,j].set_ylabel('mV');axes[1,j].set_title(region+' curvature distribution W1');axes[1,j].set_ylabel('W1 [mV]')
        for ax in axes[:,j]:ax.set_xticks(range(1,5),['q1','q2','q3','q4']);ax.set_xlabel('Own-charge matched-N region');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Common N across candidates/REAL/TRAIN reference; curvature includes shape and noise');save(fig,'conditional_curvature')
    width=m['energy_conditional_width'];centers=np.exp((np.asarray(width['edges_logQ'])[:-1]+np.asarray(width['edges_logQ'])[1:])/2);fig,axes=plt.subplots(2,3,figsize=(17,9))
    for k in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            for n in NAMES:
                vals=[r['components'][k][field] if r['valid'] and len(r['components'])==2 else None for r in width['tables'][n]['slices']]
                axes[k,j].plot(centers,[np.nan if x is None else x for x in vals],'o-',color=COLORS[n],label=n)
            ax=axes[k,j];ax.set_xscale('log');ax.set_xlabel('Signed charge slice center [mV ns]');ax.set_ylabel(field);ax.set_title(f'Mean-ordered component{k}: {field}');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Eight TRAIN-frozen narrow slices; unsupported fits are gaps, not particle truth');save(fig,'energy_peak_width')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for n,ax in zip(['R3B','JOINT','JOINT_SIGNAL'],axes):
        counts=np.array(m['source_nominal_condition'][n]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']);frac=counts/np.maximum(counts.sum(axis=1,keepdims=True),1)
        im=ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i in range(4):
            for j in range(5):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if frac[i,j]>.6 else 'black',fontsize=9)
        ax.set_xticks(range(5),['q1','q2','q3','q4','invalid']);ax.set_yticks(range(4),['q1','q2','q3','q4']);ax.set_xlabel('Observed output charge region');ax.set_ylabel('G4 nominal source charge region');ax.set_title(n)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Within-source-group fraction',shrink=.8);fig.suptitle('All 2000 parents retained; Qsource is a calibrated proxy, not incident true energy');fig.subplots_adjust(top=.83,bottom=.16,left=.05,right=.86,wspace=.35);fig.savefig(out/'parent_migration.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,14));indices=[0,500,1000,1999]
    for row,i in enumerate(indices):
        for n in ['R3B','JOINT','JOINT_SIGNAL']:
            for k,w in enumerate(waves[n]):
                d=ev.features(w[i:i+1]);axes[row,0].plot(t,d['x'][0],color=COLORS[n],alpha=1 if k==0 else .4,lw=1.3 if k==0 else .8,label=n+(' main' if k==0 else '') if k==0 else None)
                axes[row,1].plot(t,d['shape'][0],color=COLORS[n],alpha=1 if k==0 else .4,lw=1.3 if k==0 else .8,label=n if k==0 else None)
        for ax in axes[row]:ax.set_xlim(-20,220);ax.set_xlabel('Time from saved CFD [ns]');ax.legend(fontsize=8)
        axes[row,0].set_ylabel('Baseline-subtracted V');axes[row,1].set_ylabel('Own-peak normalized');axes[row,0].set_title(f'Fixed parent index{i}, hadron proxy={source["G4_condition"][i,1]:.3f}')
    fig.suptitle('Main and two prespecified random views; same new-arm draws, not three real acquisitions');save(fig,'stochastic_parent_examples')
    fig,axes=plt.subplots(2,2,figsize=(14,9));mp=metadata['JOINT_SIGNAL'][0]
    for i,ax in zip(indices,axes.ravel()):
        clean=mp['clean_V_512'][i];delta=mp['signal_delta_V_512'][i];peak=int(mp['clean_peak_index'][i]);tfull=np.arange(512)-peak
        ax.plot(tfull,clean*1000,color=COLORS['JOINT_SIGNAL'],label='Conditional clean expectation');right=ax.twinx();right.plot(tfull,delta*1000,color='#7b429d',lw=.7,label='Signal delta');ax.axvline(0,color='gray',ls='--');ax.set_xlim(-30,180)
        ax.set_title(f'Parent{i}, s(c)={mp["signal_scale_V"][i]*1000:.3f}mV');ax.set_xlabel('Time relative to first clean peak [ns]');ax.set_ylabel('Clean [mV]');right.set_ylabel('Delta [mV]');ax.legend(loc='upper left',fontsize=8);right.legend(loc='upper right',fontsize=8)
    fig.suptitle('Saved realized process: exactly zero before/at clean peak; statistical envelope hypothesis');save(fig,'signal_process')
