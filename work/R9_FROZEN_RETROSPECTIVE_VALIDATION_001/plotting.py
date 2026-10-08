from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator,MaxNLocator
import evaluation as ev
A='FULL_WAVE_STOCHASTIC';B='PULSE_STOCHASTIC_OFF'
CAND=['R3B','JOINT','JOINT_SIGNAL',A,B]
NAMES=['REAL','SIM','GEN_W0',*CAND]
COL={'REAL':'#151515','SIM':'#9a9a9a','GEN_W0':'#a87b50','R3B':'#dc8c21','JOINT':'#76b0cb','JOINT_SIGNAL':'#82ac78',A:'#2871b6',B:'#b33059'}
SHORT={'REAL':'REAL','SIM':'SIM','GEN_W0':'old W0','R3B':'R3B','JOINT':'JOINT','JOINT_SIGNAL':'J+signal',A:'FULL',B:'OFF'}
def save(fig,name,out):fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
def plots(domains,allreal,m,ref,out):
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for name,ax in zip(NAMES,axes.ravel()):
        d=domains[name];good=d['valid'];q=d['features']['Qtotal_mVns'][good];ps=d['features']['PSD'][good]
        ax.hexbin(q,ps,gridsize=50,mincnt=1,xscale='log',bins='log',cmap='viridis')
        for edge in ref['CAL_charge_quantile_edges_mV_ns']:ax.axvline(edge,color='orange',lw=.6)
        ax.set_xscale('log');ax.set_xlim(600,45000);ax.set_ylim(-.12,.7);ax.set_xlabel('Signed Qtotal [mV ns]');ax.set_ylabel('Signed PSD')
        ax.set_title(SHORT[name]+f' QC={int(good.sum())}/2000')
    fig.suptitle('R9 fixed previously viewed REAL2000; all unknown included; display tails remain in full metrics');save(fig,'charge_psd',out)
    fig,axes=plt.subplots(4,2,figsize=(17,15));t=np.arange(480)-120
    for r in range(4):
        for j,field in enumerate(['x','shape']):
            ax=axes[r,j]
            for name in ['REAL',*CAND]:
                d=domains[name];v=d[field][d['valid']&(d['region']==r)]
                if not len(v):continue
                lo,med,hi=np.quantile(v,[.1,.5,.9],axis=0);scale=1000 if j==0 else 1
                ax.plot(t,med*scale,color=COL[name],label=SHORT[name]+f' n={len(v)}')
                if name in ['REAL','R3B',A,B]:ax.fill_between(t,lo*scale,hi*scale,color=COL[name],alpha=.07)
            ax.set_xlim(-25,240);ax.set_xlabel('Time from saved CFD [ns]');ax.set_ylabel('Baseline-subtracted V [mV]' if j==0 else 'Own-peak shape')
            ax.set_title(f'Observed output charge q{r+1}');ax.legend(fontsize=7);ax.grid(alpha=.12)
    fig.suptitle('Event median and 10-90% bands, not proof of complete distribution similarity');save(fig,'conditional_waveforms',out)
    fig,axes=plt.subplots(2,3,figsize=(18,9));views=['full_historical_shape','full_voltage','prompt_local','postpeak_local','tail_local','baseline_local']
    names=[*CAND,'REAL_vs_REAL','PERTURBED_REAL'];labels=['R3B','JOINT','J+sig','FULL','OFF','R-R','positive']
    for view,ax in zip(views,axes.ravel()):
        for i,name in enumerate(names):
            row=m['C2ST'][name+'|'+view];value=row['balanced_accuracy'];ci=row['heldout_CI']['balanced_accuracy_95CI']
            ax.errorbar(i,value,yerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],fmt='o',capsize=3,color=COL.get(name,'#555555'))
        ax.axhline(.5,color='gray',ls='--');ax.set_ylim(.35,1.035);ax.set_xticks(range(7),labels,rotation=25,ha='right');ax.set_ylabel('Heldout balanced accuracy');ax.set_title(view);ax.grid(alpha=.15)
    fig.suptitle('42 frozen HGB; identical candidate evaluation indices;500 conditional group-bootstrap replicates');save(fig,'c2st',out)
    tables=m['energy_conditional_width']['tables'];edges=np.asarray(m['energy_conditional_width']['edges_logQ']);centers=np.exp((edges[:-1]+edges[1:])/2)
    fig,axes=plt.subplots(2,3,figsize=(18,9))
    for k in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for name in ['REAL',*CAND]:
                value=[s['components'][k][field] if s['valid'] and len(s['components'])==2 else np.nan for s in tables[name]['slices']]
                ax.plot(centers,value,'o-',ms=4,color=COL[name],label=SHORT[name])
            ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator(centers[::2]));ax.xaxis.set_major_formatter(FixedFormatter([f'{q:.0f}' for q in centers[::2]]));ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel('Observed output Qtotal [mV ns]');ax.set_ylabel(field);ax.set_title(f'Mean-ordered density component {k}');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('Exact saved density fits reused; gaps remain failures; components are not particle truth');save(fig,'energy_peak_width',out)
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for i,ax in enumerate(axes.ravel()):
        for name in ['REAL',*CAND]:
            table=tables[name];grid=np.asarray(table['histogram_edges_PSD']);row=table['slices'][i];density=np.asarray(row['histogram_counts'])/max(1,row['N'])/np.diff(grid)
            ax.step((grid[:-1]+grid[1:])/2,density,where='mid',color=COL[name],label=SHORT[name]+f' n={int(row["N"])}')
        ax.set_xlabel('Signed PSD');ax.set_ylabel('Full slice density');ax.set_title(f'Frozen narrow bin {i+1}');ax.legend(fontsize=6)
    fig.suptitle('Complete native charge slices, including overlap/unknown and unqualified fitted regions');save(fig,'energy_psd_slices',out)
    fig,axes=plt.subplots(2,4,figsize=(18,9));fields=['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','ACF1','baseline_RMS_over_peak']
    for field,ax in zip(fields,axes.ravel()):
        for name in ['REAL','SIM',*CAND]:
            d=domains[name];x=d['features'][field][d['valid']];x=np.sort(x[np.isfinite(x)]);scale=1000 if field=='baseline_RMS_V' else 1
            ax.plot(x*scale,np.arange(1,len(x)+1)/len(x) if len(x) else [],color=COL[name],label=SHORT[name])
        ax.set_xlabel('Baseline RMS [mV]' if field=='baseline_RMS_V' else field);ax.set_ylabel('Complete valid CDF');ax.xaxis.set_major_locator(MaxNLocator(5));ax.grid(alpha=.1)
    axes[0,0].legend(fontsize=7);fig.suptitle('All native QC-valid events; no high-confidence selection or charge reweighting');save(fig,'response_features',out)
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    for name in ['REAL',*CAND]:
        d=domains[name];good=d['valid'];base=d['x'][good,:100];power=np.mean(np.abs(np.fft.rfft(base,axis=1))**2,axis=0)[1:];power/=max(1e-30,power.sum())
        axes[0].plot(np.arange(1,21),np.nanmedian(d['acf'][good],axis=0),color=COL[name],label=SHORT[name])
        axes[1].plot(np.fft.rfftfreq(100,1e-9)[1:]/1e6,power,color=COL[name],label=SHORT[name])
        rms=np.sort(d['features']['baseline_RMS_V'][good]*1000);axes[2].plot(rms,np.arange(1,len(rms)+1)/len(rms),color=COL[name],label=SHORT[name])
    axes[0].set_xlabel('Baseline ACF lag [ns]');axes[0].set_ylabel('Median ACF; full100-point denominator');axes[1].set_xlabel('Frequency [MHz]');axes[1].set_ylabel('Normalized non-DC power');axes[2].set_xlabel('Baseline RMS [mV]');axes[2].set_ylabel('Complete CDF')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('Saved-window fluctuation comparison, not independent detector-noise calibration');save(fig,'fluctuations',out)
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for i,region in enumerate(['prompt','postpeak']):
        for j,key in enumerate([region+'_curvature_RMS_V',region+'_relative_curvature']):
            ax=axes[i,j]
            for name in ['REAL',*CAND]:
                d=domains[name];points=[]
                for r in range(4):
                    v=d['extra'][key][d['valid']&(d['region']==r)];v=v[np.isfinite(v)];points.append(np.median(v) if len(v) else np.nan)
                ax.plot(range(1,5),np.asarray(points)*(1000 if j==0 else 1),'o-',color=COL[name],label=SHORT[name])
            ax.set_xlabel('Observed output charge quartile');ax.set_ylabel('d2 RMS [mV]' if j==0 else 'd2 RMS / local RMS');ax.set_title(region);ax.legend(fontsize=7);ax.grid(alpha=.12)
    fig.suptitle('Prompt100:150 includes peak and early fall; it is not a pure rising-edge window');save(fig,'conditional_curvature',out)
    fig=plt.figure(figsize=(18,10));gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,.045],wspace=.3,hspace=.35)
    vmax=max(np.max(m['source_nominal_condition'][n]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']) for n in CAND)
    for p,name in enumerate(CAND):
        ax=fig.add_subplot(gs[p//3,p%3]);table=np.asarray(m['source_nominal_condition'][name]['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid']);im=ax.imshow(table,vmin=0,vmax=vmax,cmap='Blues')
        for (i,j),value in np.ndenumerate(table):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>vmax*.55 else 'black')
        ax.set_xticks(range(5),['q1','q2','q3','q4','invalid']);ax.set_yticks(range(4),['q1','q2','q3','q4']);ax.set_xlabel('Observed output charge');ax.set_ylabel('Nominal source charge');ax.set_title(SHORT[name])
    note=fig.add_subplot(gs[1,2]);note.axis('off');note.text(.05,.65,'All2000 source parents retained.\nSame nominal source rows across models.\nOutput bins are post-response selections.\nAssociation is not validated correspondence.',va='top',fontsize=10)
    cax=fig.add_subplot(gs[:,3]);fig.colorbar(im,cax=cax,label='All parent records');fig.suptitle('Source-nominal and output-conditioned comparisons reported together');fig.savefig(out/'parent_migration.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(18,9))
    for j,name in enumerate(['REAL',A,B]):
        d=domains[name];ix=np.flatnonzero(d['valid']);ix=ix[ev.order(d['ids'][ix],'R8C-FRONT-EXAMPLES')[:8]]
        for i in ix:
            peak=d['peak_index'][i];left=max(100,peak-30);t=np.arange(left,peak+1)-peak
            axes[0,j].plot(t,d['x'][i,left:peak+1]*1000,lw=.8);axes[1,j].plot(t,d['shape'][i,left:peak+1],lw=.8)
        for k in [0,1]:axes[k,j].set_xlabel('Time before actual first peak [ns]');axes[k,j].grid(alpha=.12)
        axes[0,j].set_title(SHORT[name]+' fixed hash examples');axes[0,j].set_ylabel('Pre-peak voltage [mV]');axes[1,j].set_ylabel('Own-peak pre-peak shape')
    fig.suptitle('Pure pre-peak examples independently selected by frozen hash; complete rise CDF above');save(fig,'pure_front_examples',out)
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for j,key in enumerate(['PSD','Qtotal_mVns']):
        for name in ['REAL',*CAND]:
            d=domains[name];x=np.sort(d['features'][key][d['valid']]);axes[j].plot(x,np.arange(1,len(x)+1)/len(x),color=COL[name],label=SHORT[name])
        x=np.sort(allreal['features'][key][allreal['valid']]);axes[j].plot(x,np.arange(1,len(x)+1)/len(x),'k--',label='All REAL7583')
        axes[j].set_xlabel(key);axes[j].set_ylabel('Complete CDF');axes[j].legend(fontsize=7)
    axes[1].set_xscale('log');ys=[m['joint_sliced_W1'][n]['W1'] for n in ['SIM','GEN_W0',*CAND]]
    axes[2].bar(range(7),ys,color=[COL[n] for n in ['SIM','GEN_W0',*CAND]])
    rr=m['REAL_REAL_joint'];axes[2].axhspan(*rr['range5_95'],color='gray',alpha=.25,label='64 R-R5-95% descriptive range');axes[2].axhline(rr['median'],color='black',ls='--')
    axes[2].set_xticks(range(7),['SIM','W0','R3B','JOINT','J+sig','FULL','OFF'],rotation=25,ha='right');axes[2].set_ylabel('2D sliced W1; fixed TRAIN scale');axes[2].legend(fontsize=7)
    fig.suptitle('Previously viewed same-acquisition main2000 and all7583 supplement; R-R is not a tolerance');save(fig,'all_real_supplement_joint',out)
