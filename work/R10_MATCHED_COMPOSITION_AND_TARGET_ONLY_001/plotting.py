import pathlib,json,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev

NAMES=['REAL','R3B','FULL_WAVE_STOCHASTIC','PULSE_STOCHASTIC_OFF','FIXED_SOURCE_STOCHASTIC']
COLORS={'REAL':'black','R3B':'tab:orange','FULL_WAVE_STOCHASTIC':'tab:blue','PULSE_STOCHASTIC_OFF':'tab:purple','FIXED_SOURCE_STOCHASTIC':'tab:green'}
def plots(A,domains,metrics,meta,root):
    figdir=root/'results/figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):fig.savefig(figdir/name,dpi=160,bbox_inches='tight');plt.close(fig)
    p=A['plan'];shape=p['shape'];fig,ax=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    keys=['TRAIN_GEN_counts','TRAIN_REAL_counts','qualified_cells','GEN_weight_per_cell','REAL_weight_per_cell','p_star']
    for a,k in zip(ax.flat,keys):
        im=a.imshow(np.asarray(p[k]).reshape(shape).T,origin='lower',aspect='auto');a.set(title=k,xlabel='REAL TRAIN logQ quantile bin',ylabel='REAL TRAIN PSD quantile bin');fig.colorbar(im,ax=a,shrink=.8)
    fig.suptitle(f'TRAIN-frozen overlap mass={p["overlap_mass"]:.4f}; DEV GEN ESS={A["DEV"]["GEN"]["ESS"]:.1f}, REAL ESS={A["DEV"]["REAL"]["ESS"]:.1f}\nOutside-support populations remain separately reported; observed matching is not physical source reweighting',fontsize=12);save(fig,'matching_support.png')
    views=['full_voltage','full_historical_shape','prompt_local','postpeak_local','tail_local','baseline_local'];fig,ax=plt.subplots(1,2,figsize=(16,5),constrained_layout=True)
    x=np.arange(6)
    for label,offset,color in [('original_R7',-.2,'gray'),('support_unweighted',0,'tab:blue'),('support_weighted',.2,'tab:green')]:
        points=[];lo=[];hi=[]
        for v in views:
            rec=A['original_full_population_cited_R7'][v] if label=='original_R7' else A['C2ST'][label+'|'+v]
            ba=rec.get('balanced_accuracy',rec.get('BA'));ci=rec.get('heldout_CI',{}).get('balanced_accuracy_95CI',rec.get('CI',{}).get('BA_95CI'))
            points.append(ba if ba is not None else np.nan);lo.append(ba-ci[0] if ci is not None else 0);hi.append(ci[1]-ba if ci is not None else 0)
        ax[0].errorbar(x+offset,points,yerr=[lo,hi],fmt='o',color=color,label=label,capsize=3)
    ax[0].axhline(.5,color='k',ls=':');ax[0].set(xticks=x,xticklabels=['V','shape','prompt','postpeak','tail','baseline'],ylabel='Balanced accuracy',ylim=(.35,1.02),title='Separate population filtering and weight effects');ax[0].legend(fontsize=8)
    for j,mode in enumerate(['support_unweighted','support_weighted']):
        rec=A['C2ST'][mode+'|coordinates'];ax[1].bar(j,rec.get('BA',np.nan),color=['tab:blue','tab:green'][j]);ax[1].text(j,.4,rec.get('reason',''),ha='center',fontsize=8,rotation=90)
    ax[1].set(xticks=[0,1],xticklabels=['support unweighted','support weighted'],ylim=(.35,1.02),ylabel='Balanced accuracy',title='logQ/PSD coordinate-only residual');ax[1].axhline(.5,color='k',ls=':');save(fig,'matching_c2st.png')
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,coord in enumerate(['logQ','PSD']):
        for col,field in enumerate(['W1','KS']):
            rec=A['coordinate_balance'][coord];y=[rec[k][field] for k in ['original','common_support_unweighted','common_support_weighted']];ax[row,col].bar(range(3),[v if v is not None else np.nan for v in y]);ax[row,col].set(xticks=range(3),xticklabels=['original','support','weighted'],ylabel=field,title=coord)
    fig.suptitle('Coarse TRAIN-bin matching does not guarantee DEV or within-bin balance');save(fig,'matching_coordinate_balance.png')
    B=metrics['B'];t=B['training'];fig,ax=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for field in ['G','D','adv','SW','CDF','baseline']:
        trace=t['loss_trace'];ax[0].plot([v['step'] for v in trace],[v[field] for v in trace],label=field)
    trace=t['validation_trace'];ax[1].plot([v['step'] for v in trace],[v['score'] for v in trace],'o-');ax[1].axvline(t['selected_step'],color='red',ls=':',label='TRAIN-val selected');ax[0].set(xlabel='updates',ylabel='TRAIN losses',title='Single fixed source, original loss');ax[1].set(xlabel='updates',ylabel='fixed TRAIN internal validation score',title='No DEV selection');ax[0].legend(fontsize=8);ax[1].legend();save(fig,'training_validation.png')
    fig,ax=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
    for a,n in zip(ax,NAMES):
        d=domains[n];u=d['valid'];a.hexbin(d['features']['Qtotal_mVns'][u],d['features']['PSD'][u],xscale='log',gridsize=36,mincnt=1,cmap='viridis');a.set(title=n.replace('_','\n')+'\nQC='+str(u.sum()),xlabel='Qtotal [mV ns]',ylabel='signed PSD',ylim=(-.12,.62))
    fig.suptitle('Complete native QC-valid populations; fixed-source rows are model draws, not independent G4 parents');save(fig,'charge_psd.png')
    widths=B['width_tables'];fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for comp in range(2):
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            for n,table in widths.items():
                xx=[];yy=[]
                for s in table['slices']:
                    xx.append(np.mean(s['logQ_edges']));yy.append(s['components'][comp][field] if s['valid'] and len(s['components'])==2 else np.nan)
                ax[comp,j].plot(xx,yy,'o-',label=n,color=COLORS[n],ms=3)
            ax[comp,j].set(xlabel='log observed Qtotal [mV ns]',ylabel=field,title=f'Ordered density component {comp}')
    ax[0,0].legend(fontsize=7);fig.suptitle('Frozen native 8-bin definition, single attempts and all unsupported gaps; not particle truth');save(fig,'energy_peak_width.png')
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);tt=np.arange(480)-120
    for n in NAMES:
        d=domains[n];u=d['valid'];col=COLORS[n]
        if not u.any():continue
        for j,key in enumerate(['x','shape']):
            v=d[key][u];median=np.median(v,axis=0);low,high=np.quantile(v,[.1,.9],axis=0);ax[0,j].plot(tt,median,color=col,label=n);ax[0,j].fill_between(tt,low,high,color=col,alpha=.08)
        for j,key in enumerate(['rise_10_90_ns','baseline_RMS_V']):
            v=np.sort(d['features'][key][u]);ax[1,j].plot(v*(1000 if j==1 else 1),np.arange(1,len(v)+1)/len(v),color=col,label=n)
    ax[0,0].set(xlim=(-30,130),xlabel='time relative to CFD [ns]',ylabel='baseline-subtracted V',title='Native median and10-90% band');ax[0,1].set(xlim=(-30,130),xlabel='time [ns]',ylabel='own-peak shape');ax[1,0].set(xlabel='pure pre-peak10-90% rise [ns]',ylabel='complete CDF');ax[1,1].set(xlabel='baseline RMS [mV]',ylabel='complete CDF');ax[0,0].legend(fontsize=6);save(fig,'waveforms_rise_noise.png')
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for n in NAMES:
        d=domains[n];u=d['valid'];b=d['V'][u,:100];b=b-b.mean(axis=1,keepdims=True);power=np.abs(np.fft.rfft(b,axis=1))[:,1:]**2;power/=np.maximum(power.sum(axis=1,keepdims=True),1e-30);ax[0].plot(np.arange(1,21),np.median(d['acf'][u,:20],axis=0),color=COLORS[n],label=n);ax[1].plot(np.fft.rfftfreq(100)*1000, np.r_[np.nan,np.median(power,axis=0)],color=COLORS[n],label=n)
    ax[0].set(xlabel='baseline ACF lag [ns]',ylabel='median ACF, full100 denominator');ax[1].set(xlabel='frequency [MHz]',ylabel='normalized non-DC power');ax[0].legend(fontsize=7);save(fig,'baseline_fluctuations.png')
    nn=np.load(root/'results/NN_PRIVATE.npz',allow_pickle=False);examples=np.load(root/'results/NN_EXAMPLES.npz',allow_pickle=False);fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for j,view in enumerate(['V','own_shape']):
        for name in ['FIXED','REAL_DEV']:
            v=np.sort(nn[view+'_'+name+'_distance']);ax[j].plot(v,np.arange(1,len(v)+1)/len(v),label=name)
        ax[j].set(xlabel='nearest TRAIN RMS distance ['+('V' if view=='V' else 'dimensionless')+']',ylabel='CDF',title=view+'; complete2592TRAIN reference');ax[j].legend()
    fig.suptitle('Nearness may reflect contraction or memory; no exact copy does not prove no memory');save(fig,'nearest_neighbours.png');fig,ax=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row,name in enumerate(['FIXED','REAL_DEV']):
        for j in range(3):
            ax[row,j].plot(tt,examples['V_'+name+'_query'][j],label=name+' query');ax[row,j].plot(tt,examples['V_'+name+'_neighbour'][j],label='nearest TRAIN',alpha=.7);ax[row,j].set(xlabel='time [ns]',ylabel='V',title='fixed hash example '+str(j+1));ax[row,j].legend(fontsize=7)
    save(fig,'nearest_fixed_examples.png');nn.close();examples.close()
