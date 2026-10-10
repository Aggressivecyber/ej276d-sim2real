from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(root):
    root=Path(root);figdir=root/'figures';figdir.mkdir(exist_ok=True)
    full=json.loads((root/'FULL_TRAIN_SUMMARY.json').read_text('utf8'))
    batches=json.loads((root/'GRADIENT_BATCH_RECORDS.json').read_text('utf8'))
    summary=json.loads((root/'GRADIENT_SUMMARY.json').read_text('utf8'))
    trace=json.loads((root/'SAVED_R59_TRAINING_TRACE_SUMMARY.json').read_text('utf8'))
    with np.load(root/'FULL_TRAIN_DIAGNOSTICS_PRIVATE.npz',allow_pickle=False)as z:d={k:z[k].copy()for k in z.files}
    with np.load(root/'GRADIENT_DIAGNOSTICS_PRIVATE.npz',allow_pickle=False)as z:g={k:z[k].copy()for k in z.files}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    names=summary['component_names'];figures=[]
    def save(fig,name):
        fig.savefig(figdir/name,dpi=140,bbox_inches='tight');plt.close(fig);figures.append('figures/'+name)
    regions=[-1,*range(8),8];labels=['below','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','above']
    fig,axs=plt.subplots(2,5,figsize=(19,8.5))
    for ax,b,label in zip(axs.flat,regions,labels):
        a=d['source_output_psd'][d['source_raw_region']==b];r=d['REAL_reference_psd'][d['REAL_raw_region']==b];c=d['REAL_output_psd'][d['REAL_raw_region']==b]
        bins=np.linspace(min(a.min(),r.min(),c.min()),max(a.max(),r.max(),c.max()),81)
        ax.hist(r,bins=bins,density=True,histtype='step',color='black',label='REAL reference')
        ax.hist(a,bins=bins,density=True,histtype='step',color='#116ac4',label='R59 SOURCE')
        ax.hist(c,bins=bins,density=True,histtype='step',color='#df7805',label='R59 on REAL')
        ax.set(title=f'Original {label}: S={len(a)}, R={len(r)}',xlabel='Signed PSD',ylabel='Density')
    handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',ncol=3,bbox_to_anchor=(.5,.93))
    fig.suptitle('Frozen R59 on all TRAIN parents: every original charge region retained',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.88]);save(fig,'full_TRAIN_PSD_regions.png')

    short=['full','PSD','order','valley','noise','REAL id','box','V480 /4','U480 /4','dU /4','d2U /4','order overlap','order resolved','full inside','full outside','PSD overall','PSD Q1-7']
    gram=g['parameter_Gram'];norm=np.sqrt(np.maximum(np.diagonal(gram,axis1=1,axis2=2),0));den=norm[:,:,None]*norm[:,None,:]
    cosine=np.divide(gram,den,out=np.full_like(gram,np.nan),where=den>1e-30)
    median=np.full((17,17),np.nan)
    for i in range(17):
        for j in range(17):
            a=cosine[:,i,j];median[i,j]=np.median(a[np.isfinite(a)])if np.isfinite(a).any()else np.nan
    fig,ax=plt.subplots(figsize=(12.5,11));cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(np.ma.masked_invalid(median),vmin=-1,vmax=1,cmap=cmap)
    ax.set_xticks(range(17),short,rotation=60,ha='right');ax.set_yticks(range(17),short)
    for i in range(17):
        for j in range(17):
            if np.isfinite(median[i,j]):ax.text(j,i,f'{median[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white'if abs(median[i,j])>.75 else'black')
    ax.set_title('Median parameter-gradient cosine over 32 fixed TRAIN batches\nGrey = undefined zero norm; raw gradients, not Adam updates',pad=15)
    fig.colorbar(im,ax=ax,shrink=.8,label='Cosine');fig.tight_layout();save(fig,'gradient_cosine_matrix.png')

    fig,axs=plt.subplots(2,2,figsize=(14,9));order=full['source_order']['groups']
    axs[0,0].bar(['Roundoff-compatible','Resolved'],[order[k]['nonzero_pair_fraction']for k in ['roundoff_compatible','resolved']],color=['#c55744','#498e68'])
    axs[0,0].set(ylim=(0,1),ylabel='Fraction of strictly nonzero ordered parent pairs',title='All 6,593 original SOURCE parents')
    x=np.arange(1,33)
    for key,label,color in [('roundoff_response_cosine','Overlap order vs response','#c55744'),('outside_inside_cosine','Outside vs inside full-window','#3579bb'),('valley_PSD_cosine','Valley vs PSD','#8963b0')]:
        vals=[np.nan if r['marker_inputs'][key]is None else r['marker_inputs'][key]for r in batches]
        axs[0,1].plot(x,vals,'.-',label=label,color=color,linewidth=1)
    axs[0,1].axhline(-.25,color='black',ls='--');axs[0,1].set(ylim=(-1.05,1.05),xlabel='Fixed batch',ylabel='Gradient cosine',title='Undefined values remain gaps');axs[0,1].legend(fontsize=8)
    for j,name in enumerate(names[:7]):axs[1,0].plot(x,norm[:,j],label=short[j],linewidth=1)
    axs[1,0].set_yscale('symlog',linthresh=1e-3);axs[1,0].set(xlabel='Fixed batch',ylabel='Parameter gradient norm',title='Original seven losses');axs[1,0].legend(ncol=4,fontsize=9)
    flags=summary['diagnostic_markers'];keys=list(flags);values=[flags[k]['batches']for k in keys]
    axs[1,1].bar(['Storage-order','Outside-region','Valley-PSD'],values,color='#52789b');axs[1,1].axhline(24,color='black',ls='--',label='Required: 24/32')
    for j,v in enumerate(values):axs[1,1].text(j,v+.5,str(v),ha='center')
    axs[1,1].set(ylim=(0,34),ylabel='Batches satisfying every marker condition',title='Predetermined diagnostic markers');axs[1,1].legend(fontsize=9)
    fig.suptitle('Storage precision and local gradient competition; no causal or acceptance claim',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94]);save(fig,'precision_order_and_flags.png')

    fig,axs=plt.subplots(2,2,figsize=(14,9));x=np.arange(10);rows=full['per_original_region']
    axs[0,0].plot(x,[r['SOURCE_PSD_W1']for r in rows],'o-',label='SOURCE')
    axs[0,0].plot(x,[r['same_treated_REAL_PSD_W1']for r in rows],'o-',label='Same-treated REAL')
    axs[0,0].set(ylabel='PSD W1',title='Complete frozen TRAIN, original regions');axs[0,0].legend()
    for name in ['V480','U480','DU479','D2U478']:
        vals=[r['weighted_contribution_to_total']for r in full['full_window']['per_region_and_feature']if r['feature']==name]
        axs[0,1].plot(x,vals,'o-',label=name)
    axs[0,1].set_yscale('log');axs[0,1].set(ylabel='Contribution to full-window loss',title='Original SOURCE region mass included');axs[0,1].legend(ncol=2)
    axs[1,0].bar(x-.18,[r['SOURCE_N']/6593 for r in rows],width=.36,label='SOURCE')
    axs[1,0].bar(x+.18,[r['REAL_N']/21047 for r in rows],width=.36,label='REAL');axs[1,0].set(ylabel='Original parent fraction',title='Input populations retained');axs[1,0].legend()
    axs[1,1].plot(x,[r['SOURCE_projection_range_fraction']for r in rows],'o-',label='Frozen output projection active')
    axs[1,1].plot(x,[r['SOURCE_original_overrange_fraction']for r in rows],'s--',label='Original input outside range')
    axs[1,1].set(ylim=(-.03,1.03),ylabel='Fraction of SOURCE parents',title='Box-charge projection by original region');axs[1,1].legend(fontsize=9)
    for ax in axs.flat:ax.set_xticks(x,labels,rotation=25);ax.set_xlabel('Original charge region')
    fig.tight_layout();save(fig,'full_window_and_projection.png')

    rms=np.sqrt(np.mean(g['raw_gradient_position_RMS'][:,:7,0,:]**2,axis=0));fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    for j in range(7):
        scale=np.sqrt(np.mean(rms[j]**2))
        if scale>0:axs[0].plot(np.arange(480),rms[j]/scale,label=short[j],linewidth=1.1)
        axs[1].plot(np.arange(480),rms[j],label=short[j],linewidth=1.1)
    axs[0].set(ylabel='RMS gradient / own full-window RMS',title='SOURCE projection-before gradient profile: all 480 samples');axs[0].legend(ncol=7,fontsize=10)
    axs[1].set_yscale('symlog',linthresh=1e-4);axs[1].set(ylabel='Raw gradient RMS',xlabel='Sample index',title='Loss weights and original-region importance already included')
    for ax in axs:
        for at in [100,150,400]:ax.axvline(at,color='black',lw=.7,ls=':')
    fig.tight_layout();save(fig,'raw_gradient_boundary_profiles.png')

    blocks=trace['blocks_of_1000'];x=[r['last_update']for r in blocks];fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    for name in names[:7]:axs[0].plot(x,[r[name]for r in blocks],label=name)
    axs[0].set_yscale('log');axs[0].set(ylabel='Mean saved loss per 1,000 updates',title='R59 saved trajectory: sole 50,000 endpoint, zero new updates');axs[0].legend(ncol=4,fontsize=9)
    axs[1].plot(x,[r['clip_fraction']for r in blocks],color='#b44637');axs[1].set(ylim=(-.03,1.03),ylabel='Fraction with pre-clip norm > 5',xlabel='Historical R59 update')
    fig.tight_layout();save(fig,'saved_R59_training_trajectory.png')

    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True);waves=full['complete_480_mean_SD']
    for name,label,color in [('SOURCE_raw_input','Raw SOURCE input','#999999'),('SOURCE_output','R59 SOURCE','#176cbb'),('REAL_reference','REAL reference','black'),('same_treated_REAL','R59 on REAL','#da7906')]:
        axs[0].plot(np.asarray(waves[name]['mean_V'])*1000,label=label,color=color)
        axs[1].plot(np.asarray(waves[name]['SD_V'])*1000,label=label,color=color)
    axs[0].set(ylabel='Mean voltage (mV)',title='Full TRAIN population waveform: original parent weights');axs[0].legend(ncol=4,fontsize=10)
    axs[1].set(ylabel='Voltage SD (mV)',xlabel='Sample index')
    for ax in axs:
        for at in [100,150,400]:ax.axvline(at,color='gray',lw=.7,ls=':')
    fig.tight_layout();save(fig,'complete_TRAIN_waveform.png')
    return figures
