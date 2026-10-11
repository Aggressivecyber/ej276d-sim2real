from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(R,C,M,gate,acceptance):
    arm=C['arms'][0];ctrl='FULL_EMPIRICAL_REAL_CONTROL';F=R/'figures';F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':145})
    def save(fig,name):
        fig.tight_layout();fig.savefig(F/name,bbox_inches='tight');plt.close(fig)
    def load(p):
        with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
    names={'BASE':'BASE','REAL':'REAL','R49_COMPLETION_SOURCE':'R49 source','R63_SOURCE':'R63 source',arm:'R65 source',ctrl:'R65 on REAL'}
    colors={'BASE':'#888888','REAL':'#111111','R49_COMPLETION_SOURCE':'#aa67b1','R63_SOURCE':'#55976e',arm:'#136bc2',ctrl:'#d77715'}
    f=load(R/'results/FEATURES_PRIVATE.npz')
    lossdata=load(R/arm/'LOSS_BY_UPDATE_PRIVATE.npz');loss=lossdata['values'];cols=lossdata['columns'].tolist()
    steps=np.arange(1,51)*1000
    fig,ax=plt.subplots(1,3,figsize=(16,4.8))
    D_names=['full_window','PSD','noise','coordinate_RMS','temporal_joint']
    C_names=['source_order','source_valley','REAL_identity','box_correction']
    for j,(keys,title) in enumerate([(D_names,'Five complete-parent D terms'),(C_names,'Four original C terms')]):
        for label in keys:
            block=loss[:,cols.index(label)].reshape(-1,1000).mean(1)
            ax[j].plot(steps,np.maximum(block,1e-12),label=label)
        ax[j].set(yscale='log',xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',title=title)
        ax[j].legend(fontsize=7)
    tr=M['training'][arm];audit=json.loads((R/'TRAINING_FROZEN_AUDIT.json').read_text())
    counter=tr['counter']
    txt='Committed effective exposures\nComplete D SOURCE: '+format(counter['full_D_SOURCE_law_rows'],',')+'\n'
    txt+='Complete D REAL: '+format(counter['full_D_REAL_law_rows'],',')+'\n'
    txt+='Original C SOURCE: '+format(counter['SOURCE_training_rows'],',')+'\n'
    txt+='Original C REAL identity: '+format(counter['REAL_identity_rows'],',')+'\n'
    txt+='Discarded micro-D references: '+format(counter['REAL_reference_rows'],',')+'\n\n'
    txt+='Completed updates: '+str(tr['updates'])+'\nTraining audit: '+('PASS' if audit['pass']else'FAIL')+'\n'
    txt+='A / B / C: '+' / '.join('PASS' if acceptance[arm][k]['pass']else'FAIL' for k in ['A','B','C'])+'\n\n'
    txt+='D/C use different parent/noise laws.\nRecovery compute is reported separately.\nLosses do not replace response acceptance.'
    ax[2].axis('off');ax[2].text(.02,.97,txt,va='top',linespacing=1.5,fontsize=9)
    save(fig,'training_and_checks.png')

    views=C['C2ST_views'];labels=['Shape 493','Normalized u480','Voltage 480','Q + PSD','Baseline']
    fig,ax=plt.subplots(figsize=(10.8,4.8));x=np.arange(5)
    for j,n in enumerate(['BASE','R49_COMPLETION_SOURCE','R63_SOURCE',arm,ctrl]):
        rows=M['C2ST']['results'][n];p=np.array([rows[v]['balanced_accuracy']for v in views])
        ci=np.array([rows[v].get('BA_95CI',rows[v]['heldout_CI']['balanced_accuracy_95CI'])for v in views]);off=(j-2)*.13
        ax.errorbar(x+off,p,yerr=np.vstack([p-ci[:,0],ci[:,1]-p]),fmt='o',ms=5,capsize=3,label=names[n],color=colors[n])
    ax.axhspan(.4,.6,color='#a6d7ab',alpha=.25);ax.axhline(.5,color='black',lw=.8,ls=':')
    ax.set(xticks=x,xticklabels=labels,ylim=(.35,1.04),ylabel='Balanced accuracy and fixed-classifier 95% CI',title='Common ADC codes; unchanged five-view acceptance')
    ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1.02),fontsize=8);save(fig,'canonical_five_view_response.png')

    fig,ax=plt.subplots(2,4,figsize=(13.4,6.5),sharex=True)
    tables=M['energy_conditional_width']['tables']
    for i,a in enumerate(ax.flat):
        failures=[]
        for n in ['REAL','BASE',arm]:
            row=tables[n]['slices'][i];edges=np.linspace(-.1,.6,129)
            q=f[n+'__Qtotal_mVns'];qedges=np.asarray(M['energy_conditional_width']['edges_logQ'])
            z=np.log(np.maximum(q,1e-30));bins=np.searchsorted(qedges,z,side='right')-1;bins[z==qedges[-1]]=7
            use=f[n+'__valid']&(bins==i)
            hist=np.histogram(f[n+'__PSD'][use],bins=edges)[0].astype(float)
            y=hist/max(hist.sum(),1)/np.diff(edges)
            a.stairs(y,edges,label=names[n],color=colors[n],lw=1.2)
            if not row['valid']:failures.append(names[n])
        a.set_title('Q'+str(i+1)+(' [required]'if i in range(1,7)else' [edge]'),fontsize=10)
        a.set(xlim=(-.1,.6),xlabel='Signed PSD',ylabel='Density')
        a.text(.02,.96,'Unqualified: '+(', '.join(failures)if failures else'none'),transform=a.transAxes,va='top',fontsize=7.7,
               bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    handles,labels=ax[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.963),ncol=3,fontsize=9)
    fig.suptitle('All fixed Q intervals retained; failed density fits remain visible',y=.998)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(F/'PSD_all_eight_bins.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(2,2,figsize=(11,7))
    for n in ['BASE',arm,ctrl]:
        ax[0,0].plot(np.arange(1,9),[r['W1']for r in M['PSD_main']['vs_REAL'][n]['bins']],'-o',ms=3,color=colors[n],label=names[n])
    ax[0,0].axhline(.01,color='black',ls=':',lw=1);ax[0,0].set(xlabel='Fixed output Q bin',ylabel='PSD W1',title='Unchanged Q1-7 mean requirement');ax[0,0].legend(fontsize=8)
    for col,component in enumerate([0,1]):
        for n in ['REAL','BASE',arm]:
            rows=tables[n]['slices'];centers=[r['components'][component]['mean']if r['valid']else np.nan for r in rows]
            ax[0,1].plot(np.arange(1,9),centers,('-o'if col==0 else'--s'),ms=3,color=colors[n],label=names[n]+(' low'if col==0 else' high'))
            widths=[r['components'][component]['sigma_PSD']if r['valid']else np.nan for r in rows]
            ax[1,col].plot(np.arange(1,9),widths,'-o',ms=3,color=colors[n],label=names[n])
        ax[1,col].set(xlabel='Fixed Q bin (gaps = failed fit)',ylabel='Raw PSD SD',title=('Low'if col==0 else'High')+' density component');ax[1,col].legend(fontsize=8)
    ax[0,1].set(xlabel='Fixed Q bin',ylabel='Density peak position',title='Solid = low; dashed = high');ax[0,1].legend(ncol=2,fontsize=7)
    save(fig,'PSD_centers_widths_and_W1.png')

    fig,ax=plt.subplots(2,2,figsize=(11,7))
    for a,field,label,scale in [(ax[0,0],'baseline_RMS_V','Baseline RMS (mV)',1000),(ax[0,1],'ACF1','Baseline ACF1',1)]:
        for n in ['REAL','BASE',arm,ctrl]:
            values=f[n+'__'+field];values=values[np.isfinite(values)]*scale
            a.hist(values,bins=60,density=True,histtype='step',color=colors[n],label=names[n])
        a.set(xlabel=label,ylabel='Density');a.legend(fontsize=8)
    for a,field,label in [(ax[1,0],'PSD','PSD'),(ax[1,1],'Qtotal_mVns','Q (mV ns)')]:
        x=f['SOURCE_INPUT__'+field];y=f[arm+'__'+field]
        a.scatter(x,y,s=4,alpha=.4,color=colors[arm],rasterized=True)
        a.set(xlabel='Original SOURCE '+label,ylabel='R65 output '+label)
    fig.suptitle('Noise response and event association (association is not particle truth)',y=1.02)
    save(fig,'noise_and_source_association.png')

    moments=load(R/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz')
    fig,ax=plt.subplots(2,1,figsize=(11,6.5),sharex=True)
    for n in ['REAL','BASE',arm,ctrl]:
        ax[0].plot(np.arange(480),moments[n+'__None__mean_V']*1000,color=colors[n],label=names[n])
        ax[1].plot(np.arange(480),moments[n+'__None__SD_V']*1000,color=colors[n],label=names[n])
    for a in ax:
        for b in [100,150,400]:a.axvline(b,color='#888888',ls=':',lw=.6)
        a.legend(ncol=4,fontsize=8)
    ax[0].set(ylabel='Mean raw voltage (mV)',title='Full 480-sample population response; all original source weights')
    ax[1].set(xlabel='Sample index',ylabel='Raw voltage SD (mV)');save(fig,'complete_waveform_population.png')

    bd=M['boundary_differences']['results'];keys=list(bd[arm]);x=np.arange(len(keys))
    fig,ax=plt.subplots(figsize=(11.5,5.0))
    for n in ['BASE','INDEPENDENT_CFM',arm]:
        ax.plot(x,[bd[n][k]['W1_V']*1000 for k in keys],'-o',ms=4,label=names.get(n,n),color=colors.get(n,'#ad9048'))
    limits=[r['limit_V']*1000 for r in gate['models'][arm]['gate4_boundary_source']['boundaries']]
    ax.plot(x,limits,'k--',label='Fixed BASE/independent limit + 1 LSB')
    ax.set(xticks=x,xticklabels=[k.replace('first_','d1 ').replace('second_center_','d2 ')for k in keys],ylabel='W1 versus REAL (mV)',title='All nine fixed boundary differences')
    ax.tick_params(axis='x',rotation=32);ax.legend(fontsize=8);save(fig,'fixed_time_boundaries.png')

    fig,ax=plt.subplots(1,3,figsize=(14.5,4.8))
    saved={n:load(R/n/'generated.npz') for n in [arm,ctrl]}
    all_zero=all(np.all(d['constraint_projection_RMS_V']==0) for d in saved.values())
    all_rails_zero=all(np.all(d['constraint_projection_rail_sample_count']==0) for d in saved.values())
    short=['SOURCE','Same-treated REAL']
    if all_zero:
        ax[0].bar(short,[2000,2000],color=[colors[arm],colors[ctrl]],width=.6)
        for i in range(2):ax[0].text(i,2030,'2,000 / 2,000',ha='center',fontsize=9)
        ax[0].set(ylim=(0,2300),ylabel='Events with zero correction',title='Exact zero correction in main outputs')
    else:
        for n,d in saved.items():
            r=d['constraint_projection_RMS_V']*1000;nonzero=r[r>0]
            if len(nonzero):ax[0].hist(nonzero,bins=40,histtype='step',label=names[n],color=colors[n])
            else:ax[0].plot([],[],label=names[n]+' (all zero)',color=colors[n])
        ax[0].set(xlim=(0,None),ylim=(0,None),xlabel='Nonzero RMS correction (mV)',ylabel='Events',title='Explicit constraint contribution');ax[0].legend(fontsize=8)
    if all_rails_zero:
        ax[1].bar(short,[2000,2000],color=[colors[arm],colors[ctrl]],width=.6)
        for i in range(2):ax[1].text(i,2030,'2,000 / 2,000',ha='center',fontsize=9)
        ax[1].set(ylim=(0,2300),ylabel='Events with no rail samples',title='No endpoint samples on voltage rails')
    else:
        for n,d in saved.items():ax[1].hist(d['constraint_projection_rail_sample_count'],bins=np.arange(-.5,481,10),histtype='step',label=names[n],color=colors[n])
        ax[1].set(xlim=(0,480),ylim=(0,None),xlabel='Rail samples per event',ylabel='Events',title='Bounded endpoint saturation');ax[1].legend(fontsize=8)
    a=M['explicit_constraint_projection'][arm];b=M['explicit_constraint_projection'][ctrl]
    trace=audit['constraint_loss_trace_summary']
    frac=trace['SOURCE_global_range_projection_fraction']['mean']
    full_frac=trace['full_D_SOURCE_range_projection_fraction']['mean']
    identity_frac=trace['REAL_range_projection_fraction']['mean']
    info='Range-active main events\nSOURCE: '+str(a['range_active_events'])+' / 2000\nREAL: '+str(b['range_active_events'])+' / 2000\n\n'
    info+='Projection remains part of the model.\nZero ADC clipping does not establish\nzero internal correction.\n\n'
    info+=f'TRAIN update-mean range activation:\ncomplete D SOURCE: {100*full_frac:.2f}%.\noriginal C SOURCE: {100*frac:.2f}%.\noriginal C REAL identity: {100*identity_frac:.2f}%.\nDifferent populations/noise laws; not pooled.'
    ax[2].axis('off');ax[2].text(.02,.97,info,va='top',linespacing=1.45,fontsize=9)
    save(fig,'explicit_constraint_projection.png')

    precision=load(R/arm/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz')
    values=precision['values'];pn=precision['columns'].tolist();steps=np.arange(1,51)*1000
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for key,label in [('original_order_loss','Original order penalty'),('omitted_order_loss','Omitted overlap penalty')]:
        ax[0].plot(steps,values[:,pn.index(key)].reshape(-1,1000).mean(1),label=label)
    ax[0].plot(steps,loss[:,cols.index('source_order')].reshape(-1,1000).mean(1),label='Retained resolved penalty')
    ax[0].set(xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',title='Original = retained + omitted');ax[0].legend(fontsize=8)
    ax[1].plot(steps,values[:,pn.index('resolved_pair_fraction')].reshape(-1,1000).mean(1),color='#136bc2')
    ax[1].set(ylim=(0,1),xlabel='Update',ylabel='Resolved fraction of original pair mass',title='Original denominator retained')
    fig.suptitle('Precision-order trace: training surrogate; no acceptance exemptions')
    save(fig,'precision_order_decomposition.png')

    fig,ax=plt.subplots(2,2,figsize=(13.2,9.2))
    for key,label in [('D_total','Complete D'),('C_total','Original C'),('total_loss','Joint D+C')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[0,0].plot(steps,np.maximum(block,1e-12),label=label)
    ax[0,0].set(yscale='log',ylabel='Mean loss per 1,000 updates',title='Complete D and original C totals')
    for key,label in [('D_parameter_gradient_L2','Complete D'),('C_parameter_gradient_L2','Original C'),('gradient_norm','Joint D+C before clipping')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[0,1].plot(steps,np.maximum(block,1e-12),label=label)
    ax[0,1].set(yscale='log',ylabel='Mean parameter-gradient L2',title='Gradient magnitudes; no directional claim')
    for key,label in [('full_D_SOURCE_range_projection_fraction','Complete D SOURCE'),('SOURCE_global_range_projection_fraction','Original C SOURCE'),('REAL_range_projection_fraction','Original C REAL identity')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[1,0].plot(steps,100*block,label=label)
    ax[1,0].set(ylabel='Mean range-active fraction (%)',ylim=(0,None),title='D and C range activation kept separate')
    for key,label in [('full_D_SOURCE_projection_RMS_V','Complete D SOURCE'),('SOURCE_global_projection_RMS_V','Original C SOURCE'),('REAL_projection_RMS_V','Original C REAL identity')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[1,1].plot(steps,1000*block,label=label)
    ax[1,1].set(ylabel='Mean projection RMS (mV)',ylim=(0,None),title='D and C correction magnitudes kept separate')
    for axis in ax.flat:
        axis.set(xlabel='Update (sole final 50,000)');axis.legend(fontsize=8)
    fig.suptitle('Committed training traces; complete D and original C have different parent/noise laws')
    save(fig,'full_D_original_C_training.png')
