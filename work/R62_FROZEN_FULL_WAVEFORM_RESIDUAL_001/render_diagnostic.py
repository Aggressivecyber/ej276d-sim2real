"""Render saved TRAIN statistics only; no model forward or data archive read."""
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from residual_math import FEATURES,REGIONS,STAGES,BOUNDARIES

def render(root):
    read=lambda n:json.loads((root/n).read_text('utf8'))
    load=lambda n:{k:v.copy()for k,v in np.load(root/n,allow_pickle=False).items()}
    path=read('PATH_SUMMARY.json');base=read('BASELINE_JOINT_SUMMARY.json');loss=read('LOSS_COVERAGE_SUMMARY.json');psd=read('PSD_SUMMARY.json');proj=read('PROJECTION_SUMMARY.json')
    p=load('PATH_STATISTICS_PRIVATE.npz');b=load('BASELINE_JOINT_PRIVATE.npz');l=load('LOSS_COVERAGE_PRIVATE.npz');s=load('PER_PARENT_PRIVATE.npz')
    out=root/'figures';out.mkdir(exist_ok=True);manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
    colors=['#607d8b','#f5a623','#cf4c54','#825cc7','#126ab3'];labels=['input','noise initial','neural raw','projected float32','canonical ADC']
    region_label=['below']+['Q'+str(i)for i in range(1,9)]+['above']
    def save(fig,name,title):
        fig.suptitle(title,fontsize=14);fig.text(.5,.012,'Frozen R61 on original TRAIN parents | descriptive diagnostics; no A-F acceptance claim',ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=[0,.035,1,.955]);file=out/(name+'.png');fig.savefig(file,facecolor='white');plt.close(fig)
        manifest.append({'path':'figures/'+file.name,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,name in zip(axes.flat,FEATURES):
        for stage,col,label in zip(STAGES,colors,labels):ax.plot(p['SOURCE_'+stage+'_all_'+name+'_W1'],color=col,label=label,lw=1.1)
        ax.plot(p['REAL_control_canonical_ADC_all_'+name+'_W1'],color='#228b65',ls='--',label='same-treated REAL',lw=1)
        ax.set(title=name+' coordinate W1',xlabel='sample / difference index',ylabel='V'if name=='V480'else'normalized units')
        for k in [100,150,400]:ax.axvline(k,color='#cccccc',lw=.7)
        ax.set_yscale('symlog',linthresh=1e-6);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,ncol=2)
    save(fig,'complete_path_coordinate_W1','Complete waveform residuals along the frozen response path')
    rows=path['rows'];select=lambda d,st,scope:next(r for r in rows if r['domain']==d and r['stage']==st and r['scope']==scope)
    fig,axes=plt.subplots(1,2,figsize=(15,6));values=[]
    for domain in ['SOURCE','REAL_control']:values.append(np.array([select(domain,st,'all')['boundary_W1_V']for st in STAGES])*16383)
    vmax=max(v.max()for v in values)
    for ax,v,d in zip(axes,values,['SOURCE','same-treated REAL']):
        im=ax.imshow(v,aspect='auto',cmap='magma',vmin=0,vmax=vmax)
        ax.set_xticks(range(9),BOUNDARIES,rotation=60,ha='right');ax.set_yticks(range(5),labels);ax.set_title(d+' vs REAL reference')
        for i in range(5):
            for j in range(9):ax.text(j,i,format(v[i,j],'.1f'),ha='center',va='center',fontsize=8,color='white'if v[i,j]<.65*vmax else'black')
        fig.colorbar(im,ax=ax,label='empirical W1 / ADC LSB',shrink=.75)
    save(fig,'fixed_boundary_path','Nine original fixed boundaries, all original TRAIN parents')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,name in zip(axes.flat,FEATURES):
        for domain,col,label in [('SOURCE','#126ab3','SOURCE'),('REAL_control','#228b65','same-treated REAL')]:
            vals=[select(domain,'canonical_ADC','region'+str(r))['feature_mean_W1'][name]for r in REGIONS]
            ax.plot(range(10),vals,'o-',color=col,label=label)
        ax.set_xticks(range(10),region_label,rotation=30);ax.set(title=name+' mean coordinate W1',ylabel='V'if name=='V480'else'normalized units');ax.grid(alpha=.2)
    axes[0,0].legend();save(fig,'original_region_residuals','Canonical output residual by unchanged original charge region')
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    names=['REAL_reference','SOURCE','REAL_control'];mats=[b['all_'+n+'_covariance_V2']for n in names];limit=max(np.abs(m).max()for m in mats)
    for ax,m,name in zip(axes.flat,mats,names):
        im=ax.imshow(m,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set(title=name+' baseline covariance',xlabel='baseline sample',ylabel='baseline sample');fig.colorbar(im,ax=ax,shrink=.75,label='V squared')
    ax=axes[1,1];br=base['rows'];x=np.arange(len(br))
    for i,row in enumerate(br):ax.scatter(np.full(32,i),row['empirical_reference_errors'],s=8,color='#aaaaaa',alpha=.45)
    ax.plot(x,[r['SOURCE_covariance_relative_Frobenius']for r in br],'o-',label='SOURCE',color='#126ab3')
    ax.plot(x,[r['REAL_control_covariance_relative_Frobenius']for r in br],'o-',label='same-treated REAL',color='#228b65')
    ax.set_xticks(x,['all']+region_label,rotation=55);ax.set(title='Matched SOURCE-N empirical reference (gray)',ylabel='relative Frobenius residual');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'baseline_covariance','Baseline joint structure and 32 empirical resamples per scope')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for name,col,label in [('SOURCE','#126ab3','SOURCE'),('REAL_control','#228b65','same-treated REAL'),('REAL_reference','#333333','REAL reference')]:
        f=b[name+'_baseline_features'];x=np.sort(f[:,0]);axes[0,0].plot(x*1e3,np.arange(1,len(x)+1)/len(x),color=col,label=label)
        axes[0,1].plot(np.arange(1,21),f[:,1:21].mean(0),'o-',ms=3,color=col,label=label)
        axes[1,0].plot(np.arange(10)*.05+.03,f[:,21:].mean(0),'o-',color=col,label=label)
        axes[1,1].plot(np.sqrt(np.diag(b['all_'+name+'_covariance_V2']))*1e3,color=col,label=label)
    axes[0,0].set(title='Per-parent baseline RMS',xlabel='mV',ylabel='empirical CDF');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Mean within-parent ACF',xlabel='lag (samples)',ylabel='ACF')
    axes[1,0].set(title='Mean normalized spectral bands',xlabel='band center (cycles/sample)',ylabel='fraction of nonzero-frequency power')
    axes[1,1].set(title='Per-coordinate baseline population SD',xlabel='baseline sample',ylabel='mV')
    for ax in axes.flat:ax.grid(alpha=.2)
    save(fig,'baseline_ACF_spectrum','Baseline noise: marginal size, temporal correlation and spectral shape')
    fig,axes=plt.subplots(2,2,figsize=(14,9));cmp=['SOURCE_REAL','REAL_REAL','REAL_control_REAL'];short=['SOURCE-REAL','REAL-REAL','control-REAL'];cc=['#126ab3','#999999','#228b65']
    axes[0,0].boxplot([l['batch_total_losses'][:,j]for j in range(3)],labels=short,showfliers=True)
    for j,n in enumerate(cmp):axes[0,0].scatter(j+1,loss['comparisons'][n]['full_empirical_loss'],marker='D',s=55,color=cc[j],zorder=5)
    axes[0,0].set(title='32 saved-endpoint batches; diamond = full law',ylabel='original standardized full-window loss')
    for j,(n,col)in enumerate(zip(cmp,cc)):
        row=loss['comparisons'][n];axes[0,1].errorbar(np.arange(4)+(j-1)*.12,row['batch_per_feature_mean'],yerr=row['batch_per_feature_SD'],fmt='o',color=col,label=short[j],capsize=3)
        axes[0,1].scatter(np.arange(4)+(j-1)*.12,row['full_per_feature'],marker='D',s=32,color=col)
        axes[1,0].plot(np.arange(10),row['full_region_contributions'],'o-',color=col,label=short[j])
    axes[0,1].set_xticks(range(4),FEATURES);axes[0,1].set(title='Batch mean +/- SD; diamond = full law',ylabel='standardized loss');axes[0,1].legend(fontsize=8)
    axes[1,0].set_xticks(range(10),region_label,rotation=35);axes[1,0].set(title='Full-law contributions with original SOURCE mass',ylabel='weighted contribution')
    excess=l['batch_total_losses'][:,0]-l['batch_total_losses'][:,1];axes[1,1].plot(np.arange(1,33),excess,'o-',ms=4);axes[1,1].axhline(0,color='#888888',ls='--');axes[1,1].set(title='SOURCE-REAL minus REAL-REAL on each batch',xlabel='fixed batch',ylabel='loss difference')
    for ax in axes.flat:ax.grid(alpha=.2)
    save(fig,'original_loss_sampling_coverage','Original marginal and 64-direction objective: complete law vs finite batches')
    fig,axes=plt.subplots(2,5,figsize=(17,8))
    for ax,r,label in zip(axes.flat,REGIONS,region_label):
        si=s['source_raw_region']==r;ri=s['REAL_raw_region']==r
        series=[s['source_original_psd'][si],s['source_output_psd'][si],s['REAL_output_psd'][ri],s['REAL_reference_psd'][ri]]
        edges=np.linspace(min(v.min()for v in series),max(v.max()for v in series),51)
        for v,col,lab in zip(series,['#607d8b','#126ab3','#228b65','#333333'],['source input','source output','REAL control','REAL reference']):ax.hist(v,bins=edges,histtype='step',density=True,color=col,label=lab,lw=1.2)
        row=next(x for x in psd['regions']if x['region']==r);ax.set(title=label+' | W1='+format(row['SOURCE_W1'],'.4f'),xlabel='PSD',ylabel='density')
        ax.text(.03,.95,'S '+str(int(si.sum()))+' / R '+str(int(ri.sum())),transform=ax.transAxes,va='top',fontsize=8)
        if r in [2,3]:
            vv=next(z for z in psd['valleys']if z['region']==r)
            for bound in vv['bounds']:ax.axvline(bound,color='#c65870',ls=':',lw=1)
    axes[0,0].legend(fontsize=7);save(fig,'PSD_all_original_regions','TRAIN PSD in all ten unchanged charge regions; density fits = 0')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for name,col,label in [('SOURCE','#126ab3','SOURCE'),('REAL_control','#228b65','same-treated REAL')]:
        axes[0,0].plot(p[name+'_canonical_ADC_mean_V'],color=col,label=label);axes[0,1].plot(p[name+'_canonical_ADC_SD_V'],color=col,label=label)
    axes[0,0].plot(p['REAL_reference_mean_V'],color='#333333',label='REAL reference');axes[0,1].plot(p['REAL_reference_SD_V'],color='#333333',label='REAL reference')
    axes[0,0].set(title='Complete canonical voltage: population mean',xlabel='sample',ylabel='V');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Complete canonical voltage: population SD',xlabel='sample',ylabel='V')
    metrics=['original_valley_mass','output_valley_mass','original_low_edge_mass','output_low_edge_mass','original_high_edge_mass','output_high_edge_mass'];x=np.arange(6)
    for j,v in enumerate(psd['valleys']):axes[1,0].bar(x+(j-.5)*.35,[100*v[k]for k in metrics],width=.35,label='Q'+str(v['region']+1)+' h=1, N='+str(v['N']))
    axes[1,0].set_xticks(x,['in valley','out valley','in low edge','out low edge','in high edge','out high edge'],rotation=30,ha='right');axes[1,0].set(title='Fixed valley and +/-0.004 edge neighborhoods',ylabel='parent mass (%)');axes[1,0].legend(fontsize=8)
    names=['SOURCE','REAL_control'];x=np.arange(2);vals=[100*proj[n]['range_active_N']/proj[n]['rows']for n in names]
    bars=axes[1,1].bar(x,vals,color=['#126ab3','#228b65'])
    for i,(bar,n)in enumerate(zip(bars,names)):
        v=proj[n];axes[1,1].text(i,vals[i],str(v['range_active_N'])+'/'+str(v['rows']),ha='center',va='bottom')
    axes[1,1].set_xticks(x,['SOURCE','same-treated REAL']);axes[1,1].set(title='Range-active constraint projection',ylabel='original parents (%)',ylim=(0,max(1.,max(vals)*1.25)))
    for ax in axes.flat:ax.grid(alpha=.15)
    save(fig,'complete_population_and_source_constraints','Population waveform, original source valleys and explicit range projection')
    return manifest
