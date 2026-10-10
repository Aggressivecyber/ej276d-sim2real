from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(R,C,M,gate,acceptance):
    arm=C['arms'][0];ctrl='JOINT_PULSE_REAL_CONTROL';F=R/'figures';F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':145})
    def save(fig,name):
        fig.tight_layout();fig.savefig(F/name,bbox_inches='tight');plt.close(fig)
    def load(p):
        with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
    names={'BASE':'BASE','REAL':'REAL','R49_COMPLETION_SOURCE':'R49 source',arm:'R53 source',ctrl:'R53 on REAL'}
    colors={'BASE':'#888888','REAL':'#111111','R49_COMPLETION_SOURCE':'#aa67b1',arm:'#136bc2',ctrl:'#d77715'}
    f=load(R/'results/FEATURES_PRIVATE.npz')
    loss=load(R/arm/'LOSS_BY_UPDATE_PRIVATE.npz')['values']
    fig,ax=plt.subplots(1,2,figsize=(11,4.1))
    for k,label,color in [(0,'SOURCE','#136bc2'),(1,'REAL identity','#d77715')]:
        block=loss[:,k].reshape(-1,1000).mean(1)
        ax[0].plot(np.arange(1,51)*1000,block,label=label,color=color)
    ax[0].set(yscale='log',xlabel='Update (fixed final 50,000)',ylabel='Mean loss per 1,000 updates',title='One training run; no checkpoint selection');ax[0].legend()
    tv=json.loads((R/'TEACHER_VALIDATION.json').read_text());tr=M['training'][arm]
    txt='Teacher structure: '+('PASS' if tv['passed']else'FAIL')+'\n'
    txt+='Charge relative error max: '+f"{tv['max_relative_q_error']:.2e}"+'\n'
    txt+='PSD change max: '+f"{tv['max_PSD_change']:.2e}"+'\n'
    txt+='Training updates: '+str(tr['updates'])+'\n'
    txt+='Original input/random streams equal: '+str(all(tr['same_R48_source_identity_and_random_streams'].values()))+'\n'
    txt+='A / B / C: '+' / '.join('PASS' if acceptance[arm][k]['pass']else'FAIL' for k in ['A','B','C'])+'\n\n'
    txt+='Teacher validity alone does not establish\nlearned response or physical correspondence.'
    ax[1].axis('off');ax[1].text(.02,.94,txt,va='top',linespacing=1.7)
    save(fig,'training_and_checks.png')

    views=C['C2ST_views'];labels=['Shape 493','Normalized u480','Voltage 480','Q + PSD','Baseline']
    fig,ax=plt.subplots(figsize=(10.8,4.8));x=np.arange(5)
    for j,n in enumerate(['BASE','R49_COMPLETION_SOURCE',arm,ctrl]):
        rows=M['C2ST']['results'][n];p=np.array([rows[v]['balanced_accuracy']for v in views])
        ci=np.array([rows[v].get('BA_95CI',rows[v]['heldout_CI']['balanced_accuracy_95CI'])for v in views]);off=(j-1.5)*.14
        ax.errorbar(x+off,p,yerr=np.vstack([p-ci[:,0],ci[:,1]-p]),fmt='o',ms=5,capsize=3,label=names[n],color=colors[n])
    ax.axhspan(.4,.6,color='#a6d7ab',alpha=.25);ax.axhline(.5,color='black',lw=.8,ls=':')
    ax.set(xticks=x,xticklabels=labels,ylim=(.35,1.04),ylabel='Balanced accuracy and fixed-classifier 95% CI',title='Common ADC codes; unchanged five-view acceptance')
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.02),fontsize=9);save(fig,'canonical_five_view_response.png')

    fig,ax=plt.subplots(2,4,figsize=(13.4,6.2),sharex=True)
    tables=M['energy_conditional_width']['tables']
    for i,a in enumerate(ax.flat):
        failures=[]
        for n in ['REAL','BASE',arm]:
            t=tables[n];row=t['slices'][i];edges=np.asarray(t.get('histogram_edges_PSD',np.linspace(-.1,.6,129)))
            if 'histogram_counts' in row:
                hist=np.asarray(row['histogram_counts'],dtype=float)
            else:
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
    ax[0,0].legend(fontsize=8);fig.suptitle('All fixed Q intervals retained; failed density fits remain visible',y=1.02)
    save(fig,'PSD_all_eight_bins.png')

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
        a.set(xlabel='Original SOURCE '+label,ylabel='R53 output '+label)
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
