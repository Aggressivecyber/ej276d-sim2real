from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import render_train

def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}

def run(root,cfg,m,gate,acceptance):
    root=Path(root);out=root/'results/figures';out.mkdir(exist_ok=True,parents=True)
    s,l=cfg['arms'];rc=['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']
    dom=['REAL','BASE',s,l];col={'REAL':'#222222','BASE':'#999999',s:'#2474ad',l:'#d45d1c',rc[0]:'#2474ad',rc[1]:'#d45d1c'}
    lab={'REAL':'REAL DEV','BASE':'Frozen BASE',s:'FIXED noise',l:'NOISE completion',rc[0]:'FIXED on REAL',rc[1]:'COMPLETION on REAL'}
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'legend.fontsize':8,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    feat=load(root/'results/FEATURES_PRIVATE.npz');cs=m['C2ST']['results'];width=m['energy_conditional_width'];edges=np.asarray(m['PSD_main']['edges_logQ'])
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    def cdf(ax,n,a,scale=1):
        a=np.sort(np.asarray(a)[np.isfinite(a)])*scale
        if len(a):ax.plot(a,np.arange(1,len(a)+1)/len(a),color=col[n],label=lab[n],lw=1.35)
    def field(n,k):return feat[n+'__'+k]
    render_train.run(root,cfg,json.loads((root/'TRAIN_SOLVER_CHECK.json').read_text('utf8')))
    views=cfg['C2ST_views']+['normalized_u480','seen_reconstruction480','complement_reconstruction480']
    x=np.arange(len(views));names=['shape493','V480','prompt','postpeak','tail','baseline','Q-PSD2D','pure u480','seen\n480 points','complement\n480 points']
    fig,ax=plt.subplots(figsize=(12.5,4.6))
    for off,n in zip([-.2,0,.2],['BASE',s,l]):ax.plot(x+off,[cs[n][v]['balanced_accuracy']for v in views],'o-',color=col[n],label=lab[n],ms=4)
    rr=[cs['RAW_REAL_REAL'][v].get('balanced_accuracy')for v in views]
    ax.scatter([i for i,v in enumerate(rr)if v is not None],[v for v in rr if v is not None],color='black',marker='x',label='Saved RAW REAL-REAL')
    ax.set_xticks(x,names);ax.set(ylabel='Fixed holdout balanced accuracy',ylim=(.4,1.025),title='Fixed600/250 pools; all10 views retained')
    ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=4);fig.subplots_adjust(top=.74,bottom=.18);save(fig,'c2st.png')
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.5),sharey=True)
    for ax,title,rows in zip(axs,['SOURCE: original parents','REAL: identical observed guides'],
                            [m['paired_primary_minus_control_C2ST'],m['paired_completion_minus_fixed_same_REAL_C2ST']]):
        for i,v in enumerate(views):
            r=rows[v];d=r['BA_delta_first_minus_second'];lo,hi=r['BA_delta_95CI']
            ax.vlines(i,lo,hi,color=col[l],lw=1.2);ax.hlines([lo,hi],i-.06,i+.06,color=col[l],lw=1.2)
            ax.plot(i,d,'o',color=col[l],ms=4)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,names,rotation=45,ha='right')
        ax.set(title=title,xlabel='Fixed diagnostic view');ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('COMPLETION minus FIXED holdout BA; 95% CI')
    fig.suptitle('Only the initial noise-amplitude rule differs; fixed500 paired group bootstraps')
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'paired_noise_rule_C2ST.png')
    q=np.concatenate([field(n,'Qtotal_mVns')[field(n,'valid')]for n in dom]);p=np.concatenate([field(n,'PSD')[field(n,'valid')]for n in dom])
    xr=[float(np.log10(q).min()),float(np.log10(q).max())]
    for full,name in [(False,'charge_psd_main_body.png'),(True,'charge_psd_full_range.png')]:
        yr=[float(p.min()),float(p.max())]if full else[0,.6]
        fig,axs=plt.subplots(1,4,figsize=(12,3.8),sharex=True,sharey=True)
        for ax,n in zip(axs,dom):
            ok=field(n,'valid');a=field(n,'Qtotal_mVns')[ok];b=field(n,'PSD')[ok]
            ax.hexbin(np.log10(a),b,gridsize=45,extent=(*xr,*yr),mincnt=1,bins='log',cmap='viridis')
            ax.set(title=lab[n]+f'\nN={len(b)}, outside={int(np.sum((b<yr[0])|(b>yr[1])))}',xlabel='log10 Qtotal (mV ns)',xlim=xr,ylim=yr)
        axs[0].set_ylabel('Signed PSD =1-Qfast/Qtotal');fig.suptitle('Identical axes; original source population; display zoom does not delete outliers')
        fig.tight_layout();save(fig,name)
    sections=load(root/'results/PSD_SECTIONS_PRIVATE.npz');fig,axs=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
    for i,ax in enumerate(axs.flat):
        count=[]
        for n in dom:
            a=sections[n+'__bin'+str(i+1)+'__PSD'];cdf(ax,n,a);count.append(len(a))
        ax.set(title=f'bin{i+1}\nR/B/F/C: '+ '/'.join(map(str,count)),xlim=(0,.6),ylim=(0,1));ax.grid(alpha=.12)
        if i>=4:ax.set_xlabel('Signed PSD')
        if i%4==0:ax.set_ylabel('CDF')
    axs[0,0].legend(fontsize=7);fig.suptitle('Fixed8 observed-Q intervals; counts and unsupported regions retained')
    fig.tight_layout();save(fig,'conditional_psd_CDF.png')
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True);cx=np.exp((edges[:-1]+edges[1:])/2)
    for k in [0,1]:
        for j,key in enumerate(['mean','sigma_PSD']):
            ax=axs[k,j]
            for n in dom:
                a=[r['components'][k][key]if r['valid']and len(r['components'])==2 else np.nan for r in width['tables'][n]['slices']]
                ax.plot(cx,a,'o-',color=col[n],label=lab[n],ms=4)
            ax.set(xscale='log',ylabel='PSD center'if j==0 else'Raw PSD SD',title=('Low'if k==0 else'High')+' density component; gaps=undefined');ax.grid(alpha=.15)
            if k:ax.set_xlabel('Frozen Q-bin geometric center (mV ns)')
    axs[0,0].legend(fontsize=7);fig.suptitle('Fixed initialization and stopping; density components are not particle truth')
    fig.tight_layout();save(fig,'energy_peak_width.png')
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for n in dom:
        ok=field(n,'valid');cdf(axs[0,0],n,field(n,'baseline_RMS_V')[ok],1000);cdf(axs[0,1],n,field(n,'ACF1')[ok])
        for ax,key,title in [(axs[1,0],'prompt_curvature_RMS_V','Prompt100:150'),(axs[1,1],'postpeak_curvature_RMS_V','Postpeak150:220')]:
            if n+'__'+key in feat:cdf(ax,n,field(n,key)[ok],1000)
            ax.set(xlabel='Second-difference RMS (mV)',ylabel='CDF',title=title+'; peak/early decline may be included')
    axs[0,0].set(xlabel='Baseline RMS (mV)',ylabel='CDF',title='Original100-point baseline');axs[0,0].legend()
    axs[0,1].set(xlabel='Baseline ACF1',ylabel='CDF',title='Full100-point centered denominator')
    fig.tight_layout();save(fig,'noise_curvature.png')
    raw=load(root/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz');fig,axs=plt.subplots(4,2,figsize=(12,11),sharex=True)
    t=np.arange(480)-120
    for i,ax in enumerate(axs.flat):
        counts=[]
        for n in dom:
            row=m['raw_voltage_moments'][n][i+1];counts.append(row['N']);key=n+'__'+str(i)+'__mean_V'
            if key in raw:ax.plot(t,raw[key]*1000,color=col[n],label=lab[n],lw=1.2)
        ax.set(title=f'Observed-Q bin{i+1}; N R/B/F/C='+ '/'.join(map(str,counts)),ylabel='Raw voltage mean (mV)');ax.grid(alpha=.12)
        if i>=6:ax.set_xlabel('Time from saved CFD reference (ns)')
    axs[0,0].legend(fontsize=7);fig.suptitle('All480 points; no domain moment transform, smoothing or separate amplitude normalization')
    fig.tight_layout();save(fig,'conditional_waveforms.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));vnames=['full_historical_shape','normalized_u480','baseline_local']
    for j,(n,colour)in enumerate(zip(rc,[col[s],col[l]])):
        for i,v in enumerate(vnames):
            r=m['C2ST']['same_treated_REAL_control_paired'][n][v];d=r['BA_delta_first_minus_second'];lo,hi=r['BA_delta_95CI']
            axs[0].errorbar(i+(-.08 if j==0 else .08),d,yerr=[[d-lo],[hi-d]],fmt='o',color=colour,capsize=4,label=lab[n]if i==0 else None)
        ident=json.loads((root/'REAL_IDENTITY_DESCRIPTIONS.json').read_text('utf8'))[n]
        summary=ident['postADC_waveform_RMS_vs_original_V'];axs[1].bar(j,summary['mean']*1000,color=colour)
    axs[0].axhline(.01,color='gray',ls=':',label='Point gate +.01');axs[0].axhline(.10,color='gray',ls='--',label='95% upper gate +.10');axs[0].axhline(0,color='black',lw=.7)
    axs[0].set_xticks(range(3),['shape493','pure u480','baseline']);axs[0].set(ylabel='New minus same-treated frozen BASE BA',title='Fixed500 group bootstrap; gates only these3 views')
    axs[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.32),ncol=2)
    axs[1].set_xticks([0,1],['FIXED','COMPLETION']);axs[1].set(ylabel='Mean postADC waveform RMS change (mV)',title='Paired REAL identity error; descriptive')
    fig.subplots_adjust(top=.74,wspace=.35,bottom=.17);save(fig,'same_treated_REAL_controls.png')
    fig,axs=plt.subplots(1,3,figsize=(12,4.2));primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
    for j,n in enumerate(['BASE',s,l]):
        vals=[primary[n][b]['valley_fraction']for b in ['bin3','bin4']]
        axs[0].bar(np.arange(2)+(j-1)*.23,[100*x if x is not None else 0 for x in vals],width=.23,color=col[n],label=lab[n])
        r=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all']
        axs[1].plot([0,1],[r[k]['Spearman']for k in ['source_Q_to_output_Q','source_PSD_to_output_PSD']],'o-',color=col[n],label=lab[n])
        sens=m['source_probes'][n]['rows']if n in m['source_probes']else None
        if sens:axs[2].bar(j-1,sens['source_roll_no_fixed_points']['waveform_RMS_vs_correct_reference_V']['mean']*1000,color=col[n])
    axs[0].set_xticks([0,1],['source bin3 (N213)','source bin4 (N181)']);axs[0].set(ylabel='Frozen valley occupancy (%)',title='Pure-had deposition proxy; not particle error');axs[0].legend(fontsize=7)
    axs[1].set_xticks([0,1],['Q rank','PSD rank']);axs[1].set(ylabel='Spearman association',title='Actual unquantized source condition')
    axs[2].set_xticks([0,1],['FIXED','COMPLETION']);axs[2].set(ylabel='Source-roll RMS change (mV)',title='Sensitivity does not establish physical correspondence')
    fig.tight_layout();save(fig,'source_preservation.png')
    fig,ax=plt.subplots(figsize=(11,4.5));bd=m['boundary_differences']['results'];keys=list(bd[s])
    for j,n in enumerate(['BASE',s,l]):ax.plot(np.arange(9)+(j-1)*.12,[bd[n][k]['W1_V']*1000 for k in keys],'o-',color=col[n],label=lab[n])
    lim=[max(bd['BASE'][k]['W1_V'],bd['INDEPENDENT_CFM'][k]['W1_V'])+1/16383 for k in keys]
    ax.plot(range(9),np.asarray(lim)*1000,'k--',label='Frozen BASE/Independent max +1 LSB')
    ax.set_xticks(range(9),[k.replace('first_','D1 ').replace('second_center_','D2 ')for k in keys],rotation=25,ha='right')
    ax.set(ylabel='Native-voltage boundary W1 (mV)',title='Exactly9 fixed boundary locations; no smoothing or post-projection');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2)
    fig.subplots_adjust(top=.73,bottom=.25);save(fig,'boundary_differences.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));initial=json.loads((root/'INITIAL_NOISE_DESCRIPTIONS.json').read_text('utf8'))
    for n in [s,l,*rc]:
        rows=initial[n]['by_actual_conditioning_bin'];values=[]
        for r in rows:
            x=r['fields']['initial_variance100_actual_V2']['mean'];values.append(np.sqrt(x)*1000 if x is not None else np.nan)
        axs[0].plot(np.arange(1,9),values,'o-'if n in[s,l]else'x--',color=col[n],label=lab[n])
    axs[0].set(xlabel='Actual conditioning Q bin',ylabel='sqrt(mean initial baseline variance) (mV)',title='Actual seed state; expected variance is not eventwise');axs[0].legend(fontsize=7)
    comps=['projected_colored_innovation','learned_increment_plus_final_f32_cast']
    for j,n in enumerate([s,l]):
        by=m['random_decomposition'][n]['groups']['joint8']['by_composition'];cat='pure_had'if 'pure_had'in by else next(iter(by))
        cv=by[cat]['three_additive_algebraic_components_covariance_V2']['baseline100']
        vals=[cv[a+'__'+b]['mean']*1e6 for a,b in[(comps[0],comps[0]),(comps[1],comps[1]),(comps[0],comps[1])]]
        axs[1].bar(np.arange(3)+(j-.5)*.3,vals,width=.3,color=col[n],label=lab[n])
    axs[1].axhline(0,color='black',lw=.7);axs[1].set_xticks(range(3),['Seed variance','Learned + cast\nvariance','Seed/learned\ncovariance']);axs[1].set(ylabel='Mean joint8 baseline covariance (mV2)',title='Pure-had8-draw algebraic decomposition');axs[1].legend(fontsize=7)
    fig.suptitle('Cross terms retained; components are not independently identified physical noises')
    fig.tight_layout();save(fig,'initial_and_random_noise.png')
    return sorted(p.name for p in out.glob('*.png'))
