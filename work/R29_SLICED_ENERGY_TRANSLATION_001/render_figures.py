from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True,parents=True);START=time.monotonic()
m=json.loads((ROOT/'metrics.json').read_text('utf8'));cfg=json.loads((ROOT/'config.json').read_text('utf8'));old=json.loads((pathlib.Path(cfg['R28_root'])/'metrics.json').read_text('utf8'))
names=['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL'];colors={'REAL':'black','SOURCE_CLEAN':'#929292','FRESH_SELECTED':'#377eb8','FRESH_FINAL':'#78b5da','ENERGY_SELECTED':'#d95f02','ENERGY_FINAL':'#f4a460'}
labels={'REAL':'REAL DEV','SOURCE_CLEAN':'SOURCE clean','FRESH_SELECTED':'Fresh selected'+str(m['control_training_reused']['selected_step']),'FRESH_FINAL':'Fresh final'+str(m['control_training_reused']['updates']),'ENERGY_SELECTED':'Energy selected'+str(m['training']['SLICED_ENERGY']['selected_step']),'ENERGY_FINAL':'Energy final'+str(m['training']['SLICED_ENERGY']['updates'])}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':145})
def save(fig,name):fig.savefig(OUT/name,bbox_inches='tight');plt.close(fig)
def load(p):
    with np.load(p,allow_pickle=False) as z:return{k:z[k].copy() for k in z.files}
def summary_mean(s):return s.get('mean')
edges=np.asarray(m['PSD_main']['edges_logQ']);waves=load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz');features=load(ROOT/'results/FEATURES_PRIVATE.npz');time_axis=np.arange(480)-120

# Frozen selection scores, actual fine groups and online D are distinct diagnostics.
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
for name,tr,color in [('fresh W1',m['control_training_reused'],'#377eb8'),('energy U',m['training']['SLICED_ENERGY'],'#d95f02')]:
    rows=[x['validation'] for x in tr['validation_trace']];steps=[r['step'] for r in rows]
    ax[0].plot(steps,[r['J'] for r in rows],'.-',color=color,label=name)
    ax[1].plot(steps,[r['fine_val'] for r in rows],'.-',color=color,label=name)
    ax[2].plot(steps,[r['online_D_weighted_BA'] for r in rows],'.-',color=color,label=name)
    ax[0].scatter([tr['selected_step']],[tr['selected_validation']['J']],marker='*',s=85,color=color)
for a,title,ylabel in zip(ax,['Original fixed J20 selection','Validation uses original directions','Online D does not select G'],['J20','fixed fine validation','weighted BA']):
    a.set(title=title,xlabel='G updates',ylabel=ylabel);a.legend()
save(fig,'training_online_D.png')

views=[*cfg['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480'];fig,ax=plt.subplots(figsize=(13,4))
for j,n in enumerate(names):
    rows=[m['C2ST']['results'][n][v] if n.startswith('ENERGY') else m['C2ST']['paired_vs_corresponding_R28']['ENERGY_SELECTED' if n=='FRESH_SELECTED' else 'ENERGY_FINAL'][v]['old_control_result'] for v in views]
    x=np.arange(len(views))+(j-1.5)*.19;ba=np.array([r['balanced_accuracy'] for r in rows]);ci=np.array([r['heldout_CI']['balanced_accuracy_95CI'] for r in rows])
    ax.bar(x,ba,.18,color=colors[n],label=labels[n]);ax.errorbar(x,ba,yerr=np.maximum(0,np.vstack([ba-ci[:,0],ci[:,1]-ba])),fmt='none',color='black',lw=.65,capsize=2)
ax.axhline(.5,color='gray',ls=':');ax.set(ylim=(.4,1.02),ylabel='holdout balanced accuracy',title='Fixed rows/groups; seven historical and three pure480 views')
ax.set_xticks(np.arange(len(views)),['shape493','full\nvoltage','prompt\nlocal','postpeak\nlocal','tail\nlocal','baseline\nlocal','Q / PSD\n2D','pure u480','seen\n480 points','complement\n480 points']);ax.legend(loc='lower right',fontsize=8,ncol=2)
save(fig,'c2st.png')

fig,ax=plt.subplots(1,5,figsize=(15,3.2),sharex=True,sharey=True);plotnames=['REAL',*names]
qs=[features[n+'__Qtotal_mVns'] for n in plotnames];positive=np.concatenate([q[q>0] for q in qs]);xmin,xmax=np.min(positive),np.max(positive)
for a,n in zip(ax,plotnames):
    q=features[n+'__Qtotal_mVns'];p=features[n+'__PSD'];valid=features[n+'__valid']&(q>0)
    a.scatter(q[valid],p[valid],s=2,alpha=.28,color=colors[n],rasterized=True);a.set_xscale('log');a.set(xlim=(xmin*.95,xmax*1.05),ylim=(-.1,.6),title=labels[n],xlabel='Q total (mV ns)');a.grid(alpha=.15)
    outside=int(np.sum(valid&((p<-.1)|(p>.6))));a.text(.02,.02,f'QC {int(valid.sum())}\nPSD outside {outside}',transform=a.transAxes,fontsize=8)
ax[0].set_ylabel('signed PSD');save(fig,'charge_psd.png')

fig,ax=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True)
with np.load(ROOT/'results/PSD_SECTIONS_PRIVATE.npz',allow_pickle=False) as z:
    for i,a in enumerate(ax.flat):
        count=[]
        for n in ['REAL',*names]:
            x=z[n+'__bin'+str(i+1)+'__PSD'];count.append(len(x))
            if len(x):a.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[n],label=labels[n],lw=1.1)
        a.set(title=f'Bin{i+1}',xlim=(-.1,.6),ylim=(0,1));a.text(.03,.04,'N '+ '/'.join(map(str,count)),transform=a.transAxes,fontsize=7)
        a.grid(alpha=.15)
for a in ax[1]:a.set_xlabel('signed PSD')
for a in ax[:,0]:a.set_ylabel('empirical CDF')
ax[0,0].legend(fontsize=7,loc='upper left');fig.suptitle('Observed-Q bins; complete CDF and counts, no density relabeling',y=1.01)
fig.tight_layout();save(fig,'conditional_psd_CDF.png')

tables=m['energy_conditional_width']['tables'];qmid=np.exp((edges[1:]+edges[:-1])/2);fig,ax=plt.subplots(2,3,figsize=(12,6),sharex=True)
for n in ['REAL',*names]:
    slices=tables[n]['slices']
    for c in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            values=[r['components'][c][field] if r['valid'] and len(r['components'])==2 else np.nan for r in slices]
            ax[c,j].plot(qmid,values,'o-',label=labels[n],color=colors[n],markersize=3)
for c in [0,1]:
    for j,title in enumerate(['component center','raw PSD width','detrended PSD width']):
        a=ax[c,j];a.set_xscale('log');a.set(title=f'{"low" if c==0 else "high"} PSD: {title}',ylabel='PSD units');a.grid(alpha=.15)
        if c==1:a.set_xlabel('Q-bin midpoint (mV ns)')
ax[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'energy_peak_width.png')

fig,ax=plt.subplots(2,4,figsize=(13,6),sharex=True)
for i,a in enumerate(ax.flat):
    for n in ['REAL',*names]:
        q=features[n+'__Qtotal_mVns'];z=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7
        use=features[n+'__valid']&(b==i);v=waves[n][use]
        if len(v):
            a.plot(time_axis,np.median(v,axis=0)*1000,color=colors[n],lw=1.1,label=labels[n])
        if len(v)<40:a.text(.02,.94-.055*list(['REAL',*names]).index(n),f'{n} N{len(v)} <40',color=colors[n],transform=a.transAxes,fontsize=6.5)
    a.set(title=f'Observed Q bin{i+1}',xlim=(-30,280));a.grid(alpha=.12)
for a in ax[1]:a.set_xlabel('saved time relative to index120 (ns)')
for a in ax[:,0]:a.set_ylabel('median voltage (mV)')
ax[0,0].legend(fontsize=6.5);fig.suptitle('Complete saved-row condition medians; differences in population are retained',y=1.01);fig.tight_layout();save(fig,'conditional_waveforms.png')

fig,ax=plt.subplots(1,4,figsize=(13,3.3))
for i,field in enumerate(['baseline_RMS_V','ACF1','prompt_curvature_RMS_V','postpeak_curvature_RMS_V']):
    # Features may use the common established RMS key; preserve undefined rather than fabricate values.
    rows=[]
    for n in ['REAL',*names]:
        key=n+'__'+field
        if key not in features and field=='baseline_RMS_V':key=n+'__baseline_RMS'
        x=features[key] if key in features else np.array([]);x=x[np.isfinite(x)]
        if len(x):ax[i].step(np.sort(x),np.arange(1,len(x)+1)/len(x),color=colors[n],label=labels[n],lw=1.1)
        else:ax[i].text(.05,.95-.08*len(rows),f'{n}: undefined',transform=ax[i].transAxes,fontsize=7)
        rows.append(n)
    ax[i].set(title=field.replace('_',' '),ylabel='empirical CDF');ax[i].grid(alpha=.15)
ax[0].legend(fontsize=6.5);fig.tight_layout();save(fig,'noise_curvature.png')

fig,ax=plt.subplots(1,3,figsize=(13,3.6));s=m['projection_distances']['common_support_summary']
for a,group in zip(ax,['shape480','diff479','baseline100_V']):
    rows=s[group]['domains'];x=np.arange(len(names)+1)
    ds=names+['TRAIN_REFERENCE'];original=[rows[n]['equal_mean_on_common_bins']['ORIGINAL'] for n in ds];fresh=[rows[n]['new_block_mean'] for n in ds]
    ranges=np.array([rows[n]['new_block_range'] for n in ds]);a.bar(x-.17,original,.32,label='original directions',color='#aaa');a.bar(x+.17,fresh,.32,label='four reused blocks',color='#666')
    a.errorbar(x+.17,fresh,yerr=np.maximum(0,np.vstack([np.asarray(fresh)-ranges[:,0],ranges[:,1]-np.asarray(fresh)])),fmt='none',color='black',capsize=2)
    a.set(title=group+'; bins'+','.join(map(str,s[group]['common_qualified_bin_ids_1based'])),ylabel='conditional quantile W1')
    a.set_xticks(x,[n.replace('HOMO','Fixed').replace('FRESH','Fresh').replace('_','\n').replace('TRAIN_REFERENCE','REAL ref') for n in ds],fontsize=7)
ax[0].legend(fontsize=7);fig.suptitle('Block ranges are direction sampling references, not confidence intervals',y=1.03);fig.tight_layout();save(fig,'projection_directions.png')

fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for n in names:
    rows=m['projection_coverage'][n]['observed_Q'][1:];ok=np.array([r['qualified'] for r in rows]);values=np.array([r.get('seen_sqnorm',np.nan) for r in rows]);comp=np.array([r.get('complement_sqnorm',np.nan) for r in rows])
    ax[0].plot(np.arange(1,9),np.where(ok,values,np.nan),'o-',color=colors[n],label=labels[n]);ax[1].plot(np.arange(1,9),np.where(ok,comp,np.nan),'o-',color=colors[n])
    if not ok.all():ax[1].text(.04,.93-.08*names.index(n),f'{n} unsupported:'+','.join(str(i+1) for i in np.flatnonzero(~ok)),transform=ax[1].transAxes,color=colors[n],fontsize=7)
for a,t in zip(ax,['Rank390 covered-space mean residual','Dimension90 complement mean residual']):a.set(title=t,xlabel='observed Q bin',ylabel='squared norm');a.grid(alpha=.15)
ax[0].legend(fontsize=7);fig.tight_layout();save(fig,'coverage_residuals.png')

fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for i,n in enumerate(names):
    source=m['source_composition']['primary_pure_had_source_bins3_4'][n]
    for j in [0,1]:
        r=source['bin'+str(j+3)];bar=i+(j-.5)*.19
        ax[0].bar(bar,r['output_PSD']['mean'],.18,color=colors[n],alpha=1 if j==0 else .55)
        ax[1].bar(bar,r['valley_fraction'],.18,color=colors[n],alpha=1 if j==0 else .55)
for a,t in zip(ax,['Pure-had source bin3/4: centers','Pure-had source bin3/4: valley fractions']):
    a.set(title=t);a.set_xticks(np.arange(4),[labels[n].replace(' ','\n') for n in names],fontsize=8);a.grid(axis='y',alpha=.15)
ax[0].set_ylabel('mean signed PSD');ax[1].set_ylabel('fraction');fig.suptitle('Dark=source-bin3 (207 parents), light=bin4 (182); complete parents retained',y=1.03);fig.tight_layout();save(fig,'pure_had_source_conditions.png')

fig,ax=plt.subplots(1,3,figsize=(12,3.5))
for i,n in enumerate(names):
    probes=m['source_probes'][n]['rows'];tags=['source_roll_no_fixed_points','fixed_source_changed_noise']
    for j,tag in enumerate(tags):
        if tag in probes:
            v=probes[tag]['waveform_RMS_vs_correct_reference_V']['mean'];ax[0].bar(i+(j-.5)*.18,v*1000,.17,color=colors[n],alpha=1 if j==0 else .5)
    joint=m['random_decomposition'][n]['groups']['joint8']
    # Entire joint summary is retained; this figure uses the known feature variation schema if available.
    blob=json.dumps(joint)
    ax[2].text(.02,.94-.19*i,f'{labels[n]}\n128 source parents ×8 repeats',transform=ax[2].transAxes,color=colors[n],fontsize=8)
    inp=ROOT/'results'/(n+'_joint8_REPEAT_PRIVATE.npz')
    if not inp.exists():inp=pathlib.Path(cfg['R28_root'])/'results'/(n+'_joint8_REPEAT_PRIVATE.npz')
    a=load(inp);p=a['PSD'].reshape(128,8);sd=p.std(1,ddof=1);ax[1].step(np.sort(sd),np.arange(1,129)/128,color=colors[n],label=labels[n])
ax[0].set(title='Corrected probes: dark=roll, light=noise',ylabel='RMS to correct reference (mV)');ax[0].set_xticks(np.arange(4),[labels[n].replace(' ','\n') for n in names],fontsize=7)
ax[1].set(title='Same-parent random PSD spread',xlabel='within-parent PSD SD',ylabel='empirical CDF');ax[1].legend(fontsize=7);ax[2].axis('off');fig.tight_layout();save(fig,'source_random.png')

files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.glob('*.png'))}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
         'figures':files,'figures_count':len(files),'fit_generation_updates':0,'source':'saved R29/R28 metrics and saved plot arrays only'}
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
