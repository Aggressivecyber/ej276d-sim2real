from pathlib import Path
import ast
R=Path(__file__).resolve().parent;old=R.parent/'R28_FRESH_PROJECTION_TRANSLATION_001/render_figures.py';s=old.read_text('utf8')
s=s.replace("names=['HOMO_SELECTED','HOMO_FINAL','FRESH_SELECTED','FRESH_FINAL'];colors={'REAL':'black','SOURCE_CLEAN':'#929292','HOMO_SELECTED':'#377eb8','HOMO_FINAL':'#78b5da','FRESH_SELECTED':'#d95f02','FRESH_FINAL':'#f4a460'}", "names=['CONTROL_SELECTED','CONTROL_FINAL','MEAN_SELECTED','MEAN_FINAL'];colors={'REAL':'black','SOURCE_CLEAN':'#929292','CONTROL_SELECTED':'#377eb8','CONTROL_FINAL':'#78b5da','MEAN_SELECTED':'#d95f02','MEAN_FINAL':'#f4a460'}")
s=s.replace("labels={'REAL':'REAL DEV','SOURCE_CLEAN':'SOURCE clean','HOMO_SELECTED':'Fixed selected2750','HOMO_FINAL':'Fixed final2950','FRESH_SELECTED':'Fresh selected2250','FRESH_FINAL':'Fresh final2950'}", "labels={'REAL':'REAL DEV','SOURCE_CLEAN':'Frozen post-ADC source','CONTROL_SELECTED':'Control selected750','CONTROL_FINAL':'Control final2950','MEAN_SELECTED':'Mean selected1750','MEAN_FINAL':'Mean final2950'}")
s=s.replace("[('fixed',m['control_training_reused'],'#377eb8'),('fresh',m['training']['FRESH_FINE'],'#d95f02')]", "[('server control',m['training']['SERVER_FRESH_CONTROL'],'#377eb8'),('conditional mean',m['training']['CONDITIONAL_MEAN'],'#d95f02')]")
oldrows="rows=[m['C2ST']['results'][n][v] if n.startswith('FRESH') else m['C2ST']['paired_vs_corresponding_R26']['FRESH_SELECTED' if n=='HOMO_SELECTED' else 'FRESH_FINAL'][v]['old_control_result'] for v in views]"
assert oldrows in s;s=s.replace(oldrows,"rows=[m['C2ST']['results'][n][v] for v in views]")
s=s.replace("fig,ax=plt.subplots(figsize=(13,4))","fig,ax=plt.subplots(figsize=(14.5,4))")
s=s.replace("[v.replace('full_historical_shape','shape493').replace('_','\\n') for v in views]", "['shape493','voltage480','prompt','postpeak','tail','baseline','Q-PSD','pure u480','seen\\n480pts','complement\\n480pts']")
s=s.replace("xlim=(-30,280)","xlim=(-120,359)")
s=s.replace("[n.replace('HOMO','Fixed').replace('FRESH','Fresh').replace('_','\\n').replace('TRAIN_REFERENCE','REAL ref') for n in ds]", "[labels[n].replace(' ','\\n') if n in labels else 'REAL\\nTRAIN ref' for n in ds]")
s=s.replace("Pure-had source bin3/4: centers","Pure-had post-ADC reference Q bin3/4: centers").replace("Pure-had source bin3/4: valley fractions","Pure-had post-ADC reference Q bin3/4: valleys")
s=s.replace("Dark=source-bin3 (207 parents), light=bin4 (182); complete parents retained","Dark=frozen post-ADC reference Q bin3 (207), light=bin4 (182); actual NN-input Q differs")
extras=r'''
# New TRAIN-only mean constraint and saved-output SD evidence.
fig,ax=plt.subplots(1,3,figsize=(12.5,3.5))
for arm,color,label in [('SERVER_FRESH_CONTROL','#377eb8','Control: diagnostic only'),('CONDITIONAL_MEAN','#d95f02','Mean: coefficient 1')]:
    vals=load(ROOT/arm/'MEAN_LOSS_BY_UPDATE_PRIVATE.npz')['values'];step=np.arange(1,len(vals)+1)
    # Every actual update is displayed; no smoothing or fit.
    ax[0].plot(step,vals[:,0],color=color,lw=.4,alpha=.65,label=label)
    ax[1].plot(step,vals[:,1],color=color,lw=.4,alpha=.65,label=label+' mean-square')
    ax[1].plot(step,vals[:,2],color=color,lw=.4,ls='--',alpha=.65,label=label+' var correction')
    ax[2].plot(step,np.cumsum(vals[:,0]<0),color=color,label=label)
for a,title,y in zip(ax,['Off-diagonal U: negatives retained','U = mean-square - variance correction','Actual negative-U batches'],['standardized mean U','standardized squared units','cumulative count']):
    a.set(title=title,xlabel='G update',ylabel=y);a.grid(alpha=.15);a.legend(fontsize=6.5)
fig.tight_layout();save(fig,'mean_constraint_training.png')

md=m['conditional_mean_SD'];mp=load(ROOT/'results/MEAN_SD_DESCRIPTIONS_PRIVATE.npz')
for field,ylab,fname in [('mean_u','peak-normalized mean u','conditional_mean_waveforms.png'),('SD_u','pointwise SD of u (ddof=1)','conditional_SD.png')]:
    fig,ax=plt.subplots(2,4,figsize=(13.5,6),sharex=True)
    for i,a in enumerate(ax.flat):
        counts=[]
        for n in ['REAL','CONTROL_FINAL','MEAN_FINAL']:
            row=md['domains'][n]['observed_Q'][i+1];counts.append(row['QC_n']);key=n+'__observed_Q__bin'+str(i+1)+'__'+field
            if key in mp:a.plot(time_axis,mp[key],color=colors[n],lw=1.15,ls='-' if row['qualified'] else ':',label=labels[n])
        a.set(title='Observed Q bin'+str(i+1),xlim=(-120,359));a.grid(alpha=.12);a.text(.03,.04,'N REAL/C/M: '+ '/'.join(map(str,counts)),transform=a.transAxes,fontsize=7)
        if i+1 not in md['common_observed_Q_bin_ids_1based']:a.text(.04,.92,'Low support: descriptive only',transform=a.transAxes,fontsize=7,color='#666')
    for a in ax[1]:a.set_xlabel('saved time relative to index120 (ns)')
    for a in ax[:,0]:a.set_ylabel(ylab)
    ax[0,0].legend(fontsize=7);fig.suptitle('Complete 480-point observed-Q descriptions; TRAIN target and DEV reference are distinct',y=1.01);fig.tight_layout();save(fig,fname)

fig,ax=plt.subplots(1,3,figsize=(13,3.5));sem=m['source_semantics']
for j,n in enumerate(['CONTROL_FINAL','MEAN_FINAL']):
    for k in [3,4]:
        rr=m['source_composition']['primary_pure_had_source_bins3_4'][n]['bin'+str(k)];ir=sem['actual_NN_input_source_Q_endpoints'][n]['primary_actual_source_bins3_4'][k-3]
        pos=j*3+(k-3)
        for a,field in zip(ax,['mean','sample_sd','valley']):
            x=rr['output_PSD'][field]if field!='valley'else rr['valley_fraction'];y=ir['output_PSD'][field]if field!='valley'else ir['valley_fraction']
            a.bar(pos-.16,x,.3,color=colors[n],alpha=.45);a.bar(pos+.16,y,.3,color=colors[n])
for a,title,y in zip(ax,['Conditional PSD center','Conditional empirical width','Frozen valley fraction'],['mean PSD','sample SD: no density refit','fraction']):
    a.set(title=title,ylabel=y);a.set_xticks([0,1,3,4],['C bin3','C bin4','M bin3','M bin4']);a.grid(axis='y',alpha=.15)
fig.suptitle('Light: ADC-reference Q (N207/182); dark: actual input Q (N213/181). Same parents, different conditions.',y=1.02);fig.tight_layout();save(fig,'source_condition_semantics.png')

fig,ax=plt.subplots(1,2,figsize=(11.5,3.6))
for n in ['CONTROL_FINAL','MEAN_FINAL']:
    rows=md['domains'][n]['observed_Q'][1:];good=np.array([r['qualified'] for r in rows]);v=np.array([r.get('full480_mean_MSE',np.nan) for r in rows]);ax[0].plot(np.arange(1,9),np.where(good,v,np.nan),'o-',color=colors[n],label=labels[n])
    row=md['domains'][n]['observed_Q'][0];regions=['baseline','prompt','postpeak','tail','late'];ax[1].plot(np.arange(5),[row['region_mean_residual_RMS'][r] for r in regions],'o-',color=colors[n],label=labels[n])
ax[0].set(title='Raw mean residual: supported observed Q',xlabel='Q-bin ID',ylabel='full480 mean MSE');ax[0].legend(fontsize=8)
ax[1].set(title='Population mean residual by fixed region',ylabel='RMS of mean difference in u');ax[1].set_xticks(np.arange(5),regions)
for a in ax:a.grid(alpha=.15)
fig.tight_layout();save(fig,'mean_residuals.png')

'''
marker="files={p.name:{'bytes'"
assert marker in s;s=s[:s.index(marker)]+extras+s[s.index(marker):]
s=s.replace("'source':'saved R28/R26 metrics and saved plot arrays only'", "'source':'saved R32 metrics/output arrays; source semantic comparisons from saved arrays only'")
assert 'paired_vs_corresponding_R26'not in s and 'control_training_reused'not in s
ast.parse(s);(R/'render_figures.py').write_text(s,'utf8');print('renderer_ready',len(s.encode()))
