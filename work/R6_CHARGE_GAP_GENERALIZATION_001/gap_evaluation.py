from __future__ import annotations
import pathlib,hashlib,json
import numpy as np
import evaluation as ev
import energy_width as ew
from common_math import distances,fom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

def take(d,indices):
    return {k:({f:v[indices] for f,v in a.items()} if k=='features' else a[indices] if isinstance(a,np.ndarray) and len(a)==len(d['V']) else a) for k,a in d.items()}
def select_q3(d):return take(d,np.flatnonzero(d['valid']&(d['region']==2)))
def assigned(d,ids,groups,kind,ref):
    d['ids']=np.asarray(ids);d['groups']=np.asarray(groups);d['kind']=kind;d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1');return d
def order(ids,prefix):return np.asarray(sorted(range(len(ids)),key=lambda i:hashlib.sha256((prefix+'|'+str(ids[i])).encode()).digest()),dtype='i8')
def c2st_family(domains,cfg,write):
    q3={name:select_q3(d) for name,d in domains.items()};splits={};supports={}
    for name,d in q3.items():
        keys=d['groups'] if d['kind']=='REAL' else d['ids'];flag=np.array([ev.hashint('R1-SPLIT|'+d['kind']+'|'+str(k))%10<7 for k in keys]);tr=np.flatnonzero(flag);te=np.flatnonzero(~flag)
        supports[name]={'total':len(d['V']),'train':len(tr),'holdout':len(te),'groups':len(np.unique(d['groups']))}
        if len(tr)>=200 and len(te)>=100:splits[name]=(tr[ev.order(d['ids'][tr],'R1-C2ST-SAMPLE')[:200]],te[ev.order(d['ids'][te],'R1-C2ST-SAMPLE')[:100]])
    if len(splits)!=len(q3):return {'defined':False,'reason':'one_or_more_fixed_q3_domains_cannot_supply200TRAIN100HOLDOUT','supports':supports,'no_replacement_or_extra_generation':True,'all_candidates_preserved':True}
    pars={k:cfg['C2ST'][k] for k in ['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','early_stopping','random_state']}
    result={};saved={}
    for name in [n for n in q3 if n!='REAL']:
        a,b=q3[name],q3['REAL'];at,av=splits[name];bt,bv=splits['REAL'];rows={}
        assert not set(a['groups'][at])&set(a['groups'][av]) and not set(b['groups'][bt])&set(b['groups'][bv])
        for view in cfg['C2ST']['views']:
            x,scalar=ev.c2st_matrix(a,view);y,_=ev.c2st_matrix(b,view);xx=np.vstack([x[at],y[bt]]);test=np.vstack([x[av],y[bv]]);label=np.r_[np.ones(200),np.zeros(200)];held=np.r_[np.ones(100),np.zeros(100)]
            imp=SimpleImputer(strategy='median',keep_empty_features=True);xx=imp.fit_transform(xx);test=imp.transform(test);model=HistGradientBoostingClassifier(**pars);model.fit(xx,label);prob=model.predict_proba(test)[:,1]
            clusters=np.r_[np.array(['SIM|'+str(g) for g in a['groups'][av]]),np.array(['REAL|'+str(g) for g in b['groups'][bv]])]
            acc=float(balanced_accuracy_score(held,prob>=.5));blocks={'scalar_PSD_rise_tail':range(4),'relative_RMS_ACF':range(4,13),'wave_baseline':range(scalar,scalar+100),'wave_prompt':range(scalar+100,scalar+150),'wave_tail':range(scalar+150,scalar+400),'wave_end':range(scalar+400,scalar+480)}
            if scalar>13:blocks['absolute_Q_peak_RMS']=range(13,scalar)
            rng=np.random.default_rng(20261008);importance={}
            for key,cols in blocks.items():
                cols=list(cols);u=test.copy();sh=rng.permutation(len(u));u[:,cols]=u[np.ix_(sh,cols)];importance[key]=acc-float(balanced_accuracy_score(held,model.predict(u)))
            rows[view]={'balanced_accuracy':acc,'AUC':float(roc_auc_score(held,prob)),'TRAIN_each':200,'HOLDOUT_each':100,'heldout_CI':ev.bootstrap(held,prob,clusters,20261008),'heldout_permutation_block_accuracy_drop':importance,'hyperparameters':pars,'not_response_optimizer':True}
            saved[name+'__'+view+'__probability']=prob;saved[name+'__'+view+'__truth']=held;saved[name+'__'+view+'__cluster']=clusters
        result[name]=rows
    write('C2ST_Q3_SPLITS.json',{n:{'train_ids':q3[n]['ids'][i[0]],'holdout_ids':q3[n]['ids'][i[1]],'source_group_intersection':0} for n,i in splits.items()})
    np.savez_compressed(ev.OUT/'C2ST_Q3_PREDICTIONS.npz',**saved)
    return {'defined':True,'supports':supports,'results':result,'selection':'same fixed whole-group or parent hash,200/100 each for allfive candidates;REAL heldout shared'}
def rr_q3(real,domains,ref,write):
    d=select_q3(real);uniq=np.unique(d['groups']);splits=[];min_available=len(d['V'])
    for r in range(64):
        groups=sorted(uniq.tolist(),key=lambda g:hashlib.sha256(f'R6-RR-group-v1|{r}|{g}'.encode()).digest());half=[groups[:len(groups)//2],groups[len(groups)//2:]]
        ix=[np.flatnonzero(np.isin(d['groups'],h)) for h in half];ix=[v[order(d['ids'][v],f'R6-RR-event-v1|{r}')] for v in ix];min_available=min(min_available,*(len(v) for v in ix));splits.append(ix)
    requested={n:len(select_q3(a)['V']) for n,a in domains.items() if n!='REAL'};common=min(min_available,*requested.values());tables={};cut=ref['slots'][3]['cut'] if ref['slots'][3]['valid'] else None
    for n in sorted(set([*requested.values(),common])):
        if n==0 or n>min_available:tables[str(n)]={'defined':False,'requested_each':n,'smallest_half_available':min_available,'reason':'disjoint_group_half_support_insufficient','no_replacement':True};continue
        records=[]
        for r,ix in enumerate(splits):
            a,b=[take(d,v[:n]) for v in ix];record={'repeat':r,'each_N':n,'groups_selected':[len(np.unique(a['groups'])),len(np.unique(b['groups']))],
                 'features':{k:distances(a['features'][k][np.isfinite(a['features'][k])],b['features'][k][np.isfinite(b['features'][k])]) for k in a['features']},'FoM':[fom(a['features']['PSD'],cut),fom(b['features']['PSD'],cut)]};records.append(record)
        summary={}
        for f in d['features']:
            summary[f]={}
            for metric in ['W1','KS']:
                v=[r['features'][f][metric] for r in records if r['features'][f][metric] is not None];summary[f][metric]={'median':float(np.median(v)) if v else None,'range5_95':np.quantile(v,[.05,.95]).tolist() if v else None,'defined_repeats':len(v)}
        tables[str(n)]={'defined':True,'each_N':n,'repeats':64,'records':records,'summary':summary}
    r={'definition':'64 fixed whole-artificial-group half splits inside EVAL QC q3;equalN only when every split has support','REAL_q3_N':len(d['V']),'REAL_q3_groups':len(uniq),'requested_candidate_N':requested,'minimum_group_half_available':min_available,'common_feasible_N':common,'references_by_N':tables,'not_independent_physical_runs_or_equivalence_CI':True};write('Q3_REAL_REAL_REFERENCE.json',r);return r
def feature_summary(d,m):
    out={}
    for k,v in d['features'].items():
        good=m&np.isfinite(v);a=v[good];out[k]={'QC_N':int(m.sum()),'defined_N':len(a),'undefined_N':int(m.sum())-len(a),'mean':float(a.mean()) if len(a) else None,'quantiles10_50_90':np.quantile(a,[.1,.5,.9]).tolist() if len(a) else None}
    return out
def evaluate(domains,ref,cfg,nominal_region,edges,write):
    ev.NAMES=list(domains);ev.COLORS={'R2':'#bb9b48','R3':'#bda0cb','R3B':'#e65c24','FULL':'#249b74','GAP':'#3579c4','REAL':'black'}
    stats,dist=ev.characterize(domains,ref);migration={};supplement={};realq3=domains['REAL'];rm=realq3['valid']&(realq3['region']==2)
    for name in ['FULL','GAP']:
        d=domains[name];table=np.zeros((4,4),dtype='i8')
        for source,out,valid in zip(nominal_region,d['region'],d['valid']):
            if valid:table[int(source),int(out)]+=1
        sm=d['valid']&(nominal_region==2)
        migration[name]={'rows_source_nominal_q1_q4_columns_output_own_Q_q1_q4':table,'source_parent_counts':np.bincount(nominal_region,minlength=4),'output_QC':int(d['valid'].sum()),'output_Q3_count':int((d['valid']&(d['region']==2)).sum()),'invalid_outputs_by_nominal_region':np.bincount(nominal_region[~d['valid']],minlength=4)}
        supplement[name]={'fixed_source_nominal_q3_parents':int((nominal_region==2).sum()),'QC_outputs_from_all_those_parents':int(sm.sum()),'output_region_counts':np.bincount(d['region'][sm],minlength=4),'features':feature_summary(d,sm),
                          'vs_same_EVAL_q3':{f:distances(d['features'][f][sm&np.isfinite(d['features'][f])],realq3['features'][f][rm&np.isfinite(realq3['features'][f])]) for f in d['features']},'no_selection_on_output_Q3_for_this_supplement':True}
    widths={name:ew.section(d['features']['PSD'][d['valid']],d['features']['Qtotal_mVns'][d['valid']],edges) for name,d in domains.items()}
    write('ENERGY_WIDTH_METRICS.json',{'edges_logQ_frozen_development':edges,'domains':widths,'no_training_target_refit':True,'density_components_not_particle_truth':True})
    c2st=c2st_family(domains,cfg,write);rr=rr_q3(domains['REAL'],domains,ref,write)
    plots(domains,stats,dist,migration,widths,edges,ref,c2st)
    return {'domains':stats,'domain_vs_EVAL_REAL':dist,'charge_migration':migration,'fixed_nominal_Q3_parent_supplement':supplement,'narrow_widths':widths,'C2ST_q3':c2st,'REAL_REAL_q3':rr,'full_vs_gap_q3':{f:{'FULL_W1':dist['FULL']['q3'][f]['W1'],'GAP_W1':dist['GAP']['q3'][f]['W1']} for f in domains['FULL']['features']}}
def plots(domains,stats,dist,migration,widths,edges,ref,c2st):
    plt=ev.plt;colors=ev.COLORS
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,['FULL','GAP']):
        t=migration[name]['rows_source_nominal_q1_q4_columns_output_own_Q_q1_q4'];im=ax.imshow(t,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(t[i,j]),ha='center',va='center',color='white' if t[i,j]>t.max()/2 else 'black')
        ax.set_xticks(range(4),['q1','q2','q3','q4']);ax.set_yticks(range(4),['q1','q2','q3','q4']);ax.set_xlabel('Generated own signed Qt region');ax.set_ylabel('Source nominal aligned Qt region');ax.set_title(name+' all2000 parents');fig.colorbar(im,ax=ax)
    fig.tight_layout();ev.savefig(fig,'charge_migration')
    fig,axes=plt.subplots(2,3,figsize=(17,8))
    names=['R2','R3','R3B','FULL','GAP','REAL']
    for ax,name in zip(axes.ravel(),names):
        d=domains[name];m=d['valid'];ax.scatter(d['features']['Qtotal_mVns'][m],d['features']['PSD'][m],s=2,alpha=.22,color=colors[name]);ax.axvspan(cfg_gap_lo:=3703.7782948464155,6711.804904974998,color='#999999',alpha=.1);ax.set_xscale('log');ax.set_ylim(-.1,.6);ax.set_title(name+f" QC {m.sum()}, ownq3 {(m&(d['region']==2)).sum()}");ax.set_xlabel('Signed Qt [mV ns]');ax.set_ylabel('Signed PSD')
    fig.suptitle('All outputs retained; grey interval is the predefined current-fit gap');fig.tight_layout();ev.savefig(fig,'charge_psd')
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    features=['PSD','rise_10_90_ns','peak_V','Qtotal_mVns','baseline_RMS_V','tail_charge_fraction']
    for ax,feature in zip(axes.ravel(),features):
        arrays=[];labels=[]
        for name,d in domains.items():
            m=d['valid']&(d['region']==2)&np.isfinite(d['features'][feature]);arrays.append(d['features'][feature][m]);labels.append(name+f' n={m.sum()}')
        ax.boxplot(arrays,tick_labels=labels,showfliers=False);ax.tick_params(axis='x',labelrotation=35);ax.set_ylabel(feature);ax.set_title('Own-charge q3; box Q1/median/Q3')
    fig.suptitle('Complete q3 subsets; exact W1/KS and tails remain in metrics');fig.tight_layout();ev.savefig(fig,'q3_response_features')
    fig,axes=plt.subplots(1,3,figsize=(17,5));subedges=np.linspace(3703.7782948464155,6711.804904974998,4);t=np.arange(480)-120
    for i,ax in enumerate(axes):
        for name,d in domains.items():
            q=d['features']['Qtotal_mVns'];m=d['valid']&(q>=subedges[i])&(q<subedges[i+1]);x=d['shape'][m]
            if len(x):ax.plot(t,x.mean(axis=0),color=colors[name],label=name+f' n={len(x)}');lo,hi=np.quantile(x,[.1,.9],axis=0);ax.fill_between(t,lo,hi,color=colors[name],alpha=.07)
        ax.set_xlim(-25,220);ax.set_title(f'Q {subedges[i]:.0f}–{subedges[i+1]:.0f} mV ns');ax.set_xlabel('Saved aligned t [ns]');ax.set_ylabel('Own-peak normalized shape');ax.legend(fontsize=7)
    fig.suptitle('Means and per-sample 10–90% ranges, not confidence bands');fig.tight_layout();ev.savefig(fig,'conditional_waveforms')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for name,d in domains.items():
        m=d['valid']&(d['region']==2);v=d['V'][m,:100];v=v-v.mean(axis=1,keepdims=True);peak=d['features']['peak_V'][m]
        if not len(v):continue
        p=np.mean(np.abs(np.fft.rfft(v/peak[:,None],axis=1))**2,axis=0)/100;axes[0].plot(np.fft.rfftfreq(100,1)*1000,p,color=colors[name],label=name)
        den=(v*v).sum(axis=1);acf=np.column_stack([np.divide((v[:,:100-l]*v[:,l:]).sum(axis=1),den,out=np.full(len(v),np.nan),where=den>0) for l in range(1,21)]);axes[1].plot(np.arange(1,21),np.nanmedian(acf,axis=0),color=colors[name],label=name)
        rms=d['features']['baseline_RMS_V'][m]/peak;axes[2].hist(rms,bins=np.linspace(0,.08,65),histtype='step',density=True,color=colors[name],label=name)
    axes[0].set_xlabel('Baseline frequency [MHz]');axes[0].set_ylabel('Mean relative baseline periodogram');axes[0].set_yscale('log');axes[1].set_xlabel('Lag [ns]');axes[1].set_ylabel('Median baseline ACF');axes[2].set_xlabel('Baseline RMS / own peak');axes[2].set_ylabel('Density (display range)')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle('Own-q3 noise; complete finite-value distances/tail counts in metrics');fig.tight_layout();ev.savefig(fig,'fluctuations')
    centers=np.exp((edges[:-1]+edges[1:])/2);fig,axes=plt.subplots(2,3,figsize=(17,9))
    for component in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[component,j]
            for name in names:
                yy=[r['components'][component][field] if r['valid'] and len(r['components'])==2 else None for r in widths[name]['slices']];ax.plot(centers,[np.nan if v is None else v for v in yy],'o-',color=colors[name],label=name)
            ax.axvspan(3703.7782948464155,6711.804904974998,color='gray',alpha=.1);ax.set_xscale('log');ax.set_xlabel('Signed Qt [mV ns]');ax.set_ylabel(field);ax.set_title(f'Ordered density component{component}')
            if j==0:ax.legend(fontsize=7)
    fig.suptitle('Existing TRAIN-frozen evaluation bins; failure gaps remain; not particle truth');fig.tight_layout();ev.savefig(fig,'energy_peak_width')
    fig,axes=plt.subplots(2,4,figsize=(18,9));grid=np.linspace(-.1,.6,129);x=(grid[:-1]+grid[1:])/2
    for i,ax in enumerate(axes.ravel()):
        for name in names:
            s=widths[name]['slices'][i];ax.step(x,np.asarray(s['histogram_counts'])/max(1,s['N'])/(grid[1]-grid[0]),where='mid',color=colors[name],label=name+f' n={int(s["N"])}',lw=1)
        ax.set_xlabel('Signed PSD');ax.set_ylabel('Full-slice density');ax.set_title(f'Q {np.exp(edges[i]):.0f}–{np.exp(edges[i+1]):.0f}');ax.legend(fontsize=6)
    fig.tight_layout();ev.savefig(fig,'energy_psd_slices')
    fig,axes=plt.subplots(1,2,figsize=(13,5));features=['PSD','rise_10_90_ns','Qtotal_mVns','peak_V','baseline_RMS_V']
    for ax,field in zip(axes,['PSD','rise_10_90_ns']):
        for name in names[:-1]:ax.plot(range(1,5),[dist[name]['q'+str(r)][field]['W1'] for r in range(1,5)],'o-',color=colors[name],label=name)
        ax.set_xticks(range(1,5));ax.set_xlabel('Own-charge CAL region');ax.set_ylabel(field+' W1 vs same EVAL region');ax.legend(fontsize=8)
    fig.suptitle('q1/q2/q4 outside-gap controls; exact unequal sample counts disclosed');fig.tight_layout();ev.savefig(fig,'outside_gap_controls')
