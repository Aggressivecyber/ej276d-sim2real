from __future__ import annotations
import os,pathlib,sys,time,json,datetime,hashlib,io,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
OUT=ROOT/'results';FIG=OUT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'joblib-cache')
START=time.monotonic()
import numpy as np
import evaluation as ev
import energy_width as ew
from common_math import distances,fom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
CFG=json.loads((ROOT/'config.json').read_text());freeze=json.loads((BASE/'FROZEN_METHOD.json').read_text());INPUTS=[]
ev.OUT=OUT;ev.FIG=FIG;ev.CFG=CFG;ev.NAMES=['SIM','GEN_W0','R2','R3','R3B','REAL']
ev.COLORS={'SIM':'#2878b5','GEN_W0':'#a33ea2','R2':'#bb9b48','R3':'#bda0cb','R3B':'#e65c24','REAL':'#181818'}
def write(n,x):ev.write_json(n,x)
def load(p,expected=None):
    p=pathlib.Path(p);b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
    if expected and sha!=expected:raise RuntimeError('frozen input SHA '+str(p))
    INPUTS.append({'path':str(p),'bytes':len(b),'sha256':sha})
    return np.load(io.BytesIO(b),allow_pickle=False)
def full_order(ids,prefix):return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256((prefix+'|'+str(ids[i])).encode()).digest()),dtype='i8')
def take(d,idx):
    return {k:({f:v[idx] for f,v in a.items()} if k=='features' else a[idx] if isinstance(a,np.ndarray) and len(a)==len(d['V']) else a) for k,a in d.items()}
def assign(d,ids,groups,kind,ref):
    d['ids']=np.asarray(ids);d['groups']=np.asarray(groups);d['kind']=kind
    d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1')
    return d
def standardized_joint(d,scaling,m=None):
    if m is None:m=d['valid']
    q=d['features']['Qtotal_mVns'][m];p=d['features']['PSD'][m]
    return (np.column_stack([np.log(q),p])-scaling['mean'])/scaling['population_sd']
def sliced(a,b,scaling):
    x=standardized_joint(a,scaling);y=standardized_joint(b,scaling)
    if not len(x) or not len(y):return {'W1':None,'counts':[len(x),len(y)]}
    theta=np.arange(128)*np.pi/128;dir=np.column_stack([np.cos(theta),np.sin(theta)])
    pa=x@dir.T;pb=y@dir.T;values=[distances(pa[:,i],pb[:,i])['W1'] for i in range(128)]
    return {'W1':float(np.mean(values)),'direction_W1':values,'counts':[len(x),len(y)],'angles':128,'standardization':'frozen TRAIN-fit mean/population SD'}
def rr_reference(allreal,ref,scaling):
    uniq=np.unique(allreal['groups']);records=[];saved={};first=None
    for repeat in range(64):
        groups=sorted(uniq.tolist(),key=lambda g:hashlib.sha256(f'R4-RR-group-v1|{repeat}|{g}'.encode()).digest())
        halves=[groups[:76],groups[76:]];selected=[]
        for half in halves:
            ix=np.flatnonzero(allreal['valid']&np.isin(allreal['groups'],half))
            if len(ix)<2000:raise RuntimeError('RR half cannot supply2000 QC events')
            ix=ix[full_order(allreal['ids'][ix],f'R4-RR-event-v1|{repeat}')[:2000]];selected.append(ix)
        a,b=[take(allreal,ix) for ix in selected]
        if set(a['groups'])&set(b['groups']):raise RuntimeError('RR group overlap')
        row={'repeat':repeat,'each_N':2000,'groups_available':[len(h) for h in halves],
             'groups_selected':[len(np.unique(d['groups'])) for d in [a,b]],'regions':{},'joint_sliced_W1':sliced(a,b,scaling)}
        for r in [None,0,1,2,3]:
            key='overall' if r is None else 'q'+str(r+1);ma=ev.mask(a,r);mb=ev.mask(b,r)
            cut=ref['slots'][0 if r is None else r+1]['cut'] if ref['slots'][0 if r is None else r+1]['valid'] else None
            row['regions'][key]={'counts':[int(ma.sum()),int(mb.sum())],
                                'features':{k:distances(a['features'][k][ma&np.isfinite(a['features'][k])],b['features'][k][mb&np.isfinite(b['features'][k])]) for k in a['features']},
                                'FoM':[fom(d['features']['PSD'][m],cut) for d,m in [(a,ma),(b,mb)]]}
        records.append(row);saved['repeat'+str(repeat)]=np.stack(selected)
        if first is None:first=(a,b)
    summary={}
    for region in ['overall','q1','q2','q3','q4']:
        summary[region]={}
        for feature in allreal['features']:
            summary[region][feature]={}
            for metric in ['W1','KS']:
                v=[r['regions'][region]['features'][feature][metric] for r in records]
                v=np.asarray([x for x in v if x is not None],dtype='f8')
                summary[region][feature][metric]={'defined_repeats':len(v),'median':float(np.median(v)) if len(v) else None,'range5_95':np.quantile(v,[.05,.95]).tolist() if len(v) else None}
    joint=np.array([r['joint_sliced_W1']['W1'] for r in records])
    np.savez_compressed(OUT/'REAL_REAL_SELECTIONS.npz',**saved)
    result={'definition':'64 disjoint whole-group half splits;2000 QC events each half;single acquisition descriptive sampling reference',
            'not_physical_equivalence_or_independent_acquisition_CI':True,'each_N':2000,'records':records,
            'summary':summary,'joint_sliced_W1':{'median':float(np.median(joint)),'range5_95':np.quantile(joint,[.05,.95]).tolist()}}
    write('REAL_REAL_REFERENCE.json',result);return result,first
def c2st(a,b,name):
    selected=[];nt=600;nv=250
    for d in [a,b]:
        keys=d['groups'] if d['kind']=='REAL' else d['ids']
        flag=np.array([ev.hashint('R1-SPLIT|'+d['kind']+'|'+str(k))%10<7 for k in keys])
        tr=np.flatnonzero(d['valid']&flag);te=np.flatnonzero(d['valid']&~flag)
        if len(tr)<nt or len(te)<nv:raise RuntimeError('frozen C2ST caps not supported '+name)
        tr=tr[ev.order(d['ids'][tr],'R1-C2ST-SAMPLE')[:nt]];te=te[ev.order(d['ids'][te],'R1-C2ST-SAMPLE')[:nv]]
        selected.append((tr,te))
    (at,av),(bt,bv)=selected
    train_groups=set((a['kind'],str(g)) for g in a['groups'][at])|set((b['kind'],str(g)) for g in b['groups'][bt])
    test_groups=set((a['kind'],str(g)) for g in a['groups'][av])|set((b['kind'],str(g)) for g in b['groups'][bv])
    if train_groups&test_groups:raise RuntimeError('source group C2ST leakage')
    write('C2ST_SPLITS_'+name+'.json',{'first_domain_train_ids':a['ids'][at],'first_domain_test_ids':a['ids'][av],
          'second_domain_train_ids':b['ids'][bt],'second_domain_test_ids':b['ids'][bv],'source_group_intersection':0,
          'train_each':600,'holdout_each':250,'fixed_R1_hash_split_order':True})
    params={k:CFG['C2ST'][k] for k in ['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','early_stopping','random_state']}
    result={};pred={}
    for view in CFG['C2ST']['views']:
        x,scalar=ev.c2st_matrix(a,view);y,_=ev.c2st_matrix(b,view)
        train=np.vstack([x[at],y[bt]]);test=np.vstack([x[av],y[bv]]);labels=np.r_[np.ones(nt),np.zeros(nt)];held=np.r_[np.ones(nv),np.zeros(nv)]
        imputer=SimpleImputer(strategy='median',keep_empty_features=True);train=imputer.fit_transform(train);test=imputer.transform(test)
        model=HistGradientBoostingClassifier(**params);start=time.monotonic();model.fit(train,labels);p=model.predict_proba(test)[:,1]
        cluster=np.r_[np.array([a['kind']+'|'+str(g) for g in a['groups'][av]]),np.array([b['kind']+'|'+str(g) for g in b['groups'][bv]])]
        acc=balanced_accuracy_score(held,p>=.5)
        blocks={'scalar_PSD_rise_tail':list(range(4)),'relative_RMS_ACF':list(range(4,13)),
                'wave_baseline_0_100':list(range(scalar,scalar+100)),'wave_prompt_100_150':list(range(scalar+100,scalar+150)),
                'wave_tail_150_400':list(range(scalar+150,scalar+400)),'wave_end_400_480':list(range(scalar+400,scalar+480))}
        if scalar>13:blocks['absolute_charge_peak_RMS']=list(range(13,scalar))
        rng=np.random.default_rng(20261008);importance={}
        for key,cols in blocks.items():
            u=test.copy();sh=rng.permutation(len(u));u[:,cols]=u[np.ix_(sh,cols)]
            importance[key]=float(acc-balanced_accuracy_score(held,model.predict(u)))
        result[view]={'balanced_accuracy':float(acc),'AUC':float(roc_auc_score(held,p)),
                      'train_each_domain':nt,'holdout_each_domain':nv,'n_features':test.shape[1],
                      'heldout_CI':ev.bootstrap(held,p,cluster,20261008),'heldout_permutation_block_accuracy_drop':importance,
                      'response_training':0,'evaluator_fit_s':time.monotonic()-start,'hyperparameters':params,'tuning':False}
        pred[view+'__probability']=p;pred[view+'__label']=held;pred[view+'__cluster']=cluster
        ev.log('R4 C2ST '+name+' '+view+' BA='+str(acc))
    np.savez_compressed(OUT/('C2ST_PREDICTIONS_'+name+'.npz'),**pred);return result
def width_plots(widths,boots,edges):
    plt=ev.plt;centers=np.exp((edges[:-1]+edges[1:])/2)
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for component in [0,1]:
        for j,field in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[component,j]
            for name in ev.NAMES:
                vals=[r['components'][component][field] if len(r['components'])==2 and r['valid'] else None for r in widths[name]['slices']]
                ax.plot(centers,[np.nan if v is None else v for v in vals],'o-',ms=4,color=ev.COLORS[name],label=name)
            bands=[r for r in boots['bands'] if r['component']==component and r['metric']==field]
            ax.fill_between(centers,[np.nan if r['CI95'] is None else r['CI95'][0] for r in bands],
                            [np.nan if r['CI95'] is None else r['CI95'][1] for r in bands],color='black',alpha=.12,label='EVAL MAIN group-bootstrap95%')
            ax.set_xscale('log');ax.set_xlabel('Signed total charge [mV ns]');ax.set_ylabel(field);ax.set_title(f'Ordered density component{component}: {field}')
            if j==0:ax.legend(fontsize=7)
    fig.suptitle('TRAIN-frozen narrow intervals; EVAL descriptive density fits; gaps preserve failures, no particle truth')
    fig.tight_layout();ev.savefig(fig,'energy_peak_width')
    fig,axes=plt.subplots(2,4,figsize=(18,9));grid=np.linspace(-.1,.6,129);x=(grid[:-1]+grid[1:])/2
    for i,ax in enumerate(axes.ravel()):
        for name in ev.NAMES:
            s=widths[name]['slices'][i];h=np.array(s['histogram_counts'])/max(1,s['N'])/(grid[1]-grid[0])
            ax.step(x,h,where='mid',lw=1,color=ev.COLORS[name],label=f'{name} n={int(s["N"])}')
        ax.set_title(f'Bin{i+1}: Q={np.exp(edges[i]):.0f}–{np.exp(edges[i+1]):.0f} mV ns');ax.set_xlabel('Signed PSD');ax.set_ylabel('Full-slice density');ax.legend(fontsize=6)
    fig.suptitle('EVAL MAIN2000 vs all five frozen source outputs; full unknown/overlap retained')
    fig.tight_layout();ev.savefig(fig,'energy_psd_slices')
def main():
    ev.log(f'R4 numerical worker PID={os.getpid()} numpy={np.__version__}; response fit0')
    ready=json.loads((ROOT/'READINESS.json').read_text())
    for s in ready['execution_sources']:
        b=pathlib.Path(s['path']).read_bytes()
        if hashlib.sha256(b).hexdigest()!=s['sha256']:raise RuntimeError('ready source drift')
    identities=json.loads((ROOT/'EVAL_IDENTITIES.json').read_text());ext=json.loads((ROOT/'EXTRACTION_RECEIPT.json').read_text())
    if ext['SSH_exit']!=0 or ext['errors'] or ext['reason']:raise RuntimeError('no confirmed extraction')
    with load(ROOT/'EVAL_EXTRACT.npz',ext['stdout_SHA256']) as z:
        allreal=ev.features(z['window_V_f32']);allreal['valid']&=z['QC_reason_u16']==0
        orig=z['original_index'].copy();qc=z['QC_reason_u16'].copy();assert np.array_equal(orig,identities['original_index'])
    ids=np.array(identities['source_id']);groups=orig//50
    k=BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001';ref=json.loads((k/'A2_ENGINEERING_001/run_001/STATISTICS.json').read_text())
    allreal=assign(allreal,ids,groups,'REAL',ref)
    ix=np.flatnonzero(allreal['valid']);main_ix=ix[full_order(ids[ix],'R4-main-real-v1')[:2000]]
    if len(main_ix)!=2000:raise RuntimeError('EVAL cannot supply2000 QC')
    real=take(allreal,main_ix)
    bindings={str(BASE/s['path']) if not pathlib.Path(s['path']).is_absolute() else s['path']:s['sha256'] for s in freeze['model_and_generation_bindings']+freeze['historical_controls']}
    meta_path=pathlib.Path(CFG['input_npz']['SIM_METADATA'])
    with load(meta_path,bindings.get(str(meta_path))) as z:meta={n:z[n] for n in ['raw','sim_event_ids','normalization_peak_V','source_label','sim_energy','edep_ee_MeV','edep_hadron_MeV','physical_r_h']}
    simids=meta['sim_event_ids'];domains={'SIM':ev.features(meta['raw'].astype(float)*meta['normalization_peak_V'][:,None])}
    p=pathlib.Path(CFG['input_npz']['GEN_W0'])
    with load(p,bindings.get(str(p))) as z:
        assert np.array_equal(z['sim_event_ids'],simids) and np.array_equal(z['before'],meta['raw'])
        domains['GEN_W0']=ev.features(z['after'].astype(float)*meta['normalization_peak_V'][:,None])
    for name,stage in [('R2','R2_SMOOTH_STOCHASTIC_RESPONSE_001'),('R3','R3_FLEXIBLE_TEMPORAL_RESPONSE_001'),('R3B','R3B_ENERGY_CONDITIONAL_WIDTH_001')]:
        p=BASE/'work'/stage/'generated.npz'
        with load(p,bindings[str(p)]) as z:
            assert np.array_equal(z['sim_event_ids'],simids);domains[name]=ev.features(z['waveforms_V'])
    for name,d in domains.items():assign(d,simids,np.arange(2000),'SIM',ref)
    domains['REAL']=real
    # Only already-authorized TRAIN fit events determine the new two-dimensional scale.
    path=k/'A2_ENGINEERING_001/run_001/window_V_f32.bin'
    with path.open('rb') as f:trainbytes=f.read(5200*480*4)
    fulltrain=np.frombuffer(trainbytes,dtype='<f4').reshape(5200,480)[2000:]
    qcpath=k/'A2_ENGINEERING_001/run_001/QC_reason_u16.bin'
    with qcpath.open('rb') as f:qcreason=np.frombuffer(f.read(5200*2),dtype='<u2')[2000:]
    with (k/'selection_group.bin').open('rb') as f:tg=np.frombuffer(f.read(5200*2),dtype='<u2')[2000:]
    valgroups=json.loads((BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/DATA_ROLES.json').read_text())['REAL_internal_validation_artificial_groups']
    fit=(qcreason==0)&~np.isin(tg,valgroups);assert fit.sum()==2592
    tf=ev.features(fulltrain[fit]);v=np.column_stack([np.log(tf['features']['Qtotal_mVns']),tf['features']['PSD']])
    scaling={'mean':v.mean(axis=0),'population_sd':v.std(axis=0,ddof=0),'n':2592,'role':'TRAIN_internal_fit','features':['logQ','PSD'],'EVAL_for_scale':0}
    assert np.all(scaling['population_sd']>0)
    write('STANDARDIZATION.json',scaling)
    stats,comparison=ev.characterize(domains,ref);allstats,_=ev.characterize({'REAL':allreal},ref)
    joint={name:sliced(d,real,scaling) for name,d in domains.items() if name!='REAL'}
    repeat,rrfirst=rr_reference(allreal,ref,scaling)
    ev.log('REAL-REAL64 equal2000 reference complete')
    classifiers={name+'_vs_REAL':c2st(d,real,name+'_vs_REAL') for name,d in domains.items() if name!='REAL'}
    classifiers['REAL_vs_REAL']=c2st(rrfirst[0],rrfirst[1],'REAL_vs_REAL')
    rng=np.random.default_rng(202610081);t=np.arange(480);tail=np.where(t>=120,np.exp(-(t-120)/80),0.)
    positive=ev.features(real['V']+.02*real['features']['peak_V'][:,None]*tail[None,:]+.015*real['features']['peak_V'][:,None]*rng.normal(size=real['V'].shape))
    assign(positive,real['ids'],real['groups'],'REAL',ref)
    classifiers['PERTURBED_REAL_vs_REAL']=c2st(positive,real,'PERTURBED_REAL_vs_REAL')
    definition=json.loads((BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/DIAGNOSTIC_DEFINITION.json').read_text());edges=np.array(definition['edges_logQ'])
    widths={name:ew.section(d['features']['PSD'][d['valid']],d['features']['Qtotal_mVns'][d['valid']],edges) for name,d in domains.items()}
    fullwidth=ew.section(allreal['features']['PSD'][allreal['valid']],allreal['features']['Qtotal_mVns'][allreal['valid']],edges)
    boots=ew.grouped_bootstrap(real['features']['PSD'][real['valid']],real['features']['Qtotal_mVns'][real['valid']],real['groups'][real['valid']],edges)
    write('ENERGY_WIDTH_METRICS.json',{'definition':definition,'main':widths,'REAL_EVAL_all':fullwidth,'REAL_MAIN_group_bootstrap':boots,'particle_truth':False})
    write('DATA_ROLES.json',{'EVAL_original_records':7600,'EVAL_artificial_groups':152,'QC_valid':int(allreal['valid'].sum()),
                          'QC_reason_bit_counts':{str(bit):int(np.sum((qc&(1<<bit))!=0)) for bit in range(9)},
                          'main_original_indices':orig[main_ix],'main_source_ids':ids[main_ix],'main_actual_groups':len(np.unique(groups[main_ix])),
                          'main_selection':'full SHA R4-main-real-v1|source_id first2000 QC','response_fit_EVAL':0,
                          'R5_R6_prespecified_before_read':True,'same_single_acquisition':True,'historical_event_exposure':'unknown',
                          'G4_parents2000_preserved':True,'legacy_source_label_counts':{str(s):int(np.sum(meta['source_label']==s)) for s in np.unique(meta['source_label'])},
                          'legacy_gamma_non2112_not_PDG22_truth':True})
    np.savez_compressed(OUT/'EVAL_MAIN_SELECTION.npz',selected_positions=main_ix,original_index=orig[main_ix],source_id=ids[main_ix])
    metric={'status':'complete','EVAL_records':7600,'EVAL_QC':int(allreal['valid'].sum()),'main_REAL_N':2000,
            'domains':stats,'domain_vs_REAL':comparison,'REAL_all_supplement':allstats['REAL'],'joint_sliced_W1':joint,
            'REAL_REAL_summary':repeat['summary'],'REAL_REAL_joint':repeat['joint_sliced_W1'],'REAL_REAL_each_N':2000,
            'C2ST':classifiers,'energy_width':'ENERGY_WIDTH_METRICS.json','CAL_reference_valid':[s['valid'] for s in ref['slots']],
            'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'response_new_fit':0,'model_selection_on_EVAL':False,
            'FROZEN_METHOD_SHA256':ready['FROZEN_METHOD_SHA256'],'single_acquisition_retrospective_holdout':True,
            'worker_PID':os.getpid(),'numeric_before_plots_s':time.monotonic()-START,'versions':{'numpy':np.__version__,'python':sys.version}}
    write('metrics.json',metric);write('INPUTS.json',INPUTS)
    ev.plots(domains,ref,stats,comparison,classifiers,meta);width_plots(widths,boots,edges)
    fig,axes=ev.plt.subplots(1,2,figsize=(13,5))
    for name,d in domains.items():
        for j,key in enumerate(['PSD','Qtotal_mVns']):
            x=np.sort(d['features'][key][d['valid']]);axes[j].plot(x,np.arange(1,len(x)+1)/len(x),label=name,color=ev.COLORS[name])
    for j,key in enumerate(['PSD','Qtotal_mVns']):
        x=np.sort(allreal['features'][key][allreal['valid']]);axes[j].plot(x,np.arange(1,len(x)+1)/len(x),'k--',label=f'All EVAL QC n={len(x)}');axes[j].set_xlabel(key);axes[j].set_ylabel('Full distribution CDF');axes[j].legend(fontsize=8)
    axes[1].set_xscale('log');fig.tight_layout();ev.savefig(fig,'all_eval_supplement')
    metric['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metric)
    (ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8')
    ev.log('R4 complete '+str(metric['numeric_elapsed_s'])+'s')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('FAILURE.json',{'worker_PID':os.getpid(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
