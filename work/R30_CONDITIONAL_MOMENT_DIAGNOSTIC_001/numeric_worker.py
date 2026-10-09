from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='4'
from pathlib import Path
from datetime import datetime, timezone
import json, time, traceback, hashlib, platform
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from diagnostic_core import bootstrap
from paired_bootstrap import paired_bootstrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
START=time.monotonic()
NAMES=['REAL','TRAIN_REFERENCE',*CFG['candidates']]
COMPARE=[*CFG['candidates'],'TRAIN_REFERENCE']
VIEWS=['RAW',*CFG['views']]
EDGES=np.asarray(CFG['edges_logQ'],'f8')
COUNTER={'HGB_attempts':0,'HGB_completed':0,'response_training':0,'response_inference':0,'response_generation':0,'EM':0,'new_REAL':0,'EVAL':0,'CAL_waveform':0,'SSH':0,'new_G4':0}

def utc():return datetime.now(timezone.utc).isoformat()
def safe(v):
    if isinstance(v,dict):return {str(k):safe(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [safe(a) for a in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(name,v):
    (ROOT/name).write_text(json.dumps(safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def check_wall():
    if time.monotonic()-START>CFG['science_budget']-10:raise TimeoutError('science budget reserve')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(CFG['deadline_UTC']):raise TimeoutError('executor deadline')
def progress(stage):
    check_wall();write('NUMERIC_PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'stage':stage,'counters':COUNTER})
def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def bins(q):
    logq=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,logq,side='right')-1
    b[logq==EDGES[-1]]=7;b[~np.isfinite(q)|(q<=0)]=255
    return b.astype('i2')
def domain(v,groups,prefix,extra=None):
    v=np.asarray(v,dtype='f8');assert v.shape==(2000,480)
    x=v-v[:,:100].mean(1,keepdims=True);peak=x.max(1);q=x[:,100:400].sum(1)*1000
    psd=1-np.divide(x[:,100:150].sum(1)*1000,q,out=np.full(len(v),np.nan),where=q>0)
    valid=np.isfinite(v).all(1)&(peak>0)&(q>0)&np.isfinite(psd)
    if extra is not None:valid&=extra
    u=x/np.maximum(peak[:,None],CFG['normalization_peak_floor_V'])
    return {'u':u,'q':q,'bin':bins(q),'valid':valid,'groups':np.asarray(groups),
            'cluster':np.asarray([prefix+'|'+str(g) for g in groups])}
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def inputs():
    a=load(CFG['inputs']['real'],['real_V','real_groups','real_QC_valid','reference_V','reference_groups'])
    domains={'REAL':domain(a['real_V'],a['real_groups'],'REAL',a['real_QC_valid']),
             'TRAIN_REFERENCE':domain(a['reference_V'],a['reference_groups'],'TRAIN_REFERENCE')}
    ids=load(CFG['inputs']['source_ids'],['source_ids'])['source_ids']
    for name in CFG['candidates']:
        g=load(CFG['generated'][name],['waveforms_V','source_ids'])
        assert np.array_equal(g['source_ids'],ids),name+' source ID order'
        domains[name]=domain(g['waveforms_V'],np.arange(2000),'G4_SOURCE')
    rows=load(CFG['inputs']['classification_rows'],[f'phase{i}_{j}' for i in (0,1) for j in (0,1,2)])
    # Verify the actual saved endpoint split arrays, without fitting or reading its predictions twice.
    for old in ['R28_FRESH_PROJECTION_TRANSLATION_001','R29_SLICED_ENERGY_TRANSLATION_001']:
        prior=load(ROOT.parent/old/'results/C2ST_PRIVATE.npz',list(rows))
        for key in rows:assert np.array_equal(rows[key],prior[key]),old+' '+key+' changed'
    selections={};private={**rows,'source_ids':ids}
    for name,d in domains.items():
        j=1 if name=='REAL' else 2 if name=='TRAIN_REFERENCE' else 0
        tr,te=rows[f'phase0_{j}'].astype('i8'),rows[f'phase1_{j}'].astype('i8')
        assert len(tr)==600 and len(te)==250 and np.all(d['valid'][np.r_[tr,te]])
        assert len(np.unique(np.r_[tr,te]))==850 and not set(d['groups'][tr])&set(d['groups'][te]),name+' group isolation'
        selections[name]=(tr,te)
        private.update({name+'__Q_mVns':d['q'],name+'__original_Qbin':d['bin'],name+'__QC':d['valid'],name+'__groups':d['groups']})
    np.savez_compressed(OUT/'INPUT_ROWS_PRIVATE.npz',**private)
    del a
    receipt={'domains':{n:{'source_count':len(d['u']),'QC_count':int(d['valid'].sum()),'all_invalid':int((~d['valid']).sum()),
                         'train_n':600,'holdout_n':250,'group_overlap':0,'train_rows_sha256':array_hash(selections[n][0]),'holdout_rows_sha256':array_hash(selections[n][1]),
                         'train_Qbin_counts':[int((d['bin'][selections[n][0]]==b).sum()) for b in range(8)],
                         'holdout_Qbin_counts':[int((d['bin'][selections[n][1]]==b).sum()) for b in range(8)]} for n,d in domains.items()},
             'same_2000_source_order_all_four':True,'source_composition_preserved_by_reference':{'pure_EE':417,'pure_had':1518,'mixed':65},
             'observed_charge_not_new_energy_calibration':True,'DEV_historically_exposed':True,'new_waveform_read':False,'no_resplit':True,'old_phase_arrays_match_R28_R29':True}
    write('ACTUAL_INPUT_RECEIPT.json',receipt)
    return domains,selections,receipt

def moments(domains,selections):
    scale=load(CFG['inputs']['fine_constants'],['shape480__scale'])['shape480__scale'].astype('f8')
    floor=np.maximum(CFG['scale_floor_multiplier']*scale,CFG['scale_floor_absolute'])
    assert floor.shape==(480,) and np.isfinite(floor).all() and (floor>0).all()
    counts={n:np.array([(d['bin'][selections[n][0]]==b).sum() for b in range(8)]) for n,d in domains.items()}
    support=np.all(np.vstack(list(counts.values()))>=CFG['conditional_train_min_each_all_six'],axis=0)
    params={};saved={'floor':floor,'common_support_bins':support}
    summaries={}
    for name,d in domains.items():
        tr,te=selections[name];u=d['u'];gm=u[tr].mean(0);gs=u[tr].std(0,ddof=1)
        assert np.isfinite(gm).all() and np.isfinite(gs).all()
        bm=np.repeat(gm[None],8,axis=0);bs=np.repeat(gs[None],8,axis=0)
        for b in range(8):
            if support[b]:
                chosen=tr[d['bin'][tr]==b];bm[b]=u[chosen].mean(0);bs[b]=u[chosen].std(0,ddof=1)
        assert np.isfinite(bm).all() and np.isfinite(bs).all()
        par={'global_mean':gm,'global_SD':gs,'bin_mean':bm,'bin_SD':bs,'floor':floor,'support':support}
        params[name]=par
        for k,v in par.items():saved[name+'__'+k]=v
        d['views']={'RAW':u}
        effective_mean,effective_sd,fallback=assign(d['bin'],par)
        d['views']['GLOBAL_CENTER']=u-gm
        d['views']['Q_CENTER']=u-effective_mean
        d['views']['Q_CENTER_SCALE']=(u-effective_mean)/np.maximum(effective_sd,floor)
        check={'global_center_train_mean_maxabs':float(np.abs(d['views']['GLOBAL_CENTER'][tr].mean(0)).max()),'conditional_bins':[]}
        for b in np.flatnonzero(support):
            ix=tr[d['bin'][tr]==b];unfloored=bs[b]>=floor
            centered=d['views']['Q_CENTER'][ix].mean(0)
            sdcheck=d['views']['Q_CENTER_SCALE'][ix].std(0,ddof=1)
            check['conditional_bins'].append({'bin':int(b+1),'N':len(ix),'train_mean_maxabs':np.abs(centered).max(),
                                            'unfloored_columns':int(unfloored.sum()),'unfloored_SD1_maxabs':np.abs(sdcheck[unfloored]-1).max() if unfloored.any() else None})
        assert check['global_center_train_mean_maxabs']<1e-12
        assert all(c['train_mean_maxabs']<1e-12 and (c['unfloored_SD1_maxabs'] is None or c['unfloored_SD1_maxabs']<1e-10) for c in check['conditional_bins'])
        summaries[name]={'train_Qbin_counts':counts[name],'common_train_supported_bins_1based':np.flatnonzero(support)+1,
                         'global_floor_columns':int((gs<floor).sum()),'bin_floor_columns':[int((bs[b]<floor).sum()) for b in range(8)],
                         'maximum_amplification_global':float((1/np.maximum(gs,floor)).max()),
                         'maximum_amplification_supported_bins':float((1/np.maximum(bs[support],floor)).max()) if support.any() else None,
                         'fallback_train_n':int(fallback[tr].sum()),'fallback_holdout_n':int(fallback[te].sum()),'fallback_valid_population_n':int((fallback&d['valid']).sum()),
                         'fallback_reasons_train':fallback_reasons(d['bin'][tr],support),
                         'fallback_reasons_holdout':fallback_reasons(d['bin'][te],support),
                         'arithmetic_checks':check}
    np.savez_compressed(OUT/'MOMENT_PARAMETERS_PRIVATE.npz',**saved)
    result={'six_domains_common_supported_bins_1based':np.flatnonzero(support)+1,'support_requires_each_train_N':20,'domains':summaries,
            'floor_min':floor.min(),'floor_max':floor.max(),'sample_SD_ddof':1,'all_events_retained_for_fallback':True,'transformation_uses_domain_identity':True,'new_PSD_on_transformed':False}
    write('MOMENT_PARAMETER_RECEIPT.json',result)
    return params,result

def assign(bins_,par):
    m=np.repeat(par['global_mean'][None],len(bins_),axis=0);s=np.repeat(par['global_SD'][None],len(bins_),axis=0)
    legal=(bins_>=0)&(bins_<8);good=np.zeros(len(bins_),bool);good[legal]=par['support'][bins_[legal]]
    m[good]=par['bin_mean'][bins_[good]];s[good]=par['bin_SD'][bins_[good]]
    return m,s,~good
def fallback_reasons(b,support):
    legal=(b>=0)&(b<8);ok=np.zeros(len(b),bool);ok[legal]=support[b[legal]]
    return {'outside_or_undefined_Q':int((~legal).sum()),'common_train_bin_below20':int((legal&~ok).sum()),'supported':int(ok.sum())}

def raw_predictions(name,d,real,selections):
    p=load(CFG['raw_predictions'][name],['probability','label','cluster','training_probability','training_label'])
    tr,te=selections[name];rt,re=selections['REAL'];labels=np.r_[np.ones(len(te)),np.zeros(len(re))];clusters=np.r_[d['cluster'][te],real['cluster'][re]]
    if not np.array_equal(p['label'],labels) or not np.array_equal(p['cluster'],clusters):raise RuntimeError(name+' original raw labels/groups/row order mismatch')
    assert np.array_equal(p['training_label'],np.r_[np.ones(len(tr)),np.zeros(len(rt))])
    r=json.loads(Path(CFG['raw_results'][name]).read_text('utf8'))
    assert abs(balanced_accuracy_score(labels,p['probability']>=.5)-r['balanced_accuracy'])<1e-12
    r.update(reused_raw_classifier=True,source_result_path=CFG['raw_results'][name],source_prediction_path=CFG['raw_predictions'][name],paired_vs_raw=None)
    return r,p

def fit(name,view,d,real,selections):
    tr,te=selections[name];rt,re=selections['REAL']
    if min(len(tr),len(rt))<150 or min(len(te),len(re))<75:raise ValueError('fixed C2ST support failure')
    a=d['views'][view];b=real['views'][view];X=np.vstack([a[tr],b[rt]]);T=np.vstack([a[te],b[re]])
    y=np.r_[np.ones(len(tr)),np.zeros(len(rt))];yt=np.r_[np.ones(len(te)),np.zeros(len(re))]
    X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);X=imputer.fit_transform(X);T=imputer.transform(T)
    scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
    model=HistGradientBoostingClassifier(**CFG['C2ST']);COUNTER['HGB_attempts']+=1;progress(name+' '+view+' fitting')
    if COUNTER['HGB_attempts']>CFG['HGB_max_fits']:raise RuntimeError('HGB cap exhausted')
    clock=time.monotonic();model.fit(X,y);p=model.predict_proba(T)[:,1];pt=model.predict_proba(X)[:,1];COUNTER['HGB_completed']+=1
    private={'probability':p,'label':yt,'cluster':np.r_[d['cluster'][te],real['cluster'][re]],
             'training_probability':pt,'training_label':y,'candidate_train':tr,'candidate_holdout':te,'REAL_train':rt,'REAL_holdout':re,
             'imputer_statistics':imputer.statistics_,'standardizer_mean':scaler.mean_,'standardizer_scale':scaler.scale_}
    np.savez_compressed(OUT/(name+'__'+view+'_PREDICTIONS_PRIVATE.npz'),**private)
    result={'balanced_accuracy':balanced_accuracy_score(yt,p>=.5),'AUC':roc_auc_score(yt,p),'training_balanced_accuracy':balanced_accuracy_score(y,pt>=.5),'training_AUC':roc_auc_score(y,pt),
            'features':480,'train_each_domain':600,'holdout_each_domain':250,'heldout_CI':None,'classification_train_only_imputer_and_standardizer':True,
            'reused_raw_classifier':False,'fit_s_before_bootstrap':time.monotonic()-clock,'view':view,'domain_identity_required':True}
    write('results/C_'+name+'__'+view+'.json',result)
    result['heldout_CI']=bootstrap(yt,p,private['cluster'],CFG['bootstrap_repeats'],CFG['bootstrap_seed'])
    result['fit_and_bootstrap_s']=time.monotonic()-clock
    write('results/C_'+name+'__'+view+'.json',result)
    progress(name+' '+view+' persisted')
    return result,private

def regional_scores(pred,qbin):
    out=[];y=pred['label'];p=pred['probability']
    for b in range(8):
        take=qbin==b;n0=int((take&(y==0)).sum());n1=int((take&(y==1)).sum());ok=min(n0,n1)>=CFG['regional_holdout_min_each']
        out.append({'bin_id':b+1,'REAL_n':n0,'candidate_n':n1,'qualified':ok,'reason':None if ok else 'a_holdout_domain_below10',
                    'balanced_accuracy':float(balanced_accuracy_score(y[take],p[take]>=.5)) if ok else None})
    return out

def classifiers(domains,selections):
    results={};private_raw={};region={};common_bins=np.ones(8,bool)
    for name in COMPARE:
        d=domains[name];tr,te=selections[name];re=selections['REAL'][1]
        qbin=np.r_[d['bin'][te],domains['REAL']['bin'][re]]
        rr,raw=raw_predictions(name,d,domains['REAL'],selections);results[name]={'RAW':rr};private_raw[name]=raw;region[name]={'RAW':regional_scores(raw,qbin)}
        for v in CFG['views']:
            r,pr=fit(name,v,d,domains['REAL'],selections)
            if np.array_equal(pr['label'],raw['label']) and np.array_equal(pr['cluster'],raw['cluster']):r['paired_vs_raw']=paired_bootstrap(pr,raw,CFG)
            else:r['paired_vs_raw']={'defined':False,'reason':'different_holdout_rows_or_groups'}
            results[name][v]=r;region[name][v]=regional_scores(pr,qbin)
            write('results/C_'+name+'__'+v+'.json',r)
        for v in VIEWS:common_bins &= np.array([x['qualified'] for x in region[name][v]])
    for name in COMPARE:
        for v in VIEWS:
            vals=[x['balanced_accuracy'] for x in region[name][v] if common_bins[x['bin_id']-1]]
            results[name][v]['common_holdout_bins_equal_mean_BA']=float(np.mean(vals)) if vals else None
            results[name][v]['common_holdout_bins_1based']=np.flatnonzero(common_bins)+1
    out={'comparisons':results,'by_original_Qbin':region,'common_holdout_bins_1based':np.flatnonzero(common_bins)+1,
         'regional_min_each':10,'no_region_fits_or_region_bootstrap':True,'overall_all_holdout_events_kept':True,
         'fixed_classifier_CI_not_physical_equivalence':True,'HGB_attempts':COUNTER['HGB_attempts']}
    write('CLASSIFIER_METRICS.json',out)
    return out

def covariance(a):
    e=a-a.mean(0);c=e.T@e/(len(a)-1);trace=float(np.trace(c));f2=float(np.sum(c*c))
    return c,{'N':len(a),'dimension':480,'ddof':1,'trace':trace,'effective_rank':trace*trace/f2 if f2>0 else None,'Frobenius':np.sqrt(f2),'sample_high_dimensional_caveat':True}

def descriptive(domains,selections):
    curves={};out={};covs={}
    for n,d in domains.items():
        for subset in ['population','holdout']:
            ix=np.flatnonzero(d['valid']) if subset=='population' else selections[n][1]
            for view in VIEWS:
                a=d['views'][view][ix];prefix=n+'__'+subset+'__'+view
                curves[prefix+'__mean']=a.mean(0);curves[prefix+'__SD']=a.std(0,ddof=1)
                if subset=='holdout' and view in ['RAW','Q_CENTER','Q_CENTER_SCALE']:covs[n,view]=covariance(a)
    for n,d in domains.items():
        out[n]={}
        for subset in ['population','holdout']:
            out[n][subset]={}
            for view in VIEWS:
                prefix=n+'__'+subset+'__'+view;mean=curves[prefix+'__mean'];sd=curves[prefix+'__SD'];ref=curves['REAL__'+subset+'__'+view+'__mean']
                sub={'N':int(d['valid'].sum()) if subset=='population' else 250,'regions':{}}
                for region,(left,right) in CFG['regions'].items():
                    delta=mean[left:right]-ref[left:right]
                    sub['regions'][region]={'mean_residual_RMS_u':np.sqrt(np.mean(delta**2)),'mean_residual_mean_u':delta.mean(),
                                            'pointwise_variance_mean_u2':np.mean(sd[left:right]**2),'pointwise_SD_mean_u':sd[left:right].mean()}
                out[n][subset][view]=sub
        # Conditional curves describe the unchanged original observed-Q assignment; no new classifiers.
        for b in range(8):
            ix=selections[n][1];ix=ix[d['bin'][ix]==b]
            if len(ix)>=10:
                for view in VIEWS:
                    prefix=n+'__holdout_bin'+str(b+1)+'__'+view;a=d['views'][view][ix]
                    curves[prefix+'__mean']=a.mean(0);curves[prefix+'__SD']=a.std(0,ddof=1)
    covariance_out={}
    for name in NAMES:
        covariance_out[name]={}
        for v in ['RAW','Q_CENTER','Q_CENTER_SCALE']:
            c,r=covs[name,v];realc,realr=covs['REAL',v];den=realr['Frobenius']
            r['relative_Frobenius_difference_to_REAL']=float(np.linalg.norm(c-realc)/den) if den>0 else None
            r['region_variance_trace']={k:float(np.trace(c[a:b,a:b])) for k,(a,b) in CFG['regions'].items()}
            covariance_out[name][v]=r
    np.savez_compressed(OUT/'FIGURE_DATA_PRIVATE.npz',**curves)
    write('DESCRIPTIVE_MOMENTS.json',out);write('HOLDOUT_COVARIANCE.json',covariance_out)
    return out,covariance_out

def run():
    write('NUMERIC_STARTED.json',{'PID':os.getpid(),'UTC':utc(),'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'scientific_scope':'saved waveforms and classifier moment transforms; no response updates or generation'})
    progress('loading saved domains and fixed classifier rows')
    domains,selections,receipt=inputs()
    params,parameter_receipt=moments(domains,selections);progress('train-only moments persisted')
    c=classifiers(domains,selections);d,cov=descriptive(domains,selections)
    result={'input_receipt':receipt,'moment_parameters':parameter_receipt,'classification':c,'descriptive_moments':d,'holdout_covariance':cov,
            'interpretation_boundaries':{'diagnostic_domain_identity_transform':True,'deployable':False,'new_transformed_PSD':False,'remaining_separability_unique_physical_cause':False,'new_blind_test':False,'same_acquisition_groups_not_independent_runs':True}}
    write('metrics.json',result)
    write('METRICS_COMPACT.json',result)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'HGB_hyperparameters':CFG['C2ST'],'all_inputs_saved_immutable':True,'original_waveform_new_decodes':0,'response_generation':0,'whole_transformed_populations_saved':False,'exit_status':'success'})
    progress('completed')
    print(json.dumps({'completed':True,'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'counters':COUNTER}),flush=True)

try:run()
except BaseException as e:
    write('NUMERIC_FAILURE.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'error':repr(e),'traceback':traceback.format_exc(),'failed_cost_retained':True})
    raise
