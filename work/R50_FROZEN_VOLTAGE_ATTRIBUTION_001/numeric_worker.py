from pathlib import Path
from datetime import datetime, timezone
import os, json, time, sys, platform, traceback
import numpy as np
import scipy
import sklearn
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
START=time.monotonic();FITS=0
def utc():return datetime.now(timezone.utc).isoformat()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(name,x):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n','utf8')
def summary(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return {'N':len(v),'mean':float(v.mean()) if len(v) else None,
            'quantile_levels':[0,.05,.5,.95,1],'quantiles':np.quantile(v,[0,.05,.5,.95,1]) if len(v) else []}
def legacy_adc(v):
    v=np.asarray(v,dtype=np.float32)
    return (-.5+np.rint((np.clip(v,-.5,.5)+.5)*16383)/16383).astype(np.float32)
def codes(v):return np.clip(np.rint((np.asarray(v,dtype='f8')+.5)*16383),0,16383).astype('i4')
def canonical(v):return -.5+codes(v).astype('f8')/16383.
def scalars(v):
    v=np.asarray(v,dtype='f8');b=v[:,:100].mean(1);s=v-b[:,None]
    qt=s[:,100:400].sum(1)*1000.;qf=s[:,100:150].sum(1)*1000.
    ps=1.-np.divide(qf,qt,out=np.full(len(v),np.nan),where=qt>0)
    return {'baseline_mean':b,'baseline_RMS':s[:,:100].std(1,ddof=1),
            'Qtotal_mVns':qt,'PSD':ps,'peak_V':s.max(1)}
def view(v,kind):
    v=np.asarray(v,dtype='f8')
    if kind=='native_voltage':return v.copy()
    if kind=='canonical_ADC_voltage':return canonical(v)
    f=scalars(v)
    if kind=='baseline_subtracted_voltage':return v-f['baseline_mean'][:,None]
    if kind in ['baseline_mean','baseline_RMS']:return f[kind][:,None]
    if kind=='pulse_Q_PSD_peak':return np.column_stack([f['Qtotal_mVns'],f['PSD'],f['peak_V']])
    raise ValueError(kind)

def run():
    global FITS
    with np.load(ROOT/'INPUT_PRIVATE.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    ref=data['reference_V'];real=data['real_V']
    domains={'RAW_REFERENCE':ref,'RAW_LEGACY_ADC':legacy_adc(ref),**{k:data[k] for k in ['FIXED_PRE','FIXED_POST','COMPLETION_PRE','COMPLETION_POST']}}
    assert all(v.shape==(2000,480) and np.isfinite(v).all() for v in [real,*domains.values()])
    tr,te=data['phase0_2'],data['phase1_2'];rt,re=data['phase0_1'],data['phase1_1']
    assert (len(tr),len(te),len(rt),len(re))==(600,250,600,250)
    assert not set(data['reference_groups'][tr]) & set(data['reference_groups'][te])
    assert not set(data['real_groups'][rt]) & set(data['real_groups'][re])
    assert not set(data['reference_groups']) & set(data['real_groups'])
    original_adc={k:bool(np.array_equal(legacy_adc(data[k+'_PRE']),data[k+'_POST'])) for k in ['FIXED','COMPLETION']}
    assert all(original_adc.values()),('saved_R49_ADC_recomputation_conflict',original_adc)
    labels=np.r_[np.ones(len(te)),np.zeros(len(re))]
    groups=np.r_[['R|'+str(g) for g in data['reference_groups'][te]],['R|'+str(g) for g in data['real_groups'][re]]]
    _,inverse=np.unique(groups,return_inverse=True);n=int(inverse.max()+1)
    rng=np.random.default_rng(CFG['bootstrap_seed'])
    weights=np.stack([np.bincount(rng.integers(n,size=n),minlength=n)[inverse] for _ in range(CFG['bootstrap_repeats'])])
    denom0=weights[:,labels==0].sum(1);denom1=weights[:,labels==1].sum(1)
    assert np.all(denom0>0) and np.all(denom1>0)
    def boot_ba(p):return .5*((weights[:,(labels==0)&(p<.5)].sum(1)/denom0)+(weights[:,(labels==1)&(p>=.5)].sum(1)/denom1))

    descriptions={};paired_change={};identity={}
    target=scalars(real);rawf=scalars(ref)
    for name,v in {**domains,'RAW_DEV':real}.items():
        f=scalars(v);rounded=canonical(v);d=np.asarray(v,dtype='f8')-rounded
        descriptions[name]={'rows':len(v),'fields':{k:summary(a) for k,a in f.items()},
            'vs_RAW_DEV_W1':{k:float(wasserstein_distance(a[np.isfinite(a)],target[k][np.isfinite(target[k])])) for k,a in f.items()},
            'ADC_lattice_residual_RMS_LSB':summary(np.sqrt(np.mean(d*d,1))*16383),
            'outside_ADC_range_samples':int(np.count_nonzero((v<-.5)|(v>.5)))}
        if name!='RAW_DEV':
            delta=v.astype('f8')-ref.astype('f8')
            paired_change[name]={'RMS_vs_same_RAW_V':summary(np.sqrt(np.mean(delta*delta,1))),
                'maxabs_vs_RAW_LSB':float(np.max(np.abs(delta))*16383),
                'code_change_sample_count':int(np.count_nonzero(codes(v)!=codes(ref))),
                'code_change_row_count':int(np.count_nonzero(np.any(codes(v)!=codes(ref),axis=1))),
                'fields_change_vs_same_RAW':{k:summary(a-rawf[k]) for k,a in f.items()}}
    for name in ['RAW_LEGACY_ADC','FIXED_POST','COMPLETION_POST']:
        before=ref if name=='RAW_LEGACY_ADC' else domains[name.replace('_POST','_PRE')]
        v=domains[name]
        identity[name]={'same_ADC_codes_before_after':bool(np.array_equal(codes(before),codes(v))),
            'changed_ADC_codes_before_after':int(np.count_nonzero(codes(before)!=codes(v))),
            'canonical_feature_exact_equal':bool(np.array_equal(canonical(before),canonical(v)))}
    write('DESCRIPTIVE_METRICS.json',{'all_2000_rows':True,'descriptions':descriptions,'paired_changes':paired_change,
        'ADC_identity':identity,'saved_R49_ADC_recomputation_bitwise_equal':original_adc})
    write('SELECTION_BINDING.json',{'reference_REAL_group_disjoint':True,'train_holdout_group_disjoint':True,
        'original_fixed_R26_phase0_2_phase1_2_vs_phase0_1_phase1_1':True,'train_each':600,'holdout_each':250,
        'whole_population_per_domain':2000,'EVAL_read':0,'CAL_decode':0,'reQC_or_resplit':False,
        'response_training':0,'new_generation':0,'environment':{'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'CPU_threads':CFG['CPU_threads']}})

    plan=[(name,kind) for kind in CFG['core_views'] for name in CFG['cases']]
    plan += [(name,kind) for kind in CFG['decomposition_views'] for name in CFG['decomposition_cases']]
    assert len(plan)==CFG['planned_HGB_fits']==24
    results={};predictions={};boots={}
    for name,kind in plan:
        slot=name+'__'+kind;clock=time.monotonic();a=view(domains[name],kind);b=view(real,kind)
        X=np.vstack([a[tr],b[rt]]);T=np.vstack([a[te],b[re]])
        y=np.r_[np.ones(len(tr)),np.zeros(len(rt))]
        X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        X=imputer.fit_transform(X);T=imputer.transform(T)
        scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
        estimator=HistGradientBoostingClassifier(**CFG['C2ST']);FITS+=1
        estimator.fit(X,y);p=estimator.predict_proba(T)[:,1]
        bs=boot_ba(p);ci=np.quantile(bs,[.025,.975])
        results[slot]={'case':name,'view':kind,'BA':float(balanced_accuracy_score(labels,p>=.5)),
            'BA_95CI':ci,'AUC':float(roc_auc_score(labels,p)),'features':X.shape[1],
            'diagnostic_CI_inside_original_engineering_band':bool(ci[0]>=.4 and ci[1]<=.6),
            'elapsed_s':time.monotonic()-clock,'original_A_status_changed':False}
        predictions[slot]=p;boots[slot]=bs
        write('RESULTS_RUNNING.json',{'completed_fits':FITS,'planned_fits':24,'results':results})
        write('PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'completed_fits':FITS,'planned_fits':24,'last_slot':slot,'elapsed_s':time.monotonic()-START})
        print(json.dumps({'fit':FITS,'slot':slot,'BA':results[slot]['BA'],'BA_95CI':ci.tolist()}),flush=True)
    assert FITS==24
    pairs=[]
    def add_pair(first,second,purpose):
        pairs.append({'first':first,'second':second,'purpose':purpose,
            'BA_first_minus_second':results[first]['BA']-results[second]['BA'],
            'paired_BA_95CI':np.quantile(boots[first]-boots[second],[.025,.975]),
            'same_holdout_groups':True,'conditional_on_fitted_classifiers':True})
    for name,kind in plan:
        if name!='RAW_REFERENCE':add_pair(name+'__'+kind,'RAW_REFERENCE__'+kind,'vs_same_view_RAW')
    for name in CFG['cases']:add_pair(name+'__canonical_ADC_voltage',name+'__native_voltage','canonical_minus_native')
    for tag in ['FIXED','COMPLETION']:
        for kind in CFG['core_views']:add_pair(tag+'_POST__'+kind,tag+'_PRE__'+kind,'postADC_minus_preADC')
    np.savez_compressed(ROOT/'PREDICTIONS_PRIVATE.npz',label=labels,cluster=groups,**predictions)
    write('METRICS.json',{'UTC':utc(),'results':results,'pairs':pairs,'HGB_fits':FITS,'response_training_runs':0,
         'new_generation_rows':0,'density_fits':0,'EVAL_read':0,'CAL_decode':0,'original_A_F_unchanged':True,
         'multiple_comparisons':'exploratory fixed contrasts; no familywise or independent-acquisition claim',
         'ADC_only_canonical_RAW_prediction_equal':bool(np.array_equal(predictions['RAW_REFERENCE__canonical_ADC_voltage'],predictions['RAW_LEGACY_ADC__canonical_ADC_voltage']))})
    plot(results,domains,real,descriptions)
    write('NUMERIC_RECEIPT.json',{'UTC':utc(),'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'HGB_fits':FITS,
        'all_fixed_comparisons_complete':True,'response_training_runs':0,'new_generation_rows':0,'EVAL_read':0,'CAL_decode':0,
        'figures':sorted(str(p.relative_to(ROOT)) for p in (ROOT/'figures').glob('*.png'))})

def plot(results,domains,real,descriptions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    names=CFG['cases'];pos=np.arange(len(names));fig,ax=plt.subplots(figsize=(10.2,5.4))
    for kind,offset,color,label in [('native_voltage',-.13,'#be4c3f','Original voltage'),('canonical_ADC_voltage',.13,'#176b9c','Common ADC reconstruction')]:
        rows=[results[n+'__'+kind] for n in names];y=np.array([r['BA'] for r in rows]);ci=np.array([r['BA_95CI'] for r in rows])
        ax.errorbar(y,pos+offset,xerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,color=color,label=label)
    ax.axvspan(.4,.6,color='#529771',alpha=.12);ax.axvline(.5,color='gray',ls=':');ax.set(yticks=pos,yticklabels=names,xlim=(.35,1.0),xlabel='Balanced accuracy with 95% group bootstrap CI',title='R50: saved REAL outputs and ADC-only control')
    ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'voltage_attribution.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.9))
    for ax,kind in zip(axes.flat,CFG['decomposition_views']):
        rows=[results[n+'__'+kind] for n in CFG['decomposition_cases']];y=np.array([r['BA'] for r in rows]);ci=np.array([r['BA_95CI'] for r in rows]);x=np.arange(3)
        ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=4,color='#176b9c')
        ax.axhspan(.4,.6,color='#529771',alpha=.12);ax.axhline(.5,color='gray',ls=':');ax.set(xticks=x,xticklabels=['RAW','Fixed post','Completion post'],ylim=(.35,1.),title=kind.replace('_',' '),ylabel='Balanced accuracy')
    fig.suptitle('R50: diagnostic decomposition (original acceptance unchanged)',y=.995);fig.tight_layout();fig.savefig(out/'baseline_pulse_decomposition.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8))
    for name,color in [('RAW_REFERENCE','#444444'),('FIXED_POST','#bd6b28'),('COMPLETION_POST','#176b9c')]:
        f=scalars(domains[name])
        for ax,k,scale in zip(axes,['baseline_mean','baseline_RMS','Qtotal_mVns'],[1000,1000,1]):
            v=np.sort(f[k][np.isfinite(f[k])]*scale);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=name,color=color,alpha=.85)
    for ax,title in zip(axes,['Baseline mean [mV]','Baseline RMS [mV]','Signed total charge [mV ns]']):ax.set(xlabel=title,ylabel='Empirical CDF');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.suptitle('All 2000 matched reference rows; no density fit');fig.tight_layout();fig.savefig(out/'paired_population_descriptions.png');plt.close(fig)

try:run()
except BaseException as exc:
    write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'HGB_fits_attempted':FITS,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START})
    raise
