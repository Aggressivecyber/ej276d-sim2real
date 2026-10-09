from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import json, hashlib, time, traceback, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from support import safe, summary, bootstrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_bytes())
START=time.monotonic()
NAMES=CFG['candidates'];EDGES=np.asarray(CFG['edges_logQ'],dtype='f8')
GROUPS=CFG['projection_draw_order']
COUNTER={'HGB_attempts':0,'response_updates':0,'response_generation':0,'EM_attempts':0,'completed_projection_group_blocks':0}

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def write(name,value):
    (ROOT/name).write_bytes((json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def progress():write('NUMERIC_PROGRESS.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER})
def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def baseline(v):return v-v[:,:100].mean(1,keepdims=True)
def normalized(v):
    x=baseline(np.asarray(v,dtype='f8'));p=np.maximum(x.max(1,keepdims=True),CFG['normalization_peak_floor_V'])
    return x/p,p
def bins(q):
    b=np.searchsorted(EDGES,np.log(np.maximum(q,1e-30)),side='right')-1
    b[np.log(np.maximum(q,1e-30))==EDGES[-1]]=7
    return b
def domain(v,groups,cluster_prefix,valid_extra=None):
    v=np.asarray(v,dtype='f8');s=baseline(v);rawpeak=s.max(1)
    q=s[:,100:400].sum(1)*1000
    psd=1-np.divide(s[:,100:150].sum(1)*1000,q,out=np.full(len(v),np.nan),where=q>0)
    valid=np.isfinite(v).all(1)&(rawpeak>0)&(q>0)&np.isfinite(psd)
    if valid_extra is not None:valid&=valid_extra
    u,p=normalized(v)
    return {'V':v,'s':s,'u':u,'p':p,'q':q,'b':bins(q),'valid':valid,
            'groups':groups,'cluster':np.asarray([cluster_prefix+'|'+str(g) for g in groups])}

def projection_space(constants,pooled):
    D=np.zeros((479,480),'f8');i=np.arange(479);D[i,i]=-1;D[i,i+1]=1
    A=np.zeros((60,480),'f8')
    for i in range(60):A[i,i*8:(i+1)*8]=1/8
    short=np.zeros((480,1));short[100:150]=1
    total=np.zeros((480,1));total[100:400]=1
    columns=[constants['shape480__projection'].astype('f8')/constants['shape480__scale'][:,None],
             D.T@(constants['diff479__projection'].astype('f8')/constants['diff479__scale'][:,None]),
             A.T@pooled.astype('f8'),np.eye(480)[:,:100],short,total]
    C=np.concatenate(columns,axis=1);length=np.linalg.norm(C,axis=0);zero=length==0
    C=C[:,~zero]/length[~zero]
    U,sigma,_=np.linalg.svd(C,full_matrices=True)
    rank=int(np.sum(sigma>CFG['rank_relative_threshold']*sigma[0]));Q=U[:,:rank]
    seen=Q@Q.T;complement=np.eye(480)-seen
    leakage=float(max(np.abs(complement[:100]).max(),np.abs(complement[:,:100]).max()))
    complement[:100,:]=0;complement[:,:100]=0
    complement=(complement+complement.T)/2;seen=np.eye(480)-complement
    bmat=constants['baseline100_V__projection'].astype('f8')/constants['baseline100_V__scale'][:,None]
    bs=np.linalg.svd(bmat,compute_uv=False);br=int(np.sum(bs>CFG['rank_relative_threshold']*bs[0]))
    checks={
        'symmetry_seen_maxabs':np.max(np.abs(seen-seen.T)),
        'symmetry_complement_maxabs':np.max(np.abs(complement-complement.T)),
        'idempotence_seen_maxabs':np.max(np.abs(seen@seen-seen)),
        'idempotence_complement_maxabs':np.max(np.abs(complement@complement-complement)),
        'seen_complement_orthogonality_maxabs':np.max(np.abs(seen@complement)),
        'C_complement_maxabs':np.max(np.abs(C.T@complement)),
        'identity_resolution_maxabs':np.max(np.abs(seen+complement-np.eye(480))),
    }
    assert max(map(float,checks.values()))<1e-9,checks
    np.savez_compressed(OUT/'PROJECTION_SPACE_PRIVATE.npz',C_unit_columns=C,singular_values=sigma,
                        Qseen_SVD=Q,Pseen=seen,Pcomplement=complement,D=D,A8=A,
                        baseline_singular_values=bs)
    receipt={'columns_before_zero_removal':sum(c.shape[1] for c in columns),'zero_columns':np.flatnonzero(zero),
             'columns':C.shape[1],'rank':rank,'complement_dimension':480-rank,
             'rank_rule':'sigma > 1e-10*sigma_max','singular_values':sigma,
             'baseline_projection_rank':br,'baseline_projection_columns':bmat.shape[1],
             'known_baseline_complement_exact_zero_before_cleanup_maxabs':leakage,
             'known_zeros_cleanup_not_data_dependent':True,'algebra_checks':checks,
             'whole_training_loss_nullspace':False}
    write('PROJECTION_SPACE_RECEIPT.json',receipt)
    log('space rank '+str(rank)+' complement '+str(480-rank))
    return seen,complement,receipt

def energy(a,b,total):
    aa=float(np.dot(a,a));bb=float(np.dot(b,b));cross=float(2*np.dot(a,b));tt=float(np.dot(total,total))
    return {'signal_sqnorm':aa,'electronic_sqnorm':bb,'twice_cross_inner_product':cross,
            'component_sum_with_cross':aa+bb+cross,'total_sqnorm':tt,'closure_error':aa+bb+cross-tt}

def means_coverage(domains,seen,complement,source_bins):
    result={};private={};real=domains['REAL'];maxreconstruct=0.
    for name,d in domains.items():
        d['seen']=d['u']@seen;d['complement']=d['u']@complement
        maxreconstruct=max(maxreconstruct,float(np.max(np.abs(d['seen']+d['complement']-d['u']))))
        private[name+'__u']=d['u'].astype('f4');private[name+'__Q_mVns']=d['q'];private[name+'__valid']=d['valid'];private[name+'__Qbin']=d['b']
    for name,d in domains.items():
        table={}
        for condition in ['observed_Q','source_Q'] if name in NAMES else ['observed_Q']:
            rows=[];b=d['b'] if condition=='observed_Q' else source_bins
            for i in [None,*range(8)]:
                mask=d['valid'] if i is None else d['valid']&(b==i)
                ref=real['valid'] if i is None else real['valid']&(real['b']==i)
                raw=np.ones(len(b),bool) if i is None else b==i
                label='overall' if i is None else 'bin'+str(i+1)
                ok=int(mask.sum())>=40 and int(ref.sum())>=40
                row={'bin_id_1based':None if i is None else i+1,'source_count':int(raw.sum()),'QC_n':int(mask.sum()),
                     'invalid_n':int((raw&~d['valid']).sum()),'REAL_observed_Q_QC_n':int(ref.sum()),
                     'qualified':ok,'reason':None if ok else 'a_domain_below40',
                     'conditioning':'same observed Q' if condition=='observed_Q' else 'candidate source Q vs REAL observed Q; not identical conditioning'}
                if mask.any():
                    m=d['u'][mask].mean(0);med=np.median(d['u'][mask],axis=0)
                    private[name+'__'+condition+'__'+label+'__mean_u']=m
                    private[name+'__'+condition+'__'+label+'__median_u']=med
                    if ref.any():
                        delta=m-real['u'][ref].mean(0);ss=delta@seen;cc=delta@complement
                        total=float(np.dot(delta,delta));st=float(np.dot(ss,ss));ct=float(np.dot(cc,cc))
                        row.update(mean_difference_sqnorm=total,seen_sqnorm=st,complement_sqnorm=ct,
                                   seen_fraction=st/total if total>0 else None,
                                   complement_fraction=ct/total if total>0 else None,
                                   energy_closure_error=st+ct-total)
                        private[name+'__'+condition+'__'+label+'__delta_u']=delta
                        private[name+'__'+condition+'__'+label+'__delta_seen']=ss
                        private[name+'__'+condition+'__'+label+'__delta_complement']=cc
                rows.append(row)
            table[condition]=rows
        result[name]=table
    assert maxreconstruct<1e-10
    np.savez_compressed(OUT/'TEMPORAL_DIAGNOSTICS_PRIVATE.npz',**private)
    result['_algebra']={'actual_u_reconstruction_maxabs':maxreconstruct,'means_are_not_full_distributions':True}
    write('A_COVERAGE_METRICS.json',result);progress()
    return result

def fine_distances(domains,constants):
    draws={'ORIGINAL':{name:constants[name+'__projection'].astype('f8') for name in GROUPS}}
    states={};save={}
    for seed in CFG['new_projection_seeds']:
        rng=np.random.Generator(np.random.PCG64(seed));label='NEW_'+str(seed);initial=rng.bit_generator.state
        draws[label]={}
        for name,dim in zip(GROUPS,[480,479,100]):
            raw=rng.standard_normal((dim,128));P=raw/np.linalg.norm(raw,axis=0)
            draws[label][name]=P;save[label+'__'+name]=P
        states[label]={'initial':initial,'final':rng.bit_generator.state}
    np.savez_compressed(OUT/'NEW_PROJECTIONS_PRIVATE.npz',**save)
    write('NEW_PROJECTION_RANDOM_STATES.json',states)
    comparisons=[*NAMES,'TRAIN_REFERENCE'];levels=np.linspace(.005,.995,128)
    results={};distances_private={}
    for label,matrices in draws.items():
        results[label]={}
        for name in GROUPS:
            quantiles={};counts={}
            for domain_name,d in domains.items():
                x=d['u'] if name=='shape480' else np.diff(d['u'],axis=1) if name=='diff479' else d['s'][:,:100]
                projected=((x-constants[name+'__mean'])/constants[name+'__scale'])@matrices[name]
                qs=[];ns=[]
                for i in [None,*range(8)]:
                    mask=d['valid'] if i is None else d['valid']&(d['b']==i)
                    ns.append(int(mask.sum()))
                    qs.append(np.quantile(projected[mask],levels,axis=0,method='linear') if mask.any() else None)
                quantiles[domain_name]=qs;counts[domain_name]=ns
            rows={}
            for domain_name in comparisons:
                parts=[]
                for j in range(9):
                    n=counts[domain_name][j];nr=counts['REAL'][j]
                    a=quantiles[domain_name][j];b=quantiles['REAL'][j]
                    diff=np.abs(a-b).mean(0) if a is not None and b is not None else None
                    if diff is not None:distances_private[label+'__'+name+'__'+domain_name+'__'+str(j)]=diff
                    parts.append({'bin_id_1based':None if j==0 else j,'candidate_N':n,'REAL_N':nr,
                                  'qualified':n>=40 and nr>=40,'W1_quantile_direction_mean':float(diff.mean()) if diff is not None else None,
                                  'reason':None if n>=40 and nr>=40 else 'a_domain_below40'})
                qualified=[j for j in range(1,9) if parts[j]['qualified']]
                rows[domain_name]={'overall':parts[0],'bins':parts[1:],
                                   'qualified_bin_ids_1based':qualified,
                                   'qualified_bin_equal_mean':float(np.mean([parts[j]['W1_quantile_direction_mean'] for j in qualified])) if qualified else None}
            results[label][name]=rows
            write('results/B_'+label+'_'+name+'.json',rows)
            COUNTER['completed_projection_group_blocks']+=1;progress()
            log('projection '+label+' '+name+' saved')
    np.savez_compressed(OUT/'PROJECTION_W1_DIRECTIONS_PRIVATE.npz',**distances_private)
    common={};blocks=list(draws)
    for name in GROUPS:
        ids=[i for i in range(1,9) if all(results[b][name][d]['bins'][i-1]['qualified'] for b in blocks for d in comparisons)]
        table={}
        for d in comparisons:
            values={b:float(np.mean([results[b][name][d]['bins'][i-1]['W1_quantile_direction_mean'] for i in ids])) if ids else None for b in blocks}
            new=[values[b] for b in blocks[1:] if values[b] is not None]
            old=values['ORIGINAL']
            table[d]={'equal_mean_on_common_bins':values,'new_block_range':[min(new),max(new)] if new else None,
                      'new_block_mean':float(np.mean(new)) if new else None,
                      'new_mean_divided_by_original':float(np.mean(new))/old if new and old and old>0 else None}
        common[name]={'common_qualified_bin_ids_1based':ids,'domains':table}
    output={'blocks':results,'common_support_summary':common,'quantile_definition':CFG['quantile_levels'],
            'new_block_range_is_projection_sampling_not_CI':True,'frozen_TRAIN_scales':True,'main_population_reweighted':False}
    write('B_PROJECTION_METRICS.json',output)
    return output

def fit_classifiers(domains):
    rows=load(CFG['inputs']['classification_rows'],[f'phase{i}_{j}' for i in (0,1) for j in (0,1,2)])
    np.savez_compressed(OUT/'CLASSIFICATION_ROWS_PRIVATE.npz',**rows)
    result={}
    for name in [*NAMES,'TRAIN_REFERENCE']:
        result[name]={};d=domains[name];r=domains['REAL']
        j=2 if name=='TRAIN_REFERENCE' else 0
        tr,te=rows[f'phase0_{j}'],rows[f'phase1_{j}'];rt,re=rows['phase0_1'],rows['phase1_1']
        supported=min(len(tr),len(rt))>=150 and min(len(te),len(re))>=75
        assert not set(d['groups'][tr])&set(d['groups'][te])
        assert not set(r['groups'][rt])&set(r['groups'][re])
        assert d['valid'][np.r_[tr,te]].all() and r['valid'][np.r_[rt,re]].all()
        for view,key in [('normalized_u480','u'),('seen_reconstruction480','seen'),('complement_reconstruction480','complement')]:
            slot=name+'__'+view
            if not supported:
                result[name][view]={'defined':False,'reason':'fixed_support_below150_75'}
                write('results/C_'+slot+'.json',result[name][view]);continue
            X=np.vstack([d[key][tr],r[key][rt]]);T=np.vstack([d[key][te],r[key][re]])
            y=np.r_[np.ones(len(tr)),np.zeros(len(rt))];yt=np.r_[np.ones(len(te)),np.zeros(len(re))]
            missing=int(np.count_nonzero(~np.isfinite(T)));X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            X=imputer.fit_transform(X);T=imputer.transform(T)
            scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
            model=HistGradientBoostingClassifier(**CFG['C2ST'])
            COUNTER['HGB_attempts']+=1;assert COUNTER['HGB_attempts']<=15;progress()
            clock=time.monotonic();model.fit(X,y)
            p=model.predict_proba(T)[:,1];ptr=model.predict_proba(X)[:,1]
            clusters=np.r_[d['cluster'][te],r['cluster'][re]]
            private={'probability':p,'label':yt,'cluster':clusters,'training_probability':ptr,'training_label':y,
                     'imputer_statistics':imputer.statistics_,'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_}
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**private)
            row={'defined':True,'training_balanced_accuracy':float(balanced_accuracy_score(y,ptr>=.5)),
                 'training_AUC':float(roc_auc_score(y,ptr)), 'balanced_accuracy':float(balanced_accuracy_score(yt,p>=.5)),
                 'AUC':float(roc_auc_score(yt,p)), 'train_each_domain':len(tr),'holdout_each_domain':len(te),'features':480,
                 'heldout_missing_values':missing,'classifier_train_only_imputer_scaler':True,
                 'fit_s':time.monotonic()-clock,'historical_493_scalar_augmented_view_reused':False,
                 'constant_training_columns':int(np.sum(np.var(X,axis=0)==0))}
            write('results/C_'+slot+'.json',row)
            row['heldout_CI']=bootstrap(yt,p,clusters,500,CFG['bootstrap_seed'])
            row['fit_and_bootstrap_s']=time.monotonic()-clock
            write('results/C_'+slot+'.json',row);result[name][view]=row
            write('C_CLASSIFIER_METRICS.json',result)
            log('classifier '+slot+' BA '+str(row['balanced_accuracy']))
    return {'results':result,'HGB_attempts':COUNTER['HGB_attempts'],'fixed_R26_indices_and_groups':True,
            'REAL_REAL_definition':'saved TRAIN reference versus DEV, not independent acquisition',
            'CI_conditions_on_fixed_classifiers_and_outputs':True,'views_have480_columns_no_scalar_append':True}

def branch_diagnostics(saved,domains,seen,complement,source_bins):
    main={};joint={};private={}
    for name in NAMES:
        a=saved[name];s=a['signal_component_V'].astype('f8');e=a['electronic_component_V'].astype('f8');y=a['pre_ADC_V'].astype('f8')
        uy,p=normalized(y);us=baseline(s)/p;ue=baseline(e)/p
        pre_res=y-s-e;adc=a['waveforms_V'].astype('f8')-y
        main[name]={'shared_normalization':True,'functional_branches_not_independent_physical_noise':True,
                    'saved_pre_vs_branchsum_maxabs_V':float(np.abs(pre_res).max()),
                    'normalized_additive_closure_maxabs':float(np.abs(us+ue-uy).max()),
                    'ADC_residual_V':summary(adc.ravel()),'ADC_row_RMS_V':summary(np.sqrt(np.mean(adc**2,axis=1)))}
        for condition in ['observed_Q','source_Q']:
            b=domains[name]['b'] if condition=='observed_Q' else source_bins;parts=[]
            for i in [None,*range(8)]:
                mask=domains[name]['valid'] if i is None else domains[name]['valid']&(b==i)
                label='overall' if i is None else 'bin'+str(i+1)
                row={'bin_id_1based':None if i is None else i+1,'QC_n':int(mask.sum()),'qualified':int(mask.sum())>=40}
                if mask.any():
                    sm=us[mask].mean(0);em=ue[mask].mean(0);tm=uy[mask].mean(0)
                    row['total']=energy(sm,em,tm);row['seen']=energy(sm@seen,em@seen,tm@seen)
                    row['complement']=energy(sm@complement,em@complement,tm@complement)
                    for suffix,values in [('signal',sm),('electronic',em),('total',tm),('signal_seen',sm@seen),('signal_complement',sm@complement),('electronic_seen',em@seen),('electronic_complement',em@complement)]:
                        private[name+'__'+condition+'__'+label+'__'+suffix]=values
                parts.append(row)
            main[name][condition]=parts
        a=load(CFG['inputs']['joint'][name],['source_parent_index','waveforms_V','signal_component_V','electronic_component_V','valid','parent_composition'])
        parent=a['source_parent_index'].reshape(128,8);assert np.all(parent==parent[:,:1]) and len(np.unique(parent[:,0]))==128
        raw_s=baseline(a['signal_component_V'].astype('f8'));raw_e=baseline(a['electronic_component_V'].astype('f8'))
        raw_y=raw_s+raw_e;y0=a['signal_component_V'].astype('f8')+a['electronic_component_V'].astype('f8')
        pp=np.maximum(raw_y.max(1,keepdims=True),.0025)
        sx=raw_s/pp;ex=raw_e/pp
        views={'raw_baseline_subtracted_V':(raw_s,raw_e),'normalized_u':(sx,ex),
               'normalized_seen':(sx@seen,ex@seen),'normalized_complement':(sx@complement,ex@complement)}
        valid=a['valid'].reshape(128,8).all(1);labels=a['parent_composition'];joint[name]={}
        for view,(ss,ee) in views.items():
            ss=ss.reshape(128,8,480);ee=ee.reshape(128,8,480);yy=ss+ee
            good=valid&np.isfinite(ss).all((1,2))&np.isfinite(ee).all((1,2))
            vs=ss.var(1,ddof=1);ve=ee.var(1,ddof=1);vt=yy.var(1,ddof=1)
            cov=((ss-ss.mean(1,keepdims=True))*(ee-ee.mean(1,keepdims=True))).sum(1)/7
            closure=vs+ve+2*cov-vt
            viewrows={}
            for cat in ['all','pure_EE','pure_had','mixed']:
                member=np.ones(128,bool) if cat=='all' else labels==cat;use=member&good
                row={'source_parent_count':int(member.sum()),'finite_parent_count':int(use.sum()),'undefined_parent_count':int((member&~good).sum()),'repeats':8,'sample_variance_ddof':1,'gates':{}}
                for gate,(lo,hi) in CFG['branch_gates'].items():
                    fields={'signal_variance':vs[:,lo:hi].mean(1),'electronic_variance':ve[:,lo:hi].mean(1),
                            'covariance':cov[:,lo:hi].mean(1),'twice_covariance':2*cov[:,lo:hi].mean(1),
                            'total_variance':vt[:,lo:hi].mean(1),'closure':closure[:,lo:hi].mean(1)}
                    row['gates'][gate]={k:summary(v[use]) for k,v in fields.items()}
                    for k,v in fields.items():private[name+'__joint__'+view+'__'+gate+'__'+k]=v
                if use.any():
                    for field,values in [('signal_variance',vs),('electronic_variance',ve),('covariance',cov),('total_variance',vt)]:
                        private[name+'__jointcurve__'+view+'__'+cat+'__'+field]=values[use].mean(0)
                    row['closure_maxabs']=float(np.abs(closure[use]).max())
                viewrows[cat]=row
            joint[name][view]=viewrows
        residual=a['waveforms_V'].astype('f8')-y0
        joint[name]['ADC_residual_V']=summary(residual.ravel())
        joint[name]['parent_composition_counts']={str(c):int((labels==c).sum()) for c in np.unique(labels)}
        joint[name]['preADC_from_saved_component_sum_not_separately_saved']=True
        write('results/D_'+name+'.json',{'main':main[name],'joint8':joint[name]})
        progress();log('branches '+name+' saved')
    np.savez_compressed(OUT/'BRANCH_DECOMPOSITION_PRIVATE.npz',**private)
    result={'main':main,'joint8':joint,'branch_covariance_retained':True,'shared_random_peak_denominator':True,
            'no_new_response_generation':True,'no_physical_independence_or_conditional_mean_claim':True}
    write('D_BRANCH_METRICS.json',result)
    return result

def run():
    constants=load(CFG['inputs']['fine_constants'])
    pooled=load(CFG['inputs']['pooled_constants'],['shape_projection'])['shape_projection']
    real=load(CFG['inputs']['real']);meta=load(CFG['inputs']['source_metadata'],['source_ids','composition','pdg_inferred'])
    source=load(CFG['inputs']['source'],['source_waveforms_V'])['source_waveforms_V']
    source_q=baseline(source.astype('f8'))[:,100:400].sum(1)*1000;source_bins=bins(source_q)
    domains={'REAL':domain(real['real_V'],real['real_groups'],'REAL',real['real_QC_valid']),
             'TRAIN_REFERENCE':domain(real['reference_V'],real['reference_groups'],'TRAIN_REFERENCE')}
    saved={}
    for name in NAMES:
        a=load(CFG['inputs']['generated'][name]);assert np.array_equal(a['source_ids'],meta['source_ids'])
        saved[name]=a;domains[name]=domain(a['waveforms_V'],np.arange(2000),'G4_SOURCE')
    inputs={'counts':{n:{'raw_n':len(d['V']),'QC_n':int(d['valid'].sum()),'observed_bin_counts':[int((d['valid']&(d['b']==i)).sum()) for i in range(8)],'outside_bin_QC_n':int((d['valid']&((d['b']<0)|(d['b']>7))).sum())} for n,d in domains.items()},
            'source_composition':{str(c):int((meta['composition']==c).sum()) for c in np.unique(meta['composition'])},
            'PDG_inferred_counts':{str(c):int((meta['pdg_inferred']==c).sum()) for c in np.unique(meta['pdg_inferred'])},
            'PDG_inference_not_new_truth_read':True,'DEV_multiple_prior_exposure':True,'population_reweighted':False}
    write('ACTUAL_INPUT_RECEIPT.json',inputs)
    seen,comp,space=projection_space(constants,pooled)
    A=means_coverage(domains,seen,comp,source_bins)
    B=fine_distances(domains,constants)
    D=branch_diagnostics(saved,domains,seen,comp,source_bins)
    C=fit_classifiers(domains)
    write('C_CLASSIFIER_METRICS.json',C)
    old=json.loads(Path(CFG['inputs']['old_metrics_compact']).read_bytes())
    metric={'stage':CFG['stage'],'status':'complete','A_space':space,'A_coverage':A,'B_projections':B,'C_classifiers':C,
            'D_branches':D,'actual_inputs':inputs,'source_Q_migration_R26_reused':old['migration'],
            'prior_PSD_widths_source_sensitivity_reference':'R26 REPORT/METRICS_COMPACT; no repeat fits',
            'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__},
            'whole_loss_blind_space_claim':False,'EVAL_read':0,'CAL_waveforms':0,'SSH':0,'fit_response_generation':0}
    write('metrics.json',metric)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
                                'counters':COUNTER,'rank':space['rank'],'status':'complete','CPU_threads':4})
    log('R27 complete '+json.dumps(COUNTER))

try:
    run()
except BaseException as exc:
    write('FAILURE_'+datetime.now(timezone.utc).strftime('%H%M%S_%f')+'.json',
          {'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
           'error':repr(exc),'traceback':traceback.format_exc(),'counters':COUNTER})
    raise
