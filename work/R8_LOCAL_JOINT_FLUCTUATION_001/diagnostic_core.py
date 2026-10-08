from __future__ import annotations
import hashlib,time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import evaluation as ev
from common_math import distances

REGIONS={'baseline':np.arange(100),'prompt':np.arange(100,150),'postpeak':np.arange(150,220),'tail':np.arange(220,400),'no_prompt':np.r_[np.arange(100),np.arange(150,480)]}

def matrix(d,view):
    if view=='full_voltage':return d['V'].copy()
    if view=='full_historical_shape':return ev.c2st_matrix(d,'shape_only')[0]
    part,kind=view.rsplit('_',1);s=d['V'][:,REGIONS[part]].copy();center=s-s.mean(axis=1,keepdims=True)
    den=(center*center).sum(axis=1);scale=np.sqrt(den/s.shape[1]);z=np.divide(center,scale[:,None],out=np.full_like(center,np.nan),where=scale[:,None]>0)
    out=s if kind=='voltage' else z
    if part=='baseline':
        acf=np.column_stack([np.divide((center[:,k:]*center[:,:-k]).sum(axis=1),den,out=np.full(len(s),np.nan),where=den>0) for k in range(1,9)])
        ps=np.abs(np.fft.rfft(center,axis=1))**2
        ps=np.divide(ps[:,1:],ps[:,1:].sum(axis=1,keepdims=True),out=np.full_like(ps[:,1:],np.nan),where=ps[:,1:].sum(axis=1,keepdims=True)>0)
        out=np.column_stack([out,acf,ps])
        if kind=='voltage':out=np.column_stack([out,np.log(np.maximum(center.std(axis=1,ddof=1),1e-30))])
    out[~np.isfinite(out)]=np.nan
    return out

def locality_checks():
    rng=np.random.default_rng(202610087);v=rng.normal(size=(5,480));original=ev.features(v)
    checks=[]
    for region,ix in REGIONS.items():
        z=v.copy();outside=np.setdiff1d(np.arange(480),ix);z[:,outside]=1000+rng.normal(size=(5,len(outside)))
        changed=ev.features(z)
        for suffix in ['voltage','local']:
            name=region+'_'+suffix;a=matrix(original,name);b=matrix(changed,name)
            same=np.array_equal(a,b,equal_nan=True);assert same,name+' leaks outside region'
            checks.append({'view':name,'outside_indices_modified':len(outside),'bitwise_unchanged':same})
    return checks

def transform(v,name):
    v=np.asarray(v,dtype='f8');edge=0;undefined=0
    if name in ['fractional_align','align_then_smooth']:
        d=ev.features(v);cross=np.array([ev.crossing(x,int(p),.5) for x,p in zip(d['x'],d['peak_index'])]);result=np.empty_like(v);t=np.arange(480)
        for i,c in enumerate(cross):
            if not np.isfinite(c):result[i]=v[i];undefined+=1;continue
            query=t+c-120;edge+=int(np.count_nonzero((query<0)|(query>479)))
            result[i]=np.interp(query,t,v[i],left=v[i,0],right=v[i,-1])
    else:result=v.copy()
    if name=='coarse_ADC4':
        result=np.clip(-.5+4/16383*np.round((result+.5)/(4/16383)),-.5,.5)
    if name in ['smooth_sigma1','align_then_smooth']:result=gaussian_filter1d(result,1,axis=1,mode='nearest',truncate=4)
    return result,{'operator':name,'edge_filled_query_points':edge,'undefined_cross50_unchanged_rows':undefined,'maxabs_change_V':float(np.max(np.abs(result-v))), 'moved_rows':int(np.count_nonzero(np.any(result!=v,axis=1)))}

def extra_features(d):
    x=d['x'];pi=d['peak_index'];cross=np.array([ev.crossing(y,int(p),.5) for y,p in zip(x,pi)])
    d['extra']={'cross50_index':cross,'cross50_phase':np.mod(cross,1),'peak_index':pi.astype(float),
                'ADC_lattice_RMS_steps':np.sqrt(np.mean((((d['V']+.5)*16383)-np.round((d['V']+.5)*16383))**2,axis=1))}
    for region in ['prompt','postpeak']:
        s=d['V'][:,REGIONS[region]];z=s-s.mean(axis=1,keepdims=True);dx=np.diff(s,axis=1);dd=np.diff(s,n=2,axis=1);den=np.sum(z*z,axis=1)
        rms=np.sqrt(np.mean(z*z,axis=1));curve=np.sqrt(np.mean(dd*dd,axis=1))
        d['extra'][region+'_local_maxima']=((dx[:,:-1]>0)&(dx[:,1:]<0)).sum(axis=1).astype(float)
        d['extra'][region+'_local_minima']=((dx[:,:-1]<0)&(dx[:,1:]>0)).sum(axis=1).astype(float)
        d['extra'][region+'_curvature_RMS_V']=curve
        d['extra'][region+'_relative_curvature']=np.divide(curve,rms,out=np.full(len(s),np.nan),where=rms>0)
        d['extra'][region+'_local_ACF1']=np.divide((z[:,1:]*z[:,:-1]).sum(axis=1),den,out=np.full(len(s),np.nan),where=den>0)
    return d

def native_statistics(d,ref):
    blocks={}
    fields={**d['features'],**d['extra']}
    from common_math import fom
    for r in [None,0,1,2,3]:
        use=d['valid'] if r is None else d['valid']&(d['region']==r)
        slot=ref['slots'][0 if r is None else r+1];key='overall' if r is None else 'q'+str(r+1)
        blocks[key]={'QC_n':int(use.sum()),'reference_valid':slot['valid'],'FoM':fom(d['features']['PSD'][use],slot['cut'] if slot['valid'] else None),'features':{k:ev.summary(v[use]) for k,v in fields.items()}}
    return {'source_n':len(d['V']),'QC_n':int(d['valid'].sum()),'regions':blocks}

def bootstrap(y,p,groups,repeats,seed):
    _,inv=np.unique(groups,return_inverse=True);n=int(inv.max()+1);rng=np.random.default_rng(seed)
    _,tie=np.unique(p,return_inverse=True);nt=int(tie.max()+1);a=[];auc=[]
    for _ in range(repeats):
        mult=np.bincount(rng.integers(n,size=n),minlength=n);w=mult[inv];n0=w[y==0].sum();n1=w[y==1].sum()
        if n0==0 or n1==0:continue
        a.append(.5*(w[(y==0)&(p<.5)].sum()/n0+w[(y==1)&(p>=.5)].sum()/n1))
        neg=np.bincount(tie[y==0],weights=w[y==0],minlength=nt);pos=np.bincount(tie[y==1],weights=w[y==1],minlength=nt)
        auc.append(float(np.dot(pos,np.cumsum(neg)-.5*neg)/(n0*n1)))
    return {'balanced_accuracy_95CI':np.quantile(a,[.025,.975]).tolist(),'AUC_95CI':np.quantile(auc,[.025,.975]).tolist(),'valid_replicates':len(a),'requested_replicates':repeats,'clusters':n,'conditional_on_fitted_classifier':True,'training_uncertainty':False}

def make_splits(domains,cfg):
    out={};nt=cfg['train_each_domain'];nv=cfg['holdout_each_domain'];counts={}
    for name,d in domains.items():
        typ='SIM' if name=='GEN' else 'REAL';keys=d['ids'] if typ=='SIM' else d['groups']
        flag=np.array([ev.hashint('R1-SPLIT|'+typ+'|'+str(s))%10<7 for s in keys])
        tr=np.flatnonzero(d['valid']&flag);te=np.flatnonzero(d['valid']&~flag)
        counts[name]={'train_available':len(tr),'holdout_available':len(te)}
        nt=min(nt,len(tr));nv=min(nv,len(te));out[name]=(tr,te)
    if nt<40 or nv<40:raise RuntimeError('fixed core C2ST support too low')
    for name,d in domains.items():
        tr,te=out[name];tr=tr[ev.order(d['ids'][tr],'R1-C2ST-SAMPLE')[:nt]];te=te[ev.order(d['ids'][te],'R1-C2ST-SAMPLE')[:nv]]
        assert not set(d['groups'][tr])&set(d['groups'][te]),name+' group/parent leakage'
        out[name]=(tr,te)
    return out,{'train_each_domain':nt,'holdout_each_domain':nv,'available':counts,'all_views_reuse_native_QC_and_indices':True,'REAL_REAL_is_TRAIN_DEV_not_half_EVAL':True}

def fit_one(candidate,real,selection_candidate,selection_real,view,cfg,counter):
    a=matrix(candidate,view);b=matrix(real,view);tr,te=selection_candidate;rt,re=selection_real
    X=np.vstack([a[tr],b[rt]]);T=np.vstack([a[te],b[re]]);y=np.r_[np.ones(len(tr)),np.zeros(len(rt))];yt=np.r_[np.ones(len(te)),np.zeros(len(re))]
    missing=int(np.count_nonzero(~np.isfinite(T)));X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    impute=SimpleImputer(strategy='median',keep_empty_features=True);X=impute.fit_transform(X);T=impute.transform(T)
    scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
    model=HistGradientBoostingClassifier(**cfg['C2ST']);counter['HGB_attempts']+=1
    if counter['HGB_attempts']>cfg['HGB_max_fits']:raise RuntimeError('HGB budget exhausted')
    clock=time.monotonic();model.fit(X,y);p=model.predict_proba(T)[:,1]
    labels=np.r_[candidate['cluster'][te],real['cluster'][re]]
    result={'balanced_accuracy':float(balanced_accuracy_score(yt,p>=.5)),'AUC':float(roc_auc_score(yt,p)),'train_each_domain':len(tr),'holdout_each_domain':len(te),'features':X.shape[1],
            'heldout_missing_values':missing,'heldout_CI':bootstrap(yt,p,labels,cfg['bootstrap_repeats'],cfg['bootstrap_seed']),
            'fit_and_bootstrap_s':time.monotonic()-clock,'classification_train_only_imputer_and_standardizer':True,'tuning':False,'region_outside_features':False if view not in ['full_voltage','full_historical_shape'] else 'complete view explicitly global'}
    return result,{'probability':p,'label':yt,'cluster':labels}

def same_N_conditional(domains):
    result={};fields=list(domains['REAL']['features'])+list(domains['REAL']['extra'])
    for r in [None,0,1,2,3]:
        key='overall' if r is None else 'q'+str(r+1);indices={}
        for name in ['REAL','GEN','TRAIN_REFERENCE']:
            d=domains[name];mask=d['valid'] if r is None else d['valid']&(d['region']==r);ix=np.flatnonzero(mask);indices[name]=ix[ev.order(d['ids'][ix],'R7-conditional-sameN')]
        n=min(len(x) for x in indices.values());result[key]={'N_per_domain':n,'supported_ge40':n>=40,'available':{k:len(v) for k,v in indices.items()},'comparison':{}}
        for name in ['GEN','TRAIN_REFERENCE']:
            fields_out={}
            for k in fields:
                a={**domains[name]['features'],**domains[name]['extra']}[k][indices[name][:n]];b={**domains['REAL']['features'],**domains['REAL']['extra']}[k][indices['REAL'][:n]]
                fields_out[k]={'candidate':ev.summary(a),'REAL':ev.summary(b),'distance':distances(a[np.isfinite(a)],b[np.isfinite(b)])}
            result[key]['comparison'][name]=fields_out
    return result
