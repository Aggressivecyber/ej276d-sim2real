"""Original fixed HGB views and grouped bootstrap; no import-time fitting."""
from __future__ import annotations
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import evaluation_primitives as ev
from common_math_primitives import distances

REGIONS={'baseline':np.arange(100),'prompt':np.arange(100,150),'postpeak':np.arange(150,220),'tail':np.arange(220,400),'no_prompt':np.r_[np.arange(100),np.arange(150,480)]}

def matrix(d,view):
    if view=='head34':return d['head34'].copy()
    if view=='Q_PSD_2D':return np.column_stack([d['features']['Qtotal_mVns'],d['features']['PSD']])
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
    from common_math_primitives import fom
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

def fit_one(candidate,real,selection_candidate,selection_real,view,cfg,counter):
    a=matrix(candidate,view);b=matrix(real,view);tr,te=selection_candidate;rt,re=selection_real
    X=np.vstack([a[tr],b[rt]]);T=np.vstack([a[te],b[re]]);y=np.r_[np.ones(len(tr)),np.zeros(len(rt))];yt=np.r_[np.ones(len(te)),np.zeros(len(re))]
    missing=int(np.count_nonzero(~np.isfinite(T)));X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    impute=SimpleImputer(strategy='median',keep_empty_features=True);X=impute.fit_transform(X);T=impute.transform(T)
    scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
    model=HistGradientBoostingClassifier(**cfg['C2ST']);counter['HGB_attempts']+=1
    if counter['HGB_attempts']>cfg['HGB_max_fits']:raise RuntimeError('HGB budget exhausted')
    clock=time.monotonic();model.fit(X,y);p=model.predict_proba(T)[:,1];ptr=model.predict_proba(X)[:,1]
    labels=np.r_[candidate['cluster'][te],real['cluster'][re]]
    result={'training_balanced_accuracy':float(balanced_accuracy_score(y,ptr>=.5)),'training_AUC':float(roc_auc_score(y,ptr)),'train_holdout_diagnostic_no_new_fit':True,'balanced_accuracy':float(balanced_accuracy_score(yt,p>=.5)),'AUC':float(roc_auc_score(yt,p)),'train_each_domain':len(tr),'holdout_each_domain':len(te),'features':X.shape[1],
            'heldout_missing_values':missing,'heldout_CI':None,
            'fit_and_bootstrap_s':time.monotonic()-clock,'classification_train_only_imputer_and_standardizer':True,'tuning':False,'region_outside_features':False if view not in ['full_voltage','full_historical_shape','head34','Q_PSD_2D'] else 'complete view explicitly global'}
    private={'probability':p,'label':yt,'cluster':labels,'training_probability':ptr,'training_label':y}
    if view=='Q_PSD_2D':private.update(training_actual_preprocessed_2columns=X,holdout_actual_preprocessed_2columns=T,training_raw_2columns=np.vstack([a[tr],b[rt]]),holdout_raw_2columns=np.vstack([a[te],b[re]]))
    if globals().get('PERSIST_HGB_CALLBACK') is not None:PERSIST_HGB_CALLBACK(result,private)
    result['heldout_CI']=bootstrap(yt,p,labels,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
    result['fit_and_bootstrap_s']=time.monotonic()-clock
    return result,private
