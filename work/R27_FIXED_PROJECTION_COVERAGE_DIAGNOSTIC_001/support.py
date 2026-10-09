import numpy as np

def safe(x):

 if isinstance(x,np.ndarray):return safe(x.tolist())

 if isinstance(x,np.generic):return safe(x.item())

 if isinstance(x,float):return x if np.isfinite(x) else None

 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}

 if isinstance(x,(tuple,list)):return [safe(v) for v in x]

 return x

def summary(v):

 a=np.asarray(v);x=a[np.isfinite(a)]

 return {'n':len(a),'defined_n':len(x),'undefined_n':len(a)-len(x),'mean':float(x.mean()) if len(x) else None,'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,'quantile_levels':[0,.05,.25,.5,.75,.95,1],'quantiles':np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist() if len(x) else None}

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
