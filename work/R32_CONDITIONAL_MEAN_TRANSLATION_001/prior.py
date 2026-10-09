from __future__ import annotations
import numpy as np
import torch

def posterior_high(p, ref):
    p=np.asarray(p,dtype=np.float64)
    w=np.asarray(ref['weights']);mu=np.asarray(ref['means']);v=np.asarray(ref['variances'])
    a=np.log(w)[None,:]-.5*np.log(2*np.pi*v)[None,:]-.5*(p[:,None]-mu[None,:])**2/v[None,:]
    a-=a.max(1,keepdims=True);a=np.exp(a);return a[:,1]/a.sum(1)

def tilt_tables(h, source_bins, real_PSD, real_bins, reference, composition):
    rows=[];lambdas=np.zeros(8);limit=np.log(10.)
    for b in range(8):
        use=source_bins==b;x=h[use];r=real_PSD[real_bins==b];ref=reference['slices'][b]
        row={'bin_id_1based':b+1,'source_N':len(x),'REAL_N':len(r),'supported':bool(len(x)and len(r)and ref['valid']),
             'raw_composition':{str(c):int(np.sum(composition[use]==c))for c in np.unique(composition)}}
        if not row['supported']:
            row.update(lambda_value=None,target_affinity=None,failure='empty_or_invalid_frozen_TRAIN_reference');rows.append(row);continue
        target=float(posterior_high(r,ref).mean())
        def affinity(l):
            w=np.exp(l*x);return float(np.dot(w,x)/w.sum())
        lower=affinity(-limit);upper=affinity(limit);lo=-limit;hi=limit
        for _ in range(60):
            middle=(lo+hi)/2
            if affinity(middle)<target:lo=middle
            else:hi=middle
        lam=-limit if target<lower else limit if target>upper else (lo+hi)/2
        w=np.exp(lam*x);lambdas[b]=lam
        row.update(lambda_value=lam,target_affinity=target,achieved_affinity=affinity(lam),affinity_bias=affinity(lam)-target,
                   reachable_lower=lower,reachable_upper=upper,boundary_failure=bool(target<lower or target>upper),
                   h_mean=float(x.mean()),h_quantiles=np.quantile(x,[0,.1,.5,.9,1]).tolist(),weight_ratio=float(w.max()/w.min()),
                   Kish_ESS=float(w.sum()**2/np.dot(w,w)),weighted_composition={str(c):float(w[composition[use]==c].sum()/w.sum())for c in np.unique(composition)})
        rows.append(row)
    return lambdas,rows

def weighted_quantile(a,w,levels):
    """Sort each coordinate. Midpoint cumulative weight with linear edge-clamped interpolation."""
    s,ix=torch.sort(a,dim=0);ww=w[:,None].expand_as(a).gather(0,ix)
    c=(ww.cumsum(0)-.5*ww)/ww.sum(0)
    ct=c.T.contiguous();st=s.T.contiguous();q=levels[None,:].expand(ct.shape[0],-1).contiguous()
    i=torch.searchsorted(ct,q).clamp(1,max(1,len(a)-1))
    if len(a)==1:return s[0].expand(len(levels),-1)
    cl=ct.gather(1,i-1);ch=ct.gather(1,i);sl=st.gather(1,i-1);sh=st.gather(1,i)
    f=((q-cl)/(ch-cl).clamp_min(1e-30)).clamp(0,1)
    return (sl+f*(sh-sl)).T

class RecordedChoice:
    def __init__(self,seed):self.rng=np.random.default_rng(seed);self.records=[]
    def choice(self,*args,**kwargs):
        x=self.rng.choice(*args,**kwargs);self.records.append(np.asarray(x,dtype='<u2').copy());return x
