from __future__ import annotations
import numpy as np

def q_np(x):
    a=np.asarray(x,dtype='f8')
    return a[...,100:400].sum(-1)-3.*a[...,:100].sum(-1)

def variance100(x):
    a=np.asarray(x,dtype='f8')[...,:100]
    e=a-a.mean(-1,keepdims=True)
    return (e*e).sum(-1)/99.

def bins_for(x,edges):
    q=q_np(x)
    if not np.isfinite(x).all() or not np.isfinite(q).all() or not (q>0).all():
        raise ValueError('nonfinite input or nonpositive original q; no deletion')
    logq=np.log(1000.*q)
    raw=np.searchsorted(edges,logq,side='right')-1
    raw[logq==edges[-1]]=7
    return np.clip(raw,0,7).astype('i8'),raw.astype('i8'),q

def continuous_logq(q,b,edges):
    q=np.asarray(q,dtype='f8');b=np.asarray(b,dtype='i8');edges=np.asarray(edges,dtype='f8')
    halfwidth=.5*(edges[b+1]-edges[b]);center=.5*(edges[b+1]+edges[b])
    if not np.isfinite(q).all() or not (q>0).all() or not (halfwidth>0).all():
        raise ValueError('invalid continuous charge condition; no deletion or clipping')
    value=(np.log(1000.*q)-center)/halfwidth
    if not np.isfinite(value).all():raise FloatingPointError('nonfinite continuous charge condition')
    return value

def kernel(epsilon,spectrum):
    e=np.asarray(epsilon,dtype='f8')
    return np.fft.irfft(np.sqrt(spectrum)*np.fft.rfft(e,axis=-1,norm='ortho'),n=480,axis=-1,norm='ortho')

def innovation(x,epsilon,mu,edges,p,kappa2,tau2,completion):
    x=np.asarray(x,dtype='f8');b,raw,q=bins_for(x,edges)
    v=variance100(x)
    amp2=np.maximum(tau2-v,0.) if completion else np.full(len(x),tau2,dtype='f8')
    n=kernel(epsilon,p)
    n=n-mu[b]*q_np(n)[...,None]
    n=n*np.sqrt(amp2/kappa2[b])[:,None]
    if not np.isfinite(n).all():raise FloatingPointError('nonfinite projected colored innovation')
    return n,{'conditioning_Qbin':b,'source_Qbin_actual_preADC':raw,'source_q_Vns':q,
              'input_variance100_V2':v,'innovation_amplitude_squared_V2':amp2,
              'inside_TRAIN_support':(raw>=0)&(raw<8),'noise_completion_triggered':amp2>0}

def summary(x):
    a=np.asarray(x,dtype='f8')
    if not np.isfinite(a).all():raise ValueError('nonfinite summary')
    return {'n':int(a.size),'min':float(a.min()),'q05':float(np.quantile(a,.05)),
            'median':float(np.median(a)),'q95':float(np.quantile(a,.95)),'max':float(a.max()),'mean':float(a.mean())}
