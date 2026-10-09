from __future__ import annotations
import numpy as np
from model import adc

REGIONS = {'baseline0_100':(0,100),'fast100_150':(100,150),
           'slow150_400':(150,400),'late400_480':(400,480)}

def project(d):
    d=np.asarray(d,dtype=np.float64)
    if d.ndim!=2 or d.shape[1]!=480 or not np.isfinite(d).all():
        raise ValueError('finite rows x480 required')
    p=d.copy();p[:,:100]=0.
    for left,right in [(100,150),(150,400)]:
        p[:,left:right]-=d[:,left:right].mean(axis=1,keepdims=True)
    return p

def charges(v):
    v=np.asarray(v,dtype=np.float64);x=v-v[:,:100].mean(1,keepdims=True)
    q=x[:,100:400].sum(1)*1000;fast=x[:,100:150].sum(1)*1000
    ps=1-np.divide(fast,q,out=np.full(len(q),np.nan),where=q>0)
    return {'Qtotal_mVns':q,'Qfast_mVns':fast,'PSD':ps}

def protect(x0,x1,origin='saved'):
    x0=np.asarray(x0,dtype=np.float64)
    if origin not in ['saved','recovered','replayed']:raise ValueError('explicit x1 provenance required')
    delta=np.asarray(x1,dtype=np.float64)-x0
    provenance='exact '+origin+' f32 x1 and x0 individually promoted to f64 before subtraction'
    keep=project(delta);removed=delta-keep;pre=x0+keep;f32=pre.astype('f4')
    post,ad=adc(f32)
    return {'pre_ADC_f64_V':pre,'ADC_input_f32_V':f32,'waveforms_V':post,
            'original_increment_f64_V':delta,'kept_increment_f64_V':keep,
            'removed_increment_f64_V':removed},ad,provenance

def selfcheck():
    rng=np.random.default_rng(20261010340);a=rng.normal(size=(7,480));b=rng.normal(size=(7,480))
    pa=project(a);pb=project(b);removed=a-pa
    rows={'linearity_maxabs':float(np.max(np.abs(project(2*a-3*b)-(2*pa-3*pb)))),
          'idempotence_maxabs':float(np.max(np.abs(project(pa)-pa))),
          'orthogonality_dot_maxabs':float(np.max(np.abs(np.sum(pa*removed,axis=1)))),
          'baseline_maxabs':float(np.max(np.abs(pa[:,:100]))),
          'fast_segment_sum_maxabs':float(np.max(np.abs(pa[:,100:150].sum(1)))),
          'slow_segment_sum_maxabs':float(np.max(np.abs(pa[:,150:400].sum(1)))),
          'unsupported_zero_increment_exact':bool(np.array_equal(project(np.zeros((4,480))),np.zeros((4,480)))),
          'rank':378,'P_uses_target_events_or_fitted_statistics':False}
    rows['pass']=all(rows[k]<1e-10 for k in ['linearity_maxabs','idempotence_maxabs','orthogonality_dot_maxabs','baseline_maxabs','fast_segment_sum_maxabs','slow_segment_sum_maxabs']) and rows['unsupported_zero_increment_exact']
    assert rows['pass'];return rows
