from __future__ import annotations
import numpy as np

class InvalidPSDKey(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('invalid raw preADC PSD key; stop registered recipe')

def keys(waveforms):
    v=np.asarray(waveforms,dtype=np.float64)
    assert v.ndim==2 and v.shape[1]==480
    s=v-v[:,:100].mean(1,keepdims=True)
    total=s[:,100:400].sum(1)*1000
    fast=s[:,100:150].sum(1)*1000
    with np.errstate(divide='ignore',invalid='ignore'):
        psd=1-fast/total
    valid=np.isfinite(v).all(1)&np.isfinite(total)&np.isfinite(fast)&np.isfinite(psd)&(total>0)
    return total,fast,psd,valid

def ordered_coupling(source_raw,target_raw,bin_ids,source_affine,target_affine):
    b=np.asarray(bin_ids,dtype=np.int64);n=len(b)
    sq,sf,sp,sv=keys(source_raw);tq,tf,tp,tv=keys(target_raw)
    if not (sv.all() and tv.all()):
        raise InvalidPSDKey({'source_invalid_positions':np.flatnonzero(~sv).tolist(),
            'target_invalid_positions':np.flatnonzero(~tv).tolist(),
            'source_Qtotal_at_invalid':[float(sq[i]) if np.isfinite(sq[i]) else None for i in np.flatnonzero(~sv)],
            'target_Qtotal_at_invalid':[float(tq[i]) if np.isfinite(tq[i]) else None for i in np.flatnonzero(~tv)]})
    matched=np.arange(n,dtype=np.int64);groups=[]
    for k in np.unique(b):
        pos=np.flatnonzero(b==k)
        source_order=pos[np.argsort(sp[pos],kind='stable')]
        target_order=pos[np.argsort(tp[pos],kind='stable')]
        matched[source_order]=target_order
        assert np.all(np.diff(sp[source_order])>=0) and np.all(np.diff(tp[matched[source_order]])>=0)
        groups.append({'bin_id_1based':int(k)+1,'N':len(pos),
            'source_tie_count':len(pos)-len(np.unique(sp[pos])),
            'target_tie_count':len(pos)-len(np.unique(tp[pos])),
            'stable_original_row_tie_rule':True,'one_to_one':True,'no_cross_Q':True,
            'source_PSD_minmax':[float(sp[pos].min()),float(sp[pos].max())],
            'target_PSD_minmax':[float(tp[pos].min()),float(tp[pos].max())],
            'mean_abs_PSD_gap_original':float(np.mean(np.abs(sp[pos]-tp[pos]))),
            'mean_abs_PSD_gap_ordered':float(np.mean(np.abs(sp[pos]-tp[matched[pos]]))),
            'no_sorted_PSD_inversion':True})
    assert np.array_equal(np.sort(matched),np.arange(n)) and np.array_equal(b[matched],b)
    a=np.asarray(source_affine,dtype=np.float64);c=np.asarray(target_affine,dtype=np.float64)
    cost=float(np.mean((a-c[matched])**2))
    return matched,cost,groups,(sq,sf,sp,tq,tf,tp)

def small_check():
    x=np.zeros((7,480),dtype=np.float32)
    x[:,100:150]=np.asarray([.01,.03,.02,.02,.015,.025,.04],dtype='f4')[:,None]
    x[:,150:400]=.01
    y=x[[1,0,3,2,5,4,6]].copy();b=np.array([0,0,1,1,2,2,7])
    p,c,g,_=ordered_coupling(x,y,b,x,y)
    assert np.array_equal(p,[1,0,2,3,5,4,6])
    assert p[6]==6 and all(row['one_to_one'] and row['no_cross_Q'] for row in g)
    assert np.array_equal(np.sort(y[p].sum(1)),np.sort(y.sum(1)))
    zero=x.copy();zero[0]=0
    refused=False
    try:ordered_coupling(zero,y,b,zero,y)
    except InvalidPSDKey as e:refused=e.details['source_invalid_positions']==[0]
    assert refused
    return {'artificial_rows':7,'stable_ties_mixed_Q_singleton':True,
        'source_original_order_and_target_row_marginal_preserved':True,'invalid_key_stops':True,
        'new_formula_model_fits':0,'no_original_flow_math_retest':True}
