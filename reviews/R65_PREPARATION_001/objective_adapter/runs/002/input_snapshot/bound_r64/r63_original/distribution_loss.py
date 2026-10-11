"""Preregistered R63 coverage extension; frozen R61 seven-term loss is retained."""
import math
import numpy as np
import torch
from R61_distribution_loss import (DistributionLoss as R61DistributionLoss, NAMES,
    CATEGORIES, feature_vectors, fit_statistics, weighted_w1, psd)
from source_flow_core import canonical_adc_torch, canonical_REAL_torch

TEMPORAL_NAMES=['RMS_V']+[f'ACF_lag{k}' for k in range(1,21)]+[f'rFFT_power_bin{1+5*k}_{5+5*k}_fraction' for k in range(10)]

def temporal_features(v,power_floor=1e-20):
    b=v[...,:100]-v[...,:100].mean(-1,keepdim=True)
    power=b.square().mean(-1);safe=power.clamp_min(power_floor)
    acf=torch.stack([(b[...,k:]*b[...,:-k]).mean(-1)/safe for k in range(1,21)],-1)
    spectrum=torch.fft.rfft(b,dim=-1)[...,1:]
    frequency_power=(spectrum.real.square()+spectrum.imag.square())/10000
    multiplicity=frequency_power.new_full((50,),2.);multiplicity[-1]=1.
    bands=(frequency_power*multiplicity).reshape(*b.shape[:-1],10,5).sum(-1)/safe[...,None]
    return torch.cat((safe.sqrt()[...,None],acf,bands),-1)

@torch.no_grad()
def fit_temporal_statistics(real,REAL_bins,cfg):
    v=canonical_REAL_torch(torch.as_tensor(real,dtype=torch.float64,device='cpu'))
    f=temporal_features(v,cfg['temporal_power_floor'])
    floor=torch.as_tensor(cfg['temporal_scale_floor'],dtype=torch.float64)
    assert f.shape==(len(real),31) and len(floor)==31 and torch.isfinite(f).all()
    means=[];scales=[];counts=[];zeros=[]
    power=(v[:,:100]-v[:,:100].mean(1,keepdim=True)).square().mean(1)
    for region in CATEGORIES:
        mask=torch.from_numpy(np.asarray(REAL_bins)==region);selected=f[mask]
        assert len(selected)>0
        means.append(selected.mean(0).numpy());scales.append(torch.maximum(selected.std(0,unbiased=False),floor).numpy())
        counts.append(len(selected));zeros.append(int((power[mask]==0).sum()))
    return {'temporal_center':np.stack(means),'temporal_scale':np.stack(scales),
            'temporal_REAL_counts':np.asarray(counts,dtype='i8'),'temporal_zero_power_counts':np.asarray(zeros,dtype='i8')}

def coordinate_w1_rms(a,b):
    """Equal-size empirical coordinate W1, pooled over all coordinates by RMS."""
    assert a.shape==b.shape and a.ndim==3
    w=(torch.sort(a,dim=1,stable=True).values-torch.sort(b,dim=1,stable=True).values).abs().mean(1)
    return torch.linalg.vector_norm(w,dim=-1)/math.sqrt(w.shape[-1])

def temporal_distribution_terms(a,b,direction):
    marginal=(torch.sort(a,dim=1,stable=True).values-torch.sort(b,dim=1,stable=True).values).abs().mean((1,2))
    sliced=(torch.sort(a@direction,dim=1,stable=True).values-torch.sort(b@direction,dim=1,stable=True).values).abs().mean((1,2))
    return marginal,sliced

class DistributionLoss(R61DistributionLoss):
    def __init__(self,statistics,cfg,device):
        super().__init__(statistics,cfg,device)
        assert cfg['temporal_feature_count']==31 and cfg['temporal_projection_directions']==64
        self.temporal_direction_calls=0
    def draw_temporal_directions(self,rng):
        p=rng.standard_normal((31,self.cfg['temporal_projection_directions']))
        p/=np.linalg.norm(p,axis=0,keepdims=True)
        self.temporal_direction_calls+=1
        return p.astype('<f4')
    def __call__(self,output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions,source_interval_low,source_interval_high,temporal_directions):
        old_total,pieces,diagnostics=super().__call__(output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions,source_interval_low,source_interval_high)
        n=10*self.B;candidate=canonical_adc_torch(output[:n],True);ref=canonical_REAL_torch(reference)
        by_feature=[];by_region=[]
        for name,fy,fr in zip(NAMES,feature_vectors(candidate),feature_vectors(ref)):
            dim=fy.shape[-1];center=self.stats[name+'_center'][:,None,:];scale=self.stats[name+'_scale'][:,None,:]
            a=((fy.reshape(10,self.B,dim)-center)/scale).to(torch.float32)
            b=((fr.reshape(10,self.B,dim)-center)/scale).to(torch.float32)
            regional=coordinate_w1_rms(a,b)
            by_region.append(regional.detach());by_feature.append((regional*self.source_mass).sum())
        pieces['coordinate_RMS']=torch.stack(by_feature).mean()
        center=self.stats['temporal_center'][:,None,:];scale=self.stats['temporal_scale'][:,None,:]
        a=((temporal_features(candidate,self.cfg['temporal_power_floor']).reshape(10,self.B,31)-center)/scale).to(torch.float32)
        b=((temporal_features(ref,self.cfg['temporal_power_floor']).reshape(10,self.B,31)-center)/scale).to(torch.float32)
        p=torch.as_tensor(temporal_directions,device=self.device,dtype=torch.float32)
        marginal,sliced=temporal_distribution_terms(a,b,p)
        m=(marginal*self.source_mass).sum();s=(sliced*self.source_mass).sum()
        pieces['temporal_joint']=.5*(m+s)
        total=old_total+self.cfg['loss_weights']['coordinate_RMS']*pieces['coordinate_RMS']+self.cfg['loss_weights']['temporal_joint']*pieces['temporal_joint']
        diagnostics.update(original_seven_total=old_total.detach(),coordinate_RMS=pieces['coordinate_RMS'].detach(),temporal_joint=pieces['temporal_joint'].detach(),
            temporal_marginal=m.detach(),temporal_sliced=s.detach(),coordinate_RMS_by_region=torch.stack(by_region),temporal_marginal_by_region=marginal.detach(),temporal_sliced_by_region=sliced.detach())
        diagnostics.update({name+'_coordinate_RMS':value.detach() for name,value in zip(NAMES,by_feature)})
        return total,pieces,diagnostics
