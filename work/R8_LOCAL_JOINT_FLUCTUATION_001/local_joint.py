from __future__ import annotations
import math
import torch
from model import observables
REGIONS=[(100,150),(150,220),(220,400)]

def vector(v,a,b,scalar=None):
    if scalar is None:scalar,_,_=observables(v)
    s=v[:,a:b];center=s-s.mean(dim=1,keepdim=True);rms=center.square().mean(dim=1).sqrt().clamp_min(1e-6)
    z=center/rms[:,None]
    return torch.cat([scalar[:,[1,0,2]],torch.log(rms)[:,None],z,z.diff(dim=1),z.diff(n=2,dim=1)],dim=1)

def fit_targets(real,seed):
    g=torch.Generator().manual_seed(seed);targets=[];scalar,_,_=observables(real)
    for a,b in REGIONS:
        v=vector(real,a,b,scalar);length=b-a;dims=[4,length,length-1,length-2]
        raw=torch.quantile(v,.75,dim=0)-torch.quantile(v,.25,dim=0)
        floors=torch.full_like(raw,.05);floors[:4]=torch.tensor([.1,.05,.1,.1])
        iqr=torch.maximum(raw,floors)
        block=torch.cat([torch.full((n,),1/math.sqrt(n)) for n in dims])
        proj=torch.randn((v.shape[1],64),generator=g);proj/=torch.linalg.vector_norm(proj,dim=0)[None,:]
        targets.append({'region':[a,b],'mean':v.mean(dim=0).detach(),'IQR':iqr.detach(),'raw_IQR':raw.detach(),'IQR_floor':floors,'block_factor':block,'projection':proj,
                        'floor_affected_count':int((raw<floors).sum()),'floor_affected_per_block':[int(x.sum()) for x in (raw<floors).split(dims)],'block_dims':dims,'fit_n':len(real)})
    return targets

def loss(fake,real,targets):
    parts=[];fs,_,_=observables(fake);rs,_,_=observables(real)
    for t in targets:
        a,b=t['region'];f=(vector(fake,a,b,fs)-t['mean'])/t['IQR']*t['block_factor'];r=(vector(real,a,b,rs)-t['mean'])/t['IQR']*t['block_factor']
        fp=f@t['projection'];rp=r@t['projection'];parts.append((fp.sort(dim=0).values-rp.sort(dim=0).values).abs().mean())
    return torch.stack(parts).mean(),{'joint_'+str(a)+'_'+str(b):p for (a,b),p in zip(REGIONS,parts)}
