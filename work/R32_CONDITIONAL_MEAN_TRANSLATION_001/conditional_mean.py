from __future__ import annotations
import numpy as np, torch

def off_diagonal(d):
    n=d.shape[0]
    if n<2:
        z=d.sum()*0
        return z,z,z
    square=d.mean(0).square().mean()
    correction=d.var(0,unbiased=True).mean()/n
    value=(d.sum(0).square()-d.square().sum(0)).sum()/(d.shape[1]*n*(n-1))
    return value,square,correction

class ConditionalMean:
    def __init__(self,mu,scale,device):
        self.mu=torch.as_tensor(mu,dtype=torch.float32,device=device)
        self.scale=torch.as_tensor(scale,dtype=torch.float32,device=device)
        assert self.mu.shape==(8,480) and self.scale.shape==(480,)
        assert torch.isfinite(self.mu).all() and torch.isfinite(self.scale).all() and (self.scale>0).all()
        assert not self.mu.requires_grad and not self.scale.requires_grad
    def __call__(self,y,bins):
        s=y-y[:,:100].mean(1,keepdim=True)
        u=s/s.amax(1,keepdim=True).clamp_min(.0025)
        b=torch.as_tensor(np.asarray(bins,dtype='i8'),device=y.device)
        value=y.sum()*0;square=value;correction=value;singletons=0
        for k in torch.unique(b).tolist():
            if k<0 or k>=8:raise ValueError('actual source bin outside frozen support')
            use=b==k;n=int(use.sum())
            if n<2:singletons+=n;continue
            d=(u[use]-self.mu[k])/self.scale
            v,m,c=off_diagonal(d);weight=n/len(y)
            value=value+weight*v;square=square+weight*m;correction=correction+weight*c
        return value,square,correction,singletons

def selfcheck():
    # Deterministic artificial arithmetic; does not consume scientific RNG streams.
    d=torch.tensor([[.2,-.4,.7],[1.1,.3,-.8],[-.9,.6,.1],[.4,-.2,-.5]],dtype=torch.float64,requires_grad=True)
    a,m,c=off_diagonal(d)
    loop=sum((d[i]*d[j]).sum() for i in range(len(d)) for j in range(len(d)) if i!=j)/(d.shape[1]*len(d)*(len(d)-1))
    ga=torch.autograd.grad(a,d,retain_graph=True)[0]
    gl=torch.autograd.grad(loop,d,retain_graph=True)[0]
    algebra=(m-c)
    v1,m1,c1=off_diagonal(d[:1]);g1=torch.autograd.grad(v1,d,retain_graph=True)[0]
    # Original mixed rule: sixteen+fifteen+one rows, last contributes zero.
    x=torch.arange(96,dtype=torch.float64).reshape(32,3)/50-1
    x.requires_grad_(True)
    v16=off_diagonal(x[:16])[0];v15=off_diagonal(x[16:31])[0]
    mixed=16/32*v16+15/32*v15
    mixed_loop=sum(sum((x[i]*x[j]).sum() for i in ids for j in ids if i!=j)/(3*len(ids)*(len(ids)-1))*len(ids)/32 for ids in [list(range(16)),list(range(16,31))])
    assert float((a-loop).abs())<1e-12 and float((a-algebra).abs())<1e-12
    assert float((ga-gl).abs().max())<1e-12 and torch.equal(g1,torch.zeros_like(d))
    assert float((mixed-mixed_loop).abs())<1e-12 and (ga.abs().sum(1)>0).all()
    return {'explicit_loop_maxabs':float((a-loop).abs().detach()),'algebra_maxabs':float((a-algebra).abs().detach()),
        'gradient_maxabs':float((ga-gl).abs().max()),'both_generated_positions_differentiate':True,
        'singleton_value_and_gradient_zero':True,'mixed_16_15_1_weight_maxabs':float((mixed-mixed_loop).abs().detach()),
        'finite_negative_value':float(a.detach()),'negative_not_clamped':float(a.detach())<0,'constant_requires_grad':False,'response_updates':0}
