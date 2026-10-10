from __future__ import annotations
import numpy as np, torch
from network import FlowNetwork
from noise_core import q_np,innovation,bins_for,variance100

def q_torch(x):return x[...,100:400].sum(-1)-3.*x[...,:100].sum(-1)
def project(x,mu):return x-mu*q_torch(x)[...,None]

class TangentField:
    def __init__(self,network,mu,scale,guide):
        self.network=network;self.mu=mu;self.scale=scale;self.guide=guide;self.calls=0;self.rows=0
    def __call__(self,y,t,b):
        self.calls+=1;self.rows+=len(y)
        f=self.network(y,t,b,self.guide).to(torch.float64)
        return project(self.scale[b]*f,self.mu[b])/self.scale[b]

def heun(field,y,b,steps):
    h=1./steps
    for j in range(steps):
        t=y.new_full((len(y),),j*h)
        k1=field(y,t,b);k2=field(y+h*k1,t+h,b)
        y=y+(h*.5)*(k1+k2)
    return y

def formula_check():
    rng=np.random.Generator(np.random.PCG64(48));mu=rng.normal(size=(4,480))
    mu[:,100:400]+=.1;mu=mu/q_np(mu)[:,None]
    mt=torch.from_numpy(mu);v=torch.from_numpy(rng.normal(size=(4,480))).requires_grad_(True)
    p=project(v,mt);e=float(q_torch(p).abs().max());assert e<1e-10
    p.square().mean().backward();assert v.grad is not None and torch.isfinite(v.grad).all() and torch.count_nonzero(v.grad)>0
    a=torch.zeros((4,480),dtype=torch.float64);b=torch.arange(4)
    out=heun(lambda y,t,b:p.detach(),a,b,64);diff=float((out-p.detach()).abs().max());assert diff<1e-11
    return {'artificial_only':True,'q_projection_maxabs':e,'constant_Heun_maxabs':diff,
            'autograd_finite_nonzero':True,'no_fitted_response':True}

@torch.no_grad()
def solve(network,source,epsilon,mu,scale,edges,noise,completion,steps=64,chunk=64,details=True):
    x=np.asarray(source,dtype='f4');epsilon=np.asarray(epsilon,dtype='f4')
    if x.shape!=epsilon.shape or x.shape[1]!=480:raise ValueError('input/epsilon shape')
    b,raw,q=bins_for(x,edges)
    dev=next(network.parameters()).device
    mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    parts=[];meta={};max64=max32=0.;calls=rows=0
    for j in range(0,len(x),chunk):
        sl=slice(j,min(j+chunk,len(x)))
        n,ni=innovation(x[sl],epsilon[sl],mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),completion)
        bt=torch.as_tensor(b[sl],dtype=torch.int64,device=dev);qt=torch.as_tensor(q[sl],device=dev)
        xt=torch.as_tensor(x[sl],dtype=torch.float64,device=dev)
        g=(xt/qt[:,None]-mt[bt])/st[bt]
        a=(torch.as_tensor(x[sl].astype('f8')+n,device=dev)/qt[:,None]-mt[bt])/st[bt]
        field=TangentField(network,mt,st,g)
        y=heun(field,a,bt,steps);out=qt[:,None]*(mt[bt]+st[bt]*y);pp=out.to(torch.float32)
        e64=(q_torch(out)-qt).abs()/qt;e32=(q_torch(pp.to(torch.float64))-qt).abs()/qt
        max64=max(max64,float(e64.max()));max32=max(max32,float(e32.max()))
        if not torch.isfinite(out).all() or max32>1e-5:raise FloatingPointError(('preADC charge error',max64,max32))
        parts.append(pp.cpu().numpy());calls+=field.calls;rows+=field.rows
        if details:
            data=dict(ni,preADC_charge_relative_error_f64=e64.cpu().numpy(),preADC_charge_relative_error_f32=e32.cpu().numpy(),
                initial_variance100_actual_V2=variance100(x[sl].astype('f8')+n),
                innovation_variance100_actual_V2=variance100(n),
                initial_q_relative_error=np.abs(q_np(x[sl].astype('f8')+n)-q[sl])/q[sl],
                guide_q_unit_residual=np.abs(q_np(x[sl].astype('f8')/q[sl,None])-1),
                neural_flow_applied_all_rows=np.ones(len(n),bool))
            for k,v in data.items():meta.setdefault(k,[]).append(v)
    return np.concatenate(parts),{k:np.concatenate(v)for k,v in meta.items()},dict(field_calls=calls,field_forward_rows=rows,
        max_relative_q_error_f64=max64,max_relative_q_error_f32=max32,steps=steps,chunk=chunk,
        fixed_guide=True,completion=bool(completion),all_rows_new_flow=True)
