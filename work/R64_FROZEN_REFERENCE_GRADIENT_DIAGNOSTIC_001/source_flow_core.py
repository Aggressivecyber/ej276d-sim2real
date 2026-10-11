from __future__ import annotations
import numpy as np
import torch
from network import FlowNetwork
from noise_core import q_np,innovation,bins_for,variance100,continuous_logq

def q_torch(x):return x[...,100:400].sum(-1)-3*x[...,:100].sum(-1)
def project(x,mu):return x-mu*q_torch(x)[...,None]
def charge_vector(x):
    a=x.new_zeros(480);a[:100]=-3;a[100:400]=1
    return a

class BoxChargeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v,q):
        if v.dtype!=torch.float64 or q.dtype!=torch.float64:raise TypeError('constraint construction requires float64')
        if not torch.isfinite(v).all() or not torch.isfinite(q).all():raise FloatingPointError('nonfinite projection input')
        if (q.abs()>300).any():raise ValueError('absolute ADC charge constraint infeasible; no event deletion')
        a=charge_vector(v);answer=v.clone()
        change=((v<-.5)|(v>.5)).any(1)|(torch.abs(q_torch(v)-q)>1e-12)
        if change.any():
            u=v[change];qq=q[change];aa=a[:400]
            lower=torch.where(aa>0,(u[:,:400]-.5)/aa,(u[:,:400]+.5)/aa).amin(1)
            upper=torch.where(aa>0,(u[:,:400]+.5)/aa,(u[:,:400]-.5)/aa).amax(1)
            for _ in range(64):
                mid=.5*(lower+upper);s=q_torch(torch.clamp(u-mid[:,None]*a,-.5,.5))
                lower=torch.where(s>qq,mid,lower);upper=torch.where(s>qq,upper,mid)
            answer[change]=torch.clamp(u-.5*(lower+upper)[:,None]*a,-.5,.5)
        free=(answer>-.5)&(answer<.5);ctx.save_for_backward(free,a)
        return answer
    @staticmethod
    def backward(ctx,gradient):
        free,a=ctx.saved_tensors;active=free.to(gradient);den=(active*a.square()).sum(1)
        coefficient=(gradient*active*a).sum(1)/den.clamp_min(1.)
        gv=active*(gradient-coefficient[:,None]*a)
        gq=torch.where(den>0,coefficient,torch.zeros_like(coefficient))
        return gv,gq

def bounded_charge(v,q):return BoxChargeProjection.apply(v,q)

def canonical_adc_torch(v,straight_through=False):
    # Match the existing float32 adc() output, then R51's common float64 codes.
    f=v.to(torch.float32);legacy=(-.5+torch.round((f.clamp(-.5,.5)+.5)*16383)/16383).to(torch.float32)
    canonical=-.5+torch.round((legacy.to(torch.float64)+.5)*16383).clamp(0,16383)/16383
    return v+(canonical-v).detach() if straight_through else canonical

def canonical_REAL_torch(v):return -.5+torch.round((v.to(torch.float64)+.5)*16383).clamp(0,16383)/16383

def constant_heun(initial,displacement,steps):
    if steps not in [128,256]:raise ValueError('only the preregistered candidate/reference steps are permitted')
    y=initial.clone();h=1./steps
    for _ in range(steps):
        k1=displacement;k2=displacement
        y=y+(h*.5)*(k1+k2)
    return y

def displacement(network,source,noise,q,b,condition,mu,scale):
    guide=(source/q[:,None]-mu[b])/scale[b]
    initial=source+noise;noisy=(initial/q[:,None]-mu[b])/scale[b]
    f=network(noisy,b,guide,condition).to(torch.float64)
    delta=q[:,None]*project(scale[b]*f,mu[b])
    return initial,delta

def endpoint(network,source,noise,q,b,condition,mu,scale):
    initial,delta=displacement(network,source,noise,q,b,condition,mu,scale)
    raw=initial+delta;out=bounded_charge(raw,q)
    return out,raw

@torch.no_grad()
def solve(network,source,epsilon,mu,scale,edges,noise,completion,steps=128,chunk=64,details=True):
    x=np.asarray(source,dtype='f4');epsilon=np.asarray(epsilon,dtype='f4')
    if x.shape!=epsilon.shape or x.ndim!=2 or x.shape[1]!=480:raise ValueError('input/epsilon shape')
    b,rawbin,q=bins_for(x,edges);charge_condition=continuous_logq(q,b,edges)
    dev=next(network.parameters()).device;mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    parts=[];meta={};max64=max32=0.;calls=rows=net_calls=0;projection_events=0
    for j in range(0,len(x),chunk):
        sl=slice(j,min(j+chunk,len(x)));n,ni=innovation(x[sl],epsilon[sl],mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),completion)
        bt=torch.as_tensor(b[sl],dtype=torch.int64,device=dev);qt=torch.as_tensor(q[sl],device=dev)
        xt=torch.as_tensor(x[sl],dtype=torch.float64,device=dev);nt=torch.as_tensor(n,device=dev);ht=torch.as_tensor(charge_condition[sl],device=dev)
        initial,delta=displacement(network,xt,nt,qt,bt,ht,mt,st)
        preconstraint=constant_heun(initial,delta,steps);out=bounded_charge(preconstraint,qt);pp=out.to(torch.float32)
        e64=(q_torch(out)-qt).abs()/qt;e32=(q_torch(pp.to(torch.float64))-qt).abs()/qt
        max64=max(max64,float(e64.max()));max32=max(max32,float(e32.max()))
        if not torch.isfinite(out).all() or max32>1e-5:raise FloatingPointError(('preADC charge error',max64,max32))
        if ((pp<-.5)|(pp>.5)).any():raise FloatingPointError('bounded output exceeded ADC range')
        excursion=((preconstraint<-.5)|(preconstraint>.5));correction=out-preconstraint
        parts.append(pp.cpu().numpy());calls+=2*steps;rows+=2*steps*len(pp);net_calls+=1;projection_events+=int(excursion.any(1).sum())
        if details:
            data=dict(ni,continuous_logq_condition=charge_condition[sl],preADC_charge_relative_error_f64=e64.cpu().numpy(),preADC_charge_relative_error_f32=e32.cpu().numpy(),
                initial_variance100_actual_V2=variance100(x[sl].astype('f8')+n),innovation_variance100_actual_V2=variance100(n),
                initial_q_relative_error=np.abs(q_np(x[sl].astype('f8')+n)-q[sl])/q[sl],guide_q_unit_residual=np.abs(q_np(x[sl].astype('f8')/q[sl,None])-1),
                neural_flow_applied_all_rows=np.ones(len(n),bool),pre_constraint_out_of_range_sample_count=excursion.sum(1).cpu().numpy(),
                constraint_projection_range_active=excursion.any(1).cpu().numpy(),constraint_projection_RMS_V=correction.square().mean(1).sqrt().cpu().numpy(),
                constraint_projection_maxabs_V=correction.abs().amax(1).cpu().numpy(),constraint_projection_rail_sample_count=(out.abs()==.5).sum(1).cpu().numpy(),
                pre_constraint_charge_relative_error=((q_torch(preconstraint)-qt).abs()/qt).cpu().numpy())
            for k,v in data.items():meta.setdefault(k,[]).append(v)
    return np.concatenate(parts),{k:np.concatenate(v) for k,v in meta.items()},dict(field_calls=calls,field_forward_rows=rows,actual_network_calls=net_calls,actual_network_rows=len(x),
        cached_time_and_state_constant_displacement=True,max_relative_q_error_f64=max64,max_relative_q_error_f32=max32,steps=steps,chunk=chunk,
        fixed_guide=True,completion=bool(completion),all_rows_new_flow=True,post_integral_box_charge_projection=True,box_projection_iterations=64,range_active_projection_events=projection_events)
