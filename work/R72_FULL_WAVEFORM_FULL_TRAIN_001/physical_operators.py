"""Frozen R65 Box and ADC operators, verbatim AST extraction."""
import torch

def q_torch(x):return x[...,100:400].sum(-1)-3*x[...,:100].sum(-1)

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
