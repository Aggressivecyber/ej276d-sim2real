from __future__ import annotations
import numpy as np,torch
from network import FlowNetwork

def q_np(x):
    a=np.asarray(x,dtype='f8');return a[...,100:400].sum(-1)-3.*a[...,:100].sum(-1)
def q_torch(x):return x[...,100:400].sum(-1)-3.*x[...,:100].sum(-1)
def project(x,mu):return x-mu*q_torch(x)[...,None]
def initial(source,white,mu,scale):
    x=source.to(torch.float64);w=white.to(torch.float64);q=q_torch(x)
    if not torch.isfinite(x).all()or not torch.isfinite(q).all()or not(q>0).all():raise ValueError('nonpositive/nonfinite source charge')
    gaussian=project(.001*w/q[:,None],mu)
    w0=x/q[:,None]+gaussian
    return (w0-mu)/scale,q,gaussian
class TangentField:
    def __init__(self,network,mu,scale):self.network=network;self.mu=mu;self.scale=scale;self.calls=0;self.rows=0
    def __call__(self,y,t,b):
        self.calls+=1;self.rows+=len(y)
        mu=self.mu[b];s=self.scale[b];f=self.network(y,t,b).to(torch.float64)
        return project(s*f,mu)/s
def heun(field,y,b,steps):
    h=1./steps
    for j in range(steps):
        t=y.new_full((len(y),),j*h);k1=field(y,t,b);k2=field(y+h*k1,t+h,b)
        y=y+(h*.5)*(k1+k2)
    return y
def formula_check():
    rng=np.random.Generator(np.random.PCG64(42));mu=rng.normal(size=(4,480));mu[:,100:400]+=.02
    qm=q_np(mu);mu=mu/qm[:,None];v=rng.normal(size=(4,480))
    mt=torch.from_numpy(mu);vt=torch.from_numpy(v);p=project(vt,mt)
    err=float(q_torch(p).abs().max());assert err<1e-10
    x=torch.zeros((4,480),dtype=torch.float64);b=torch.arange(4)
    got=heun(lambda y,t,b:p,x,b,32);e=float((got-p).abs().max());assert e<1e-12
    return {'small_artificial_only':True,'q_projection_maxabs':err,'constant_velocity_Heun_maxabs':e,
        'field_projection_preserves_gradients':'static differentiable construction; no separate autograd test','coordinate_arithmetic':'float64; NN float32','new_response_fits':0}

@torch.no_grad()
def solve(network,source,white,mu,scale,edges,steps=32,chunk=64):
    source=np.asarray(source,dtype='f4');white=np.asarray(white,dtype='f4');q=q_np(source)
    if not np.isfinite(q).all()or not(q>0).all():raise ValueError('nonfinite or nonpositive full source q')
    logq=np.log(1000.*q);raw=np.searchsorted(edges,logq,side='right')-1;raw[logq==edges[-1]]=7;condition=np.clip(raw,0,7)
    device=next(network.parameters()).device;mt=torch.as_tensor(mu,dtype=torch.float64,device=device);st=torch.as_tensor(scale,dtype=torch.float64,device=device)
    details={};parts=[];max64=0.;max32=0.;calls=0;rows=0
    for j in range(0,len(source),chunk):
        sl=slice(j,min(j+chunk,len(source)));bt=torch.as_tensor(condition[sl],dtype=torch.int64,device=device)
        x=torch.as_tensor(source[sl],device=device);w=torch.as_tensor(white[sl],device=device)
        a,qt,g=initial(x,w,mt[bt],st[bt]);field=TangentField(network,mt,st);y=heun(field,a,bt,steps)
        out=qt[:,None]*(mt[bt]+st[bt]*y);pre=out.to(torch.float32)
        e64=(q_torch(out)-qt).abs()/qt; e32=(q_torch(pre.to(torch.float64))-qt).abs()/qt
        max64=max(max64,float(e64.max()));max32=max(max32,float(e32.max()))
        if not torch.isfinite(out).all()or max32>1e-5:raise FloatingPointError(('preADC charge/nonfinite failure',max64,max32))
        noise=(qt[:,None]*g).cpu().numpy();base64=source[sl].astype('f8')+noise;base=base64.astype('f4');pp=pre.cpu().numpy();o=out.cpu().numpy()
        row={'base_pre_ADC_V':base,'base_signal_V':source[sl].copy(),'base_electronic_V':noise.astype('f4'),
            'base_sum_rounding_f64_V':base.astype('f8')-source[sl].astype('f8')-noise.astype('f4').astype('f8'),
            'flow_increment_V':(pp.astype('f8')-base.astype('f8')).astype('f4'),'flow_increment_f64_V':pp.astype('f8')-base.astype('f8'),
            'projected_initial_Gaussian_f64_V':noise,'initial_state_f64_V':base64,'learned_flow_increment_f64_V':o-base64,
            'float64_final_preADC_V':o,'final_cast_rounding_f64_V':pp.astype('f8')-o,
            'inside_TRAIN_support':((raw[sl]>=0)&(raw[sl]<8)),'source_Qbin_actual_preADC':raw[sl],'conditioning_Qbin':condition[sl],
            'neural_flow_applied_all_rows':np.ones(len(pp),bool),'source_q_Vns':q[sl],'preADC_charge_relative_error_f64':e64.cpu().numpy(),'preADC_charge_relative_error_f32':e32.cpu().numpy()}
        parts.append(pp)
        for k,v in row.items():details.setdefault(k,[]).append(v)
        calls+=field.calls;rows+=field.rows
    return np.concatenate(parts),{k:np.concatenate(v)for k,v in details.items()},{'field_calls':calls,'field_forward_rows':rows,'max_relative_q_error_f64':max64,'max_relative_q_error_f32':max32,'steps':steps,'chunk':chunk}
