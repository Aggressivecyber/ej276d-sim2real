"""Artificial full-population objective, complete VJP and Adam checks only."""
import copy
import numpy as np
import torch
from public_charge_head import PublicConditionalChargeHead,window_charge
from physical_operators import canonical_REAL_torch
from waveform_response import ShapeResidualNetwork,prepare_static,slice_prepared,forward_prepared
from full_waveform_loss import FullWaveformLoss
from precision_order_reference import rounding_psd_interval,psd_np

def fixture(device,seed):
    rng=np.random.Generator(np.random.PCG64(seed));edgeq=np.array([.1,.2,.4,.8,1.6,3.2,6.4,12.8,25.6]);edges=np.log(1000*edgeq)
    levels=np.r_[.06,np.sqrt(edgeq[:-1]*edgeq[1:]),36.]
    def make(repeats):
        q=np.repeat(levels,repeats);n=len(q);base=rng.normal(0,3e-5,(n,480));t=np.arange(300)
        decay=85+np.arange(n)%4*7;profile=np.exp(-t[None,:]/decay[:,None]);profile/=profile.sum(1,keepdims=True)
        base[:,100:400]+=q[:,None]*profile
        return base.astype('f4')
    source=make(2);real=make(3);nS=len(source)
    tensor=lambda v:torch.as_tensor(v,device=device,dtype=torch.float64)
    x=tensor(np.concatenate([source,real]));mu=np.zeros((8,480));profile=np.exp(-np.arange(300)/90);profile/=profile.sum()
    mu[:,100:400]=profile;scale=np.full((8,480),.001);p=np.linspace(.2,1.4,241);tau2=8e-9
    head=PublicConditionalChargeHead(tensor(np.zeros(480)),tensor(np.ones(480)*.01)).to(device).double();head.requires_grad_(False)
    prepared=prepare_static(head,x,*[tensor(v)for v in [mu,scale,edges,p]],tau2);feature=prepared['features482'][nS:]
    torch.manual_seed(seed);shape=ShapeResidualNetwork(feature.mean(0),feature.std(0,unbiased=False).clamp_min(1e-6)).to(device)
    with torch.no_grad():
        shape.output.weight.copy_(tensor(rng.normal(0,.003,(480,128))).to(torch.float32))
        shape.output.bias.copy_(tensor(rng.normal(0,.002,480)).to(torch.float32))
    lo,hi=rounding_psd_interval(source);h=np.ones(nS);h[::2]=.4
    cfg=dict(PSD_loss_scale=.01,valley_delta=.02,valley_smoothing=.002,
        feature_scale_floor={'V480':1/16383,'U480':.001,'DU479':.001,'D2U478':.001},
        noise_scale_floor=[.1/16383,.01],projection_directions=64,temporal_feature_count=31,
        temporal_projection_directions=64,temporal_scale_floor=[.1/16383]+[.01]*30,temporal_power_floor=1e-20,
        loss_weights={k:1. for k in ['full_window','coordinate_RMS','PSD','temporal_joint','noise','source_order','source_valley','REAL_identity','box_correction']},
        loss_feature_chunk=64)
    obj=FullWaveformLoss(canonical_REAL_torch(x[nS:]),tensor(edges),tensor(psd_np(source)),tensor(lo),tensor(hi),prepared['original_bin_raw'][:nS],tensor(h),tensor([[.3,.45],[.3,.45]]),cfg)
    eps=torch.as_tensor(rng.standard_normal(x.shape).astype('f4'),device=device)
    dirs=[]
    for dim in [480,480,479,478,31]:
        a=rng.standard_normal((dim,64));a/=np.linalg.norm(a,axis=0,keepdims=True);dirs.append(torch.as_tensor(a.astype('f4'),device=device))
    return shape,prepared,eps,nS,obj,dirs[:4],dirs[4]

def joint_check(device,seed):
    shape,prepared,eps,nS,obj,dirs,temporal=fixture(device,seed);N=len(eps);chunk=13
    initial={k:v.detach().clone()for k,v in shape.state_dict().items()};params=list(shape.parameters())
    def forward():
        collections=[[],[],[]]
        for begin in range(0,N,chunk):
            pre,adc,raw,_=forward_prepared(shape,slice_prepared(prepared,begin,min(N,begin+chunk)),eps[begin:begin+chunk])
            for j,v in enumerate([adc,pre,raw]):collections[j].append(v)
        return [torch.cat(v)for v in collections]
    # Full graph and common chunks form the exact declared numerical reference.
    actual=forward();args=[v[sl]for v in actual for sl in [slice(0,nS),slice(nS,None)]]
    total,terms,_=obj(*args,dirs,temporal);direct=torch.autograd.grad(total,params)
    reference_value=float(total.detach());del actual,args,total,terms
    with torch.no_grad():saved=forward()
    leaves=[v[sl].detach().requires_grad_(True)for v in saved for sl in [slice(0,nS),slice(nS,None)]]
    accum=[torch.zeros_like(v,dtype=torch.float64)for v in leaves];observed=0.;names=[]
    for name,value,_ in obj.iter_terms(*leaves,dirs,temporal):
        names.append(name);observed+=float(value.detach());g=torch.autograd.grad(value,leaves,allow_unused=True)
        for a,b in zip(accum,g):
            if b is not None:a.add_(b.to(torch.float64))
    assert len(names)==len(set(names))==9
    gfull=[torch.cat([accum[2*j],accum[2*j+1]])for j in range(3)];replayed=[torch.zeros_like(p,dtype=torch.float64)for p in params]
    rows=0;chunks=0
    for begin in range(0,N,chunk):
        end=min(N,begin+chunk);pre,adc,raw,_=forward_prepared(shape,slice_prepared(prepared,begin,end),eps[begin:end]);outputs=[adc,pre,raw]
        assert all(torch.equal(v.detach(),saved[j][begin:end])for j,v in enumerate(outputs))
        gradient=torch.autograd.grad(outputs,params,grad_outputs=[g[begin:end].to(v.dtype)for g,v in zip(gfull,outputs)])
        for a,b in zip(replayed,gradient):a.add_(b.to(torch.float64))
        rows+=end-begin;chunks+=1
    candidate=[g.to(p.dtype)for g,p in zip(replayed,params)]
    flat=lambda values:torch.cat([v.detach().reshape(-1).to(torch.float64)for v in values])
    a,b=flat(direct),flat(candidate);delta=a-b
    relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a).clamp_min(1e-30));absolute=float(delta.abs().max())
    value_error=abs(reference_value-observed);assert value_error<=1e-7*max(abs(reference_value),1.),value_error
    assert relative<=1e-5,(relative,absolute)
    assert rows==N and chunks==4 and N%chunk!=0
    # Compare one clipped Adam update as well, including every original parameter.
    opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999))
    for p,g in zip(params,direct):p.grad=g.clone()
    torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();expected=flat(params).clone()
    shape.load_state_dict(initial);opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999))
    for p,g in zip(params,candidate):p.grad=g.clone()
    torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();step_error=float((expected-flat(params)).abs().max())
    assert step_error<=2e-6,step_error
    assert all(not p.requires_grad for p in prepared.values())
    return dict(device=str(device),artificial_only=True,rows=N,SOURCE_N=nS,REAL_N=N-nS,nondivisible_tail_checked=True,
      objective_terms=names,value_error=value_error,parameter_gradient_relative_L2=relative,parameter_gradient_maxabs=absolute,
      Adam_parameter_maxabs_difference=step_error,firstpass_replay_bitwise=True,all_rows_parameter_VJP=True,pass_gate=True)

def run_checks(seed):
    from check_waveform_response import run_artificial as response_check
    from check_full_waveform_loss import run_checks as loss_check
    loss=loss_check(require_cuda=True,require_numba=True)
    assert loss['status']=='PASS_ARTIFICIAL_ONLY'and loss['CUDA']=='passed'
    result={}
    for device in ['cpu','cuda']:
        response=response_check(device);assert response['pass_all']
        result[device]=dict(response=response,joint=joint_check(device,seed))
    return dict(pass_all=True,artificial_only=True,science_arrays_decoded=0,checks=result,loss=loss)
