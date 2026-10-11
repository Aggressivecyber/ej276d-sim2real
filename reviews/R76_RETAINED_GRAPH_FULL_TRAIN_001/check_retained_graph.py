"""Artificial unchanged-law check of retained versus repeated full forward."""
import copy
import torch
from artificial_checks import fixture
from waveform_response import forward_prepared,slice_prepared


def check(device,seed,chunk):
    model,prepared,epsilon,nS,objective,directions,td=fixture(device,seed)
    params=list(model.parameters());initial=copy.deepcopy(model.state_dict());n=len(epsilon)
    def forward():
        found=[[],[],[]]
        for start in range(0,n,chunk):
            pre,adc,raw,_=forward_prepared(model,slice_prepared(prepared,start,min(start+chunk,n)),epsilon[start:start+chunk],straight_through=True)
            for col,value in zip(found,[adc,pre,raw]):col.append(value)
        return [torch.cat(col)for col in found]
    with torch.no_grad():saved=forward()
    retained=forward()
    assert all(torch.equal(a,b.detach())for a,b in zip(saved,retained))
    leaves=[v[sl].detach().requires_grad_(True)for v in saved for sl in [slice(0,nS),slice(nS,None)]]
    accum=[torch.zeros_like(v,dtype=torch.float64)for v in leaves];values={}
    for name,term,_ in objective.iter_terms(*leaves,directions,td):
        gradient=torch.autograd.grad(term,leaves,allow_unused=True)
        for a,g in zip(accum,gradient):
            if g is not None:a.add_(g.detach().to(torch.float64))
        values[name]=float(term.detach())
    assert len(values)==9
    allg=[torch.cat(accum[2*j:2*j+2])for j in range(3)]
    repeated=forward();assert all(torch.equal(a,b.detach())for a,b in zip(saved,repeated))
    ga=torch.autograd.grad(repeated,params,grad_outputs=[g.to(v.dtype)for g,v in zip(allg,repeated)])
    gb=torch.autograd.grad(retained,params,grad_outputs=[g.to(v.dtype)for g,v in zip(allg,retained)])
    flat=lambda vs:torch.cat([v.detach().flatten().to(torch.float64)for v in vs])
    a,b=flat(ga),flat(gb)
    relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30))
    absolute=float((a-b).abs().max());assert relative<=1e-6
    after=[]
    for gradients in [ga,gb]:
        model.load_state_dict(initial);opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999))
        for p,g in zip(params,gradients):p.grad=g.clone()
        torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();after.append(flat(params).clone())
    adam=float((after[0]-after[1]).abs().max());assert adam<=1e-6
    return dict(device=str(device),rows=n,source_N=nS,REAL_N=n-nS,chunk=chunk,pass_all=True,
        equal_saved_endpoints=True,nine_unchanged_terms=True,parameter_gradient_relative_L2=relative,
        parameter_gradient_maxabs=absolute,Adam_maxabs=adam,scientific_arrays_decoded=False)


def run(seed,require_cuda=False):
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    devices=['cpu']+(['cuda']if torch.cuda.is_available()else[])
    return dict(pass_all=True,checks=[check(device,seed,chunk)for device in devices for chunk in [50,13]])
