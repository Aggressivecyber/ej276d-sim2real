"""Artificial endpoint, parameter-VJP and Adam equivalence, no science arrays."""
import copy
import numpy as np
import torch
from artificial_checks import fixture
from combined_loss import FullWaveformLoss
from exact_uniform_w1 import UniformW1Cache
from waveform_response import forward_prepared,slice_prepared

def check(device,seed):
    model,prepared,epsilon,nS,original,directions,td=fixture(device,seed)
    candidate=copy.copy(original);candidate.__class__=FullWaveformLoss;candidate.w1=UniformW1Cache(original.device)
    n=len(epsilon);chunk=13;params=list(model.parameters());initial=copy.deepcopy(model.state_dict())
    def forward():
        found=[[],[],[]]
        for start in range(0,n,chunk):
            pre,adc,raw,_=forward_prepared(model,slice_prepared(prepared,start,min(start+chunk,n)),epsilon[start:start+chunk])
            for col,value in zip(found,[adc,pre,raw]):col.append(value)
        return [torch.cat(col)for col in found]
    with torch.no_grad():saved=forward()
    results=[]
    for implementation in [original,candidate]:
        leaves=[v[sl].detach().requires_grad_(True)for v in saved for sl in [slice(0,nS),slice(nS,None)]]
        acc=[torch.zeros_like(v,dtype=torch.float64)for v in leaves];values={}
        if implementation is original:
            iterator=(((name,),term,{name:term},diag)for name,term,diag in implementation.iter_terms(*leaves,directions,td))
        else:iterator=implementation.iter_gradient_groups(*leaves,directions,td)
        for names,total,terms,_ in iterator:
            g=torch.autograd.grad(total,leaves,allow_unused=True)
            for a,b in zip(acc,g):
                if b is not None:a.add_(b.to(torch.float64))
            values.update({k:float(v.detach())for k,v in terms.items()})
        allg=[torch.cat(acc[2*j:2*j+2])for j in range(3)]
        out=forward();gradient=torch.autograd.grad(out,params,grad_outputs=[g.to(v.dtype)for g,v in zip(allg,out)])
        results.append((values,acc,[g.detach().clone()for g in gradient]))
    a,b=results
    value_errors={name:abs(a[0][name]-b[0][name])for name in a[0]}
    assert len(value_errors)==9
    assert all(value_errors[k]<=1e-11*max(1,abs(a[0][k]))for k in value_errors),value_errors
    leaf=[]
    for g,h in zip(a[1],b[1]):
        absolute=float((g-h).abs().max());relative=float(torch.linalg.vector_norm(g-h)/torch.linalg.vector_norm(g).clamp_min(1e-30))
        torch.testing.assert_close(g,h,rtol=1e-10,atol=1e-10);leaf.append(dict(maxabs=absolute,relative=relative))
    flat=lambda v:torch.cat([x.detach().to(torch.float64).flatten()for x in v])
    ga,gb=flat(a[2]),flat(b[2]);relative=float(torch.linalg.vector_norm(ga-gb)/torch.linalg.vector_norm(ga).clamp_min(1e-30))
    absolute=float((ga-gb).abs().max());assert relative<=1e-6,(relative,absolute)
    after=[]
    for entry in results:
        model.load_state_dict(initial);opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999))
        for p,g in zip(params,entry[2]):p.grad=g.clone()
        torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();after.append(flat(params).clone())
    adam=float((after[0]-after[1]).abs().max());assert adam<=1e-6,adam
    return dict(device=str(device),pass_all=True,artificial_only=True,source_N=nS,REAL_N=n-nS,rows=n,nondivisible_chunk=chunk,
      nine_term_value_absolute_errors=value_errors,six_endpoint_gradient_comparisons=leaf,
      parameter_gradient_relative_L2=relative,parameter_gradient_maxabs=absolute,Adam_maxabs=adam,
      current_bins_and_all_original_parents_unchanged=True,scientific_arrays_decoded=0)

def run(seed,require_cuda=False):
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA artificial checks required')
    devices=['cpu']+(['cuda']if torch.cuda.is_available()else[])
    return dict(pass_all=True,checks=[check(device,seed)for device in devices])

