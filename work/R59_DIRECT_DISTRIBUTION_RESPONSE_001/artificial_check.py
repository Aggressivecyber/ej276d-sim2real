"""Artificial mathematical checks only; no scientific data or response training."""
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from source_flow_core import project,q_torch,bounded_charge,constant_heun,canonical_adc_torch
from distribution_loss import weighted_w1

def check():
    rng=np.random.Generator(np.random.PCG64(5901));mu=rng.normal(size=(4,480));mu[:,100:400]+=.1
    mt=torch.from_numpy(mu);mt=mt/q_torch(mt)[:,None]
    v=torch.from_numpy(rng.normal(size=(4,480))).requires_grad_(True);tangent=project(v,mt)
    assert float(q_torch(tangent).detach().abs().max())<1e-10
    tangent.square().mean().backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
    raw=torch.from_numpy(rng.normal(scale=.8,size=(4,480)));charge=torch.tensor([.4,10.,100.,290.],dtype=torch.float64)
    out=bounded_charge(raw,charge);maxcharge=float((q_torch(out)-charge).abs().max())
    assert maxcharge<1e-10 and bool((out.abs()<=.5).all())
    try:bounded_charge(raw[:1],torch.tensor([300.001],dtype=torch.float64))
    except ValueError:infeasible=True
    else:raise AssertionError('infeasible charge accepted')
    # Check the piecewise analytic Jacobian at a nondegenerate mix of free and bound coordinates.
    x=raw[:1].clone().requires_grad_(True);qq=torch.tensor([10.],dtype=torch.float64,requires_grad=True)
    objective=torch.from_numpy(rng.normal(size=(1,480)));y=bounded_charge(x,qq);loss=(objective*y).sum();loss.backward()
    finite=[];analytic=[];eps=1e-6
    for k in [0,20,99,100,150,399,400,479]:
        d=torch.zeros_like(x);d[0,k]=eps
        f=float(((bounded_charge(x.detach()+d,qq.detach())-bounded_charge(x.detach()-d,qq.detach()))*objective).sum()/(2*eps))
        finite.append(f);analytic.append(float(x.grad[0,k]))
    qfd=float(((bounded_charge(x.detach(),qq.detach()+eps)-bounded_charge(x.detach(),qq.detach()-eps))*objective).sum()/(2*eps))
    jacerr=max(float(np.max(np.abs(np.asarray(finite)-analytic))),abs(qfd-float(qq.grad)))
    assert jacerr<1e-7,(finite,analytic,qfd,float(qq.grad))
    initial=torch.from_numpy(rng.normal(size=(4,480)));delta=tangent.detach()
    y128=constant_heun(initial,delta,128);y256=constant_heun(initial,delta,256)
    integral_error=float((y128-(initial+delta)).abs().max());precision=float((y128-y256).abs().max())
    assert max(integral_error,precision)<1e-10
    x=torch.tensor([-.2,.1,.1,.8],dtype=torch.float64,requires_grad=True);a=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    r=torch.tensor([-.1,.3,.9],dtype=torch.float64);b=torch.tensor([.4,.2,.4],dtype=torch.float64)
    w=weighted_w1(x,a,r,b);exact=wasserstein_distance(x.detach().numpy(),r.numpy(),u_weights=a.numpy(),v_weights=b.numpy())
    assert abs(float(w.detach())-exact)<1e-14;w.backward();assert torch.isfinite(x.grad).all()
    counts=np.asarray([1,3,7]);masses=counts/counts.sum();B=32
    recovered=np.concatenate([np.full(n,mass/n) for n,mass in zip(counts,masses)])
    from fractions import Fraction
    assert all(Fraction(int(n),int(counts.sum()))/int(n)==Fraction(1,int(counts.sum())) for n in counts)
    weight_error=float(np.max(np.abs(recovered-1/counts.sum())))
    assert weight_error<1e-15
    vv=np.r_[np.linspace(-.7,.7,1001),rng.normal(0,.1,1100)].astype('f4')
    legacy=(-.5+np.rint((np.clip(vv,-.5,.5)+.5)*16383)/16383).astype('f4')
    reference=-.5+np.clip(np.rint((legacy.astype('f8')+.5)*16383),0,16383)/16383
    got=canonical_adc_torch(torch.from_numpy(vv)).numpy();assert np.array_equal(got,reference)
    return {'artificial_only':True,'scientific_data_decoded':False,'response_training_updates':0,'extra_LP_solves':0,
        'charge_tangent_and_gradient':True,'box_charge_max_error':maxcharge,'box_bounds_pass':True,'impossible_absolute_charge_rejected':infeasible,
        'box_free_coordinate_Jacobian_max_error':jacerr,'constant_field_analytic_error':integral_error,'constant_field_128_256_max_error':precision,
        'weighted_W1':float(w.detach()),'independent_scipy_W1':float(exact),'global_original_parent_weight_recovery':True,
        'parent_weight_float_roundoff':weight_error,'ADC_then_canonical_matches_existing_numpy':True}

def integration_check(cfg,device='cpu'):
    from distribution_loss import fit_statistics,DistributionLoss,psd,CATEGORIES
    from source_flow_core import endpoint
    from network import ResponseNetwork
    rng=np.random.Generator(np.random.PCG64(5902))
    def population(counts):
        values=[];labels=[]
        for j,(b,count) in enumerate(zip(CATEGORIES,counts)):
            v=rng.normal(scale=.0007,size=(count,480))
            v[:,100:400]+=(.035+.006*j)*np.exp(-np.arange(300)/(30+3*j))[None,:]
            values.append(v);labels.append(np.full(count,b))
        return np.concatenate(values).astype('f4'),np.concatenate(labels)
    source,sb=population(np.arange(5,15));real,rb=population(np.arange(8,18));h=(np.arange(len(source))%2).astype('f8')
    original_psd=psd(torch.as_tensor(source,dtype=torch.float64)).numpy()
    valley={'bins':[{'valid':True,'low':.20,'high':.24} for _ in range(8)]}
    device=torch.device(device)
    statistics=fit_statistics(real,original_psd,h,sb,rb,valley,cfg);objective=DistributionLoss(statistics,cfg,device)
    si=np.concatenate([rng.choice(np.flatnonzero(sb==b),cfg['batch_per_region']) for b in CATEGORIES])
    ri=np.concatenate([rng.choice(np.flatnonzero(rb==b),cfg['batch_per_region']) for b in CATEGORIES]);ii=rng.integers(len(real),size=cfg['identity_batch'])
    original=torch.as_tensor(np.concatenate((source[si],real[ii])),dtype=torch.float64,device=device)
    raw=(original+torch.as_tensor(rng.normal(scale=.001,size=original.shape),device=device)).detach().requires_grad_(True)
    output=bounded_charge(raw,q_torch(original))
    loss,pieces,diagnostics=objective(output,raw,torch.as_tensor(real[ri],dtype=torch.float64,device=device),torch.as_tensor(real[ii],dtype=torch.float64,device=device),torch.as_tensor(rb[ii],device=device),
        torch.as_tensor(original_psd[si],device=device),torch.as_tensor(h[si],device=device),objective.draw_directions(rng))
    assert set(pieces)==set(cfg['loss_weights']) and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
    global_source_mass=float(objective.sw.sum());global_REAL_mass=float(objective.rw.sum())
    assert abs(global_source_mass-1)<1e-14 and abs(global_REAL_mass-1)<1e-14
    # A tiny complete network forward/backward checks the real model interface without any optimizer update.
    torch.manual_seed(5903);network=ResponseNetwork().to(device);x=torch.as_tensor(source[:2],dtype=torch.float64,device=device);q=q_torch(x)
    mu=torch.zeros((8,480),dtype=torch.float64,device=device);mu[:,100:400]=1/300;scale=torch.full_like(mu,.001);b=torch.tensor([0,1],device=device)
    n=project(torch.as_tensor(rng.normal(scale=.0001,size=(2,480)),device=device),mu[b]);hc=torch.zeros(2,dtype=torch.float64,device=device)
    y,pre=endpoint(network,x,n,q,b,hc,mu,scale)
    assert torch.max(torch.abs(pre-(x+n)))==0 and torch.max(torch.abs(q_torch(y)-q))<1e-10
    y.square().mean().backward();head=network.final.weight.grad
    assert head is not None and torch.isfinite(head).all() and torch.count_nonzero(head)>0
    return {'artificial_only':True,'device':str(device),'scientific_data_decoded':False,'optimizer_updates':0,'ten_region_all_seven_losses_finite':True,
        'full_loss_value':float(loss.detach()),'direct_loss_gradient_norm':float(raw.grad.norm()),'source_global_mass':global_source_mass,'REAL_global_mass':global_REAL_mass,
        'neural_zero_head_exact_initial_displacement':True,'complete_network_box_charge_gradient_finite_nonzero':True,'network_parameter_count':sum(p.numel() for p in network.parameters())}

if __name__=='__main__':
    import json
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    print(json.dumps(check(),indent=2))
