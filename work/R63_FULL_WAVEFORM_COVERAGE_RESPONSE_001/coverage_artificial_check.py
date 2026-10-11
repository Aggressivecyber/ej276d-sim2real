"""Artificial checks only. No original TRAIN, optimizer updates, or evaluation reads."""
import numpy as np
import torch
from R61_artificial_check import check
from R61_distribution_loss import DistributionLoss as R61Loss
from distribution_loss import (fit_statistics,fit_temporal_statistics,DistributionLoss,
    psd,CATEGORIES,temporal_features,coordinate_w1_rms,temporal_distribution_terms)
from precision_order import rounding_psd_interval,artificial_precision_check
from source_flow_core import bounded_charge,q_torch,project,endpoint
from network import ResponseNetwork

def numpy_temporal(v):
    b=np.asarray(v,dtype='f8')[...,:100];b=b-b.mean(-1,keepdims=True)
    power=np.mean(b*b,axis=-1);safe=np.maximum(power,1e-20)
    acf=np.stack([np.mean(b[...,k:]*b[...,:-k],axis=-1)/safe for k in range(1,21)],axis=-1)
    f=np.fft.rfft(b,axis=-1);bands=[]
    for low in range(1,51,5):
        cols=np.arange(low,low+5);weights=np.where(cols==50,1.,2.)
        bands.append(np.sum(np.abs(f[...,cols])**2*weights,axis=-1)/10000/safe)
    return np.concatenate((np.sqrt(safe)[...,None],acf,np.stack(bands,axis=-1)),axis=-1)

def finite_difference(fn,x,indices,eps=1e-7):
    x=x.detach().clone().requires_grad_(True);scalar=fn(x)
    gradient=torch.autograd.grad(scalar,x)[0];errors=[]
    for index in indices:
        delta=torch.zeros_like(x);delta[index]=eps
        fd=float((fn(x.detach()+delta)-fn(x.detach()-delta))/(2*eps));analytic=float(gradient[index])
        errors.append(abs(fd-analytic)/(1+abs(fd)+abs(analytic)))
    error=max(errors);assert error<2e-6,(error,errors)
    return error

def formula_check(device='cpu'):
    device=torch.device(device);rng=np.random.Generator(np.random.PCG64(63001))
    v=rng.normal(scale=.003,size=(4,480));time=np.arange(100)
    v[0,:100]=.02*np.cos(2*np.pi*7*time/100);v[1,:100]=.02*(-1.)**time;v[2,:100]=0
    vt=torch.as_tensor(v,dtype=torch.float64,device=device);f=temporal_features(vt);reference=numpy_temporal(v)
    err=float(np.max(np.abs(f.detach().cpu().numpy()-reference)));assert err<1e-12,err
    assert abs(float(f[0,0])-.02/np.sqrt(2))<1e-14 and abs(float(f[1,0])-.02)<1e-14
    assert abs(float(f[0,22])-1)<1e-12 and abs(float(f[1,30])-1)<1e-12
    assert torch.allclose(f[[0,1,3],21:].sum(1),torch.ones(3,dtype=f.dtype,device=device),rtol=0,atol=1e-12)
    zero_rms_roundoff=abs(float(f[2,0])-1e-10)
    assert zero_rms_roundoff<=np.finfo(np.float64).eps*1e-10 and torch.count_nonzero(f[2,1:])==0
    zero=torch.zeros((2,480),dtype=torch.float64,device=device,requires_grad=True)
    temporal_features(zero).sum().backward();assert torch.isfinite(zero.grad).all() and torch.count_nonzero(zero.grad)==0
    a=torch.zeros((2,7,4),dtype=torch.float64,device=device);b=a.clone();a[0,:,1]=8.;a[1,:,:]=3.
    assert torch.allclose(coordinate_w1_rms(a,b),a.new_tensor([4.,3.]),atol=0,rtol=0)
    z=torch.zeros((2,7,4),dtype=torch.float64,device=device,requires_grad=True)
    coordinate_w1_rms(z,z.detach()).sum().backward();assert torch.isfinite(z.grad).all() and torch.count_nonzero(z.grad)==0
    x=torch.as_tensor(rng.normal(size=(2,7,9)),dtype=torch.float64,device=device);target=torch.as_tensor(rng.normal(size=x.shape),dtype=torch.float64,device=device)
    erms=finite_difference(lambda xx:coordinate_w1_rms(xx,target).sum(),x,[(0,0,0),(0,3,5),(1,1,1),(1,5,8)])
    xp=torch.as_tensor(rng.normal(scale=.003,size=(2,480)),dtype=torch.float64,device=device);fw=torch.as_tensor(rng.normal(size=(2,31)),dtype=torch.float64,device=device)
    efeat=finite_difference(lambda xx:(temporal_features(xx)*fw).sum(),xp,[(0,0),(0,37),(0,99),(1,61),(1,140)],eps=1e-8)
    x=torch.as_tensor(rng.normal(size=(2,7,31)),dtype=torch.float64,device=device);target=torch.as_tensor(rng.normal(size=x.shape),dtype=torch.float64,device=device)
    directions=rng.normal(size=(31,64));directions/=np.linalg.norm(directions,axis=0);p=torch.as_tensor(directions,dtype=torch.float64,device=device)
    m,s=temporal_distribution_terms(x,target,p);xn=x.cpu().numpy();tn=target.cpu().numpy()
    mn=np.abs(np.sort(xn,axis=1)-np.sort(tn,axis=1)).mean((1,2));sn=np.abs(np.sort(xn@directions,axis=1)-np.sort(tn@directions,axis=1)).mean((1,2))
    assert np.allclose(m.cpu().numpy(),mn,rtol=0,atol=1e-13) and np.allclose(s.cpu().numpy(),sn,rtol=0,atol=1e-13)
    ejoint=finite_difference(lambda xx:sum(t.sum()for t in temporal_distribution_terms(xx,target,p)),x,[(0,0,0),(0,3,17),(1,1,1),(1,5,30)])
    return {'device':str(device),'zero_RMS_roundoff':zero_rms_roundoff,'independent_NumPy_temporal_max_abs_error':err,'ordinary_and_Nyquist_band_power_pass':True,
        'Parseval_10_band_sum_pass':True,'zero_power_and_zero_RMS_residual_finite_zero_gradient':True,'coordinate_constant_and_localized_residual_pass':True,
        'finite_difference_max_scaled_errors':{'coordinate_RMS':erms,'continuous_temporal_features':efeat,'continuous_temporal_W1':ejoint},
        'finite_differences_do_not_cross_ADC':True,'scientific_data_decoded':False,'optimizer_updates':0}

def integration_check(cfg,device='cpu'):
    rng=np.random.Generator(np.random.PCG64(5902))
    def population(counts):
        values=[];labels=[]
        for j,(b,count) in enumerate(zip(CATEGORIES,counts)):
            v=rng.normal(scale=.0007,size=(count,480));v[:,100:400]+=(.035+.006*j)*np.exp(-np.arange(300)/(30+3*j))[None,:]
            values.append(v);labels.append(np.full(count,b))
        return np.concatenate(values).astype('f4'),np.concatenate(labels)
    source,sb=population(np.arange(5,15));real,rb=population(np.arange(8,18));h=(np.arange(len(source))%2).astype('f8')
    original_psd=psd(torch.as_tensor(source,dtype=torch.float64)).numpy();interval_low,interval_high=rounding_psd_interval(source)
    valley={'bins':[{'valid':True,'low':.20,'high':.24}for _ in range(8)]};device=torch.device(device)
    statistics=fit_statistics(real,original_psd,h,sb,rb,valley,cfg);old_statistics=dict(statistics);temporal=fit_temporal_statistics(real,rb,cfg);statistics.update(temporal)
    objective=DistributionLoss(statistics,cfg,device)
    assert temporal['temporal_center'].shape==(10,31) and temporal['temporal_scale'].shape==(10,31)
    assert np.array_equal(temporal['temporal_REAL_counts'],statistics['REAL_counts'])
    si=np.concatenate([rng.choice(np.flatnonzero(sb==b),cfg['batch_per_region'])for b in CATEGORIES]);ri=np.concatenate([rng.choice(np.flatnonzero(rb==b),cfg['batch_per_region'])for b in CATEGORIES]);ii=rng.integers(len(real),size=cfg['identity_batch'])
    original=torch.as_tensor(np.concatenate((source[si],real[ii])),dtype=torch.float64,device=device)
    raw=(original+torch.as_tensor(rng.normal(scale=.001,size=original.shape),device=device)).detach().requires_grad_(True)
    output=bounded_charge(raw,q_torch(original));directions=objective.draw_directions(rng)
    trng=np.random.Generator(np.random.PCG64(cfg['PCG64_seeds']['baseline_projection_directions']));tp=objective.draw_temporal_directions(trng)
    assert tp.shape==(31,64) and np.max(np.abs(np.linalg.norm(tp.astype('f8'),axis=0)-1))<1e-7
    args=(output,raw,torch.as_tensor(real[ri],dtype=torch.float64,device=device),torch.as_tensor(real[ii],dtype=torch.float64,device=device),torch.as_tensor(rb[ii],device=device),
        torch.as_tensor(original_psd[si],device=device),torch.as_tensor(h[si],device=device),directions,torch.as_tensor(interval_low[si],device=device),torch.as_tensor(interval_high[si],device=device))
    loss,pieces,diagnostics=objective(*args,tp);old_loss,old_pieces,_=R61Loss(old_statistics,cfg,device)(*args)
    assert len(old_pieces)==7 and len(pieces)==9 and all(torch.equal(pieces[k],old_pieces[k])for k in old_pieces)
    new_gradient=torch.autograd.grad(sum(pieces[k]for k in old_pieces),raw,retain_graph=True)[0];old_gradient=torch.autograd.grad(old_loss,raw,retain_graph=True)[0];assert torch.equal(new_gradient,old_gradient)
    assert torch.equal(loss,old_loss+pieces['coordinate_RMS']+pieces['temporal_joint']) and torch.allclose(loss,sum(pieces.values()),rtol=1e-14,atol=1e-14)
    assert torch.equal(diagnostics['original_seven_total'],old_loss.detach())
    for name in ['coordinate_RMS','temporal_joint']:
        g=torch.autograd.grad(pieces[name],raw,retain_graph=True)[0];assert torch.isfinite(g).all()and torch.count_nonzero(g)>0,name
    precision_record=artificial_precision_check(device);assert set(pieces)==set(cfg['loss_weights']) and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(raw.grad).all()and raw.grad.abs().sum()>0
    global_source_mass=float(objective.sw.sum());global_REAL_mass=float(objective.rw.sum());assert abs(global_source_mass-1)<1e-14 and abs(global_REAL_mass-1)<1e-14
    torch.manual_seed(5903);network=ResponseNetwork().to(device);x=torch.as_tensor(source[:2],dtype=torch.float64,device=device);q=q_torch(x)
    mu=torch.zeros((8,480),dtype=torch.float64,device=device);mu[:,100:400]=1/300;scale=torch.full_like(mu,.001);b=torch.tensor([0,1],device=device)
    n=project(torch.as_tensor(rng.normal(scale=.0001,size=(2,480)),device=device),mu[b]);hc=torch.zeros(2,dtype=torch.float64,device=device)
    y,pre=endpoint(network,x,n,q,b,hc,mu,scale);assert torch.max(torch.abs(pre-(x+n)))==0 and torch.max(torch.abs(q_torch(y)-q))<1e-10
    y.square().mean().backward();head=network.final.weight.grad;assert head is not None and torch.isfinite(head).all()and torch.count_nonzero(head)>0
    return {'new_formula_checks':formula_check(device),'precision_order':precision_record,'all_seven_loss_values_and_gradients_exactly_match_R61':True,
        'artificial_only':True,'device':str(device),'scientific_data_decoded':False,'optimizer_updates':0,'ten_region_all_nine_losses_finite':True,
        'loss_components':{k:float(v.detach())for k,v in pieces.items()},'full_loss_value':float(loss.detach()),'direct_loss_gradient_norm':float(raw.grad.norm()),
        'source_global_mass':global_source_mass,'REAL_global_mass':global_REAL_mass,'neural_zero_head_exact_initial_displacement':True,
        'complete_network_box_charge_gradient_finite_nonzero':True,'network_parameter_count':sum(p.numel()for p in network.parameters())}
