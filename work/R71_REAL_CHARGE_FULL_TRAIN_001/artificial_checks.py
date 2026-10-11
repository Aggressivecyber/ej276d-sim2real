"""Artificial CPU/CUDA implementation admission before scientific array reads."""
import numpy as np
import torch
from public_charge_head import PublicConditionalChargeHead,window_charge,normalized_template_bank
from common_response import projected_noise_variance,common_response
from charge_losses import FullChargeObjective
from numpy_reference import reference

def run_checks(seed):
    rng=np.random.Generator(np.random.PCG64(seed));torch.manual_seed(seed)
    edges=np.linspace(np.log(300),np.log(100000),9)
    mu=np.zeros((8,480));mu[:,100:400]=1/300
    mu+=rng.normal(0,1e-5,mu.shape)
    scale=np.full((8,480),.003)
    p=.1+np.linspace(0,1,241)**2
    x=mu[[0,2,3,7,1,4,6,5]]*np.arange(1,9)[:,None]
    x=x-x[:,:100].mean(1,keepdims=True)+.001
    eps=rng.standard_normal(x.shape)
    m=normalized_template_bank(torch.tensor(mu))
    a=np.zeros(480);a[:100]=-3;a[100:400]=1
    ix=np.arange(480);K=np.fft.irfft(p,n=480)[(ix[:,None]-ix[None,:])%480]
    B=np.eye(480)[:100]-np.eye(480)[:100].mean(0)[None,:]
    expected=[]
    for row in m.numpy():
        P=np.eye(480)-row[:,None]*a[None,:]
        expected.append(np.trace(B@P@K@P.T@B.T)/99)
    analytic=projected_noise_variance(m,torch.tensor(p)).numpy()
    cov_error=float(np.max(np.abs(analytic-expected)));assert cov_error<1e-10
    model=PublicConditionalChargeHead(torch.zeros(480),torch.ones(480))
    with torch.no_grad():
        model.output.weight.copy_(torch.tensor(rng.normal(0,.001,model.output.weight.shape)))
        model.output.bias.copy_(torch.tensor([.08,.03]))
    state={k:v.detach().numpy().copy()for k,v in model.state_dict().items()}
    independent=reference(state,x,eps,mu,scale,edges,p,2e-6)
    cpu=common_response(model,torch.tensor(x),torch.tensor(eps),torch.tensor(mu),torch.tensor(scale),torch.tensor(edges),torch.tensor(p),2e-6)
    cpuRMS=np.sqrt(np.mean((cpu[0].detach().numpy().astype('f8')-independent[0].astype('f8'))**2,1))*16383
    assert float(cpuRMS.max())<=1
    cpstate={k:v.detach().clone()for k,v in model.state_dict().items()}
    cuda=PublicConditionalChargeHead(torch.zeros(480,device='cuda'),torch.ones(480,device='cuda'));cuda.load_state_dict(cpstate)
    args=[torch.tensor(v,device='cuda')for v in (x,eps,mu,scale,edges,p)]
    gpu=common_response(cuda,*args,2e-6)
    gpuRMS=np.sqrt(np.mean((gpu[0].detach().cpu().numpy().astype('f8')-independent[0].astype('f8'))**2,1))*16383
    assert float(gpuRMS.max())<=1
    # No-completion branch has a finite, explicitly zero noise-amplitude derivative.
    xx=torch.tensor(x,requires_grad=True)
    zero=common_response(model,xx,torch.tensor(eps),torch.tensor(mu),torch.tensor(scale),torch.tensor(edges),torch.tensor(p),0.)
    zero[0].double().square().sum().backward();assert bool(torch.isfinite(xx.grad).all())
    qs=np.array([.7,1.,1.,3.,8.]);qr=np.array([.8,1.2,2.,3.,4.,9.,11.])
    predS=np.array([1.1,.9,1.4,4.,7.]);predR=qr*np.array([1.01,.99,1.03,1.02,.98,1.04,1.01])
    values=[];grads=[]
    for device in ['cpu','cuda']:
        s=torch.tensor(predS,device=device,requires_grad=True);r=torch.tensor(predR,device=device,requires_grad=True)
        objective=FullChargeObjective(torch.tensor(qs,device=device),torch.tensor(qr,device=device))
        value,terms=objective(s,r,torch.tensor(qr,device=device));value.backward()
        values.append(float(value.detach()));grads.append(np.r_[s.grad.cpu().numpy(),r.grad.cpu().numpy()])
        assert set(['charge_distribution','REAL_identity','source_charge_order']).issubset(terms)
    loss_error=abs(values[0]-values[1]);gradient_error=float(np.max(np.abs(grads[0]-grads[1])))
    assert loss_error<1e-10 and gradient_error<1e-9
    return dict(pass_all=True,artificial_only=True,dense_covariance_maxabs=cov_error,
        independent_CPU_max_RMS_LSB=float(cpuRMS.max()),independent_CUDA_max_RMS_LSB=float(gpuRMS.max()),
        CPU_CUDA_loss_abs_error=loss_error,CPU_CUDA_loss_gradient_maxabs=gradient_error,
        no_completion_gradient_finite=True,scientific_arrays_read=False)
