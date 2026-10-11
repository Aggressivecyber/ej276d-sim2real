"""One public Q-response, lifting, continuous noise, Box, and ADC path."""
import numpy as np
import torch
from public_charge_head import common_charge_coordinates,window_charge,target_charge_condition
from physical_operators import bounded_charge,canonical_adc_torch


def projected_noise_variance(template,p480):
    """Exact E[var_100((I-m a^T) K^1/2 epsilon)] for current m(Q*)."""
    p=torch.as_tensor(p480,dtype=torch.float64,device=template.device)
    if p.shape!=(241,) or not bool(torch.isfinite(p).all()) or bool((p<0).any()):
        raise ValueError('invalid fixed noise spectrum')
    a=template.new_zeros(480);a[:100]=-3.;a[100:400]=1.
    k=torch.fft.irfft(p,n=480)
    ka=torch.fft.irfft(p*torch.fft.rfft(a),n=480)
    kaa=(a*ka).sum()
    lag=torch.arange(1,100,dtype=torch.float64,device=template.device)
    sum_cov=100*k[0]+2*((100-lag)*k[1:100]).sum()
    base=(100*k[0]-sum_cov/100)/99
    m=template[:,:100]-template[:,:100].mean(1,keepdim=True)
    centered_ka=ka[:100]-ka[:100].mean()
    result=base-2*(m*centered_ka).sum(1)/99+kaa*m.square().sum(1)/99
    if not bool(torch.isfinite(result).all()) or bool((result<=0).any()):
        raise FloatingPointError('nonpositive analytic projected noise variance')
    return result


def common_response(head,x,epsilon,mu,scale,edges,p480,tau2):
    coordinates=common_charge_coordinates(head,x,mu,scale,edges)
    lifted=coordinates['lifted_waveform'];m=coordinates['unit_template'];qs=coordinates['q_target']
    eps=torch.as_tensor(epsilon,dtype=torch.float64,device=x.device)
    p=torch.as_tensor(p480,dtype=torch.float64,device=x.device)
    if eps.shape!=x.shape or not bool(torch.isfinite(eps).all()):raise ValueError('invalid epsilon')
    if not np.isfinite(tau2) or tau2<0:raise ValueError('invalid frozen target variance')
    var=lifted[:,:100].var(1,unbiased=True)
    amp2=(tau2-var).clamp_min(0)
    kappa2=projected_noise_variance(m,p)
    n=torch.fft.irfft(torch.sqrt(p)*torch.fft.rfft(eps,norm='ortho'),n=480,norm='ortho')
    n=n-window_charge(n)[:,None]*m
    active=amp2>0
    ratio=torch.where(active,amp2/kappa2,torch.ones_like(amp2))
    amplitude=torch.where(active,torch.sqrt(ratio),torch.zeros_like(amp2))
    n=n*amplitude[:,None]
    raw=lifted+n
    pre64=bounded_charge(raw,qs);pre32=pre64.to(torch.float32)
    error64=(window_charge(pre64)-qs).abs()/qs
    error32=(window_charge(pre32)-qs).abs()/qs
    if not bool(torch.isfinite(pre64).all()) or float(error64.max())>1e-5 or float(error32.max())>1e-5:
        raise FloatingPointError('Q* implementation precision failed; no row deletion')
    if bool(((pre32<-.5)|(pre32>.5)).any()):raise FloatingPointError('preADC physical range failure')
    adc=canonical_adc_torch(pre32)
    preq=window_charge(pre32);adcq=window_charge(adc)
    before=target_charge_condition(preq,edges);after=target_charge_condition(adcq,edges)
    meta={
      'Q0':coordinates['q_input'],'Qstar':qs,'QpreADC':preq,'QADC':adcq,
      'original_bin_raw':coordinates['input_bin_raw'],'target_bin_raw':coordinates['target_bin_raw'],
      'preADC_bin_raw':before['bin_raw'],'ADC_bin_raw':after['bin_raw'],
      'original_inside_support':(coordinates['input_bin_raw']>=0)&(coordinates['input_bin_raw']<8),
      'target_inside_support':(coordinates['target_bin_raw']>=0)&(coordinates['target_bin_raw']<8),
      'ADC_inside_support':(after['bin_raw']>=0)&(after['bin_raw']<8),
      'noise_condition_Qstar':qs,'continuous_condition':coordinates['continuous_condition'],
      'noise_kappa2':kappa2,'noise_amplitude_squared':amp2,'lifted_baseline_variance':var,
      'fixed_unit_Q_slope':coordinates['fixed_unit_q_slope'],
      'Q_relative_error_f64':error64,'Q_relative_error_f32':error32,
      'range_projection_active':((raw<-.5)|(raw>.5)).any(1),
      'projection_RMS_V':(pre64-raw).square().mean(1).sqrt(),
      'target_to_input_Q_ratio':qs/coordinates['q_input'],'ADC_to_input_Q_ratio':adcq/coordinates['q_input'],
    }
    return pre32,adc,meta


@torch.no_grad()
def generate(head,x,epsilon,mu,scale,edges,p480,tau2,chunk=256):
    device=next(head.parameters()).device
    stats=[torch.as_tensor(a,dtype=torch.float64,device=device)for a in(mu,scale,edges,p480)]
    pre=[];adc=[];meta={}
    for start in range(0,len(x),chunk):
        v=torch.as_tensor(x[start:start+chunk],dtype=torch.float64,device=device)
        e=torch.as_tensor(epsilon[start:start+chunk],dtype=torch.float64,device=device)
        pp,aa,mm=common_response(head,v,e,*stats,tau2)
        pre.append(pp.cpu().numpy());adc.append(aa.cpu().numpy())
        for k,value in mm.items():meta.setdefault(k,[]).append(value.cpu().numpy())
    return np.concatenate(pre),np.concatenate(adc),{k:np.concatenate(v)for k,v in meta.items()}
