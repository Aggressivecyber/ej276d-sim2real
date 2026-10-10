"""Observe the frozen solve path, making exactly one network call per chunk."""
import numpy as np
import torch
from noise_core import bins_for,continuous_logq,innovation
from source_flow_core import displacement,constant_heun,bounded_charge,q_torch
from residual_math import canonical,STAGES,BOUNDARIES,boundaries,array_sha

@torch.no_grad()
def traced_solve(network,source,epsilon,mu,scale,edges,noise,chunk=64):
    x=np.asarray(source,dtype='f4');epsilon=np.asarray(epsilon,dtype='f4')
    assert x.shape==epsilon.shape and x.ndim==2 and x.shape[1]==480
    b,rawbin,q=bins_for(x,edges);condition=continuous_logq(q,b,edges)
    dev=next(network.parameters()).device;mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    stages={name:np.empty(x.shape,dtype='f8')for name in STAGES}
    private={k:np.empty(len(x))for k in ['delta_RMS_V','projection_RMS_V','preADC_relative_Q_error_f32','preADC_relative_Q_error_f64','range_active','rail_samples']}
    private['delta_boundaries_V']=np.empty((len(x),9));private['projection_boundaries_V']=np.empty((len(x),9))
    calls=0;max32=max64=0.
    for j in range(0,len(x),chunk):
        sl=slice(j,min(j+chunk,len(x)));n,ni=innovation(x[sl],epsilon[sl],mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        bt=torch.as_tensor(b[sl],dtype=torch.int64,device=dev);qt=torch.as_tensor(q[sl],device=dev)
        xt=torch.as_tensor(x[sl],dtype=torch.float64,device=dev);nt=torch.as_tensor(n,device=dev);ht=torch.as_tensor(condition[sl],device=dev)
        initial,delta=displacement(network,xt,nt,qt,bt,ht,mt,st)
        raw=constant_heun(initial,delta,128);out=bounded_charge(raw,qt);pp=out.to(torch.float32)
        e64=(q_torch(out)-qt).abs()/qt;e32=(q_torch(pp.to(torch.float64))-qt).abs()/qt
        max64=max(max64,float(e64.max()));max32=max(max32,float(e32.max()))
        assert torch.isfinite(out).all()and max32<=1e-5 and not((pp<-.5)|(pp>.5)).any()
        stages['input'][sl]=x[sl];stages['noise_initial'][sl]=initial.cpu().numpy();stages['neural_raw'][sl]=raw.cpu().numpy()
        stages['projected_f32'][sl]=pp.cpu().numpy();stages['canonical_ADC'][sl]=canonical(pp.cpu().numpy(),True)
        correction=out-raw
        private['delta_RMS_V'][sl]=delta.square().mean(1).sqrt().cpu().numpy();private['projection_RMS_V'][sl]=correction.square().mean(1).sqrt().cpu().numpy()
        private['preADC_relative_Q_error_f32'][sl]=e32.cpu().numpy();private['preADC_relative_Q_error_f64'][sl]=e64.cpu().numpy()
        private['range_active'][sl]=((raw<-.5)|(raw>.5)).any(1).cpu().numpy();private['rail_samples'][sl]=(out.abs()==.5).sum(1).cpu().numpy()
        private['delta_boundaries_V'][sl]=boundaries(delta.cpu().numpy());private['projection_boundaries_V'][sl]=boundaries(correction.cpu().numpy())
        calls+=1
    receipt={'actual_network_rows':len(x),'actual_network_calls':calls,'mathematical_Heun_field_rows':len(x)*256,'solver_steps':128,'max_relative_Q_f64':max64,'max_relative_Q_f32':max32,'range_active_N':int(private['range_active'].sum()),'stage_SHA256':{k:array_sha(v)for k,v in stages.items()},'instrumentation_no_extra_neural_forward':True}
    return stages,private,receipt
