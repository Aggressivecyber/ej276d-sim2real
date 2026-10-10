"""Checks on artificial arrays only; no scientific data or response network."""
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from distribution_loss import DistributionLoss,NAMES,CATEGORIES
from source_flow_core import canonical_adc_torch
from diagnostic_math import rounding_psd_interval,psd_np,exact_w1_columns,canonical_np,decompose,COMPONENTS,gradient_geometry

def check(cfg,device='cpu'):
    rng=np.random.Generator(np.random.PCG64(20261011699));t=np.arange(480,dtype='f8');n=384
    true=rng.normal(0,.0006,(n,480))+.16*np.exp(-.5*((t-126)/12)**2)[None,:]
    stored=true.astype('f4');lo,hi=rounding_psd_interval(stored);truth=psd_np(true)
    assert np.all((truth>=lo)&(truth<=hi))
    x=rng.normal(size=(7,4));y=rng.normal(size=(13,4));got=exact_w1_columns(x,y)
    expected=np.asarray([wasserstein_distance(x[:,j],y[:,j])for j in range(4)])
    assert np.max(np.abs(got-expected))<1e-13
    a=canonical_np(stored,True);b=canonical_adc_torch(torch.from_numpy(stored).double()).numpy()
    assert np.array_equal(a,b)
    dev=torch.device(device);B=32;sn=320
    stats={'source_counts':np.arange(1,11),'REAL_counts':np.arange(10,0,-1),
      'noise_mean':np.array([.0006,0]),'noise_scale':np.array([.0001,.15]),
      'valley_bounds':np.array([[.18,.24],[.18,.24]]),'source_h_one_fraction':np.array([.5,.5]),'source_raw_valley_fraction':np.zeros(2)}
    for name,dim in zip(NAMES,[480,480,479,478]):
        stats[name+'_center']=np.zeros((10,dim));stats[name+'_scale']=np.full((10,dim),.03 if name=='V480' else .1)
    train_cfg={'batch_per_region':B,'projection_directions':64,'PSD_loss_scale':.01,'valley_smoothing':.002,'valley_delta':.02,
      'loss_weights':{k:1. for k in COMPONENTS[:7]}}
    obj=DistributionLoss(stats,train_cfg,dev)
    theta=torch.tensor([.001,.002,.003],device=dev,dtype=torch.float64,requires_grad=True)
    base=torch.as_tensor(stored.astype('f8'),device=dev)
    wave=torch.as_tensor(np.stack([np.sin(t/11),np.cos(t/17),np.sin(t/31)]),device=dev)
    raw=base+(theta[:,None]*wave).sum(0);out=raw+.3*theta[2]*wave[1]
    reference=torch.as_tensor(np.roll(true[:sn],1,axis=1),device=dev)
    identity=base[sn:];bins=torch.tensor(np.resize(CATEGORIES,64),device=dev)
    sp=np.repeat(np.linspace(.18,.36,160),2)+np.tile([0,1e-10],160)
    source_psd=torch.as_tensor(sp,device=dev);h=torch.as_tensor(np.tile([0,1],160),device=dev)
    directions=obj.draw_directions(rng)
    total,pieces,diagnostic,fraction=decompose(obj,out,raw,reference,identity,bins,source_psd,h,directions,source_psd-1e-8,source_psd+1e-8)
    values=np.asarray([float(pieces[k].detach())for k in COMPONENTS]);vectors=[]
    for k in COMPONENTS:vectors.append(torch.autograd.grad(pieces[k],theta,retain_graph=True)[0].detach().cpu().numpy())
    geometry=gradient_geometry(np.asarray(vectors),values,cfg)
    assert np.isfinite(values).all()and np.isfinite(vectors).all()and all(v['pass']for v in geometry['checks'].values())
    assert fraction['roundoff_compatible_fraction_of_nonzero_mass']>0
    assert np.isclose(float(total.detach()),values[:7].sum(),rtol=1e-12,atol=1e-12)
    return {'pass':True,'device':str(dev),'artificial_float32_intervals_contain_unrounded_PSD':True,
      'unequal_sample_W1_max_error':float(np.max(np.abs(got-expected))),'canonical_ADC_matches_original':True,
      'seventeen_component_value_and_gradient_recompositions':geometry['checks'],'all_original_seven_sum_to_total':True,
      'scientific_data_read':False,'response_network_forwards':0,'response_fits_or_updates':0}
