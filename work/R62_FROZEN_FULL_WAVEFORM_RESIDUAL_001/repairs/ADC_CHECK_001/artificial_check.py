"""Artificial-only checks: no project waveform archive is opened here."""
import json
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from noise_core import q_np
from source_flow_core import solve,canonical_adc_torch,canonical_REAL_torch
from distribution_loss import feature_vectors,noise_features
from residual_math import *
from traced_response import traced_solve

class ArtificialField(torch.nn.Module):
    def __init__(self):
        super().__init__();self.amplitude=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float32))
    def forward(self,noisy,b,guide,condition):
        t=torch.arange(480,device=noisy.device,dtype=torch.float32)
        return self.amplitude*torch.sin(t[None]/9)+.03*torch.tanh(noisy.float())+.01*torch.tanh(guide.float())+.002*condition.float()[:,None]

def check(device='cpu',integration=True):
    rng=np.random.Generator(np.random.PCG64(620017));checks={}
    a=rng.normal(size=(17,7));b=rng.normal(size=(23,7));a[:3]=0;b[:5]=0
    got=exact_w1_columns(a,b);expected=np.array([wasserstein_distance(a[:,j],b[:,j])for j in range(7)])
    assert np.allclose(got,expected,rtol=2e-14,atol=2e-14)
    assert np.array_equal(exact_w1_columns(a,a),np.zeros(7))
    checks['unequal_exact_W1_max_abs_error']=float(np.max(np.abs(got-expected)))
    x=rng.normal(0,.003,(6,480));x[:,100:400]+=.04
    tx=torch.as_tensor(x,device=device,dtype=torch.float64)
    assert all(np.allclose(f,g.cpu().numpy(),atol=2e-13,rtol=2e-13)for f,g in zip(features(x),feature_vectors(tx)))
    assert np.array_equal(canonical(x,False),canonical_REAL_torch(tx).cpu().numpy())
    assert np.array_equal(canonical(x,True),canonical_adc_torch(tx).cpu().numpy())
    nf,base,power=baseline_features(x)
    assert np.allclose(nf[:,:2],noise_features(tx).cpu().numpy(),atol=1e-14,rtol=1e-12)
    assert np.allclose(nf[:,21:].sum(1),1.,atol=1e-14)
    assert np.allclose(covariance(base),np.cov(base,rowvar=False,ddof=1),atol=1e-18)
    assert covariance_error(covariance(base),covariance(base))==0
    assert covariance_error(np.zeros((2,2)),np.zeros((2,2)))is None
    checks['four_features_canonical_ADC_noise_and_covariance']=True
    t=np.arange(100);waves=np.zeros((3,480));waves[0,:100]=np.sin(2*np.pi*3*t/100);waves[1,:100]=(-1.)**t
    f,base,p=baseline_features(waves)
    assert np.allclose(p,[.5,1,0],atol=1e-14)
    assert np.allclose(f[0,21],1,atol=1e-14)and np.allclose(f[1,30],1,atol=1e-14)
    assert np.count_nonzero(f[2])==0
    raw=np.abs(np.fft.rfft(base,axis=1)[:,1:])**2;raw[:,:-1]*=2
    assert np.allclose(raw.sum(1)/100**2,p,atol=1e-14)
    impulse=np.zeros((1,480));impulse[0,150]=1
    assert np.array_equal(boundaries(impulse)[0],[0,1,0,0,0,1,-2,0,0])
    checks['spectral_Parseval_Nyquist_zero_power_and_boundary_indices']=True
    checks['numpy_scalar_JSON']=json.loads(json.dumps(clean({'b':np.bool_(True),'n':np.int64(2),'v':np.array([1.,2.])}),allow_nan=False))
    rejected=False
    try:clean({'v':np.float64(np.nan)})
    except ValueError:rejected=True
    assert rejected;checks['nonfinite_JSON_rejected']=True
    mu=np.zeros((8,480));mu[:,100:400]=1/300;scale=np.full((8,480),.01);edges=np.linspace(np.log(1000),np.log(300000),9)
    source=(np.array([2.,5.,25.,75.,170.,299.])[:,None]*mu[0]).astype('f4')
    source[:,:100]+=rng.normal(0,.0003,(6,100)).astype('f4')
    eps=rng.standard_normal(source.shape).astype('f4');noise={'p480':np.ones(241),'kappa2':np.ones(8),'tau2':np.array(4e-6)}
    field=ArtificialField().to(device).eval()
    original,meta,receipt=solve(field,source,eps,mu,scale,edges,noise,True,128,chunk=2)
    paths,private,traced=traced_solve(field,source,eps,mu,scale,edges,noise,chunk=2)
    assert np.array_equal(original,paths['projected_f32'].astype('f4'))
    assert np.array_equal(canonical(original,True),paths['canonical_ADC'])
    assert traced['actual_network_rows']==receipt['actual_network_rows']==6
    assert traced['actual_network_calls']==receipt['actual_network_calls']==3
    assert traced['range_active_N']==receipt['range_active_projection_events']>0
    assert np.array_equal(private['projection_RMS_V'],meta['constraint_projection_RMS_V'])
    assert traced['max_relative_Q_f32']<=1e-5
    checks['traced_frozen_solve_bitwise']={'pass':True,'rows_each_implementation':6,'calls_each_implementation':3,'range_active_N':traced['range_active_N'],'projected_SHA256':array_sha(original)}
    if integration:
        from residual_analysis import path_description,baseline_description,loss_description,psd_description
        ss=rng.normal(0,.001,(30,480));rr=rng.normal(0,.001,(40,480));ss[:,100:400]+=.03;rr[:,100:400]+=.04
        ss=canonical(ss,True);rr=canonical(rr,False);cc=canonical(rr+.0002*rng.normal(size=rr.shape),True)
        bs=np.repeat(REGIONS,3);br=np.repeat(REGIONS,4)
        stats={'source_counts':np.full(10,3),'REAL_counts':np.full(10,4),'valley_bounds':np.array([[.2,.4],[.2,.4]])}
        for n,v in zip(FEATURES,features(rr)):
            stats[n+'_center']=np.stack([v[br==j].mean(0)for j in REGIONS]);stats[n+'_scale']=np.stack([np.maximum(v[br==j].std(0),.0025)for j in REGIONS])
        dirs=draw_directions(rng)
        sf=normalized_feature_bank(ss,bs,stats);rf=normalized_feature_bank(rr,br,stats)
        si=np.stack([np.flatnonzero(bs==j)[[0,1,2,0]]for j in REGIONS]);ri=np.stack([np.flatnonzero(br==j)for j in REGIONS])
        table=objective_table(sf,rf,si,ri,dirs)
        for j,p in enumerate(dirs):
            aa=torch.as_tensor(sf[j][si],device=device);bb=torch.as_tensor(rf[j][ri],device=device);pt=torch.as_tensor(p,device=device)
            marginal=(torch.sort(aa,dim=1).values-torch.sort(bb,dim=1).values).abs().mean((1,2)).cpu().numpy()
            sliced=(torch.sort(aa@pt,dim=1).values-torch.sort(bb@pt,dim=1).values).abs().mean((1,2)).cpu().numpy()
            assert np.allclose(table[:,j,0],marginal,atol=2e-6,rtol=2e-6)
            assert np.allclose(table[:,j,1],sliced,atol=2e-6,rtol=2e-6)
        consume=lambda name,value:None
        streams={k:np.random.Generator(np.random.PCG64(620100+i))for i,k in enumerate(['batch_SOURCE_rows','batch_REAL_A_rows','batch_REAL_B_rows','batch_REAL_control_rows'])}
        p,pa=path_description({'SOURCE':{k:ss for k in STAGES},'REAL_control':{k:cc for k in STAGES}},rr,bs,br)
        n,na=baseline_description(ss,cc,rr,bs,br,rng,consume)
        l,la=loss_description(ss,cc,rr,bs,br,stats,dirs,streams,consume)
        z,za=psd_description(ss,cc,rr,ss,np.arange(30),np.ones(30),bs,br,stats)
        assert len(p['rows'])==110 and len(n['rows'])==11 and la['batch_tables'].shape==(32,3,10,4,2)
        assert l['comparisons']['REAL_REAL']['full_empirical_loss']==0
        assert sum(v.size for k,v in pa.items()if k.endswith('_W1'))==110*(480+480+479+478)
        for obj in [p,n,l,z,checks]:json.loads(json.dumps(clean(obj),allow_nan=False))
        checks['all_summary_paths_JSON_and_original_torch_loss_equivalence']=True
    return {'pass':True,'device':str(device),'checks':checks,'artificial_only':True,'TRAIN_rows_read':0,'new_optimizers':0,'new_parameter_updates':0,'torch_version':torch.__version__,'numpy_version':np.__version__}
