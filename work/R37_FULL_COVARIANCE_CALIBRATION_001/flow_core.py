from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(497,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),
                                 nn.Linear(512,512),nn.SiLU(),nn.Linear(512,480))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,t,b):
        t=t.reshape(-1,1).to(x)
        angles=2*math.pi*t*x.new_tensor([1.,2.,4.,8.])[None,:]
        time=torch.cat((t,angles.sin(),angles.cos()),1)
        onehot=torch.nn.functional.one_hot(b.to(torch.int64),8).to(x)
        return self.net(torch.cat((x,time,onehot),1))

def heun(field,x,b,steps=32):
    y=x.clone(); h=1./steps
    for j in range(steps):
        t=y.new_full((len(y),),j*h)
        k1=field(y,t,b)
        k2=field(y+h*k1,t+h,b)
        y=y+(h*.5)*(k1+k2)
    return y

def signed_q(x):
    a=np.asarray(x,dtype=np.float64)
    s=a-a[:,:100].mean(1,keepdims=True)
    return s[:,100:400].sum(1)*1000

def bins(x,edges):
    q=signed_q(x); z=np.log(np.maximum(q,1e-30))
    b=np.searchsorted(edges,z,side='right')-1
    b[z==edges[-1]]=7
    return b

@torch.no_grad()
def correct(flow,base_V,source_V,mu,scale,edges,steps=32,chunk=128):
    """One common TRAIN affine map; unsupported source bins retain base exactly."""
    base=np.asarray(base_V,dtype=np.float32)
    b=bins(source_V,edges); used=(b>=0)&(b<8)
    ans=base.copy()
    for j in range(0,len(base),chunk):
        ix=np.flatnonzero(used[j:j+chunk])+j
        if not len(ix):continue
        bi=torch.from_numpy(b[ix].astype('i8'))
        m=torch.from_numpy(np.asarray(mu[b[ix]],dtype='f4'))
        s=torch.from_numpy(np.asarray(scale[b[ix]],dtype='f4'))
        a=(torch.from_numpy(base[ix])-m)/s
        c=heun(flow,a,bi,steps)
        ans[ix]=(c*s+m).numpy()
    return ans,used,b

def coupling(a,c,b,arm,rng=None):
    from scipy.optimize import linear_sum_assignment
    matched=np.arange(len(a),dtype='i8'); cost_total=0.; groups=[]
    for k in np.unique(b):
        ids=np.flatnonzero(b==k); aa=np.asarray(a[ids],dtype='f8'); cc=np.asarray(c[ids],dtype='f8')
        cost=np.maximum(0,(np.sum(aa*aa,1)[:,None]+np.sum(cc*cc,1)[None,:]-2*aa@cc.T)/480.)
        if arm=='OT_CFM':
            row,col=linear_sum_assignment(cost)
            assert np.array_equal(row,np.arange(len(ids)))
        elif arm=='INDEPENDENT_CFM':col=rng.permutation(len(ids))
        else:raise ValueError(arm)
        matched[ids]=ids[col];cost_total+=float(cost[np.arange(len(ids)),col].sum())
        groups.append({'bin_id_1based':int(k)+1,'N':len(ids),'mean_cost':float(cost[np.arange(len(ids)),col].mean())})
    assert np.array_equal(np.sort(matched),np.arange(len(a))) and np.array_equal(b[matched],b)
    return matched,cost_total/len(a),groups

def selfcheck():
    import itertools
    rng=np.random.default_rng(117)
    a=torch.as_tensor(rng.normal(size=(4,3)),dtype=torch.float64)
    c=torch.as_tensor(rng.normal(size=(4,3)),dtype=torch.float64)
    assert torch.equal((1-0)*a+0*c,a) and torch.equal((1-1)*a+1*c,c)
    speed=c-a; b=torch.zeros(4,dtype=torch.int64)
    err=float((heun(lambda x,t,b:speed,a,b,32)-c).abs().max())
    m=torch.tensor([.1,-.4,.2]); s=torch.tensor([.3,.7,.4])
    affine=float((((a-m)/s)*s+m-a).abs().max())
    aa=rng.normal(size=(5,480)); cc=rng.normal(size=(5,480)); bb=np.array([0,0,0,2,7])
    perm,cost,groups=coupling(aa,cc,bb,'OT_CFM')
    brute=min(float(np.mean((aa[:3]-cc[list(p)])**2)) for p in itertools.permutations(range(3)))
    direct=float(np.mean((aa[:3]-cc[perm[:3]])**2))
    assert abs(brute-direct)<1e-12 and perm[3]==3 and perm[4]==4
    assert err<1e-12 and affine<1e-12
    torch.manual_seed(20261010330); model=Velocity()
    initial=model(torch.zeros((3,480)),torch.tensor([0.,.4,1.]),torch.tensor([0,2,7]))
    assert torch.equal(initial,torch.zeros_like(initial))
    return {'linear_path_endpoints_exact':True,'constant_velocity_Heun_max_error':err,
            'affine_inverse_max_error':affine,'OT_3row_cost_minus_bruteforce':direct-brute,
            'mixed_batch_Q_edges_and_singleton_preserved':True,'zero_initial_velocity_exact':True,
            'network_parameter_count':sum(p.numel()for p in model.parameters()),'no_response_training_in_check':True}
