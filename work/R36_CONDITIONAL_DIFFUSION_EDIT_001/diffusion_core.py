from __future__ import annotations
import math, hashlib
import numpy as np
import torch
from torch import nn

def schedule():
    t=np.arange(101,dtype='f8')
    f=np.cos(((t/100.+.008)/1.008)*np.pi/2.)**2
    target=f/f[0]
    beta=np.r_[0.,np.minimum(1.-target[1:]/target[:-1],.999)]
    alpha=1.-beta; abar=np.cumprod(alpha)
    variance=np.zeros(101,'f8'); variance[1:]=beta[1:]*(1.-abar[:-1])/(1.-abar[1:])
    assert abar[0]==1. and variance[1]==0. and np.all(beta[1:]>0)
    return dict(beta=beta,alpha=alpha,abar=abar,posterior_variance=variance,unclipped_target_abar=target)

def physical_condition(wave):
    a=np.asarray(wave,dtype='f8'); s=a-a[:,:100].mean(1,keepdims=True)
    q=s[:,100:400].sum(1)*1000.; fast=s[:,100:150].sum(1)*1000.
    if not np.isfinite(q).all() or not (q>0).all():
        bad=np.flatnonzero(~np.isfinite(q)|(q<=0)); raise ValueError(('invalid_Q_condition',bad.tolist(),q[bad].tolist()))
    c=np.column_stack((np.log(q),1.-fast/q))
    if not np.isfinite(c).all():raise ValueError('nonfinite condition; no clipping/fallback')
    return c

class ResidualBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.n1=nn.GroupNorm(8,cin);self.c1=nn.Conv1d(cin,cout,3,padding=1)
        self.n2=nn.GroupNorm(8,cout);self.film=nn.Linear(128,2*cout);self.c2=nn.Conv1d(cout,cout,3,padding=1)
        self.skip=nn.Identity()if cin==cout else nn.Conv1d(cin,cout,1)
    def forward(self,x,e):
        h=self.c1(torch.nn.functional.silu(self.n1(x)))
        scale,shift=self.film(e).chunk(2,1)
        h=self.n2(h)*(1.+scale[:,:,None])+shift[:,:,None]
        return self.skip(x)+self.c2(torch.nn.functional.silu(h))

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128);self.attn=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
    def forward(self,x):
        z=x.transpose(1,2);n=self.norm(z);a=self.attn(n,n,n,need_weights=False)[0]
        return (z+a).transpose(1,2)

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Sequential(nn.Linear(27,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.register_buffer('position',torch.linspace(-1.,1.,480).reshape(1,1,480))
        self.register_buffer('frequency',torch.tensor([1.,2.,4.,8.,16.,32.,64.,128.]))
        self.input=nn.Conv1d(2,32,3,padding=1)
        self.enc=nn.ModuleList([nn.ModuleList([ResidualBlock(c,c),ResidualBlock(c,c)])for c in [32,64,128,128]])
        self.down=nn.ModuleList([nn.Conv1d(a,b,3,stride=2,padding=1)for a,b in [(32,64),(64,128),(128,128)]])
        self.attention=Attention()
        self.up=nn.ModuleList([nn.Conv1d(a,b,3,padding=1)for a,b in [(128,128),(128,64),(64,32)]])
        self.dec=nn.ModuleList([nn.ModuleList([ResidualBlock(2*c,c),ResidualBlock(c,c)])for c in [128,64,32]])
        self.final_norm=nn.GroupNorm(8,32);self.final=nn.Conv1d(32,1,3,padding=1)
        nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)
    def forward(self,y,t,b,c):
        time=t.to(y).reshape(-1,1)/100.;angle=2*math.pi*time*self.frequency[None,:].to(y)
        onehot=torch.nn.functional.one_hot(b.to(torch.int64),8).to(y)
        e=self.embedding(torch.cat((time,angle.sin(),angle.cos(),onehot,c.to(y)),1))
        h=self.input(torch.cat((y[:,None,:],self.position.to(y).expand(len(y),-1,-1)),1));skips=[]
        for j,blocks in enumerate(self.enc):
            h=blocks[0](h,e)
            if j==3:h=self.attention(h)
            h=blocks[1](h,e);skips.append(h)
            if j<3:h=self.down[j](h)
        for j,(up,blocks)in enumerate(zip(self.up,self.dec)):
            h=up(torch.nn.functional.interpolate(h,scale_factor=2,mode='nearest'))
            h=torch.cat((h,skips[2-j]),1);h=blocks[1](blocks[0](h,e),e)
        return self.final(torch.nn.functional.silu(self.final_norm(h)))[:,0,:]

def record_noise(seed):
    # Exactly one row-major (initial + reverse t50..2) x 480 normal matrix.
    return np.random.Generator(np.random.PCG64(int(seed))).normal(size=(50,480)).astype('f4')

@torch.no_grad()
def reverse_chain(model,guide,b,c,mu,scale,seeds,coeff,chunk=64,fixed_condition_raw=None):
    guide=np.asarray(guide,dtype='f4');b=np.asarray(b,dtype='i8');used=(b>=0)&(b<8)
    result=guide.copy();raw_c=np.full((len(guide),2),np.nan,'f8');hashes=np.full(len(guide),'not_consumed',dtype='<U64')
    ix_all=np.flatnonzero(used)
    if len(ix_all):
        if fixed_condition_raw is None:raw_c[ix_all]=physical_condition(guide[ix_all])
        else:
            supplied=np.asarray(fixed_condition_raw,dtype='f8');assert supplied.shape==(len(guide),2)
            if not np.isfinite(supplied[ix_all]).all():raise ValueError('fixed probe condition not finite')
            raw_c[ix_all]=supplied[ix_all]
    condition=(raw_c-c['mean'])/c['sample_sd']
    for j in range(0,len(ix_all),chunk):
        ix=ix_all[j:j+chunk];noise=np.stack([record_noise(seeds[i])for i in ix],0)
        hashes[ix]=[hashlib.sha256(n.tobytes()).hexdigest()for n in noise]
        bi=torch.from_numpy(b[ix]);m=torch.from_numpy(np.asarray(mu[b[ix]],dtype='f4'));s=torch.from_numpy(np.asarray(scale[b[ix]],dtype='f4'))
        cc=torch.from_numpy(condition[ix].astype('f4'));a=(torch.from_numpy(guide[ix])-m)/s
        abar=float(coeff['abar'][50]);y=(abar**.5)*a+((1.-abar)**.5)*torch.from_numpy(noise[:,0,:])
        for t in range(50,0,-1):
            tt=torch.full((len(ix),),t,dtype=torch.int64)
            eps=model(y,tt,bi,cc);beta=float(coeff['beta'][t]);alpha=float(coeff['alpha'][t]);at=float(coeff['abar'][t])
            y=(y-(beta/(1.-at)**.5)*eps)/(alpha**.5)
            if t>1:y=y+float(coeff['posterior_variance'][t])**.5*torch.from_numpy(noise[:,51-t,:])
        result[ix]=(y*s+m).numpy()
        if not np.isfinite(result[ix]).all():raise FloatingPointError('nonfinite edited wave; preserve and stop')
    return result,used,raw_c,condition,hashes

def formula_check():
    coeff=schedule();rng=np.random.Generator(np.random.PCG64(17))
    x=rng.normal(size=(5,7));eps=rng.normal(size=x.shape);rows=[]
    for t in [1,50,100]:
        a=coeff['abar'][t];ap=coeff['abar'][t-1];be=coeff['beta'][t];al=coeff['alpha'][t]
        y=np.sqrt(a)*x+np.sqrt(1.-a)*eps;recovered=(y-np.sqrt(1.-a)*eps)/np.sqrt(a)
        m1=(y-be/np.sqrt(1.-a)*eps)/np.sqrt(al)
        m2=np.sqrt(ap)*be/(1.-a)*x+np.sqrt(al)*(1.-ap)/(1.-a)*y
        rows.append({'t':t,'known_epsilon_recovery_maxabs':float(np.max(np.abs(recovered-x))),
            'posterior_mean_equivalence_maxabs':float(np.max(np.abs(m1-m2)))})
    assert all(r['known_epsilon_recovery_maxabs']<1e-10 and r['posterior_mean_equivalence_maxabs']<1e-10 for r in rows)
    assert coeff['posterior_variance'][1]==0.
    return {'small_artificial_only':True,'rows':rows,'v1_exact_zero':True,'abar_is_actual_beta_cumprod':True,
        'last_beta':float(coeff['beta'][-1]),'t100_target_and_actual_abar':[float(coeff['unclipped_target_abar'][-1]),float(coeff['abar'][-1])],
        'no_scientific_response_training':True}
