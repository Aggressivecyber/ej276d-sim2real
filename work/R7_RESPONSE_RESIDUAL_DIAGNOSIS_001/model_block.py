from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


class FlexibleResponse(nn.Module):
    """G4-conditioned positive smooth expectation, random signal parameters and noise.

    The input condition is derived from a clean G4 expectation, not a REAL event.
    Q is an explicitly learned charge scale in V ns. Integer CFD windowing and
    ADC quantization reproduce the saved interface, with straight-through ADC
    gradients only during training. No real waveform is a generator input.
    """
    def __init__(self, log_gain, noise_log_mean, noise_log_sd, noise_spectrum):
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,10))
        self.random = nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,10))
        nn.init.zeros_(self.mean[-1].weight);nn.init.zeros_(self.mean[-1].bias)
        nn.init.zeros_(self.random[-1].weight);nn.init.zeros_(self.random[-1].bias)
        self.log_gain=nn.Parameter(torch.tensor(float(log_gain)))
        self.energy_slope=nn.Parameter(torch.tensor(0.0))
        self.random_scale=nn.Parameter(torch.tensor([.7,.25,.2]))
        self.decay_scale=nn.Parameter(torch.zeros(7))
        self.noise_log_mean=nn.Parameter(torch.tensor(float(noise_log_mean)))
        self.noise_log_sd=nn.Parameter(torch.tensor(float(noise_log_sd)))
        self.register_buffer('noise_spectrum',torch.tensor(noise_spectrum,dtype=torch.float32))
        self.register_buffer('time',torch.arange(512*20,dtype=torch.float32)/20)
        self.register_buffer('taus',torch.tensor([8.,13.,35.,59.,120.,270.,460.]))
        centers=np.r_[np.arange(0,34,2),np.arange(40,104,8),np.arange(112,401,32)].astype('f4')
        widths=np.r_[np.full(17,3.),np.full(8,12.),np.full(len(centers)-25,40.)].astype('f4')
        self.register_buffer('time_centers_ns',torch.tensor(centers))
        self.register_buffer('time_widths_ns',torch.tensor(widths))
        nknots=len(centers)
        self.temporal_mean=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,nknots))
        self.temporal_random=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,nknots))
        for net in [self.temporal_mean,self.temporal_random]:
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        self.last_time_coefficients=None

    def forward(self,c,q_source,z,white,quantize=True):
        n=len(c);mu=self.mean(c);ran=self.random(torch.cat([c,z],dim=1))
        # The visible hadron fraction enters the mixture directly as well as
        # both networks. Changes of G4 composition cannot be erased by a z-only
        # real-template sampling operation.
        h=c[:,1].clamp(0,1)
        ee=torch.tensor([.01,.68,.15,.01,.01,.13,.01],device=c.device)
        had=torch.tensor([.01,.48,.01,.10,.01,.01,.38],device=c.device)
        base=(1-h[:,None])*ee+h[:,None]*had
        latent=ran[:,:7]+z[:,0,None]*torch.tensor([-1.,-1.,-.3,.5,.4,.7,1.],device=c.device)*self.random_scale[0]
        logits=torch.log(base.clamp_min(1e-4))+mu[:,:7]+latent
        weights=torch.softmax(logits,dim=1)
        rise=2.+12.*torch.sigmoid(mu[:,7]+ran[:,7]+self.random_scale[1]*z[:,1]-.8)
        width=torch.exp(.30*torch.tanh(mu[:,8]+ran[:,8]+.3*z[:,2]))
        tau=self.taus[None,:]*torch.exp(.6*torch.tanh(self.decay_scale))[None,:]*width[:,None]
        # Continuous sub-ns event phase; fixed predeclared distribution, not an
        # integer upsampling grid. The same phase acts on pulse and time basis.
        phase_ns=.5*torch.tanh(z[:,5])
        u=(self.time[None,None,:]-128.-phase_ns[:,None,None]).clamp_min(0)
        r=rise[:,None,None]
        # Stable positive difference-exponential, including near tau==rise.
        delta=tau[:,:,None]-r
        close=delta.abs()<.001
        den=torch.where(close,torch.ones_like(delta),delta)
        kernels=(torch.exp(-u/tau[:,:,None])-torch.exp(-u/r))/den
        limit=u/(r*r)*torch.exp(-u/r)
        kernels=torch.where(close,limit,kernels).clamp_min(0)
        shape=(weights[:,:,None]*kernels).sum(dim=1)
        tc=self.temporal_mean(c)+self.temporal_random(torch.cat([c,z],dim=1))
        basis=torch.exp(-.5*((u-self.time_centers_ns[None,:,None])/self.time_widths_ns[None,:,None])**2)
        basis=basis/basis.sum(dim=1,keepdim=True).clamp_min(1e-8)
        correction=(tc[:,:,None]*basis).sum(dim=1).clamp(-4,4)
        shape=shape*torch.exp(correction)
        self.last_time_coefficients=tc
        slope=1.+.4*torch.tanh(self.energy_slope)
        q=q_source*torch.exp(self.log_gain+(slope-1)*c[:,0]+.35*torch.tanh(mu[:,9])+
                            self.random_scale[2]*z[:,3]+.3*torch.tanh(ran[:,9]))
        # Normalize the clean expectation by its full recorded integral. The
        # learned gain includes finite-window charge losses during matching.
        shape=shape.reshape(n,512,20).mean(dim=2)
        clean=shape/shape.sum(dim=1,keepdim=True).clamp_min(1e-8)*q[:,None]
        # Stationary covariance learned only from TRAIN pre-trigger statistics.
        ft=torch.fft.rfft(white,dim=1)*self.noise_spectrum[None,:]
        noise=torch.fft.irfft(ft,n=512,dim=1)
        noise=noise/noise.std(dim=1,keepdim=True).clamp_min(1e-8)
        rms=torch.exp(self.noise_log_mean+self.noise_log_sd.clamp(0,.8)*z[:,4])
        full=clean+rms[:,None]*noise
        if quantize:
            code=((full+.5)*16383.).round().clamp(0,16383.)
            quant=code/16383.-.5
            full=full+(quant-full).detach() if self.training else quant
        # Full pre-trigger baseline for CFD, exactly the main-peak connected
        # below-threshold branch used by the saved interface.
        b=full[:,:102].mean(dim=1);norm=full-b[:,None]
        peak=norm.max(dim=1).values;pi=norm.argmax(dim=1)
        tt=torch.arange(512,device=c.device)[None,:]
        below=(tt<pi[:,None])&(norm<.5*peak[:,None])
        tp=torch.where(below,tt,torch.full_like(tt,-1)).max(dim=1).values+1
        tp=torch.where(below.any(dim=1),tp,pi)
        left=tp-120;indices=(left[:,None]+torch.arange(480,device=c.device)[None,:]).clamp(0,511)
        out=torch.gather(full,1,indices)
        meta={'weights':weights,'rise_parameter_ns':rise,'width_factor':width,'Qfull_Vns':q,
              'noise_RMS_parameter_V':rms,'event_phase_ns':phase_ns,'time_coefficients':tc,'CFD_full_index':tp,'window_left':left,
              'padding':(left<0)|(left+480>512),'quantization_clip':(full.abs()>=.5).any(dim=1)}
        return out,meta


def observables(v):
    """Differentiable common saved-window features; no paired waveform loss."""
    x=v-v[:,:100].mean(dim=1,keepdim=True)
    peak=x.max(dim=1).values.clamp_min(1e-6)
    shape=x/peak[:,None]
    qt=x[:,100:400].sum(dim=1).clamp_min(1e-6)
    qf=x[:,100:150].sum(dim=1)
    ps=1.-qf/qt
    sigma=x[:,:100].std(dim=1)
    tail=x[:,200:400].sum(dim=1)/qt
    tailmean=x[:,300:400].mean(dim=1)/peak
    pi=x.argmax(dim=1);tt=torch.arange(480,device=v.device)[None,:]
    crosses=[]
    for fraction in [.1,.9]:
        target=peak*fraction
        hit=(x[:,:-1]<=target[:,None])&(x[:,1:]>=target[:,None])&(x[:,1:]>x[:,:-1])
        hit&=(tt[:,:-1]>=100)&(tt[:,:-1]<pi[:,None])
        ix=torch.where(hit,tt[:,:-1],torch.full_like(tt[:,:-1],99)).max(dim=1).values
        low=x.gather(1,ix[:,None]).squeeze(1);high=x.gather(1,(ix+1)[:,None]).squeeze(1)
        crosses.append(ix.float()+(target-low)/(high-low).clamp_min(1e-6))
    rise=(crosses[1]-crosses[0]).clamp(0,50)
    scalar=torch.stack([ps,torch.log(qt*1000),torch.log(peak),rise,
                        tail,tailmean,torch.log(sigma.clamp_min(1e-7)),sigma/peak],dim=1)
    return scalar,shape,x


def distribution_loss(fake,real,scale,projection):
    fs,fw,_=observables(fake);rs,rw,_=observables(real)
    # Sorted minibatch marginal discrepancies plus random projections of
    # smooth shape coordinates retain the joint response distribution.
    f=fs/scale; r=rs/scale
    marginal=(f.sort(dim=0).values-r.sort(dim=0).values).abs().mean()
    shape=(fw@projection);target=(rw@projection)
    sliced=(shape.sort(dim=0).values-target.sort(dim=0).values).abs().mean()
    joint=torch.cat([f,fw[:,100:220:4]],dim=1)
    other=torch.cat([r,rw[:,100:220:4]],dim=1)
    # Cross-covariances supplement marginal distances without event pairing.
    fc=joint-joint.mean(dim=0);rc=other-other.mean(dim=0)
    cov=((fc.T@fc-rc.T@rc)/max(1,len(fake)-1)).square().mean().sqrt()
    regions=[(0,100,.05),(100,150,1.),(150,220,.6),(220,400,.25),(400,480,.05)]
    fm=fw.sort(dim=0).values;rm=rw.sort(dim=0).values
    time_marginal=sum(weight*(fm[:,a:b]-rm[:,a:b]).abs().mean() for a,b,weight in regions)/sum(s[2] for s in regions)
    # Time-point marginals compare distributions, not unpaired event targets.
    return marginal+.35*sliced+.08*cov+4.*time_marginal,{'marginal':marginal,'shape_SW':sliced,'joint_cov':cov,'balanced_time_marginal':time_marginal}


def noise_statistics(real):
    b=real[:,:100].astype(np.float64);b-=b.mean(axis=1,keepdims=True)
    sigma=b.std(axis=1,ddof=1);ls=np.log(sigma[sigma>0])
    power=np.mean(np.abs(np.fft.rfft(b,axis=1))**2,axis=0)
    # 100ns data cannot resolve frequencies below10MHz: hold first positive
    # ordinate down to the first512-point frequency, zero DC explicitly.
    power[0]=power[1]
    spec=np.sqrt(np.interp(np.fft.rfftfreq(512),np.fft.rfftfreq(100),power))
    spec[0]=0;spec/=np.sqrt(np.mean(spec[1:]**2))
    return float(ls.mean()),float(ls.std(ddof=1)),spec.astype(np.float32),power

# Constructor alias retains the compact common run/replay interface.
SmoothResponse=FlexibleResponse
