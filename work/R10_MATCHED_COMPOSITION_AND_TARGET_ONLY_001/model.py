from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from observable_source import observables


class Block(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.a=nn.Conv1d(24,24,5,padding=2*dilation,dilation=dilation)
        self.b=nn.Conv1d(24,24,5,padding=2*dilation,dilation=dilation)
    def forward(self,x):return (x+self.b(F.silu(self.a(F.silu(x)))))/math.sqrt(2.)


class WholeResponse(nn.Module):
    def __init__(self,log_gain,noise_log_mean,noise_log_sd,initial_fir,logq_center,pulse_enabled=True):
        super().__init__();self.pulse_enabled=bool(pulse_enabled)
        self.encoder=nn.Sequential(nn.Conv1d(8,24,5,padding=2),*[Block(d) for d in [1,2,4,8,16,32,64]],nn.SiLU())
        self.gain=nn.Sequential(nn.Linear(4,16),nn.SiLU(),nn.Linear(16,1))
        self.mu_out=nn.Conv1d(24,1,5,padding=2)
        self.latent=nn.Linear(20,24);self.pulse_white=nn.Conv1d(1,24,5,padding=2)
        self.pulse_net=nn.Sequential(*[Block(d) for d in [1,4,16]],nn.SiLU(),nn.Conv1d(24,1,5,padding=2))
        self.base=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,32))
        for layer in [self.gain[-1],self.mu_out,self.pulse_net[-1],self.base[-1]]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.gain[-1].bias.fill_(float(log_gain))
            self.base[-1].bias[:31].copy_(torch.tensor(initial_fir,dtype=torch.float32))
            self.base[-1].bias[31]=float(noise_log_mean)
        self.base_latent_logsd=nn.Parameter(torch.tensor([noise_log_sd,0.,0.,0.],dtype=torch.float32))
        self.register_buffer('logq_center',torch.tensor(float(logq_center)))
        self.register_buffer('time',(torch.arange(512,dtype=torch.float32)-128.)/256.)
        self.register_buffer('mask',F.pad(torch.ones(480),(8,24)))

    def forward(self,x,c,q,zp,zb,wp,wb,with_meta=False):
        padded=F.pad(x,(8,24));n=len(x)
        ch=torch.cat([padded[:,None,:]/.1,self.mask[None,None,:].expand(n,1,512),
            self.time[None,None,:].expand(n,1,512),c[:,:,None].expand(n,4,512),
            (q.clamp_min(1e-8).log()-self.logq_center)[:,None,None].expand(n,1,512)],dim=1)
        h=self.encoder(ch);gain=self.gain(c).exp().squeeze(1)
        mean=gain[:,None]*padded+.1*self.mu_out(h).squeeze(1)
        if self.pulse_enabled:
            ph=h+self.latent(torch.cat([zp,zb],dim=1))[:,:,None]+self.pulse_white(wp[:,None,:])
            pulse=.1*self.pulse_net(ph).squeeze(1)
        else:pulse=torch.zeros_like(mean)
        bp=self.base(torch.cat([c,zb],dim=1));fir=bp[:,:31]
        fir=fir/torch.linalg.vector_norm(fir,dim=1,keepdim=True).clamp_min(1e-8)
        rms=(bp[:,31]+(zb*self.base_latent_logsd[None,:]).sum(dim=1)).exp()
        # Mathematical convolution, centered zero padding; no forced zero sum.
        base=F.conv1d(wb[None,:,:],fir.flip(1)[:,None,:],padding=15,groups=n).squeeze(0)*rms[:,None]
        raw=mean+pulse+base
        raw_code=((raw+.5)*16383.).round();code=raw_code.clamp(0,16383.)
        quant=code/16383.-.5
        full=raw+(quant-raw).detach() if self.training else quant
        b=full[:,:102].mean(dim=1);norm=full-b[:,None]
        peak=norm.max(dim=1).values;pi=norm.argmax(dim=1)
        tt=torch.arange(512,device=x.device)[None,:]
        below=(tt<pi[:,None])&(norm<.5*peak[:,None])
        tp=torch.where(below,tt,torch.full_like(tt,-1)).max(dim=1).values+1
        tp=torch.where(below.any(dim=1),tp,pi)
        left=tp-120;ix=(left[:,None]+torch.arange(480,device=x.device)[None,:]).clamp(0,511)
        out=torch.gather(full,1,ix)
        if not with_meta:return out
        return out,{'raw_full_V':raw,'quantized_full_V':full,'mean_full_V':mean,'pulse_full_V':pulse,
            'baseline_full_V':base,'gain':gain,'baseline_RMS_parameter_V':rms,'baseline_FIR':fir,
            'CFD_full_index':tp,'window_left':left,'padding':(left<0)|(left+480>512),
            'ADC_clip_count':((raw_code<0)|(raw_code>16383)).sum(dim=1),
            'invalid_full_peak':(~torch.isfinite(peak))|(peak<=0),'no_prepeak_halfcross':~below.any(dim=1)}


class CriticScale(nn.Module):
    def __init__(self):
        super().__init__();sn=nn.utils.spectral_norm
        self.net=nn.Sequential(sn(nn.Conv1d(2,24,5,padding=2)),nn.LeakyReLU(.2),
            *sum(([sn(nn.Conv1d(24,24,5,padding=2*d,dilation=d)),nn.LeakyReLU(.2)] for d in [1,2,4]),[]),
            sn(nn.Conv1d(24,1,1)))
    def forward(self,x):return self.net(x).mean(dim=(1,2))


class Critic(nn.Module):
    def __init__(self):super().__init__();self.scales=nn.ModuleList([CriticScale() for _ in range(3)])
    def forward(self,v):
        _,shape,_=observables(v);x=torch.stack([v/.1,shape],dim=1)
        return sum(net(x if k==1 else F.avg_pool1d(x,k)) for k,net in zip([1,2,4],self.scales))/3.


def train_initialization(real,xtrain,qtrain):
    b=np.asarray(real[:,:100],dtype='f8');b=b-b.mean(axis=1,keepdims=True)
    ls=np.log(b.std(axis=1,ddof=1));power=np.mean(np.abs(np.fft.rfft(b,axis=1))**2,axis=0)
    # First positive ordinate holds below the 100ns record's spectral resolution.
    power[0]=power[1]
    spec=np.sqrt(np.interp(np.fft.rfftfreq(512),np.fft.rfftfreq(100),power))
    impulse=np.fft.irfft(spec,n=512);fir=np.r_[impulse[-15:],impulse[:16]]
    fir=fir/np.linalg.norm(fir)
    r=np.asarray(real,dtype='f8');r=r-r[:,:100].mean(axis=1,keepdims=True)
    s=np.asarray(xtrain,dtype='f8');s=s-s[:,:100].mean(axis=1,keepdims=True)
    gain=float(np.median(r[:,100:400].sum(axis=1))/np.median(s[:,100:400].sum(axis=1)))
    return {'log_gain':float(np.log(gain)),'noise_log_mean':float(ls.mean()),'noise_log_sd':float(ls.std(ddof=1)),
        'initial_fir':fir.astype('f4').tolist(),'logq_center':float(np.log(qtrain).mean())}
