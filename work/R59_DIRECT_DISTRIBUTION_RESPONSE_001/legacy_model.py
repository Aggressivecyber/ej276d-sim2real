from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def input_channels(x):
    b=x[:,:100].mean(1,keepdim=True);s=x-b
    peak=s.amax(1,keepdim=True).clamp_min(.0025)
    return torch.stack((x/.1,s/peak),1)

def up2(x):
    left=torch.cat((x[:,:,:1],x[:,:,:-1]),dim=2)
    right=torch.cat((x[:,:,1:],x[:,:,-1:]),dim=2)
    return torch.stack((.75*x+.25*left,.75*x+.25*right),dim=-1).flatten(2,3)

class Block(nn.Module):
    def __init__(self,c,dilation=1):
        super().__init__();self.body=nn.Sequential(nn.Conv1d(c,c,5,padding=2*dilation,dilation=dilation),nn.LeakyReLU(.2),nn.Conv1d(c,c,3,padding=dilation,dilation=dilation))
    def forward(self,x):return F.leaky_relu(x+self.body(x),.2)

class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0=nn.Sequential(nn.Conv1d(2,16,7,padding=3),nn.LeakyReLU(.2),Block(16))
        self.e1=nn.Sequential(nn.Conv1d(16,32,5,stride=2,padding=2),nn.LeakyReLU(.2),Block(32))
        self.e2=nn.Sequential(nn.Conv1d(32,64,5,stride=2,padding=2),nn.LeakyReLU(.2),*[Block(64,d) for d in (1,2,4,8)])
        self.d1=nn.Sequential(nn.Conv1d(96,32,5,padding=2),nn.LeakyReLU(.2),Block(32))
        self.d0=nn.Sequential(nn.Conv1d(48,16,5,padding=2),nn.LeakyReLU(.2),Block(16))
        self.mean=nn.Conv1d(16,1,7,padding=3)
        self.noise=nn.Sequential(nn.Conv1d(25,16,5,padding=2),nn.LeakyReLU(.2),nn.Conv1d(16,16,5,padding=2),nn.LeakyReLU(.2),nn.Conv1d(16,1,5,padding=2))
        self.project=nn.ModuleList([nn.Conv1d(c,32,1) for c in (16,32,64)])
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.noise[-1].weight);nn.init.zeros_(self.noise[-1].bias)
    def encode(self,x):
        a=self.e0(input_channels(x));b=self.e1(a);c=self.e2(b);return [a,b,c]
    def forward(self,x,z,white):
        a,b,c=self.encode(x);d=self.d1(torch.cat((up2(c),b),1))
        h=self.d0(torch.cat((up2(d),a),1))
        mean=x+.1*self.mean(h)[:,0]
        random=.001*white+.005*torch.tanh(self.noise(torch.cat((h,z[:,:,None].expand(-1,-1,480),white[:,None,:]),1))[:,0])
        return mean+random,mean,random

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(2,16,7,stride=2,padding=3),nn.LeakyReLU(.2),nn.Conv1d(16,32,7,stride=2,padding=3),nn.LeakyReLU(.2),nn.Conv1d(32,64,7,stride=2,padding=3),nn.LeakyReLU(.2),nn.Conv1d(64,1,7,padding=3))
    def forward(self,x):return self.net(input_channels(x)).flatten(1).mean(1)

def charge_psd(x):
    b=x[:,:100].mean(1,keepdim=True);s=x-b;q=s[:,100:400].sum(1)
    p=1-s[:,100:150].sum(1)/q.clamp_min(.01)
    return q,p,s

def physical(x):
    q,p,s=charge_psd(x);peak=s.amax(1).clamp_min(.0025)
    rms=s[:,:100].square().mean(1).sqrt().clamp_min(1e-7)
    d2=torch.diff(s,n=2,dim=1)
    prompt=d2[:,100:148].square().mean(1).sqrt().clamp_min(1e-7)
    post=d2[:,150:218].square().mean(1).sqrt().clamp_min(1e-7)
    return torch.stack((p,q.clamp_min(.01).log(),peak.log(),rms.log(),prompt.log(),post.log()),1)

def pooled_shape(x):
    _,_,s=charge_psd(x);peak=s.amax(1,keepdim=True).clamp_min(.0025)
    return F.avg_pool1d((s/peak)[:,None,:],8)[:,0]

def rank_losses(source,out,edges):
    q,p,_=charge_psd(source);qo,po,_=charge_psd(out)
    lq=q.clamp_min(.01).log();lqo=qo.clamp_min(.01).log()
    dq=lq[:,None]-lq[None,:];dp=p[:,None]-p[None,:]
    bins=torch.bucketize((q.clamp_min(.01)*1000).log(),torch.as_tensor(edges,device=q.device,dtype=q.dtype),right=True)-1
    narrow=(bins[:,None]==bins[None,:])&(bins[:,None]>=0)&(bins[:,None]<8)
    def loss(delta,after,threshold,cap,factor,extra=None):
        use=delta.abs()>threshold
        if extra is not None:use=use&extra
        desired=(factor*delta.abs()).clamp_max(cap)
        violation=F.relu(desired-delta.sign()*(after[:,None]-after[None,:]))
        return violation[use].mean() if use.any() else violation.sum()*0
    return (lqo-lq).square().mean(),loss(dq,lqo,.05,.2,.5),loss(dp,po,.005,.04,.3,narrow),F.relu(.01-qo).mean()

def nce(model,x,y,cfg,rng):
    source=model.encode(x);target=model.encode(y);loss=0
    for k,(s,t) in enumerate(zip(source,target)):
        start=max(1,int(80*s.shape[-1]/480));stop=min(s.shape[-1],int(420*s.shape[-1]/480))
        idx=torch.as_tensor(rng.choice(range(start,stop),cfg['patches_per_layer'],replace=False),device=x.device)
        key=F.normalize(model.project[k](s)[:,:,idx].transpose(1,2).detach(),dim=2)
        query=F.normalize(model.project[k](t)[:,:,idx].transpose(1,2),dim=2)
        logits=torch.bmm(query,key.transpose(1,2))/cfg['NCE_temperature']
        label=torch.arange(len(idx),device=x.device)[None,:].expand(len(x),-1)
        loss=loss+F.cross_entropy(logits.flatten(0,1),label.flatten())
    return loss/len(source)

def sw(a,b,projection):
    quantiles=torch.linspace(.005,.995,128,device=a.device,dtype=a.dtype)
    return (torch.quantile(a@projection,quantiles,dim=0)-torch.quantile(b@projection,quantiles,dim=0)).abs().mean()

def adc(x):
    import numpy as np
    v=np.asarray(x,dtype=np.float32);clipped=np.count_nonzero((v<-.5)|(v>.5))
    out=(-.5+np.rint((np.clip(v,-.5,.5)+.5)*16383)/16383).astype(np.float32)
    return out,{'clip_count':int(clipped),'pre_ADC_finite':bool(np.isfinite(v).all()),'bits':14,'Vrange':[-.5,.5]}

def infer(model,x,z,w,chunk=128):
    import numpy as np
    model.eval();ans=[];means=[];noise=[]
    with torch.no_grad():
        for i in range(0,len(x),chunk):
            y,m,n=model(torch.from_numpy(np.asarray(x[i:i+chunk],dtype='f4')),torch.from_numpy(z[i:i+chunk]),torch.from_numpy(w[i:i+chunk]))
            ans.append(y.numpy());means.append(m.numpy());noise.append(n.numpy())
    return np.concatenate(ans),np.concatenate(means),np.concatenate(noise)
