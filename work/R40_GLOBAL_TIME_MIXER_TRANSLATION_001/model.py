from __future__ import annotations
from legacy_model import *
from legacy_model import Translator as LegacyTranslator
import numpy as np

class OriginalTranslator(LegacyTranslator):
    def __init__(self):
        super().__init__()
        self.style_film=nn.Sequential(nn.Linear(8,128),nn.LeakyReLU(.2),nn.Linear(128,224))
        nn.init.zeros_(self.style_film[-1].weight);nn.init.zeros_(self.style_film[-1].bias)
    def apply_time_mixer(self,h):return h
    def forward(self,x,z,white,style=None):
        if style is None:style=x.new_zeros((len(x),8))
        p=self.style_film(style)-self.style_film(torch.zeros_like(style))
        g64,b64,g32,b32,g16,b16=torch.split(p,[64,64,32,32,16,16],dim=1)
        a,b,c=self.encode(x);c=c*(1+g64[:,:,None])+b64[:,:,None]
        d=self.d1(torch.cat((up2(c),b),1));d=d*(1+g32[:,:,None])+b32[:,:,None]
        h=self.d0(torch.cat((up2(d),a),1));h=h*(1+g16[:,:,None])+b16[:,:,None]
        h=self.apply_time_mixer(h)
        signal=x+.1*self.mean(h)[:,0]
        electronic=.001*white+.005*torch.tanh(self.noise(torch.cat((h,z[:,:,None].expand(-1,-1,480),white[:,None,:]),1))[:,0])
        return signal+electronic,signal,electronic

class Translator(OriginalTranslator):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(20261010040)
            self.time_mixer=nn.Sequential(nn.Linear(480,128,bias=True),nn.GELU(approximate='none'),nn.Linear(128,480,bias=True))
            nn.init.zeros_(self.time_mixer[-1].weight);nn.init.zeros_(self.time_mixer[-1].bias)
        assert sum(p.numel() for p in self.time_mixer.parameters())==123488
        self.disable_time_mixer=False
        self.last_time_mixer_output=None
    def apply_time_mixer(self,h):
        delta=self.time_mixer(h)
        self.last_time_mixer_output=delta.detach()
        return h if self.disable_time_mixer else h+delta

def style_view(x):
    s=x-x[:,:100].mean(1,keepdim=True)
    u=s/s.amax(1,keepdim=True).clamp_min(.0025)
    return F.avg_pool1d(u[:,None,100:400],5)[:,0]

class StyleReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(120,128),nn.LeakyReLU(.2),nn.Linear(128,128),nn.LeakyReLU(.2),nn.Linear(128,8))
    def forward(self,x,signal):return self.net(torch.cat((style_view(x),style_view(signal)),1))

def infer(model,x,z,w,style,chunk=128):
    model.eval();answer=[];signals=[];electronics=[]
    with torch.no_grad():
        for i in range(0,len(x),chunk):
            arrays=[torch.from_numpy(np.asarray(a[i:i+chunk],dtype='f4'))for a in [x,z,w,style]]
            y,s,n=model(*arrays)
            answer.append(y.numpy());signals.append(s.numpy());electronics.append(n.numpy())
    return np.concatenate(answer),np.concatenate(signals),np.concatenate(electronics)
