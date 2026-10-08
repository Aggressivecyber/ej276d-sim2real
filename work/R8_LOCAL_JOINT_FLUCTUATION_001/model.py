from __future__ import annotations
import math
import torch
from torch import nn
from response_base import FlexibleResponse,observables,distribution_loss,noise_statistics

class Response(FlexibleResponse):
    def __init__(self,log_gain,noise_log_mean,noise_log_sd,noise_spectrum,signal_enabled=False):
        super().__init__(log_gain,noise_log_mean,noise_log_sd,noise_spectrum)
        self.signal_w=nn.Parameter(torch.zeros(4))
        self.signal_b=nn.Parameter(torch.tensor(math.log(.1/.9)))
        self.signal_a=nn.Parameter(torch.zeros(2))
        self.signal_enabled=bool(signal_enabled)
    def signal_scale(self,c):return .02*torch.sigmoid(self.signal_b+c@self.signal_w)
    def signal_kernel(self):
        k=torch.stack([.5*torch.tanh(self.signal_a[0]),torch.ones_like(self.signal_a[0]),.5*torch.tanh(self.signal_a[1])])
        return k/torch.linalg.vector_norm(k)
    def signal_term(self,c,clean,white):
        if not self.signal_enabled:return torch.zeros_like(clean)
        if white is None or white.shape!=clean.shape:raise ValueError('fixed independent signal white missing')
        kernel=self.signal_kernel()
        xi=torch.nn.functional.conv1d(white[:,None,:],kernel.flip(0)[None,None,:],padding=1)[:,0,:]
        peak=clean.argmax(dim=1).detach()
        lag=(torch.arange(512,device=clean.device)[None,:]-peak[:,None]).clamp_min(0)
        gate=-torch.expm1(-lag.float()/10.)
        positive=clean>0
        root=torch.where(positive,torch.sqrt(torch.where(positive,clean,torch.ones_like(clean))/.1),torch.zeros_like(clean))
        return self.signal_scale(c)[:,None]*root*gate*xi

SmoothResponse=Response
