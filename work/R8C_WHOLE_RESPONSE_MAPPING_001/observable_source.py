import torch
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
