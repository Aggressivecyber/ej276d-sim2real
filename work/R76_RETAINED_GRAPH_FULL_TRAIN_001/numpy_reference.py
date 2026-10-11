"""Independent NumPy float64 head/response reference, no Torch imports."""
import numpy as np

def q(x):return x[...,100:400].sum(-1)-3*x[...,:100].sum(-1)
def sigmoid(x):
    e=np.exp(-np.abs(x))
    return np.where(x>=0,1/(1+e),e/(1+e))

def reference(state,x,eps,mu,scale,edges,p,tau2):
    x=np.asarray(x,dtype='f8');eps=np.asarray(eps,dtype='f8');q0=q(x)
    if x.ndim!=2 or x.shape[1]!=480 or eps.shape!=x.shape:raise ValueError('independent input shape')
    if not np.isfinite(x).all()or not np.isfinite(eps).all()or not((q0>0)&(q0<300)).all():raise ValueError('independent input support')
    if not np.isfinite(p).all()or not(p>=0).all()or not np.isfinite(tau2)or tau2<0:raise ValueError('independent noise constants')
    if not all(np.isfinite(v).all()for v in state.values())or not(state['feature_scale']>0).all():raise ValueError('independent head state')
    unit=(x-x[:,:100].mean(1,keepdims=True))/q0[:,None]
    f=(unit-state['feature_center'])/state['feature_scale']
    hidden=f@state['hidden.weight'].T+state['hidden.bias'];hidden=hidden*sigmoid(hidden)
    ab=hidden@state['output.weight'].T+state['output.bias']
    slope=np.exp(ab[:,1])
    if not np.isfinite(slope).all()or not(slope>0).all():raise ValueError('independent monotone slope')
    qt=300*sigmoid(ab[:,0]+slope*(np.log(q0)-np.log(300-q0)))
    if not np.isfinite(qt).all()or not((qt>0)&(qt<300)).all():raise ValueError('independent Q* failure')
    mu=np.asarray(mu,dtype='f8');mu=mu-mu[:,:100].mean(1,keepdims=True);mu=mu/q(mu)[:,None]
    centers=(edges[:-1]+edges[1:])*.5;lq=np.log(1000*qt)
    insertion=np.searchsorted(centers,lq,side='right');left=np.clip(insertion-1,0,7);right=np.clip(insertion,0,7)
    alpha=np.zeros_like(lq);mask=left!=right
    alpha[mask]=(lq[mask]-centers[left[mask]])/(centers[right[mask]]-centers[left[mask]])
    m=(1-alpha[:,None])*mu[left]+alpha[:,None]*mu[right]
    m=m-m[:,:100].mean(1,keepdims=True);m=m/q(m)[:,None]
    lifted=x+(qt-q0)[:,None]*m
    a=np.zeros(480);a[:100]=-3;a[100:400]=1
    # Independent covariance-column construction; differs from analytic shortcut.
    k=np.fft.irfft(p,n=480)
    idx=np.arange(480);K=k[(idx[:,None]-idx[None,:])%480]
    B=np.eye(480)[:100]-np.ones((100,1))@np.eye(480)[:100].mean(0,keepdims=True)
    base=np.trace(B@K@B.T)/99
    bka=B@(K@a);ama=a@K@a;bm=m@B.T
    kappa=base-2*(bm*bka).sum(1)/99+ama*(bm*bm).sum(1)/99
    if not(kappa>0).all():raise ValueError('independent noise variance failure')
    amp=np.maximum(tau2-np.var(lifted[:,:100],axis=1,ddof=1),0.)
    noise=np.fft.irfft(np.sqrt(p)*np.fft.rfft(eps,axis=1,norm='ortho'),n=480,axis=1,norm='ortho')
    noise=(noise-q(noise)[:,None]*m)*np.sqrt(amp/kappa)[:,None]
    raw=lifted+noise;pre=raw.copy()
    change=((raw<-.5)|(raw>.5)).any(1)|(np.abs(q(raw)-qt)>1e-12)
    if change.any():
        v=raw[change];qq=qt[change];aa=a[:400]
        lo=np.where(aa>0,(v[:,:400]-.5)/aa,(v[:,:400]+.5)/aa).min(1)
        hi=np.where(aa>0,(v[:,:400]+.5)/aa,(v[:,:400]-.5)/aa).max(1)
        for _ in range(64):
            mid=(lo+hi)*.5;charge=q(np.clip(v-mid[:,None]*a,-.5,.5))
            lo=np.where(charge>qq,mid,lo);hi=np.where(charge>qq,hi,mid)
        pre[change]=np.clip(v-.5*(lo+hi)[:,None]*a,-.5,.5)
    pre=pre.astype('f4')
    legacy=(-np.float32(.5)+np.rint((np.clip(pre,-.5,.5)+np.float32(.5))*np.float32(16383))/np.float32(16383)).astype('f4')
    adc=-.5+np.clip(np.rint((legacy.astype('f8')+.5)*16383),0,16383)/16383
    if not np.isfinite(pre).all()or not np.isfinite(adc).all():raise ValueError('independent output finite failure')
    return pre,adc,qt
