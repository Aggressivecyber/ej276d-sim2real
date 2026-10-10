"""Fixed full-window statistics; no fitting, row deletion, or optimizer."""
import hashlib
import numpy as np
from noise_core import q_np

FEATURES=['V480','U480','DU479','D2U478']
REGIONS=list(range(-1,9))
STAGES=['input','noise_initial','neural_raw','projected_f32','canonical_ADC']
BOUNDARIES=['d1_99_100','d1_149_150','d1_399_400','d2_99','d2_100','d2_149','d2_150','d2_399','d2_400']

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float)and not np.isfinite(v):raise ValueError('nonfinite JSON statistic')
    return v

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def canonical(v,generated):
    if generated:
        f=np.asarray(v,dtype='f4');v=(-.5+np.round((np.clip(f,-.5,.5)+.5)*16383)/16383).astype('f4')
    return -.5+np.clip(np.round((np.asarray(v,dtype='f8')+.5)*16383),0,16383)/16383

def features(v):
    x=np.asarray(v,dtype='f8');s=x-x[:,:100].mean(1,keepdims=True);u=s/np.maximum(s.max(1,keepdims=True),.0025)
    return [x,u,np.diff(u,axis=1),np.diff(u,n=2,axis=1)]

def psd(v):
    x=np.asarray(v,dtype='f8');q=q_np(x)
    assert np.all(q>0)
    return 1-(x[:,100:150].sum(1)-.5*x[:,:100].sum(1))/q

def exact_w1_columns(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8')
    if a.ndim==1:a=a[:,None]
    if b.ndim==1:b=b[:,None]
    assert a.shape[1]==b.shape[1]and len(a)>0 and len(b)>0
    assert np.isfinite(a).all()and np.isfinite(b).all()
    sa=np.sort(a,axis=0,kind='stable');sb=np.sort(b,axis=0,kind='stable')
    ca=np.arange(1,len(a)+1,dtype='f8')/len(a);cb=np.arange(1,len(b)+1,dtype='f8')/len(b)
    edges=np.sort(np.r_[0.,ca,cb],kind='stable');mid=.5*(edges[1:]+edges[:-1]);mass=np.diff(edges)
    ia=np.minimum(np.searchsorted(ca,mid),len(a)-1);ib=np.minimum(np.searchsorted(cb,mid),len(b)-1)
    return (np.abs(sa[ia]-sb[ib])*mass[:,None]).sum(0)

def boundaries(v):
    x=np.asarray(v,dtype='f8');d1=np.diff(x,axis=1);d2=np.diff(x,n=2,axis=1)
    return np.c_[d1[:,[99,149,399]],d2[:,[98,99,148,149,398,399]]]

def baseline_features(v):
    b=np.asarray(v,dtype='f8')[:,:100];b=b-b.mean(1,keepdims=True);power=(b*b).mean(1)
    acf=np.column_stack([(b[:,lag:]*b[:,:-lag]).mean(1)/np.maximum(power,1e-20)for lag in range(1,21)])
    spectrum=np.abs(np.fft.rfft(b,axis=1)[:,1:])**2;spectrum[:,:-1]*=2
    total=spectrum.sum(1);bands=spectrum.reshape(len(b),10,5).sum(2)/np.maximum(total[:,None],1e-20)
    return np.c_[np.sqrt(power),acf,bands],b,power

def covariance(b):
    x=np.asarray(b,dtype='f8');assert len(x)>=2
    centered=x-x.mean(0);return centered.T@centered/(len(x)-1)

def covariance_error(c,reference):
    den=np.linalg.norm(reference)
    if den==0:return None
    return float(np.linalg.norm(c-reference)/den)

def draw_directions(rng):
    answer=[]
    for dim in [480,480,479,478]:
        p=rng.standard_normal((dim,64));p/=np.linalg.norm(p,axis=0,keepdims=True);answer.append(p.astype('<f4'))
    return answer

def normalized_feature_bank(v,bins,stats):
    answer=[]
    for name,f in zip(FEATURES,features(v)):
        answer.append(((f-stats[name+'_center'][bins+1])/stats[name+'_scale'][bins+1]).astype('f4'))
    return answer

def objective_table(a,b,ia,ib,directions):
    out=np.zeros((10,4,2))
    for region in range(10):
        for j,p in enumerate(directions):
            x=a[j][ia[region]];y=b[j][ib[region]]
            out[region,j,0]=exact_w1_columns(x,y).mean()
            out[region,j,1]=exact_w1_columns(x@p,y@p).mean()
    return out

def summary(x):
    x=np.asarray(x,dtype='f8');assert x.size and np.isfinite(x).all()
    return {'N':int(x.size),'mean':float(x.mean()),'SD':float(x.std(ddof=1))if x.size>1 else 0.,'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
