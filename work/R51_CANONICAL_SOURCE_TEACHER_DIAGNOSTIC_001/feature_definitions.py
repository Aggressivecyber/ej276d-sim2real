import numpy as np
import sys
ev=sys.modules[__name__]
REGIONS={"baseline":np.arange(100)}


def crossing(x,peak,frac):

 if peak<=100:return np.nan

 target=x[peak]*frac;s=x[100:peak+1];idx=np.flatnonzero((s[:-1]<=target)&(s[1:]>=target)&(s[1:]>s[:-1]))

 if not len(idx):return np.nan

 i=int(idx[-1])+100;return i+(target-x[i])/(x[i+1]-x[i])

def features(v):

 v=np.asarray(v,dtype=np.float64);b=v[:,:100].mean(axis=1);x=v-b[:,None];peak=x.max(axis=1);pi=x.argmax(axis=1)

 qt=x[:,100:400].sum(axis=1)*1000;qf=x[:,100:150].sum(axis=1)*1000;ps=np.divide(qf,qt,out=np.full(len(v),np.nan),where=qt>0);ps=1-ps

 sigma=x[:,:100].std(axis=1,ddof=1);shape=np.divide(x,peak[:,None],out=np.full_like(x,np.nan),where=peak[:,None]>0)

 rise=np.array([crossing(y,int(p),.9)-crossing(y,int(p),.1) for y,p in zip(x,pi)])

 tail=np.divide(x[:,200:400].sum(axis=1)*1000,qt,out=np.full(len(v),np.nan),where=qt>0)

 tailmean=np.divide(x[:,300:400].mean(axis=1),peak,out=np.full(len(v),np.nan),where=peak>0)

 den=(x[:,:100]**2).sum(axis=1)

 acf=np.column_stack([np.divide((x[:,k:100]*x[:,:100-k]).sum(axis=1),den,out=np.full(len(v),np.nan),where=den>0) for k in range(1,21)])

 valid=np.all(np.isfinite(v),axis=1)&(peak>0)&(qt>0)&np.isfinite(ps)

 rel=np.divide(sigma,peak,out=np.full(len(v),np.nan),where=peak>0)

 f={'PSD':ps,'Qtotal_mVns':qt,'peak_V':peak,'baseline_RMS_V':sigma,'baseline_RMS_over_peak':rel,'rise_10_90_ns':rise,'tail_charge_fraction':tail,'tail_mean_over_peak':tailmean,'ACF1':acf[:,0],'baseline_mean_V':b}

 return {'V':v,'x':x,'shape':shape,'acf':acf,'features':f,'valid':valid,'peak_index':pi}

def c2st_matrix(d,view):

 f=d['features'];columns=['PSD','rise_10_90_ns','tail_charge_fraction','tail_mean_over_peak','baseline_RMS_over_peak'];scalar=np.column_stack([f[k] for k in columns]+[d['acf'][:,i] for i in range(8)])

 if view=='shape_plus_amplitude_proxy':

  scalar=np.column_stack([scalar,*[np.log(np.where(f[k]>0,f[k],np.nan)) for k in ['Qtotal_mVns','peak_V','baseline_RMS_V']]])

 a=np.column_stack([scalar,d['shape']]);a[~np.isfinite(a)]=np.nan;return a,scalar.shape[1]

def matrix(d,view):
    if view=='head34':return d['head34'].copy()
    if view=='Q_PSD_2D':return np.column_stack([d['features']['Qtotal_mVns'],d['features']['PSD']])
    if view=='full_voltage':return d['V'].copy()
    if view=='full_historical_shape':return ev.c2st_matrix(d,'shape_only')[0]
    part,kind=view.rsplit('_',1);s=d['V'][:,REGIONS[part]].copy();center=s-s.mean(axis=1,keepdims=True)
    den=(center*center).sum(axis=1);scale=np.sqrt(den/s.shape[1]);z=np.divide(center,scale[:,None],out=np.full_like(center,np.nan),where=scale[:,None]>0)
    out=s if kind=='voltage' else z
    if part=='baseline':
        acf=np.column_stack([np.divide((center[:,k:]*center[:,:-k]).sum(axis=1),den,out=np.full(len(s),np.nan),where=den>0) for k in range(1,9)])
        ps=np.abs(np.fft.rfft(center,axis=1))**2
        ps=np.divide(ps[:,1:],ps[:,1:].sum(axis=1,keepdims=True),out=np.full_like(ps[:,1:],np.nan),where=ps[:,1:].sum(axis=1,keepdims=True)>0)
        out=np.column_stack([out,acf,ps])
        if kind=='voltage':out=np.column_stack([out,np.log(np.maximum(center.std(axis=1,ddof=1),1e-30))])
    out[~np.isfinite(out)]=np.nan
    return out
