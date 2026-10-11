"""Original finite waveform features/CAL descriptions, without plots or execution."""
from __future__ import annotations
import numpy as np
from common_math_primitives import distances,posterior,fom

def safe(x):

 if isinstance(x,np.ndarray):return safe(x.tolist())

 if isinstance(x,np.generic):return safe(x.item())

 if isinstance(x,float):return x if np.isfinite(x) else None

 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}

 if isinstance(x,(tuple,list)):return [safe(v) for v in x]

 return x

def summary(v):

 a=np.asarray(v);x=a[np.isfinite(a)]

 return {'n':len(a),'defined_n':len(x),'undefined_n':len(a)-len(x),'mean':float(x.mean()) if len(x) else None,'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,'quantile_levels':[0,.05,.25,.5,.75,.95,1],'quantiles':np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist() if len(x) else None}

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

def mask(d,r=None):return d['valid'] if r is None else d['valid']&(d['region']==r)

def characterize(domains,ref):

 stats={};comparisons={};real=domains['REAL']

 for name,d in domains.items():

  label=np.zeros(len(d['V']),dtype='u1');post=np.full((len(d['V']),2),np.nan)

  for r,slot in enumerate(ref['slots'][1:]):

   m=mask(d,r)

   if slot['valid'] and np.any(m):

    p,_=posterior(d['features']['PSD'][m],np.array(slot['weights']),np.array(slot['means']),np.array(slot['variances']));post[m]=p;winner=p.argmax(axis=1);high=p.max(axis=1)>=.99;label[m]=np.where(high,winner+1,0)

  blocks={}

  for r in [None,0,1,2,3]:

   key='overall' if r is None else 'q'+str(r+1);m=mask(d,r);mr=mask(real,r);slot=ref['slots'][0 if r is None else r+1];cut=slot['cut'] if slot['valid'] else None

   blocks[key]={'source_n':len(d['V']) if r is None else int(np.sum(d['region']==r)),'QC_n':int(m.sum()),'HC_gamma_like_n':int(np.sum(label[m]==1)),'HC_neutron_like_n':int(np.sum(label[m]==2)),'unknown_n':int(np.sum(label[m]==0)),'reference_valid':bool(slot['valid']),'frozen_reference_means':slot['means'] if slot['valid'] else None,'FoM':fom(d['features']['PSD'][m],cut),'features':{k:summary(v[m]) for k,v in d['features'].items()}}

   if name!='REAL':comparisons.setdefault(name,{})[key]={k:distances(v[m&np.isfinite(v)],real['features'][k][mr&np.isfinite(real['features'][k])]) for k,v in d['features'].items()}

  stats[name]={'QC_valid':int(d['valid'].sum()),'invalid':int((~d['valid']).sum()),'regions':blocks}

  d['label']=label;d['posterior']=post

 return stats,comparisons

def c2st_matrix(d,view):

 f=d['features'];columns=['PSD','rise_10_90_ns','tail_charge_fraction','tail_mean_over_peak','baseline_RMS_over_peak'];scalar=np.column_stack([f[k] for k in columns]+[d['acf'][:,i] for i in range(8)])

 if view=='shape_plus_amplitude_proxy':

  scalar=np.column_stack([scalar,*[np.log(np.where(f[k]>0,f[k],np.nan)) for k in ['Qtotal_mVns','peak_V','baseline_RMS_V']]])

 a=np.column_stack([scalar,d['shape']]);a[~np.isfinite(a)]=np.nan;return a,scalar.shape[1]
