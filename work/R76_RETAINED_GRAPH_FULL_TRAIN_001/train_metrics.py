"""Preserved descriptive TRAIN metrics; no model inference at import."""

import json

import numpy as np

from scipy.stats import wasserstein_distance,ks_2samp

from numpy_reference import q



def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')

def canonical(v):return -.5+np.clip(np.rint((np.asarray(v,dtype='f8')+.5)*16383),0,16383)/16383

def psd(v):return 1-(v[:,100:150].sum(1)-.5*v[:,:100].sum(1))/q(v)

def bins(qs,edges):
    logq=np.log(1000*qs);b=np.searchsorted(edges,logq,side='right')-1;b[logq==edges[-1]]=7;return b

def rms(v):return np.sqrt(np.mean((v[:,:100]-v[:,:100].mean(1,keepdims=True))**2,1))

def acf(v):
    b=v[:,:100]-v[:,:100].mean(1,keepdims=True);power=np.mean(b*b,1)
    return np.mean(b[:,:-1]*b[:,1:],1)/np.maximum(power,1e-20)

def boundary(v):
    data={f'first_{i}_{j}':v[:,j]-v[:,i]for i,j in[(99,100),(149,150),(399,400)]}
    data.update({f'second_{i}':v[:,i-1]-2*v[:,i]+v[:,i+1]for i in[99,100,149,150,399,400]})
    return data

def summary(x):
    return dict(n=len(x),min=float(np.min(x)),median=float(np.median(x)),mean=float(np.mean(x)),max=float(np.max(x)))

def compare(v,ref,edges):
    qs,qr=q(v),q(ref);p,pr=psd(v),psd(ref);b,br=bins(qs,edges),bins(qr,edges)
    bv,bref=boundary(v),boundary(ref)
    by=[]
    for j in range(8):
        sx=b==j;rx=br==j
        by.append(dict(bin_1based=j+1,candidate_N=int(sx.sum()),reference_N=int(rx.sum()),
            PSD_W1=float(wasserstein_distance(p[sx],pr[rx]))if sx.any()and rx.any()else None))
    return dict(N=len(v),weight_sum=1.,Q_mVns=summary(qs*1000),Q_W1_mVns=float(wasserstein_distance(qs*1000,qr*1000)),
      logQ_W1=float(wasserstein_distance(np.log(qs),np.log(qr))),Q_KS=float(ks_2samp(qs,qr).statistic),
      output_bin_counts=[int((b==j).sum())for j in range(-1,9)],output_bin_weights=[float((b==j).mean())for j in range(-1,9)],
      outside_support_N=int(((b<0)|(b>7)).sum()),outside_support_mass=float(((b<0)|(b>7)).mean()),
      overall_PSD_W1=float(wasserstein_distance(p,pr)),output_condition_PSD=by,
      Q1_Q7_PSD_W1_mean=float(np.mean([row['PSD_W1']for row in by[:7]]))if all(row['PSD_W1']is not None for row in by[:7])else None,
      baseline_RMS_W1_V=float(wasserstein_distance(rms(v),rms(ref))),baseline_ACF1_W1=float(wasserstein_distance(acf(v),acf(ref))),
      boundary_W1_V={k:float(wasserstein_distance(bv[k],bref[k]))for k in bv})

def migrations(meta):
    names=['original_bin_raw','target_bin_raw','preADC_bin_raw','ADC_bin_raw']
    joint=np.column_stack([meta[n]for n in names]);rows,counts=np.unique(joint,axis=0,return_counts=True)
    return dict(coordinates=names,N=len(joint),rows=[dict(bins=row.tolist(),count=int(n),weight=float(n/len(joint)))for row,n in zip(rows,counts)])
