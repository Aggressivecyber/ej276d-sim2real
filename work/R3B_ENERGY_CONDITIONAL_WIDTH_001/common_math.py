import numpy as np
import math

def distances(a,b):
    if not len(a) or not len(b):return {'W1':None,'KS':None,'counts':[len(a),len(b)]}
    aa=np.sort(a);bb=np.sort(b);points=np.unique(np.concatenate([aa,bb]));ks=float(np.max(np.abs(np.searchsorted(aa,points,side='right')/len(aa)-np.searchsorted(bb,points,side='right')/len(bb))))
    # Unequal valid denominators are descriptive only; exact W1 from merged ECDF.
    if len(a)==len(b):w1=float(np.mean(np.abs(aa-bb)))
    else:
        left=points[:-1];gap=np.diff(points);w1=float(np.sum(np.abs(np.searchsorted(aa,left,side='right')/len(aa)-np.searchsorted(bb,left,side='right')/len(bb))*gap))
    return {'W1':w1,'KS':ks,'counts':[len(a),len(b)]}

def posterior(x,weights,means,variances):
    lp=np.log(weights)[None,:]-.5*(math.log(2*math.pi)+np.log(variances))[None,:]-.5*(x[:,None]-means[None,:])**2/variances[None,:]
    top=np.max(lp,axis=1);norm=top+np.log(np.exp(lp-top[:,None]).sum(axis=1));return np.exp(lp-norm[:,None]),float(np.mean(norm))

def fom(x,cut):
    if cut is None:return {'defined':False,'reason':'REAL_reference_or_unique_cut_unavailable'}
    bands=[x[x<=cut],x[x>cut]];stats=[]
    for values in bands:stats.append({'count':len(values),'proportion_of_valid':len(values)/len(x) if len(x) else None,'mean':float(values.mean()) if len(values) else None,'sample_sd_ddof1':float(values.std(ddof=1)) if len(values)>1 else None})
    result={'defined':False,'cut':cut,'low_includes_equality':True,'bands':stats,'unknown_included':True,'valid_denominator':len(x)}
    if any(len(v)<20 for v in bands):result['reason']='a_band_below20';return result
    denom=math.sqrt(8*math.log(2))*(stats[0]['sample_sd_ddof1']+stats[1]['sample_sd_ddof1'])
    if not math.isfinite(denom) or denom<=0:result['reason']='nonpositive_width';return result
    result.update(defined=True,reason=None,FoM=abs(stats[1]['mean']-stats[0]['mean'])/denom);return result
