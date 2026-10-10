import numpy as np
from scipy.stats import wasserstein_distance
from noise_core import q_np
from teacher_core import short_q

KEYS=['PSD','logQ','baseline_RMS_V','baseline_ACF1','peak_V','Q_Vns']
CATEGORIES=[-1,*range(8),8]

def desc(v,edges):
    v=np.asarray(v,dtype='f8');q=q_np(v);valid=np.isfinite(v).all(1)&np.isfinite(q)&(q>0)
    psd=np.full(len(v),np.nan);logq=psd.copy();psd[valid]=1-short_q(v)[valid]/q[valid];logq[valid]=np.log(1000*q[valid])
    e=v[:,:100]-v[:,:100].mean(1,keepdims=True);energy=np.sum(e*e,axis=1)
    acf=np.divide(np.sum(e[:,:-1]*e[:,1:],axis=1),energy,out=np.full(len(v),np.nan),where=energy>0)
    raw=np.full(len(v),-99,dtype='i8');raw[valid]=np.searchsorted(edges,logq[valid],side='right')-1;raw[logq==edges[-1]]=7
    return {'PSD':psd,'logQ':logq,'baseline_RMS_V':np.sqrt(energy/99),'baseline_ACF1':acf,'peak_V':np.max(v-v[:,:100].mean(1,keepdims=True),axis=1),'Q_Vns':q,
        'rawbin':raw,'valid':valid,'range_exceeded':np.any((v<-.5)|(v>.5),axis=1)}

def stats(a,w):
    a=np.asarray(a);w=np.asarray(w);valid=np.isfinite(a);total=float(w.sum());good=bool(len(a)>0 and valid.all() and total>0)
    mean=float(np.dot(w,a)/total) if good else None
    return {'edges':len(a),'mass':total,'undefined_edges':int((~valid).sum()),'undefined_mass':float(w[~valid].sum()),'mean':mean,'population_SD':float(np.sqrt(np.dot(w,(a-mean)**2)/total)) if good else None}

def w1(a,b,wa,wb):
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all() or min(wa.sum(),wb.sum())<=0:return None
    return float(wasserstein_distance(a,b,u_weights=wa,v_weights=wb))

def compare(a,b,wa,wb,source_bins=None):
    ba=a['rawbin'] if source_bins is None else source_bins;rows=[]
    for bin0 in [None,*CATEGORIES]:
        ix=np.ones(len(ba),bool) if bin0 is None else ba==bin0;ir=np.ones(len(b['rawbin']),bool) if bin0 is None else b['rawbin']==bin0
        rows.append({'raw_bin0':bin0,'candidate_edges':int(ix.sum()),'reference_edges':int(ir.sum()),'candidate_mass':float(wa[ix].sum()),'reference_mass':float(wb[ir].sum()),
            'invalid_mass':float(wa[ix&~a['valid']].sum()),'reference_invalid_mass':float(wb[ir&~b['valid']].sum()),
            'features':{k:{'candidate':stats(a[k][ix],wa[ix]),'reference':stats(b[k][ir],wb[ir]),'W1':w1(a[k][ix],b[k][ir],wa[ix],wb[ir])} for k in KEYS}})
    return rows

def ranks(x,w):
    order=np.argsort(x,kind='stable');s=x[order];ww=w[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    masses=np.add.reduceat(ww,starts);end=np.cumsum(masses);mid=(end-.5*masses)/end[-1]
    answer=np.empty(len(x));answer[order]=np.repeat(mid,np.diff(np.r_[starts,len(x)]));return answer

def correlation(x,y,w):
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():return None
    a=ranks(x,w);b=ranks(y,w);a-=np.average(a,weights=w);b-=np.average(b,weights=w)
    denominator=float(np.sqrt(np.dot(w,a*a)*np.dot(w,b*b)))
    return float(np.dot(w,a*b)/denominator) if denominator else None

def boundary(v):
    v=np.asarray(v,dtype='f8');out={}
    for c in [100,150,400]:
        out[str(c)+'_first']=v[:,c]-v[:,c-1]
        out[str(c)+'_second_left']=v[:,c]-2*v[:,c-1]+v[:,c-2]
        out[str(c)+'_second_right']=v[:,c+1]-2*v[:,c]+v[:,c-1]
    return out

def wave_zero():return {'mass':np.zeros(11),'sumV':np.zeros((11,480)),'sumV2':np.zeros((11,480)),'sumU':np.zeros((11,480)),'sumU2':np.zeros((11,480))}
def wave_add(out,x,bins,w):
    x=np.asarray(x,dtype='f8');v=x-x[:,:100].mean(1,keepdims=True);u=v/np.maximum(v.max(1,keepdims=True),1e-12)
    for j,bin0 in enumerate([*CATEGORIES,None]):
        ix=np.ones(len(x),bool) if bin0 is None else bins==bin0
        if not ix.any():continue
        out['mass'][j]+=w[ix].sum()
        for k,a in [('V',x),('U',u)]:
            out['sum'+k][j]+=(a[ix]*w[ix,None]).sum(0);out['sum'+k+'2'][j]+=(a[ix]*a[ix]*w[ix,None]).sum(0)
def wave_finish(out):
    res={'mass':out['mass']}
    for key in ['V','U']:
        mass=out['mass'][:,None];mean=np.divide(out['sum'+key],mass,out=np.zeros_like(out['sum'+key]),where=mass>0)
        var=np.divide(out['sum'+key+'2'],mass,out=np.zeros_like(mean),where=mass>0)-mean*mean
        res[key+'_mean']=mean;res[key+'_SD']=np.sqrt(np.maximum(var,0))
    return res
def wave_once(x,b,w):
    obj=wave_zero();wave_add(obj,x,b,w);return wave_finish(obj)
