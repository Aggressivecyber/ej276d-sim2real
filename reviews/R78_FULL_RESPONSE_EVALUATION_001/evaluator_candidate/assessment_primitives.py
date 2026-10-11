"""Original common-canonical domain packaging and paired fixed-prediction bootstrap."""
from __future__ import annotations
import numpy as np
import evaluation_primitives as ev
import diagnostic_primitives as dc
from canonical_primitives import canonical

def data(v,ids,groups,typ,ref,qc=None,adc_domain=True):
    d=ev.features(canonical(v) if adc_domain else v);d['ids']=np.asarray(ids);d['groups']=np.asarray(groups);d['cluster']=np.array([typ+'|'+str(g) for g in groups])
    q=d['features']['Qtotal_mVns'];d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],q,side='right')
    d['region'][~np.isfinite(q)]=255
    if qc is not None:d['valid'] &= np.asarray(qc)==0
    return dc.extra_features(d)

def paired_bootstrap(a,b,cfg):
    assert np.array_equal(a['label'],b['label']) and np.array_equal(a['cluster'],b['cluster'])
    y=a['label'];p=a['probability'];q=b['probability'];_,inv=np.unique(a['cluster'],return_inverse=True);n=int(inv.max()+1)
    rng=np.random.default_rng(cfg['bootstrap_seed']);acc=[];aucs=[]
    _,tp=np.unique(p,return_inverse=True);_,tq=np.unique(q,return_inverse=True)
    def score(prob,tie,w):
        n0=w[y==0].sum();n1=w[y==1].sum()
        ba=.5*(w[(y==0)&(prob<.5)].sum()/n0+w[(y==1)&(prob>=.5)].sum()/n1)
        neg=np.bincount(tie[y==0],weights=w[y==0],minlength=int(tie.max()+1));pos=np.bincount(tie[y==1],weights=w[y==1],minlength=int(tie.max()+1))
        auc=float(np.dot(pos,np.cumsum(neg)-.5*neg)/(n0*n1));return ba,auc
    point_a=score(p,tp,np.ones(len(y)));point_b=score(q,tq,np.ones(len(y)))
    for _ in range(cfg['bootstrap_repeats']):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[inv]
        if w[y==0].sum()==0 or w[y==1].sum()==0:continue
        aa=score(p,tp,w);bb=score(q,tq,w);acc.append(aa[0]-bb[0]);aucs.append(aa[1]-bb[1])
    return {'BA_delta_first_minus_second':point_a[0]-point_b[0],'BA_delta_95CI':np.quantile(acc,[.025,.975]),'AUC_delta_first_minus_second':point_a[1]-point_b[1],'AUC_delta_95CI':np.quantile(aucs,[.025,.975]),'valid_replicates':len(acc),'clusters':n,'paired_test_rows_and_groups':True,'conditional_on_two_fitted_classifiers':True,'response_training_uncertainty':False}
