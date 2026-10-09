from __future__ import annotations
import time
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import evaluation as ev
import diagnostic_core as dc

def coordinates(d):
    return np.column_stack([np.log(np.maximum(d['features']['Qtotal_mVns'],1e-300)),d['features']['PSD']])

def learn(g,r):
    rc=coordinates(r);gc=coordinates(g);edges=[];raw=[];merged=[]
    for j in range(2):
        z=np.quantile(rc[r['valid'],j],np.linspace(0,1,7));raw.append(z);u=np.unique(z[1:-1]);edges.append(np.r_[-np.inf,u,np.inf]);merged.append(5-len(u))
    shape=(len(edges[0])-1,len(edges[1])-1)
    def cell(x):return np.searchsorted(edges[0][1:-1],x[:,0],side='right')*shape[1]+np.searchsorted(edges[1][1:-1],x[:,1],side='right')
    ng=np.bincount(cell(gc[g['valid']]),minlength=np.prod(shape));nr=np.bincount(cell(rc[r['valid']]),minlength=np.prod(shape));pg=ng/ng.sum();pr=nr/nr.sum()
    ok=(ng>=20)&(nr>=20);overlap=np.where(ok,np.minimum(pg,pr),0);total=overlap.sum();star=overlap/total if total>0 else overlap
    wg=np.divide(star,pg,out=np.zeros_like(pg),where=pg>0);wr=np.divide(star,pr,out=np.zeros_like(pr),where=pr>0)
    return {'edges':edges,'raw_quantiles':raw,'duplicate_internal_edges_merged':merged,'shape':shape,'TRAIN_GEN_n':int(ng.sum()),'TRAIN_REAL_n':int(nr.sum()),'TRAIN_GEN_counts':ng,'TRAIN_REAL_counts':nr,'TRAIN_GEN_mass':pg,'TRAIN_REAL_mass':pr,'qualified_cells':ok,'overlap_mass':float(total),'p_star':star,'GEN_weight_per_cell':wg,'REAL_weight_per_cell':wr,'TRAIN_only':True}

def apply(d,plan,domain):
    x=coordinates(d);i=np.searchsorted(plan['edges'][0][1:-1],x[:,0],side='right')*plan['shape'][1]+np.searchsorted(plan['edges'][1][1:-1],x[:,1],side='right')
    w=plan[domain+'_weight_per_cell'][i].copy();w[~d['valid']]=0
    return w,i

def ess(w):return float(w.sum()**2/np.sum(w*w)) if np.sum(w*w)>0 else 0.

def population(d,w,cell,plan):
    use=d['valid'];support=use&(w>0);mass=np.bincount(cell,weights=w,minlength=len(plan['p_star']));mass=mass/mass.sum() if mass.sum() else mass
    return {'source_n':len(w),'QC_n':int(use.sum()),'QC_invalid':int((~use).sum()),'common_support_n':int(support.sum()),'retained_fraction_of_QC':float(support.sum()/use.sum()) if use.any() else None,'unsupported_QC_n':int((use&~support).sum()),'weight_sum':float(w.sum()),'ESS':ess(w),'artificial_groups_or_draws':int(np.unique(d['groups'][support]).size),'DEV_cell_counts':np.bincount(cell[use],minlength=len(mass)),'DEV_weighted_cell_mass':mass,'L1_cell_mass_vs_TRAIN_pstar':float(np.abs(mass-plan['p_star']).sum())}

def weighted_distance(a,b,wa,wb):
    maska=np.isfinite(a)&(wa>0);maskb=np.isfinite(b)&(wb>0);a=a[maska];b=b[maskb];wa=wa[maska];wb=wb[maskb]
    if not len(a) or not len(b):return {'W1':None,'KS':None,'N':[len(a),len(b)]}
    ia=np.argsort(a);ib=np.argsort(b);aa=a[ia];bb=b[ib];ac=np.r_[0,np.cumsum(wa[ia])/wa.sum()];bc=np.r_[0,np.cumsum(wb[ib])/wb.sum()];x=np.unique(np.r_[aa,bb]);ks=np.abs(ac[np.searchsorted(aa,x,side='right')]-bc[np.searchsorted(bb,x,side='right')]).max()
    return {'W1':float(wasserstein_distance(a,b,wa,wb)),'KS':float(ks),'N':[len(a),len(b)]}

def choose(a,b,originala,originalb,wa,wb):
    pairs=[]
    for i in range(2):
        ia=np.asarray(originala[i]);ib=np.asarray(originalb[i]);ia=ia[a['valid'][ia]&(wa[ia]>0)];ib=ib[b['valid'][ib]&(wb[ib]>0)]
        n=min(len(ia),len(ib));pairs.append((ia[:n],ib[:n]))
    sa=(pairs[0][0],pairs[1][0]);sb=(pairs[0][1],pairs[1][1]);info={'train_each_domain':len(sa[0]),'holdout_each_domain':len(sa[1]),'GEN_ESS_train':ess(wa[sa[0]]),'GEN_ESS_holdout':ess(wa[sa[1]]),'REAL_ESS_train':ess(wb[sb[0]]),'REAL_ESS_holdout':ess(wb[sb[1]]),'groups':{k:{'train':len(np.unique(d['groups'][s[0]])),'holdout':len(np.unique(d['groups'][s[1]]))} for k,d,s in [('first',a,sa),('second',b,sb)]}}
    info['weighted_supported']=min(info['GEN_ESS_train'],info['REAL_ESS_train'])>=150 and min(info['GEN_ESS_holdout'],info['REAL_ESS_holdout'])>=75
    for d,s in [(a,sa),(b,sb)]:assert not set(d['groups'][s[0]])&set(d['groups'][s[1]])
    return sa,sb,info

def score(y,p,w):
    n0=w[y==0].sum();n1=w[y==1].sum()
    if min(n0,n1)<=0:return None
    ba=.5*(w[(y==0)&(p<.5)].sum()/n0+w[(y==1)&(p>=.5)].sum()/n1)
    return float(ba),float(roc_auc_score(y,p,sample_weight=w))

def bootstrap(a,b,cfg):
    y=a['label'];clusters=a['cluster'];_,inv=np.unique(clusters,return_inverse=True);n=int(inv.max()+1);rng=np.random.default_rng(cfg['bootstrap_seed']);av=[];bv=[];delta=[]
    if b is not None:assert np.array_equal(y,b['label']) and np.array_equal(clusters,b['cluster'])
    for _ in range(500):
        m=np.bincount(rng.integers(n,size=n),minlength=n)[inv];s=score(y,a['probability'],a['weight']*m)
        if s is None:continue
        av.append(s)
        if b is not None:
            t=score(y,b['probability'],b['weight']*m)
            if t is not None:bv.append(t);delta.append(np.asarray(s)-t)
    out={'valid_repeats':len(av),'clusters':n,'conditional_on_weights_and_fitted_classifiers':True,'includes_weight_estimation_or_training_or_acquisition_uncertainty':False,'BA_95CI':np.quantile(np.array(av)[:,0],[.025,.975]),'AUC_95CI':np.quantile(np.array(av)[:,1],[.025,.975])}
    if delta:out['first_minus_second']={'BA':score(y,a['probability'],a['weight'])[0]-score(y,b['probability'],b['weight'])[0],'BA_95CI':np.quantile(np.array(delta)[:,0],[.025,.975]),'AUC_95CI':np.quantile(np.array(delta)[:,1],[.025,.975])}
    return out

def fit(a,b,sa,sb,wa,wb,view,cfg,counter):
    ta,ea=sa;tb,eb=sb;am=coordinates(a) if view=='coordinates' else dc.matrix(a,view);bm=coordinates(b) if view=='coordinates' else dc.matrix(b,view)
    X=np.vstack([am[ta],bm[tb]]);T=np.vstack([am[ea],bm[eb]]);y=np.r_[np.ones(len(ta)),np.zeros(len(tb))];yt=np.r_[np.ones(len(ea)),np.zeros(len(eb))]
    wtrain=np.r_[wa[ta]/wa[ta].sum()*len(ta),wb[tb]/wb[tb].sum()*len(tb)];wtest=np.r_[wa[ea],wb[eb]]
    X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan;im=SimpleImputer(strategy='median',keep_empty_features=True);X=im.fit_transform(X);T=im.transform(T);sc=StandardScaler();X=sc.fit_transform(X);T=sc.transform(T)
    counter['HGB_attempts']+=1
    if counter['HGB_attempts']>26:raise RuntimeError('R10HGB cap')
    clock=time.monotonic();h=HistGradientBoostingClassifier(**cfg['C2ST']);h.fit(X,y,sample_weight=wtrain);prob=h.predict_proba(T)[:,1];v=score(yt,prob,wtest)
    pred={'probability':prob,'label':yt,'cluster':np.r_[a['cluster'][ea],b['cluster'][eb]],'weight':wtest}
    out={'supported':True,'BA':v[0],'AUC':v[1],'train_each_domain':len(ta),'holdout_each_domain':len(ea),'features':X.shape[1],'sample_weight_applied_train_and_holdout':True,'elapsed_s':time.monotonic()-clock,'CI':bootstrap(pred,None,cfg)}
    return out,pred

def analyze(genfit,realfit,devgen,devreal,rr,selection,cfg,counter,old):
    plan=learn(genfit,realfit);wg,cg=apply(devgen,plan,'GEN');wr,cr=apply(devreal,plan,'REAL');wt,ct=apply(rr,plan,'REAL');splits={k:(np.array(v['train']),np.array(v['holdout'])) for k,v in selection['indices'].items()}
    sg,sr,support=choose(devgen,devreal,splits['GEN'],splits['REAL'],wg,wr);st,srr,rrsupport=choose(rr,devreal,splits['TRAIN_REFERENCE'],splits['REAL'],wt,wr)
    out={'plan':plan,'DEV':{'GEN':population(devgen,wg,cg,plan),'REAL':population(devreal,wr,cr,plan),'RR_TRAIN_reference':population(rr,wt,ct,plan)},'evaluation_support':support,'RR_evaluation_support':rrsupport,'C2ST':{},'paired_weighted_minus_unweighted':{},'coordinate_balance':{},'original_full_population_cited_R7':{view:old['C2ST']['native|'+view+'|GEN_vs_REAL'] for view in cfg['C2ST_views']},'weights_not_identified_physical_source_correction':True}
    preds={};onesg=np.ones(len(wg));onesr=np.ones(len(wr))
    for view in [*cfg['C2ST_views'],'coordinates']:
        if min(len(sg[0]),len(sg[1]))<40:out['C2ST']['support_unweighted|'+view]={'supported':False,'reason':'fewer_than40_train_or_holdout'}
        else:out['C2ST']['support_unweighted|'+view],preds['U|'+view]=fit(devgen,devreal,sg,sr,onesg,onesr,view,cfg,counter)
        if not support['weighted_supported']:out['C2ST']['support_weighted|'+view]={'supported':False,'reason':'TRAIN_ESS_below150_or_holdout_below75'}
        else:out['C2ST']['support_weighted|'+view],preds['W|'+view]=fit(devgen,devreal,sg,sr,wg,wr,view,cfg,counter)
        if 'W|'+view in preds and 'U|'+view in preds:out['paired_weighted_minus_unweighted'][view]=bootstrap(preds['W|'+view],preds['U|'+view],cfg)
    for view in cfg['C2ST_views']:
        if rrsupport['weighted_supported']:out['C2ST']['RR_weighted|'+view],preds['RR|'+view]=fit(rr,devreal,st,srr,wt,wr,view,cfg,counter)
        else:out['C2ST']['RR_weighted|'+view]={'supported':False,'reason':'TRAIN_ESS_below150_or_holdout_below75'}
    for j,label in enumerate(['logQ','PSD']):
        a=coordinates(devgen)[:,j];b=coordinates(devreal)[:,j]
        out['coordinate_balance'][label]={'original':weighted_distance(a,b,devgen['valid'].astype(float),devreal['valid'].astype(float)),'common_support_unweighted':weighted_distance(a,b,(wg>0).astype(float),(wr>0).astype(float)),'common_support_weighted':weighted_distance(a,b,wg,wr)}
    private={'GEN_weight':wg,'REAL_weight':wr,'RR_TRAIN_weight':wt,'GEN_cell':cg,'REAL_cell':cr,'RR_TRAIN_cell':ct,'GEN_train':sg[0],'GEN_holdout':sg[1],'REAL_train':sr[0],'REAL_holdout':sr[1]}
    return out,private,preds
