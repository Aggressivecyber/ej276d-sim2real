from __future__ import annotations
import math,time,warnings
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import density as den
import evaluation as ev

def encoded_heads(v,rep):
    h,r,valid=den.encode(v,rep)
    x=(h-rep['head_mean'])/rep['head_scale']
    return x,h,r,valid

def transform(head,x,chunk=512):
    zz=[];ll=[]
    for start in range(0,len(x),chunk):
        with torch.no_grad():z,ld=head(torch.from_numpy(x[start:start+chunk]))
        zz.append(z.numpy());ll.append(ld.numpy())
    return np.concatenate(zz),np.concatenate(ll)

def gaussian_logpdf(z,p):
    white=solve_triangular(p['cholesky'],(z-p['mean']).T,lower=True,check_finite=True).T
    return -.5*(34*math.log(2*math.pi)+(white*white).sum(axis=1))-np.log(np.diag(p['cholesky'])).sum()

def fit_gaussian(z):
    mean=z.mean(axis=0);center=z-mean;cov=center.T@center/len(z)+np.eye(34)*1e-6;chol=np.linalg.cholesky(cov)
    p={'mean':mean,'covariance':cov,'cholesky':chol}
    return p,{'status':'complete','eligible':True,'TRAIN_N':len(z),'denominator':len(z),'diagonal_addition':1e-6,'shrinkage':False,'min_covariance_eigenvalue':float(np.linalg.eigvalsh(cov)[0])}

def fit_gmm(z,cfg):
    options=dict(cfg['latent_GMM']);options['random_state']=np.random.RandomState(np.random.MT19937(np.random.SeedSequence(cfg['base_fit_seed'])))
    g=GaussianMixture(**options)
    with warnings.catch_warnings(record=True) as caught:g.fit(z)
    p={'weights':g.weights_,'means':g.means_,'covariances':g.covariances_,'cholesky':np.stack([np.linalg.cholesky(c) for c in g.covariances_])}
    if not all(np.isfinite(a).all() for a in p.values()) or not np.all(p['weights']>0):raise FloatingPointError('GMM parameters illegal')
    rec={'status':'complete' if g.converged_ else 'nonconverged_diagnostic','eligible':bool(g.converged_),'TRAIN_N':len(z),'seed':cfg['base_fit_seed'],'sklearn_random_state_adapter':'RandomState(MT19937(SeedSequence(full_integer_seed))); no truncated integer seed','converged':bool(g.converged_),'iterations':int(g.n_iter_),'lower_bound':float(g.lower_bound_),'warnings':[str(w.message) for w in caught],'n_init':1,'reg_covar':1e-6,'min_component_covariance_eigenvalue':float(min(np.linalg.eigvalsh(c)[0] for c in p['covariances'])),'lower_bounds_trace':getattr(g,'lower_bounds_',None)}
    return p,rec

def formula_checks(gaussian,gmm,zval):
    ix=np.array([0,199,399,598]);z=zval[ix];rows=[]
    a=gaussian_logpdf(z,gaussian);b=multivariate_normal.logpdf(z,mean=gaussian['mean'],cov=gaussian['covariance']);error=float(np.max(np.abs(a-b)))
    rows.append({'base':'GAUSSIAN','validation_positions':ix.tolist(),'manual_vs_scipy_maxabs_nat':error,'pass':error<=1e-7})
    if gmm is not None:
        a=den.gmm_logpdf(z,gmm);b=logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(z,mean=mu,cov=cov) for w,mu,cov in zip(gmm['weights'],gmm['means'],gmm['covariances'])]),axis=0);error=float(np.max(np.abs(a-b)));rows.append({'base':'GMM8','validation_positions':ix.tolist(),'manual_vs_scipy_mixture_maxabs_nat':error,'pass':error<=1e-7})
    return {'checks':rows,'pass':all(r['pass'] for r in rows),'prior_head_FACTOR_residual_audit_repeated':False}

def latent_statistics(z):
    mean=z.mean(axis=0);center=z-mean;cov=center.T@center/len(z);delta=cov-np.eye(34)
    return {'N':len(z),'dimension':34,'covariance_denominator':'N','mean':mean,'mean_absmax':float(np.max(np.abs(mean))),'mean_l2':float(np.linalg.norm(mean)),'covariance':cov,'covariance_minus_identity_absmax':float(np.max(np.abs(delta))),'covariance_minus_identity_Frobenius':float(np.linalg.norm(delta)),'covariance_eigenvalues':np.linalg.eigvalsh(cov),'marginal_quantile_levels':[0,.05,.25,.5,.75,.95,1],'marginal_quantiles':np.quantile(z,[0,.05,.25,.5,.75,.95,1],axis=0),'diagnostic_only':True}

def full_scores(z,ld,oldparts,gaussian,gmm):
    bases={'STANDARD':-.5*(34*np.log(2*np.pi)+(z*z).sum(axis=1)),'GAUSSIAN':gaussian_logpdf(z,gaussian)}
    if gmm is not None:bases['GMM8']=den.gmm_logpdf(z,gmm)
    summary={};private={}
    for name,lp in bases.items():
        hp=lp+ld;complete=hp+oldparts['tail_logdensity']-oldparts['Jacobian']
        if not np.isfinite(complete).all():
            summary[name]={'all_rows':len(z),'finite_rows':int(np.isfinite(complete).sum()),'complete_NLL':None,'status':'nonfinite_score'}
        else:
            summary[name]={'all_rows':len(z),'finite_rows':len(z),'complete_NLL':float(-complete.mean()),'mean_base_logdensity':float(lp.mean()),'mean_head_flow_logdet':float(ld.mean()),'mean_head_std_logdensity':float(hp.mean()),'mean_tail_logdensity':float(oldparts['tail_logdensity'].mean()),'mean_logJacobian':float(oldparts['Jacobian'].mean()),'arithmetic':'CPUfloat64; fixed R15 conditional tail and FACTOR terms'}
        for k,v in {'complete_logdensity':complete,'head_logdensity':hp,'base_logdensity':lp,'head_flow_logdet':ld,'tail_logdensity':oldparts['tail_logdensity'],'Jacobian':oldparts['Jacobian']}.items():private[name+'__'+k]=v
    return summary,private

def sample_heads(name,p,head,inputs,uniform=None,chunk=128):
    white=inputs['head_white'];components=None
    if name=='GAUSSIAN':latent=p['mean']+white@p['cholesky'].T
    else:
        cdf=np.cumsum(p['weights']);components=np.searchsorted(cdf,uniform,side='right')
        if not np.all((components>=0)&(components<8)):raise ValueError('component CDF does not cover actual uniforms')
        latent=np.empty_like(white)
        for k in range(8):
            use=components==k;latent[use]=p['means'][k]+white[use]@p['cholesky'][k].T
    xs=[]
    for start in range(0,len(latent),chunk):
        with torch.no_grad():xx,_=head.inverse(torch.from_numpy(latent[start:start+chunk]))
        xs.append(xx.numpy())
    return np.concatenate(xs),latent,components

def psd_contribution(domains,rep):
    summary={};private={}
    for name,d in domains.items():
        x,h,r,legal=encoded_heads(d['V'],rep);b=h[:,1];q=np.exp(h[:,0]);head_only=b[:,None]+q[:,None]*(rep['mean']+h[:,2:]@rep['basis'][:,:32].T);full=head_only+r@rep['basis'][:,32:].T
        head_only[~legal]=np.nan;full[~legal]=np.nan
        fhead=ev.features(head_only);ffull=ev.features(full);delta=ffull['features']['PSD']-fhead['features']['PSD'];defined=legal&d['valid']&fhead['valid']&ffull['valid']&np.isfinite(delta);blocks={}
        for region in [None,0,1,2,3]:
            use=defined if region is None else defined&(d['region']==region);source=d['valid'] if region is None else d['valid']&(d['region']==region);key='overall' if region is None else 'q'+str(region+1)
            blocks[key]={'source_QC_n':int(source.sum()),'defined_n':int(use.sum()),'undefined_n':int(source.sum()-use.sum()),'PSD_full':ev.summary(ffull['features']['PSD'][use]),'PSD_head_only':ev.summary(fhead['features']['PSD'][use]),'PSD_full_minus_head_only':ev.summary(delta[use]),'difference_RMS':float(np.sqrt(np.mean(delta[use]**2))) if use.any() else None}
        summary[name]={'definition':'head-only b+Q(mean+U_head h32), full adds U_tail r; difference full minus head-only','head_only_not_complete_response':True,'regions_by_fixed_full_output_charge':blocks}
        private.update({name+'__full_PSD':ffull['features']['PSD'],name+'__head_PSD':fhead['features']['PSD'],name+'__delta':delta,name+'__defined':defined,name+'__region':d['region']})
        x[~legal]=np.nan;d['head34']=x
    return summary,private
