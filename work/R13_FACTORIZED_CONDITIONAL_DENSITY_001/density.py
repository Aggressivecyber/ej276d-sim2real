from __future__ import annotations
import numpy as np,time,math,warnings
from scipy.linalg import qr,solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf

def encode(v,rep):
    if rep['kind']=='RAW_PCA_GMM':
        z=(v-rep['mean'])@rep['basis'];head=z[:,:32];tail=z[:,32:];valid=np.isfinite(v).all(axis=1)
    else:
        b=v[:,:100].mean(axis=1);q=(v[:,100:400]-b[:,None]).sum(axis=1);valid=np.isfinite(v).all(axis=1)&(q>0)
        qq=np.where(valid,q,1.);s=(v-b[:,None])/qq[:,None];z=(s-rep['mean'])@rep['basis'];head=np.column_stack([np.log(qq),b,z[:,:32]]);tail=z[:,32:]*qq[:,None]
    return head,tail,valid

def inverse_full(v,rep,rank):
    if rep['kind']=='RAW_PCA_GMM':
        z=(v-rep['mean'])@rep['basis'];r=rep['basis'].shape[1] if rank=='full' else rank
        return rep['mean']+z[:,:r]@rep['basis'][:,:r].T
    b=v[:,:100].mean(axis=1);q=(v[:,100:400]-b[:,None]).sum(axis=1);s=(v-b[:,None])/q[:,None];z=(s-rep['mean'])@rep['basis'];r=rep['basis'].shape[1] if rank=='full' else rank
    return b[:,None]+q[:,None]*(rep['mean']+z[:,:r]@rep['basis'][:,:r].T)

def reconstruction(v,rep,ranks):
    records=[]
    b=v[:,:100].mean(axis=1);q=(v[:,100:400]-b[:,None]).sum(axis=1);ps=1-(v[:,100:150]-b[:,None]).sum(axis=1)/q
    good=np.isfinite(v).all(axis=1)&((q>0) if rep['kind']=='FACTOR_PCA_GMM' else True)
    for rank in ranks:
        vv=v[good];hat=inverse_full(vv,rep,rank);err=hat-vv;hb=hat[:,:100].mean(axis=1);hq=(hat[:,100:400]-hb[:,None]).sum(axis=1);hps=1-(hat[:,100:150]-hb[:,None]).sum(axis=1)/hq
        rms=np.sqrt(np.mean(err*err,axis=1));defined=np.isfinite(hps)
        records.append({'rank':rank,'rows':len(v),'defined_rows':int(good.sum()),'excluded_illegal_rows':int((~good).sum()),'RMS_mean_V':float(np.mean(rms)),'RMS_quantiles_V':np.quantile(rms,[0,.05,.5,.95,1]).tolist(),'maxabs_V':float(np.max(np.abs(err))),'baseline_maxabs_V':float(np.max(np.abs(hb-b[good]))),'charge_maxabs_Vns':float(np.max(np.abs(hq-q[good]))),'PSD_error_defined_n':int(defined.sum()),'PSD_abs_error_mean':float(np.mean(np.abs(hps[defined]-ps[good][defined]))) if defined.any() else None,'charge_abs_error_mean_Vns':float(np.mean(np.abs(hq-q[good])))})
    return records

def representation(v,kind,cfg):
    n=len(v);rank=32
    if kind=='RAW_PCA_GMM':
        p=PCA(n_components=480,svd_solver='full',whiten=False);p.fit(v);basis=p.components_.T.copy();mean=p.mean_.copy();constant=0.;constraints={};fitvalid=np.ones(n,bool)
    else:
        a=np.r_[np.full(100,.01),np.zeros(380)];w=np.r_[np.full(100,-3.),np.ones(300),np.zeros(80)];qfull,_=qr(np.column_stack([a,w]),mode='full');Z=qfull[:,2:].copy()
        for j in range(Z.shape[1]):
            if Z[np.argmax(np.abs(Z[:,j])),j]<0:Z[:,j]*=-1
        s0=np.r_[np.zeros(100),np.full(300,1/300),np.zeros(80)];b=v[:,:100].mean(axis=1);charge=(v[:,100:400]-b[:,None]).sum(axis=1);fitvalid=np.isfinite(v).all(axis=1)&(charge>0)
        s=(v[fitvalid]-b[fitvalid,None])/charge[fitvalid,None];p=PCA(n_components=478,svd_solver='full',whiten=False);p.fit((s-s0)@Z);basis=Z@p.components_.T;mean=s0+Z@p.mean_
        sign,constant=np.linalg.slogdet(np.column_stack([np.ones(480),mean,basis]))
        if not np.isfinite(constant) or sign==0:raise RuntimeError('factor inverse matrix singular')
        constraints={'a_mean':float(a@mean),'w_mean':float(w@mean),'maxabs_a_basis':float(np.max(np.abs(a@basis))),'maxabs_w_basis':float(np.max(np.abs(w@basis))),'Z_orthogonality_maxabs':float(np.max(np.abs(Z.T@Z-np.eye(478)))),'basis_orthogonality_maxabs':float(np.max(np.abs(basis.T@basis-np.eye(478)))),'Jacobian_constant_logabsdet':float(constant)}
    rep={'kind':kind,'basis':basis,'mean':mean,'constant_logdet':float(constant),'explained_variance':p.explained_variance_,'explained_variance_ratio':p.explained_variance_ratio_}
    h,r,legal=encode(v,rep);h=h[legal];r=r[legal];hm=h.mean(axis=0);raw=h.std(axis=0,ddof=1);scale=np.maximum(raw,1e-12);x=(h-hm)/scale
    X=np.column_stack([np.ones(len(x)),x]);pen=np.eye(X.shape[1])*.001;pen[0,0]=0;coef=np.linalg.solve(X.T@X+pen,X.T@r);residual=r-X@coef;lw=LedoitWolf(assume_centered=False,store_precision=False);lw.fit(residual);coef[0]+=lw.location_
    cov=lw.covariance_+np.eye(r.shape[1])*1e-12;chol=np.linalg.cholesky(cov)
    rep.update(head_mean=hm,head_scale=scale,head_raw_scale=raw,tail_intercept=coef[0],tail_B=coef[1:],tail_cov=cov,tail_cholesky=chol,head_log_scale=float(np.log(scale).sum()),LW_shrinkage=float(lw.shrinkage_),LW_residual_location=lw.location_)
    meta={'kind':kind,'fit_rows':n,'fit_legal_rows':len(h),'fit_illegal_rows':int((~legal).sum()),'head_dim':h.shape[1],'tail_dim':r.shape[1],'head_rank':32,'head_std_floor_affected':int((raw<1e-12).sum()),'tail_LW_shrinkage':float(lw.shrinkage_),'tail_residual_center_maxabs_V':float(np.max(np.abs(lw.location_))),'tail_cov_min_eigenvalue_V2':float(np.linalg.eigvalsh(cov)[0]),'constraints':constraints,'head_log_standardization_scale':rep['head_log_scale'],'Jacobian_constant_logabsdet':float(constant),'total_explained_variance_first32':float(p.explained_variance_ratio_[:32].sum()),'no_tail_residual_templates':True}
    return rep,x,meta

def gaussian_logpdf(r,rep,x):
    err=r-rep['tail_intercept']-x@rep['tail_B'];z=solve_triangular(rep['tail_cholesky'],err.T,lower=True,check_finite=False).T;dim=r.shape[1]
    return -.5*(dim*np.log(2*np.pi)+2*np.log(np.diag(rep['tail_cholesky'])).sum()+np.sum(z*z,axis=1))

def gmm_logpdf(x,p):
    vals=[]
    for weight,mean,chol in zip(p['weights'],p['means'],p['cholesky']):
        z=solve_triangular(chol,(x-mean).T,lower=True,check_finite=False).T
        vals.append(np.log(weight)-.5*(x.shape[1]*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+np.sum(z*z,axis=1)))
    return logsumexp(np.stack(vals),axis=0)

def fit_models(v,rep,x,cfg,index,counter,log):
    h,r,valid=encode(v,rep);xv=(h[valid]-rep['head_mean'])/rep['head_scale'];tail=gaussian_logpdf(r[valid],rep,xv);q=(v[valid,100:400]-v[valid,:100].mean(axis=1)[:,None]).sum(axis=1);jac=rep['head_log_scale']+ (33*np.log(q)+rep['constant_logdet'] if rep['kind']=='FACTOR_PCA_GMM' else 0.)
    models={};records=[];scores={};checks=[]
    for k in [1,4,8]:
        counter['GMM_fit_attempts']+=1
        if counter['GMM_fit_attempts']>6:raise RuntimeError('GMM fit cap')
        start=time.monotonic();seed=cfg['seed']+1000*index+k
        g=GaussianMixture(n_components=k,random_state=seed,**cfg['GMM'])
        try:
            with warnings.catch_warnings(record=True) as warnings_seen:g.fit(x)
            gp={'weights':g.weights_,'means':g.means_,'covariances':g.covariances_,'cholesky':np.stack([np.linalg.cholesky(c) for c in g.covariances_])};head=g.score_samples(xv);score=np.full(len(v),-np.inf);score[valid]=head+tail-jac;finite=np.isfinite(score);complete=float(-np.mean(score)) if finite.all() else math.inf
            delta=float(np.max(np.abs(gmm_logpdf(xv[:4],gp)-head[:4])));checks.append({'components':k,'head_logpdf_formula_maxabs_nat':delta});models[k]=gp;scores[k]=score
            rec={'components':k,'seed':seed,'converged':bool(g.converged_),'iterations':int(g.n_iter_),'lower_bound_head_std':float(g.lower_bound_),'complete_validation_NLL':complete,'eligible_for_selection':bool(g.converged_ and finite.all()),'validation_all_n':len(v),'validation_finite_n':int(finite.sum()),'validation_illegal_q_or_nonfinite_n':int((~valid).sum()),'validation_mean_head_std_logdensity':float(head.mean()),'validation_mean_tail_logdensity':float(tail.mean()),'validation_mean_log_Jacobian':float(np.mean(jac)),'elapsed_s':time.monotonic()-start,'warnings':[str(w.message) for w in warnings_seen]}
        except (np.linalg.LinAlgError,ValueError,FloatingPointError) as e:rec={'components':k,'seed':seed,'converged':False,'eligible_for_selection':False,'failure':repr(e),'elapsed_s':time.monotonic()-start}
        records.append(rec);log(f'{rep["kind"]} GMM{k} {rec}')
    eligible=[r for r in records if r['eligible_for_selection']];selected=min(eligible,key=lambda r:(r['complete_validation_NLL'],r['components']))['components'] if eligible else None
    return models,records,selected,scores,checks

def sample_inputs(rep,gmm,n,seed):
    rng=np.random.default_rng(seed);return {'component':rng.choice(len(gmm['weights']),size=n,p=gmm['weights']),'head_white':rng.standard_normal((n,len(rep['head_mean']))),'tail_white':rng.standard_normal((n,rep['tail_cholesky'].shape[0])),'seed':np.array(seed)}

def invert_samples(rep,gmm,inp,chunk=128):
    out=[]
    for start in range(0,len(inp['component']),chunk):
        sl=slice(start,start+chunk);c=inp['component'][sl];z=inp['head_white'][sl];x=np.empty_like(z)
        for k in range(len(gmm['weights'])):
            m=c==k
            if m.any():x[m]=gmm['means'][k]+z[m]@gmm['cholesky'][k].T
        h=x*rep['head_scale']+rep['head_mean'];tail=rep['tail_intercept']+x@rep['tail_B']+inp['tail_white'][sl]@rep['tail_cholesky'].T
        if rep['kind']=='RAW_PCA_GMM':v=rep['mean']+h@rep['basis'][:,:32].T+tail@rep['basis'][:,32:].T
        else:
            with np.errstate(over='ignore',invalid='ignore'):q=np.exp(h[:,0]);v=h[:,1,None]+q[:,None]*(rep['mean']+h[:,2:]@rep['basis'][:,:32].T)+tail@rep['basis'][:,32:].T
        out.append(v)
    return np.concatenate(out)

def adc(v):
    code=np.rint((v+.5)*16383);clip=(code<0)|(code>16383);post=(np.clip(code,0,16383)/16383-.5).astype('<f4')
    return post,{'ADC_clip_points':int(clip.sum()),'ADC_clip_rows':int(clip.any(axis=1).sum()),'pre_nonfinite_rows':int((~np.isfinite(v).all(axis=1)).sum()),'post_nonfinite_rows':int((~np.isfinite(post).all(axis=1)).sum()),'ADC_arithmetic':'float64 nearest-even, postf32','no_CFD_recrop':True,'rejection_resampling':False}

def selfcheck(v,rep,gmm,cfg):
    ix=np.array([0,len(v)//3,2*len(v)//3,len(v)-1]);vv=v[ix];h,r,ok=encode(vv,rep);h=h[ok];r=r[ok];x=(h-rep['head_mean'])/rep['head_scale'];ll=gaussian_logpdf(r,rep,x);mean=rep['tail_intercept']+x@rep['tail_B'];direct=np.array([multivariate_normal.logpdf(r[i],mean=mean[i],cov=rep['tail_cov']) for i in range(len(r))]);full=inverse_full(vv[ok],rep,'full');err=float(np.max(np.abs(full-vv[ok])))
    check={'fixed_validation_positions':ix.tolist(),'full_inverse_maxabs_V':err,'tail_Gaussian_formula_maxabs_nat':float(np.max(np.abs(ll-direct))),'full_inverse_pass':err<=cfg['inverse_check_maxabs_V'],'tail_density_pass':np.max(np.abs(ll-direct))<=cfg['density_check_abs_nat']}
    if rep['kind']=='FACTOR_PCA_GMM':
        hd=h[0];q=np.exp(hd[0]);UH=rep['basis'][:,:32];UT=rep['basis'][:,32:];J=np.column_stack([q*(rep['mean']+UH@hd[2:]),np.ones(480),q*UH,UT]);sign,logdet=np.linalg.slogdet(J);expected=rep['constant_logdet']+33*np.log(q);delta=float(abs(logdet-expected));numeric=[]
        def inverse_coords(hd,rt):return hd[1]+np.exp(hd[0])*(rep['mean']+UH@hd[2:])+UT@rt
        for j in [0,1,2,33]:
            step=1e-5*rep['head_scale'][j];hp=hd.copy();hm=hd.copy();hp[j]+=step;hm[j]-=step;fd=(inverse_coords(hp,r[0])-inverse_coords(hm,r[0]))/(2*step);numeric.append({'head_column':j,'maxabs_derivative_error':float(np.max(np.abs(fd-J[:,j])))})
        check.update(Jacobian_constant_formula_maxabs_logdet=delta,Jacobian_pass=delta<=cfg['Jacobian_logdet_check_abs'],Jacobian_finite_difference=numeric)
    if not check['full_inverse_pass'] or not check['tail_density_pass'] or not check.get('Jacobian_pass',True):raise RuntimeError('density/inverse selfcheck failed '+str(check))
    return check
