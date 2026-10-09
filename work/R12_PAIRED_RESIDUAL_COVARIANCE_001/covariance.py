from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_normal
import density as den
import evaluation as ev

def recover(old,cfg):
    alpha=float(old['LW_shrinkage']);eps=cfg['covariance_epsilon_V2'];C=old['tail_cov'];d=len(C)
    if not 0<=alpha<1:raise ValueError('invalid/singular shrinkage fraction')
    C0=C-eps*np.eye(d);mu=float(np.trace(C0)/d);S=(C0-alpha*mu*np.eye(d))/(1-alpha);S=(S+S.T)/2;back=(1-alpha)*S+alpha*mu*np.eye(d);err=back-C0
    fro=float(np.linalg.norm(err)/np.linalg.norm(C0));ma=float(np.max(np.abs(err)));eig=np.linalg.eigvalsh(S);newC=S+eps*np.eye(d);newEig=np.linalg.eigvalsh(newC)
    if fro>cfg['recovery_relative_Frobenius_tol'] or ma>cfg['recovery_maxabs_V2_tol']:raise ValueError('covariance inverse recovery tolerance')
    new=dict(old);new['tail_cov']=newC;new['tail_cholesky']=np.linalg.cholesky(newC)
    fields={'alpha':alpha,'mu_V2':mu,'dimension':d,'epsilon_V2':eps,'S_min_eigenvalue_V2':float(eig[0]),'S_max_eigenvalue_V2':float(eig[-1]),'new_min_eigenvalue_V2':float(newEig[0]),'new_max_eigenvalue_V2':float(newEig[-1]),'new_condition_number':float(newEig[-1]/newEig[0]),'old_condition_number':float(np.linalg.cond(C)),'recovered_relative_Frobenius_error':fro,'recovered_maxabs_V2_error':ma,'analytic_recovery':True,'symmetrization_only':True,'TRAIN_statistics_refit':0}
    return new,S,fields

def validate(val,rep,gmm):
    h,r,good=den.encode(val,rep);x=(h[good]-rep['head_mean'])/rep['head_scale'];head=den.gmm_logpdf(x,gmm);tail=den.gaussian_logpdf(r[good],rep,x)
    jac=np.full(len(head),rep['head_log_scale'])
    if rep['kind']=='FACTOR_PCA_GMM':jac+=33*h[good,0]+rep['constant_logdet']
    score=np.full(len(val),-np.inf);score[good]=head+tail-jac
    out={'all_validation_rows':len(val),'legal_rows':int(good.sum()),'illegal_rows':int((~good).sum()),'complete_NLL':float(-score.mean()),'mean_head_std_logdensity':float(head.mean()),'mean_tail_logdensity':float(tail.mean()),'mean_logJacobian':float(jac.mean())}
    ix=np.array([0,len(x)//3,2*len(x)//3,len(x)-1]);means=rep['tail_intercept']+x[ix]@rep['tail_B'];direct=np.array([multivariate_normal.logpdf(r[good][j],mean=means[k],cov=rep['tail_cov']) for k,j in enumerate(ix)]);delta=float(np.max(np.abs(tail[ix]-direct)))
    out['new_tail_density_selfcheck_maxabs_nat']=delta
    if delta>1e-7:raise RuntimeError('tail density formula check')
    return out,score

def fixed_mean(rep,gmm,inp,chunk=128):
    vs=[];heads=[];xs=[]
    for start in range(0,len(inp['component']),chunk):
        sl=slice(start,start+chunk);c=inp['component'][sl];z=inp['head_white'][sl];x=np.empty_like(z)
        for k in range(len(gmm['weights'])):
            use=c==k
            if use.any():x[use]=gmm['means'][k]+z[use]@gmm['cholesky'][k].T
        h=x*rep['head_scale']+rep['head_mean'];tailmean=rep['tail_intercept']+x@rep['tail_B'];UT=rep['basis'][:,32:]
        if rep['kind']=='RAW_PCA_GMM':v=rep['mean']+h@rep['basis'][:,:32].T+tailmean@UT.T
        else:v=h[:,1,None]+np.exp(h[:,0,None])*(rep['mean']+h[:,2:]@rep['basis'][:,:32].T)+tailmean@UT.T
        vs.append(v);heads.append(h);xs.append(x)
    return np.concatenate(vs),np.concatenate(heads),np.concatenate(xs)

def decomposition(old,S,mean,preold,prenew):
    UT=old['basis'][:,32:];alpha=float(old['LW_shrinkage']);eps=1e-12;d=len(S);C0=old['tail_cov']-eps*np.eye(d);mu=float(np.trace(C0)/d);out={};private={}
    for name,lo,hi,order in [('baseline',0,100,0),('prompt_curvature',100,150,2),('postpeak_curvature',150,220,2),('tail_curvature',220,400,2)]:
        A=UT[lo:hi];M=mean[:,lo:hi];oldV=preold[:,lo:hi];newV=prenew[:,lo:hi]
        if order:
            A=np.diff(A,n=2,axis=0);M=np.diff(M,n=2,axis=1);oldV=np.diff(oldV,n=2,axis=1);newV=np.diff(newV,n=2,axis=1);div=hi-lo-2
        else:
            A=A-A.mean(axis=0);M=M-M.mean(axis=1,keepdims=True);oldV=oldV-oldV.mean(axis=1,keepdims=True);newV=newV-newV.mean(axis=1,keepdims=True);div=99
        fixed=(M*M).sum(axis=1)/div;emp=float(np.sum((A@S)*A)/div);shrink=float(alpha*mu*np.sum(A*A)/div);floor=float(eps*np.sum(A*A)/div);oldtail=float(np.sum((A@old['tail_cov'])*A)/div);newtail=emp+floor
        oldactual=(oldV*oldV).sum(axis=1)/div;newactual=(newV*newV).sum(axis=1)/div;delta=float(np.max(np.abs(oldtail-((1-alpha)*emp+shrink+floor))))
        if delta>1e-12:raise RuntimeError('variance decomposition inconsistency')
        out[name]={'window':[lo,hi],'operator':'centered sample variance' if not order else 'second difference mean-square','denominator':div,'fixed_conditional_mean_term_V2':ev.summary(fixed),'empirical_covariance_term_V2':emp,'old_scaled_empirical_term_V2':(1-alpha)*emp,'old_shrinkage_term_V2':shrink,'numeric_floor_term_V2':floor,'old_expected_population_mean_V2':float(fixed.mean()+oldtail),'new_expected_population_mean_V2':float(fixed.mean()+newtail),'expected_new_minus_old_V2':newtail-oldtail,'actual_old_pre_ADC_mean_square_V2':ev.summary(oldactual),'actual_new_pre_ADC_mean_square_V2':ev.summary(newactual),'actual_paired_mean_delta_V2':float(np.mean(newactual-oldactual)),'decomposition_maxabs_V2':delta,'heads':len(mean),'extra_Monte_Carlo_draws':0}
        private[name+'__old']=oldactual;private[name+'__new']=newactual;private[name+'__fixed_mean_term']=fixed
    return out,private
