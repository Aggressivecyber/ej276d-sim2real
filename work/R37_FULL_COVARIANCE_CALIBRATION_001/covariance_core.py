from __future__ import annotations
import numpy as np


def symmetric(a):
    return (a + a.T) * .5


class Moments:
    def __init__(self, dimension=480):
        self.n=0; self.mean=np.zeros(dimension,'f8'); self.M2=np.zeros((dimension,dimension),'f8')

    def update(self, values):
        x=np.asarray(values,dtype='f8'); m=len(x)
        if m==0:return
        assert np.isfinite(x).all()
        mean=x.mean(0); centered=x-mean; batch_M2=centered.T@centered
        if self.n==0:self.n=m;self.mean=mean;self.M2=batch_M2;return
        delta=mean-self.mean; new_n=self.n+m
        self.M2+=batch_M2+np.outer(delta,delta)*(self.n*m/new_n)
        self.mean+=delta*(m/new_n);self.n=new_n

    def covariance(self):
        assert self.n>1
        return symmetric(self.M2/(self.n-1))


def regularize(c, fraction=1e-6, negative_limit=1e-8):
    c=symmetric(np.asarray(c,dtype='f8'));assert np.isfinite(c).all()
    w,v=np.linalg.eigh(c); trace=float(np.trace(c)); dimension=len(c)
    if not trace>0 or not w[-1]>0:raise FloatingPointError('nonpositive covariance trace')
    if w[0]<-negative_limit*w[-1]:raise FloatingPointError('original covariance excessive negative eigenvalue')
    tau=fraction*trace/dimension; clipped=np.maximum(w,tau)
    reg=symmetric((v*clipped)@v.T)
    root=symmetric((v*np.sqrt(clipped))@v.T)
    inverse_root=symmetric((v*(1/np.sqrt(clipped)))@v.T)
    info={'trace':trace,'eigenfloor':tau,'original_min_eigenvalue':float(w[0]),
          'original_max_eigenvalue':float(w[-1]),'floor_coordinate_count':int((w<tau).sum()),
          'added_trace_fraction':float((clipped.sum()-w.sum())/trace),
          'regularized_condition_number':float(clipped[-1]/clipped[0])}
    return c,reg,root,inverse_root,w,clipped,info


def gaussian_map(cs,ct,fraction=1e-6,negative_limit=1e-8):
    source=regularize(cs,fraction,negative_limit);target=regularize(ct,fraction,negative_limit)
    cs0,csr,x,xi,ws,wrs,sd=source;ct0,ctr,y,yi,wt,wrt,td=target
    middle=symmetric(x@ctr@x);mid_eigen=np.linalg.eigvalsh(middle)
    if mid_eigen[0]<-negative_limit*mid_eigen[-1]:raise FloatingPointError('intermediate excessive negative eigenvalue')
    # H H.T = Cs_reg^.5 Ct_reg Cs_reg^.5. SVD computes its same principal
    # square root without squaring its small singular values in the eigensolver.
    # This is a stable evaluation of the frozen formula, not a new floor/rotation.
    h=x@y;u,s,_=np.linalg.svd(h,full_matrices=False)
    principal=symmetric((u*s)@u.T)
    a=symmetric(xi@principal@xi);aw=np.linalg.eigvalsh(a)
    residual=float(np.linalg.norm(a@csr@a.T-ctr,'fro')/np.linalg.norm(ctr,'fro'))
    diagnostics={'source':sd,'target':td,'middle_min_eigenvalue':float(mid_eigen[0]),
        'middle_max_eigenvalue':float(mid_eigen[-1]),'middle_negative_eigen_count':int((mid_eigen<0).sum()),
        'principal_root_evaluation':'SVD factor H=Cs_reg^.5 Ct_reg^.5; same unique principal square root',
        'principal_root_relative_residual':float(np.linalg.norm(principal@principal-middle,'fro')/np.linalg.norm(middle,'fro')),
        'A_symmetry_maxabs':float(np.max(np.abs(a-a.T))),
        'A_min_eigenvalue':float(aw[0]),'A_max_eigenvalue':float(aw[-1]),
        'A_condition_number':float(aw[-1]/aw[0]) if aw[0]>0 else None,
        'regularized_covariance_relative_Frobenius_residual':residual,
        'unregularized_covariance_relative_Frobenius_residual':float(np.linalg.norm(a@cs0@a.T-ct0,'fro')/np.linalg.norm(ct0,'fro')),
        'regularized_mean_covariance_closure_not_distribution_truth':True,
        'positive_definite':bool(aw[0]>0),'numerical_pass':bool(aw[0]>0 and residual<=1e-6)}
    payload={'A':a,'Cs':cs0,'Ct':ct0,'Cs_reg':csr,'Ct_reg':ctr,
             'Cs_eigenvalues':ws,'Ct_eigenvalues':wt,'Cs_reg_eigenvalues':wrs,'Ct_reg_eigenvalues':wrt,
             'A_eigenvalues':aw}
    return payload,diagnostics


def selfcheck():
    s=np.array([[2.,.4,.2],[.4,1.,.1],[.2,.1,.8]])
    same,sd=gaussian_map(s,s);diag,dd=gaussian_map(np.diag([1.,4.,9.]),np.diag([4.,9.,16.]))
    t=np.array([[1.,.2,-.1],[.2,2.,.3],[-.1,.3,1.5]])
    generic,gd=gaussian_map(s,t)
    check={'small_artificial_only':True,'identical_A_maxabs_minus_identity':float(np.max(np.abs(same['A']-np.eye(3)))),
           'diagonal_A_maxabs_expected':float(np.max(np.abs(diag['A']-np.diag([2.,1.5,4/3])))),
           'generic_relative_covariance_error':gd['regularized_covariance_relative_Frobenius_residual'],
           'all_positive':all(d['positive_definite']for d in [sd,dd,gd])}
    check['pass']=bool(check['all_positive']and max(check[k]for k in ['identical_A_maxabs_minus_identity','diagonal_A_maxabs_expected','generic_relative_covariance_error'])<1e-10)
    assert check['pass'];return check


def calibrate(guide, b, maps, mu, scale, ms, mt):
    base=np.asarray(guide,dtype='f4');out64=base.astype('f8');b=np.asarray(b,dtype='i8')
    applied=(b>=0)&(b<8)
    for k in range(8):
        use=np.flatnonzero(b==k)
        if len(use):
            normalized=(base[use].astype('f8')-mu[k])/scale[k]
            out64[use]=mu[k]+scale[k]*(mt[k]+(normalized-ms[k])@maps[k].T)
    if not np.isfinite(out64).all():raise FloatingPointError('nonfinite calibration output')
    return out64.astype('f4'),out64,applied
