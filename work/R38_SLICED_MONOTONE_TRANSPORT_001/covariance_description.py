from __future__ import annotations
import numpy as np

def describe_covariance(domains,names,out):
    payload={};result={}
    def moment(d,mask):
        x=d['V'][mask].astype('f8');x-=x.mean(0,keepdims=True)
        c=(x.T@x)/(len(x)-1);sd=np.sqrt(np.maximum(np.diag(c),0))
        den=sd[:,None]*sd[None,:];r=np.divide(c,den,out=np.zeros_like(c),where=den>0)
        return c,r,int(np.sum(sd==0))
    def summary(c,r,rc,rr):
        off=~np.eye(480,dtype=bool)
        return {'covariance_relative_Frobenius_vs_REAL':float(np.linalg.norm(c-rc)/np.linalg.norm(rc)),
            'off_diagonal_covariance_relative_Frobenius_vs_REAL':float(np.linalg.norm((c-rc)[off])/np.linalg.norm(rc[off])),
            'off_diagonal_correlation_RMS_difference':float(np.sqrt(np.mean((r[off]-rr[off])**2))),
            'diagonal_variance_RMS_difference_V2':float(np.sqrt(np.mean((np.diag(c)-np.diag(rc))**2)))}
    for i in [None,*range(8)]:
        key='population' if i is None else 'observedQbin'+str(i+1)
        real=domains['REAL'];rm=real['valid'] if i is None else real['valid']&(real['b']==i)
        rows={};rp=int(rm.sum())
        if rp>=40:
            rc,rr,rz=moment(real,rm);payload[key+'__REAL_Cov_V2']=rc;payload[key+'__REAL_Correlation']=rr
            rows['REAL']={'N':rp,'qualified':True,'zero_variance_coordinates':rz}
            for n in names:
                d=domains[n];use=d['valid'] if i is None else d['valid']&(d['b']==i);N=int(use.sum())
                row={'N':N,'REAL_N':rp,'qualified':N>=40}
                if N>=40:
                    c,r,z=moment(d,use);payload[key+'__'+n+'_Cov_V2']=c;payload[key+'__'+n+'_Correlation']=r
                    row.update(zero_variance_coordinates=z,**summary(c,r,rc,rr))
                rows[n]=row
        else:rows={'REAL':{'N':rp,'qualified':False}}
        result[key]=rows
    np.savez_compressed(out/'DEV_COVARIANCE_DESCRIPTION_PRIVATE.npz',**payload)
    return {'sample_covariance_ddof':1,'matrix_dimension':480,'per_bin_common_support_N':40,'tables':result,
        'observed_output_Q_condition':True,'no_parameter_fit_or_deployment_use':True,
        'constant_coordinate_correlation_set0_and_counted':True,'DEV_used_for_description_only':True}
