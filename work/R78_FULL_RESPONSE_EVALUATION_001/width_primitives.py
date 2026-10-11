"""Original fixed two-density slot mathematics; no Torch, bootstrap scan, or fitting on import."""
from __future__ import annotations
import math
import numpy as np
from common_math_primitives import posterior,distances

EM = {'max_updates': 500, 'parameter_atol': 1e-8, 'variance_floor': 1e-8,
      'initial_weights': [.5, .5], 'initial_means_quantiles': [.25, .75],
      'initial_variances': 'full population variance, floored',
      'min_component_effective_count': 20,
      'min_mean_gap_over_pooled_sigma': 1.0,
      'bootstrap_repeats': 200, 'bootstrap_seed': 2026100838,
      'min_bootstrap_valid_for_CI': 100,
      'component_order': 'ascending mean at every update',
      'not_frozen_CAL_reference_refit': True}

FWHM = math.sqrt(8 * math.log(2))

def fit_slice(psd, logq, multiplicity=None):
    x = np.asarray(psd, dtype='f8'); q = np.asarray(logq, dtype='f8')
    a = np.ones(len(x)) if multiplicity is None else np.asarray(multiplicity, dtype='f8')
    keep = np.isfinite(x) & np.isfinite(q) & (a > 0)
    x, q, a = x[keep], q[keep], a[keep]
    n = float(a.sum()); reasons = []
    result = {'N': n, 'distinct_records_with_positive_weight': len(x),
              'valid': False, 'converged': False, 'fit_attempts': 0,
              'restarts': 0, 'failure_reasons': [], 'components': []}
    if n < 40:
        result['failure_reasons'] = ['total_support_below40']; return result
    # Weighted quantiles exactly match type7 quantiles of the expanded integer
    # group bootstrap sample, without materializing repeated waveform records.
    order = np.argsort(x, kind='stable'); sx=x[order]; sa=a[order]
    cumulative = np.cumsum(sa)
    def quantile(p):
        t=(n-1)*p; lo=math.floor(t); hi=math.ceil(t)
        i=np.searchsorted(cumulative,lo,side='right');j=np.searchsorted(cumulative,hi,side='right')
        return float(sx[i]*(hi-t)+sx[j]*(t-lo)) if hi!=lo else float(sx[i])
    mean=float(np.dot(a,x)/n); variance=float(np.dot(a,(x-mean)**2)/n)
    w=np.array([.5,.5]); mu=np.array([quantile(.25),quantile(.75)])
    var=np.full(2,max(variance,EM['variance_floor'])); floor=variance<EM['variance_floor']
    resp,ll=posterior(x,w,mu,var); delta=None; result['fit_attempts']=1
    initial={'weights':w.tolist(),'means':mu.tolist(),'variances':var.tolist(),'average_loglik':float(np.dot(a,np.log(np.maximum(np.sum(w[None,:]/np.sqrt(2*np.pi*var)[None,:]*np.exp(-.5*(x[:,None]-mu[None,:])**2/var[None,:]),axis=1),1e-300)))/n)}
    for step in range(1,EM['max_updates']+1):
        old=np.r_[w,mu,var]; ar=resp*a[:,None]; nk=ar.sum(axis=0)
        if np.any(nk<=0) or not np.all(np.isfinite(nk)):
            reasons.append('nonpositive_effective_count');break
        w=nk/n;mu=(ar*x[:,None]).sum(axis=0)/nk
        raw=(ar*(x[:,None]-mu[None,:])**2).sum(axis=0)/nk
        floor=bool(floor or np.any(raw<EM['variance_floor']));var=np.maximum(raw,EM['variance_floor'])
        ix=np.argsort(mu,kind='stable');w,mu,var=w[ix],mu[ix],var[ix]
        if not np.all(np.isfinite(np.r_[w,mu,var])):
            reasons.append('nonfinite_parameters');break
        resp,ll=posterior(x,w,mu,var)
        delta=float(np.max(np.abs(np.r_[w,mu,var]-old)))
        if delta<=EM['parameter_atol']:
            result['converged']=True;break
    if not result['converged']:reasons.append('not_converged500')
    if floor:reasons.append('variance_floor_touched')
    separation=float((mu[1]-mu[0])/np.sqrt(np.mean(var)))
    if separation < EM['min_mean_gap_over_pooled_sigma']:reasons.append('density_components_not_separated')
    result.update(initial=initial,updates=step,weights=w.tolist(),means=mu.tolist(),variances=var.tolist(),
                  effective_counts=(n*w).tolist(),parameter_delta=delta,
                  variance_floor_ever=floor,mean_gap_over_pooled_sigma=separation)
    # Responsibilities remain those of the PSD-only density fit. Trend removal
    # is weighted local regression in log charge, without truth assignments.
    for k in range(2):
        ar=a*resp[:,k]; s=ar.sum(); xm=float(np.dot(ar,q)/s);ym=float(np.dot(ar,x)/s)
        den=float(np.dot(ar,(q-xm)**2));slope=float(np.dot(ar,(q-xm)*(x-ym))/den) if den>1e-14 else None
        residual=x-ym if slope is None else x-ym-slope*(q-xm)
        residual_sigma=float(np.sqrt(np.dot(ar,residual**2)/s))
        good=result['converged'] and not floor and separation>=EM['min_mean_gap_over_pooled_sigma'] and n*w[k]>=20
        c={'order':k,'mean':float(mu[k]),'weight':float(w[k]),
           'effective_N_times_weight':float(n*w[k]),
           'responsibility_Kish_N':float(s*s/np.dot(ar,ar)),
           'width_defined':bool(good),
           'sigma_PSD':float(np.sqrt(var[k])) if good else None,
           'FWHM_PSD':float(FWHM*np.sqrt(var[k])) if good else None,
           'diagnostic_unqualified_sigma':float(np.sqrt(var[k])),
           'weighted_logQ_center':xm,'weighted_PSD_center':ym,
           'local_PSD_per_logQ_slope':slope,
           'detrended_sigma_PSD':residual_sigma if good and slope is not None else None,
           'detrended_FWHM_PSD':FWHM*residual_sigma if good and slope is not None else None,
           'diagnostic_unqualified_detrended_sigma':residual_sigma}
        if n*w[k]<20:reasons.append(f'component{k}_support_below20')
        result['components'].append(c)
    result['valid']=all(c['width_defined'] for c in result['components'])
    result['failure_reasons']=reasons
    return result

def bin_indices(q, edges):
    z=np.log(np.asarray(q,dtype='f8'))
    bins=np.searchsorted(edges,z,side='right')-1
    bins[z==edges[-1]]=7
    return z,bins
