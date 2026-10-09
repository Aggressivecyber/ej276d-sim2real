"""Frozen charge-conditioned density diagnostics, not particle labeling."""
from __future__ import annotations
import collections
import math
import numpy as np
import torch
from common_math import posterior, distances


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


def section(psd,q,edges,multiplicity=None,with_histogram=True):
    x=np.asarray(psd);q=np.asarray(q);z,bins=bin_indices(q,edges)
    mult=np.ones(len(x),dtype='f8') if multiplicity is None else multiplicity
    out=[];grid=np.linspace(-.1,.6,129)
    for k in range(8):
        select=(bins==k)&(mult>0)
        r=fit_slice(x[select],z[select],mult[select])
        r.update(bin=k,logQ_edges=edges[k:k+2].tolist(),charge_edges_mVns=np.exp(edges[k:k+2]).tolist(),
                 actual_charge_range_mVns=[float(q[select].min()),float(q[select].max())] if select.any() else None)
        if with_histogram:
            hist,_=np.histogram(x[select],bins=grid,weights=mult[select]);r['histogram_counts']=hist.tolist()
            r['histogram_PSD_outside_grid_count']=float(mult[select&( (x<grid[0]) | (x>grid[-1]) )].sum())
        out.append(r)
    return {'N':float(mult.sum()),'below_logQ_range':float(mult[bins<0].sum()),
            'above_logQ_range':float(mult[bins>7].sum()),
            'PSD_below_minus0_1':float(mult[x<-.1].sum()),'PSD_above0_6':float(mult[x>.6].sum()),
            'slices':out,'histogram_edges_PSD':grid.tolist() if with_histogram else None}


def slope_over_bins(slices, centers, component, metric):
    points=[(centers[k],r['components'][component][metric]) for k,r in enumerate(slices)
            if len(r['components'])==2 and r['components'][component][metric] is not None
            and r['components'][component][metric]>0]
    if len(points)<3:return None
    x,y=np.asarray(points).T;return float(np.dot(x-x.mean(),np.log(y)-np.log(y).mean())/np.dot(x-x.mean(),x-x.mean()))


def grouped_bootstrap(psd,q,groups,edges):
    uniq,inv=np.unique(groups,return_inverse=True);rng=np.random.default_rng(EM['bootstrap_seed'])
    samples=[];fail=collections.Counter();centers=(edges[:-1]+edges[1:])/2
    for _ in range(EM['bootstrap_repeats']):
        mult=np.bincount(rng.integers(len(uniq),size=len(uniq)),minlength=len(uniq))[inv]
        s=section(psd,q,edges,mult,with_histogram=False)
        for r in s['slices']:
            for e in r['failure_reasons']:fail[e]+=1
        samples.append(s['slices'])
    bands=[];trends=[]
    for component in [0,1]:
        for metric in ['mean','sigma_PSD','detrended_sigma_PSD']:
            for k in range(8):
                v=[a[k]['components'][component][metric] for a in samples
                   if len(a[k]['components'])==2 and a[k]['components'][component]['width_defined']
                   and a[k]['components'][component][metric] is not None]
                bands.append({'component':component,'metric':metric,'bin':k,'valid_replicates':len(v),
                              'CI95':np.quantile(v,[.025,.975]).tolist() if len(v)>=EM['min_bootstrap_valid_for_CI'] else None})
        for metric in ['sigma_PSD','detrended_sigma_PSD']:
            slopes=[slope_over_bins(a,centers,component,metric) for a in samples]
            slopes=[x for x in slopes if x is not None]
            trends.append({'component':component,'metric':metric,'definition':'OLS log(width) versus logQ center over >=3 qualified bins',
                           'valid_replicates':len(slopes),'CI95':np.quantile(slopes,[.025,.975]).tolist() if len(slopes)>=100 else None,
                           'fraction_negative_among_defined':float(np.mean(np.array(slopes)<0)) if slopes else None})
    return {'replicates':200,'artificial_groups':len(uniq),'not_independent_acquisitions':True,
            'fixed_initialization_each_fit':True,'failure_occurrences':dict(fail),'bands':bands,'trend_slopes':trends}


def parent_width(features,edges):
    psd=np.stack([f['features']['PSD'] for f in features]);q=np.stack([f['features']['Qtotal_mVns'] for f in features])
    good=np.logical_and.reduce([f['valid'] for f in features]);sd=psd.std(axis=0,ddof=1);meanq=q.mean(axis=0)
    _,bins=bin_indices(meanq,edges);out=[]
    for k in range(8):
        v=sd[good&(bins==k)]
        out.append({'bin':k,'parent_count':len(v),'within_parent_PSD_SD_median':float(np.median(v)) if len(v) else None,
                    'within_parent_PSD_SD_q10_q90':np.quantile(v,[.1,.9]).tolist() if len(v) else None})
    return {'views_per_parent':len(features),'not_independent_events':True,'slices':out}


def conditional_targets(v,centers,width):
    from model import observables
    scalar,_,_=observables(v)
    logq=scalar[:,1];psd=scalar[:,0]
    grid=torch.linspace(-.1,.6,128,dtype=v.dtype)
    w=torch.exp(-.5*((logq[:,None]-centers[None,:])/width)**2)
    step=torch.sigmoid((grid[None,:]-psd[:,None])/.0025)
    cdf=(w.T@step)/w.sum(dim=0).clamp_min(1e-8)[:,None]
    mass=w.mean(dim=0)
    return {'CDF':cdf.detach(),'window_occupancy':mass.detach(),
            'PSD_grid':grid,'logQ_centers':centers,'logQ_window_width':width,
            'sample_n':len(v),'PSD_outside_grid_n':int(((psd<-.1)|(psd>.6)).sum()),
            'all_samples_included':True}


def conditional_loss(fake,target):
    from model import observables
    scalar,_,_=observables(fake);logq=scalar[:,1];psd=scalar[:,0]
    w=torch.exp(-.5*((logq[:,None]-target['logQ_centers'][None,:])/target['logQ_window_width'])**2)
    step=torch.sigmoid((target['PSD_grid'][None,:]-psd[:,None])/.0025)
    cdf=(w.T@step)/w.sum(dim=0).clamp_min(1e-8)[:,None]
    cdf_error=(cdf-target['CDF']).abs().mean()
    occupancy_error=(w.mean(dim=0)-target['window_occupancy']).abs().mean()
    return 4*cdf_error+occupancy_error,{'conditional_CDF':cdf_error,'conditional_occupancy':occupancy_error}


def plot_widths(pre,post,bootstrap,edges,colors,out):
    import matplotlib.pyplot as plt
    centers=np.exp((edges[:-1]+edges[1:])/2)
    domains={**pre,'R3B_DEV':post};names=['REAL_DEV','SIM_DEV','GEN_W0_DEV','R2_DEV','R3_DEV','R3B_DEV']
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for k in range(2):
        for j,metric in enumerate(['mean','sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for name in names:
                values=[r['components'][k][metric] if len(r['components'])==2 else None for r in domains[name]['slices']]
                values=np.array([np.nan if v is None else v for v in values])
                if metric=='mean':
                    values=np.array([v if r['valid'] else np.nan for v,r in zip(values,domains[name]['slices'])])
                ax.plot(centers,values,'o-',ms=4,label=name,color=colors[name])
            bands=[r for r in bootstrap['REAL_DEV']['bands'] if r['component']==k and r['metric']==metric]
            low=np.array([np.nan if r['CI95'] is None else r['CI95'][0] for r in bands])
            high=np.array([np.nan if r['CI95'] is None else r['CI95'][1] for r in bands])
            ax.fill_between(centers,low,high,color='black',alpha=.12,label='REAL DEV group-bootstrap95%')
            ax.set_xscale('log');ax.set_xlabel('Signed total charge [mV ns]');ax.set_ylabel(metric)
            ax.set_title(f'Mean-ordered density component {k}: {metric}');ax.grid(alpha=.15)
            if j==0:ax.legend(fontsize=7)
    fig.suptitle('TRAIN-frozen narrow logQ slices; gaps denote unqualified fits; not particle truth')
    fig.tight_layout();fig.savefig(out/'energy_peak_width.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,9));grid=np.linspace(-.1,.6,129);xx=(grid[:-1]+grid[1:])/2
    for b,ax in enumerate(axes.ravel()):
        for name in names:
            s=domains[name]['slices'][b];hh=np.array(s['histogram_counts'])
            ax.step(xx,hh/max(1,s['N'])/(grid[1]-grid[0]),where='mid',label=f'{name} n={int(s["N"])}',color=colors[name],lw=1)
        rr=domains['REAL_DEV']['slices'][b]
        if rr.get('weights'):
            density=sum(w/np.sqrt(2*np.pi*v)*np.exp(-.5*(xx-m)**2/v) for w,m,v in zip(rr['weights'],rr['means'],rr['variances']))
            ax.plot(xx,density,'k--',lw=1,label='REAL two-component diagnostic'+(' (unqualified)' if not rr['valid'] else ''))
        ax.set_xlabel('Signed PSD');ax.set_ylabel('Density (full slice denominator)')
        ax.set_title(f'Bin{b+1}: Q {np.exp(edges[b]):.0f}–{np.exp(edges[b+1]):.0f} mV ns');ax.legend(fontsize=6)
    fig.suptitle('All QC-valid overlap/unknown retained; no width/FoM selection')
    fig.tight_layout();fig.savefig(out/'energy_psd_slices.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for k in range(2):
        for j,metric in enumerate(['sigma_PSD','detrended_sigma_PSD']):
            ax=axes[k,j]
            for name in ['REAL_TRAIN_fit','REAL_TRAIN_validation','REAL_DEV']:
                s=pre[name]['slices'];v=[r['components'][k][metric] if len(r['components'])==2 else None for r in s]
                ax.plot(centers,[np.nan if z is None else z for z in v],'o-',label=name)
                bands=[r for r in bootstrap[name]['bands'] if r['component']==k and r['metric']==metric]
                ax.fill_between(centers,[np.nan if r['CI95'] is None else r['CI95'][0] for r in bands],
                                [np.nan if r['CI95'] is None else r['CI95'][1] for r in bands],alpha=.1)
            ax.set_xscale('log');ax.set_title(f'Component{k}: {metric}');ax.set_xlabel('Charge [mV ns]');ax.set_ylabel('PSD width');ax.legend(fontsize=8)
    fig.suptitle('REAL same-acquisition artificial-group bootstrap200; unsupported curves remain gaps')
    fig.tight_layout();fig.savefig(out/'REAL_width_uncertainty.png',dpi=150,bbox_inches='tight');plt.close(fig)
