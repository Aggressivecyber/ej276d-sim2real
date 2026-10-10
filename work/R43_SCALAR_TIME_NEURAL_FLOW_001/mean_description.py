from __future__ import annotations
import numpy as np

REGIONS={'full480':(0,480),'baseline':(0,100),'prompt':(100,150),'postpeak':(150,250),'tail':(250,400),'late':(400,480)}
def describe(domains,names,source_bins,train_mu):
    real=domains['REAL'];summary={};private={}
    for name in ['REAL','TRAIN_REFERENCE',*names]:
        d=domains[name];table={}
        conditions=['observed_Q'] if name in ['REAL','TRAIN_REFERENCE'] else ['observed_Q','source_Q']
        for condition in conditions:
            bins=d['b'] if condition=='observed_Q' else source_bins;rows=[]
            for k in [None,*range(8)]:
                use=d['valid'] if k is None else d['valid']&(bins==k)
                target=real['valid'] if k is None else real['valid']&(real['b']==k)
                n=int(use.sum());nr=int(target.sum());label='overall' if k is None else 'bin'+str(k+1)
                row={'bin_id_1based':None if k is None else k+1,'source_count':len(use) if k is None else int((bins==k).sum()),
                     'QC_n':n,'invalid_n':int((~d['valid']).sum()) if k is None else int(((bins==k)&~d['valid']).sum()),
                     'REAL_observed_Q_QC_n':nr,'qualified':n>=40 and nr>=40,'reason':None if n>=40 and nr>=40 else 'a_domain_below40',
                     'conditioning':'same observed-output Q' if condition=='observed_Q' else 'source-Q candidate versus observed-Q REAL; distinct conditions'}
                if n:
                    m=d['u'][use].mean(0);sd=d['u'][use].std(0,ddof=1) if n>1 else None
                    private[name+'__'+condition+'__'+label+'__mean_u']=m
                    if sd is not None:private[name+'__'+condition+'__'+label+'__SD_u']=sd
                    row['pointwise_SD_summary']={'defined':sd is not None,'mean':float(sd.mean()) if sd is not None else None,
                        'quantiles':np.quantile(sd,[0,.1,.5,.9,1]).tolist() if sd is not None else None}
                    if nr:
                        rm=real['u'][target].mean(0);delta=m-rm;row['full480_mean_MSE']=float(np.mean(delta**2))
                        row['region_mean_residual_RMS']={r:float(np.sqrt(np.mean(delta[a:b]**2))) for r,(a,b)in REGIONS.items()}
                        if sd is not None and nr>1:
                            rsd=real['u'][target].std(0,ddof=1)
                            row['region_SD_difference_RMS']={r:float(np.sqrt(np.mean((sd[a:b]-rsd[a:b])**2))) for r,(a,b)in REGIONS.items()}
                        private[name+'__'+condition+'__'+label+'__delta_REAL_u']=delta
                    if condition=='source_Q' and k is not None:
                        delta=m-train_mu[k];row['full480_mean_MSE_vs_frozen_TRAIN_mu']=float(np.mean(delta**2))
                        row['region_RMS_vs_frozen_TRAIN_mu']={r:float(np.sqrt(np.mean(delta[a:b]**2)))for r,(a,b)in REGIONS.items()}
                rows.append(row)
            table[condition]=rows
        summary[name]=table
    common=[i for i in range(8) if all(summary[n]['observed_Q'][i+1]['qualified'] for n in ['REAL',*names])]
    common_summary={n:float(np.mean([summary[n]['observed_Q'][i+1]['full480_mean_MSE']for i in common])) if common else None for n in names}
    return {'domains':summary,'common_observed_Q_bin_ids_1based':[i+1 for i in common],
        'common_bin_equal_mean_full480_MSE':common_summary,'regions':{r:list(v)for r,v in REGIONS.items()},
        'SD_ddof':1,'no_transformed_classification':True,'DEV_descriptions_not_used_in_training':True},private
