def means_coverage(domains,seen,complement,source_bins):
    result={};private={};real=domains['REAL'];maxreconstruct=0.
    for name,d in domains.items():
        d['seen']=d['u']@seen;d['complement']=d['u']@complement
        maxreconstruct=max(maxreconstruct,float(np.max(np.abs(d['seen']+d['complement']-d['u']))))
        private[name+'__u']=d['u'].astype('f4');private[name+'__Q_mVns']=d['q'];private[name+'__valid']=d['valid'];private[name+'__Qbin']=d['b']
    for name,d in domains.items():
        table={}
        for condition in ['observed_Q','source_Q'] if name in CFG.get('source_condition_domains',NAMES) else ['observed_Q']:
            rows=[];b=d['b'] if condition=='observed_Q' else source_bins
            for i in [None,*range(8)]:
                mask=d['valid'] if i is None else d['valid']&(b==i)
                ref=real['valid'] if i is None else real['valid']&(real['b']==i)
                raw=np.ones(len(b),bool) if i is None else b==i
                label='overall' if i is None else 'bin'+str(i+1)
                ok=int(mask.sum())>=40 and int(ref.sum())>=40
                row={'bin_id_1based':None if i is None else i+1,'source_count':int(raw.sum()),'QC_n':int(mask.sum()),
                     'invalid_n':int((raw&~d['valid']).sum()),'REAL_observed_Q_QC_n':int(ref.sum()),
                     'qualified':ok,'reason':None if ok else 'a_domain_below40',
                     'conditioning':'same observed Q' if condition=='observed_Q' else 'candidate source Q vs REAL observed Q; not identical conditioning'}
                if mask.any():
                    m=d['u'][mask].mean(0);med=np.median(d['u'][mask],axis=0)
                    private[name+'__'+condition+'__'+label+'__mean_u']=m
                    private[name+'__'+condition+'__'+label+'__median_u']=med
                    if ref.any():
                        delta=m-real['u'][ref].mean(0);ss=delta@seen;cc=delta@complement
                        total=float(np.dot(delta,delta));st=float(np.dot(ss,ss));ct=float(np.dot(cc,cc))
                        row.update(mean_difference_sqnorm=total,seen_sqnorm=st,complement_sqnorm=ct,
                                   seen_fraction=st/total if total>0 else None,
                                   complement_fraction=ct/total if total>0 else None,
                                   energy_closure_error=st+ct-total)
                        private[name+'__'+condition+'__'+label+'__delta_u']=delta
                        private[name+'__'+condition+'__'+label+'__delta_seen']=ss
                        private[name+'__'+condition+'__'+label+'__delta_complement']=cc
                rows.append(row)
            table[condition]=rows
        result[name]=table
    assert maxreconstruct<1e-10
    np.savez_compressed(OUT/'TEMPORAL_DIAGNOSTICS_PRIVATE.npz',**private)
    result['_algebra']={'actual_u_reconstruction_maxabs':maxreconstruct,'means_are_not_full_distributions':True}
    write('A_COVERAGE_METRICS.json',result);progress()
    return result

def fine_distances(domains,constants):
    draws={'ORIGINAL':{name:constants[name+'__projection'].astype('f8') for name in GROUPS}}
    actual=load(pathlib.Path(CFG['R27_root'])/'results/NEW_PROJECTIONS_PRIVATE.npz')
    for seed in [20261009271,20261009272,20261009273,20261009274]:
        label='NEW_'+str(seed);draws[label]={name:actual[label+'__'+name] for name in GROUPS}
    comparisons=[*NAMES,'TRAIN_REFERENCE'];levels=np.linspace(.005,.995,128)
    results={};distances_private={}
    for label,matrices in draws.items():
        results[label]={}
        for name in GROUPS:
            quantiles={};counts={}
            for domain_name,d in domains.items():
                x=d['u'] if name=='shape480' else np.diff(d['u'],axis=1) if name=='diff479' else d['s'][:,:100]
                projected=((x-constants[name+'__mean'])/constants[name+'__scale'])@matrices[name]
                qs=[];ns=[]
                for i in [None,*range(8)]:
                    mask=d['valid'] if i is None else d['valid']&(d['b']==i)
                    ns.append(int(mask.sum()))
                    qs.append(np.quantile(projected[mask],levels,axis=0,method='linear') if mask.any() else None)
                quantiles[domain_name]=qs;counts[domain_name]=ns
            rows={}
            for domain_name in comparisons:
                parts=[]
                for j in range(9):
                    n=counts[domain_name][j];nr=counts['REAL'][j]
                    a=quantiles[domain_name][j];b=quantiles['REAL'][j]
                    diff=np.abs(a-b).mean(0) if a is not None and b is not None else None
                    if diff is not None:distances_private[label+'__'+name+'__'+domain_name+'__'+str(j)]=diff
                    parts.append({'bin_id_1based':None if j==0 else j,'candidate_N':n,'REAL_N':nr,
                                  'qualified':n>=40 and nr>=40,'W1_quantile_direction_mean':float(diff.mean()) if diff is not None else None,
                                  'reason':None if n>=40 and nr>=40 else 'a_domain_below40'})
                qualified=[j for j in range(1,9) if parts[j]['qualified']]
                rows[domain_name]={'overall':parts[0],'bins':parts[1:],
                                   'qualified_bin_ids_1based':qualified,
                                   'qualified_bin_equal_mean':float(np.mean([parts[j]['W1_quantile_direction_mean'] for j in qualified])) if qualified else None}
            results[label][name]=rows
            write('results/B_'+label+'_'+name+'.json',rows)
            COUNTER['completed_projection_group_blocks']+=1;progress()
            log('projection '+label+' '+name+' saved')
    np.savez_compressed(OUT/'PROJECTION_W1_DIRECTIONS_PRIVATE.npz',**distances_private)
    common={};blocks=list(draws)
    for name in GROUPS:
        ids=[i for i in range(1,9) if all(results[b][name][d]['bins'][i-1]['qualified'] for b in blocks for d in comparisons)]
        table={}
        for d in comparisons:
            values={b:float(np.mean([results[b][name][d]['bins'][i-1]['W1_quantile_direction_mean'] for i in ids])) if ids else None for b in blocks}
            new=[values[b] for b in blocks[1:] if values[b] is not None]
            old=values['ORIGINAL']
            table[d]={'equal_mean_on_common_bins':values,'new_block_range':[min(new),max(new)] if new else None,
                      'new_block_mean':float(np.mean(new)) if new else None,
                      'new_mean_divided_by_original':float(np.mean(new))/old if new and old and old>0 else None}
        common[name]={'common_qualified_bin_ids_1based':ids,'domains':table}
    output={'blocks':results,'common_support_summary':common,'quantile_definition':{'first':.005,'last':.995,'count':128,'method':'linear'},
            'new_block_range_is_projection_sampling_not_CI':True,'frozen_TRAIN_scales':True,'main_population_reweighted':False}
    write('B_PROJECTION_METRICS.json',output)
    return output
