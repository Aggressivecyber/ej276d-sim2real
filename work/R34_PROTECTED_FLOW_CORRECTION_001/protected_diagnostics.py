from __future__ import annotations
import numpy as np
import evaluation as ev
from scipy.stats import wasserstein_distance
from projection_core import project,protect,charges,REGIONS
from flow_runtime import infer
from model import adc

def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

def precision_report(payload,x0,base_post,applied):
    pre=payload['pre_ADC_f64_V'];keep=payload['kept_increment_f64_V'];removed=payload['removed_increment_f64_V'];delta=payload['original_increment_f64_V']
    f32=payload['ADC_input_f32_V'];post=payload['waveforms_V'];x0=np.asarray(x0,dtype='f8')
    stages={'protected_preADC_f64':pre,'ADC_input_f32':f32,'post_ADC':post}
    bases={'protected_preADC_f64':x0,'ADC_input_f32':x0.astype('f4'),'post_ADC':base_post}
    errors={};private={}
    for name,v in stages.items():
        a=charges(v);b=charges(bases[name]);errors[name]={}
        for key in a:
            d=a[key]-b[key];errors[name][key]={'difference':ev.summary(d),'maxabs':float(np.max(np.abs(d))) if np.isfinite(d).all()else None}
            private[name+'__'+key]=a[key];private[name+'__BASE__'+key]=b[key];private[name+'__difference__'+key]=d
    residual={'baseline_preADC_bitwise_BASE':bool(np.array_equal(pre[:,:100],x0[:,:100])),
              'baseline_f32_bitwise_BASE':bool(np.array_equal(f32[:,:100],x0.astype('f4')[:,:100])),
              'baseline_postADC_bitwise_BASE':bool(np.array_equal(post[:,:100],base_post[:,:100])),
              'fast_segment_sum_maxabs_V':float(np.max(np.abs(keep[:,100:150].sum(1)))),
              'slow_segment_sum_maxabs_V':float(np.max(np.abs(keep[:,150:400].sum(1)))),
              'unsupported_zero_increment_exact':bool(np.array_equal(keep[~applied],np.zeros_like(keep[~applied]))),
              'idempotence_maxabs_V':float(np.max(np.abs(project(keep)-keep))),
              'orthogonality_dot_maxabs_V2':float(np.max(np.abs(np.sum(keep*removed,1)))),
              'finite_all_precision_levels':bool(np.isfinite(pre).all()and np.isfinite(f32).all()and np.isfinite(post).all()),
              'ADC_input_f32_rounding_maxabs_V':float(np.max(np.abs(f32.astype('f8')-pre)))}
    residual['constraint_pass']=all(residual[k]for k in ['baseline_preADC_bitwise_BASE','baseline_f32_bitwise_BASE','baseline_postADC_bitwise_BASE','unsupported_zero_increment_exact','finite_all_precision_levels']) and max(residual['fast_segment_sum_maxabs_V'],residual['slow_segment_sum_maxabs_V'])<=1e-10
    energy=np.sum(delta*delta,1);ke=np.sum(keep*keep,1);re=np.sum(removed*removed,1);zero=energy==0
    ratio=np.divide(ke,energy,out=np.full(len(energy),np.nan),where=~zero)
    residual.update(delta_zero_rows=int(zero.sum()),retained_energy_fraction_nonzero=ev.summary(ratio[~zero]),
                    energy_closure_maxabs_V2=float(np.max(np.abs(ke+re-energy))))
    regions={}
    for name,(left,right)in REGIONS.items():
        regions[name]={}
        for key,a in [('original',delta),('kept',keep),('removed',removed)]:
            r=np.sqrt(np.mean(a[:,left:right]**2,1));private[name+'__'+key+'_RMS_V']=r;regions[name][key+'_RMS_V']=ev.summary(r)
    return {'rows':len(pre),'actual_constraints':residual,'charge_PSD_precision_levels':errors,'projection_regions':regions},private

def restore(pipeline,a,old,chunk):
    pre,signal,electronic=infer(pipeline,a['source_waveforms_V'],a['z'],a['white'],a['style'],chunk)
    detail=pipeline.details
    checks={k:bool(np.array_equal(detail[k],old[k]))for k in ['base_pre_ADC_V','base_signal_V','base_electronic_V','flow_increment_V','flow_applied','source_Qbin_actual_preADC']}
    post,_=adc(pre);checks['original_post_ADC_V']=bool(np.array_equal(post,old['waveforms_V']))
    assert all(checks.values()),checks
    return pre,detail,{'rows':len(pre),'checks':checks,'exact_recovery_pass':True,'original_auxiliary_x1_not_stored':True,'new_model_fit':0}

def probes(pipeline,main,mainstyle,name,root,out,cfg,edges,counter):
    oldroot=__import__('pathlib').Path(cfg['R33_root']);orig=cfg['endpoint_origins'][name]
    a=load(oldroot/'PROBE_INPUTS_CORRECTED.npz');a['style']=mainstyle[a['noise_origin_parent_index']]
    old=load(oldroot/'results'/(orig+'_SOURCE_PROBES_PRIVATE.npz'))
    pre,detail,recovery=restore(pipeline,a,old,cfg['generation_chunk']);counter['auxiliary_recovery_inference_rows']+=len(pre)
    payload,ad,provenance=protect(detail['base_pre_ADC_V'],pre,origin='recovered')
    counter['source_probe_rows']+=len(pre);post=payload['waveforms_V'];f=ev.features(post);sf=ev.features(a['source_waveforms_V']);reference=main[a['reference_parent_index']];rf=ev.features(reference);tags=a['probe_kind'];rows={}
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),
            'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-reference[use])**2,1))),
            'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-rf['features']['PSD'][use]),
            'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),
            'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use]),'flow_applied_N':int(detail['flow_applied'][use].sum())}
    keys=['source_parent_index','source_parent_ids','noise_origin_parent_index','reference_parent_index','source_waveform_modified_counterfactual','probe_kind']
    np.savez_compressed(out/(name+'_SOURCE_PROBES_PRIVATE.npz'),**{k:a[k]for k in keys},style=a['style'],
        **payload,base_pre_ADC_V=detail['base_pre_ADC_V'],base_signal_V=detail['base_signal_V'],base_electronic_V=detail['base_electronic_V'],
        original_pre_ADC_f32_V=pre,flow_applied=detail['flow_applied'],source_Qbin_actual_preADC=detail['source_Qbin_actual_preADC'],PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise';base_post,_=adc(detail['base_pre_ADC_V'])
    check,_=precision_report(payload,detail['base_pre_ADC_V'],base_post,detail['flow_applied'])
    return {'rows':rows,'deterministically_projected_rows':len(pre),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),
        'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),
        'corrected_parent_semantics_reused':True,'physical_final_branch_split':'unknown','ADC':ad,'recovery':recovery,'constraints':check,'provenance':provenance,'nonzero_source_sensitivity_not_correct_physics':True}

def joint(pipeline,source,comp,name,root,out,cfg,edges,valley,counter):
    oldroot=__import__('pathlib').Path(cfg['R33_root']);orig=cfg['endpoint_origins'][name]
    a=load(oldroot/'JOINT8_INPUTS_PRIVATE.npz');old=load(oldroot/'results'/(orig+'_joint8_REPEAT_PRIVATE.npz'));parent=a['source_parent_index'];pick=parent.reshape(128,8)[:,0]
    assert np.all(parent.reshape(128,8)==pick[:,None])
    pre,detail,recovery=restore(pipeline,a,old,cfg['generation_chunk']);counter['auxiliary_recovery_inference_rows']+=len(pre)
    payload,ad,provenance=protect(detail['base_pre_ADC_V'],pre,origin='recovered');counter['repeat_noise_rows']+=len(pre);post=payload['waveforms_V'];f=ev.features(post)
    from flow_core import bins
    sb=bins(source[pick],edges);p=f['features']['PSD'].reshape(128,8);q=f['features']['Qtotal_mVns'].reshape(128,8);valid=f['valid'].reshape(128,8)
    signal=detail['base_signal_V'].astype('f8');electronic=detail['base_electronic_V'].astype('f8');base_rounding=detail['base_pre_ADC_V'].astype('f8')-signal-electronic
    pieces=np.stack([signal,electronic,base_rounding,payload['kept_increment_f64_V'],payload['removed_increment_f64_V']],0).reshape(5,128,8,480)
    centered=pieces-pieces.mean(2,keepdims=True);cov=np.einsum('ignt,jgnt->gijt',centered,centered)/7
    totalvar=np.var(payload['pre_ADC_f64_V'].reshape(128,8,480),axis=1,ddof=1)
    originalvar=np.var(pre.astype('f8').reshape(128,8,480),axis=1,ddof=1)
    protected_closure=float(np.max(np.abs(cov[:,:4,:4].sum((1,2))-totalvar)))
    original_closure=float(np.max(np.abs(cov.sum((1,2))-originalvar)))
    assert max(protected_closure,original_closure)<1e-10
    rows=[];by={};regions={'full480':(0,480),'baseline100':(0,100),'prompt100_150':(100,150),'postpeak150_220':(150,220),'tail220_400':(220,400)}
    terms=['base_signal','base_electronic','base_sum_rounding','kept_increment','removed_increment']
    for j,idx in enumerate(pick):
        de=valley['bins'][sb[j]]if 0<=sb[j]<8 else None;pp=p[j,valid[j]];qq=q[j,valid[j]]
        rows.append({'source_parent_index':int(idx),'composition':str(comp[idx]),'valid_draws':int(valid[j].sum()),'output_PSD':ev.summary(pp),'output_Q_mVns':ev.summary(qq),
            'both_sides_of_frozen_valley':bool(de and de['valid']and len(pp)and(pp<de['low']).any()and(pp>de['high']).any())})
    for category in np.unique(comp[pick]):
        use=comp[pick]==category;selected=[r for r in rows if r['composition']==category];cs={}
        for region,(left,right)in regions.items():
            cs[region]={terms[i]+'__'+terms[j]:ev.summary(cov[use,i,j,left:right].mean(1))for i in range(5)for j in range(5)}
        by[str(category)]={'parents':int(use.sum()),'within_parent_PSD_SD':ev.summary(np.array([r['output_PSD']['sample_sd']for r in selected if r['output_PSD']['sample_sd']is not None])),
            'within_parent_Q_SD_mVns':ev.summary(np.array([r['output_Q_mVns']['sample_sd']for r in selected if r['output_Q_mVns']['sample_sd']is not None])),
            'parents_straddling_valley':sum(r['both_sides_of_frozen_valley']for r in selected),
            'within_parent_waveform_RMS_V':{region:ev.summary(np.sqrt(totalvar[use,left:right].mean(1)))for region,(left,right)in regions.items()},'five_algebraic_components_covariance_V2':cs}
    basepost,_=adc(detail['base_pre_ADC_V']);constraint,_=precision_report(payload,detail['base_pre_ADC_V'],basepost,detail['flow_applied'])
    np.savez_compressed(out/(name+'_joint8_REPEAT_PRIVATE.npz'),source_parent_index=parent,**payload,original_pre_ADC_f32_V=pre,
        base_pre_ADC_V=detail['base_pre_ADC_V'],base_signal_V=detail['base_signal_V'],base_electronic_V=detail['base_electronic_V'],
        flow_applied=detail['flow_applied'],source_Qbin_actual_preADC=detail['source_Qbin_actual_preADC'],PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'],valid=f['valid'],covariance_V2=cov,total_variance_preADC_V2=totalvar,original_total_variance_preADC_V2=originalvar)
    summary={'parent_count':128,'draws_each':8,'rows':rows,'by_composition':by,'variance_denominator':7,
        'protected_covariance_closure_maxabs_V2':protected_closure,'original_covariance_closure_maxabs_V2':original_closure,
        'physical_final_signal_electronic_split':'unknown','cross_terms_retained':True,'base_rounding_is_not_physical_noise':True,'component_order':terms,'ADC':ad,'recovery':recovery,'constraints':constraint,'provenance':provenance,'repeats_not_independent_source_events':True}
    return {'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0}

def boundary_metrics(domains,out):
    locations={'first_99_100':(99,100),'first_149_150':(149,150),'first_399_400':(399,400)}
    data={};results={}
    for n,d in domains.items():
        if n in ['SOURCE_CLEAN','SOURCE_INPUT']:continue
        a=d['V'];data[n]={}
        for key,(i,j)in locations.items():data[n][key]=a[:,j]-a[:,i]
        for i in [99,100,149,150,399,400]:data[n]['second_center_'+str(i)]=a[:,i-1]-2*a[:,i]+a[:,i+1]
    for n,values in data.items():
        results[n]={}
        for key,a in values.items():
            target=data['REAL'][key];good=np.isfinite(a)&domains[n]['valid'];rgood=np.isfinite(target)&domains['REAL']['valid']
            results[n][key]={'candidate':ev.summary(a[good]),'REAL':ev.summary(target[rgood]),'W1_V':float(wasserstein_distance(a[good],target[rgood]))if good.any()and rgood.any()else None}
    np.savez_compressed(out/'BOUNDARY_DIFFERENCES_PRIVATE.npz',**{n+'__'+key:a for n,values in data.items()for key,a in values.items()})
    return {'fixed_locations_only':True,'native_voltage_V':True,'first_definition':'v[j]-v[i]','second_definition':'v[i-1]-2*v[i]+v[i+1]','ADC_LSB_V':1/16383,'results':results}
