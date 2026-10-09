from __future__ import annotations
import numpy as np
import evaluation as ev
from flow_runtime import infer
from model import adc

def probes(model,main,mainstyle,name,root,out,cfg,edges,counter):
    with np.load(root/'PROBE_INPUTS_CORRECTED.npz',allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
    x=a['source_waveforms_V'];style=mainstyle[a['noise_origin_parent_index']]
    pre,signal,electronic=infer(model,x,a['z'],a['white'],style,cfg['generation_chunk']);detail=model.details
    post,ad=adc(pre);counter['source_probe_rows']+=len(post)
    f=ev.features(post);sf=ev.features(x);tags=a['probe_kind'];reference=main[a['reference_parent_index']];rf=ev.features(reference);rows={}
    for tag in np.unique(tags):
        use=tags==tag
        rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),
            'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-reference[use])**2,1))),
            'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-rf['features']['PSD'][use]),
            'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),
            'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use]),'flow_applied_N':int(detail['flow_applied'][use].sum())}
    payload={k:a[k]for k in ['source_parent_index','source_parent_ids','noise_origin_parent_index','reference_parent_index',
                           'source_waveform_modified_counterfactual','probe_kind']}
    np.savez_compressed(out/(name+'_SOURCE_PROBES_PRIVATE.npz'),**payload,style=style,waveforms_V=post,
        **detail,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return {'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),
        'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),
        'corrected_parent_semantics_reused':True,'physical_final_branch_split':'unknown','ADC':ad,'nonzero_source_sensitivity_not_correct_physics':True}

def joint(model,source,comp,name,root,out,cfg,edges,valley,counter):
    with np.load(root/'JOINT8_INPUTS_PRIVATE.npz',allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
    parent=a['source_parent_index'];pick=parent.reshape(128,8)[:,0]
    assert np.all(parent.reshape(128,8)==pick[:,None])
    pre,signal,electronic=infer(model,a['source_waveforms_V'],a['z'],a['white'],a['style'],cfg['generation_chunk']);detail=model.details
    post,ad=adc(pre);counter['repeat_noise_rows']+=len(post);f=ev.features(post)
    from flow_core import bins
    sb=bins(source[pick],edges);p=f['features']['PSD'].reshape(128,8);q=f['features']['Qtotal_mVns'].reshape(128,8);valid=f['valid'].reshape(128,8)
    pieces=np.stack([signal,electronic,detail['flow_increment_V']],0).reshape(3,128,8,480)
    centered=pieces-pieces.mean(2,keepdims=True)
    covariance=np.einsum('ignt,jgnt->gijt',centered,centered)/7
    total_var=np.var(pre.reshape(128,8,480).astype('f8'),axis=1,ddof=1)
    closure=float(np.max(np.abs(covariance.sum((1,2))-total_var)))
    rows=[];by={};regions={'full480':(0,480),'baseline100':(0,100),'prompt100_150':(100,150),'postpeak150_220':(150,220),'tail220_400':(220,400)}
    for j,idx in enumerate(pick):
        de=valley['bins'][sb[j]] if 0<=sb[j]<8 else None
        pp=p[j,valid[j]];qq=q[j,valid[j]]
        rows.append({'source_parent_index':int(idx),'composition':str(comp[idx]),'valid_draws':int(valid[j].sum()),
            'output_PSD':ev.summary(pp),'output_Q_mVns':ev.summary(qq),
            'both_sides_of_frozen_valley':bool(de and de['valid'] and len(pp) and (pp<de['low']).any()and(pp>de['high']).any())})
    for category in np.unique(comp[pick]):
        use=comp[pick]==category;selected=[r for r in rows if r['composition']==category]
        cov_summary={}
        for region,(left,right)in regions.items():
            cov_summary[region]={}
            for i,n1 in enumerate(['base_signal','base_electronic','flow_increment']):
                for j,n2 in enumerate(['base_signal','base_electronic','flow_increment']):
                    cov_summary[region][n1+'__'+n2]=ev.summary(covariance[use,i,j,left:right].mean(1))
        by[str(category)]={'parents':int(use.sum()),
            'within_parent_PSD_SD':ev.summary(np.array([r['output_PSD']['sample_sd']for r in selected if r['output_PSD']['sample_sd'] is not None])),
            'within_parent_Q_SD_mVns':ev.summary(np.array([r['output_Q_mVns']['sample_sd']for r in selected if r['output_Q_mVns']['sample_sd'] is not None])),
            'parents_straddling_valley':sum(r['both_sides_of_frozen_valley']for r in selected),
            'within_parent_waveform_RMS_V':{region:ev.summary(np.sqrt(total_var[use,left:right].mean(1)))for region,(left,right)in regions.items()},
            'three_additive_algebraic_components_covariance_V2':cov_summary}
    summary={'parent_count':128,'draws_each':8,'rows':rows,'by_composition':by,'variance_denominator':7,
        'base_signal_base_electronic_plus_increment_closure_maxabs_V2':closure,
        'physical_final_signal_electronic_split':'unknown','cross_terms_retained':True,'ADC':ad,
        'repeats_not_independent_source_events':True}
    np.savez_compressed(out/(name+'_joint8_REPEAT_PRIVATE.npz'),source_parent_index=parent,waveforms_V=post,
        **detail,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'],valid=f['valid'],covariance_V2=covariance,total_variance_preADC_V2=total_var)
    return {'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0}

def source_tables(domains,actual_source_bins,reference_source_bins,comp,valley,edges):
    result={}
    from flow_core import signed_q
    for name,d in domains.items():
        if name=='REAL':continue
        result[name]={}
        ob=np.searchsorted(edges,np.log(np.maximum(d['features']['Qtotal_mVns'],1e-30)),side='right')-1
        for category in ['pure_EE','pure_had','mixed']:
            result[name][category]={}
            for mode,b in [('actual_NN_source_Q_preADC',actual_source_bins),('historical_SOURCE_CLEAN_Q_postADC',reference_source_bins),('observed_output_Q',ob)]:
                tab=[]
                for i in range(8):
                    population=(comp==category)&(b==i);use=population&d['valid'];p=d['features']['PSD'][use];de=valley['bins'][i]
                    tab.append({'bin_id_1based':i+1,'source_count':int(population.sum()),'N':int(use.sum()),
                        'invalid_count':int((population&~d['valid']).sum()),'output_PSD':ev.summary(p),
                        'valley_fraction':float(np.mean((p>=de['low'])&(p<=de['high'])))if len(p)and de['valid']else None})
                result[name][category][mode]=tab
    primary={n:{'bin'+str(i+1):result[n]['pure_had']['actual_NN_source_Q_preADC'][i]for i in [2,3]}for n in result}
    historical={n:{'bin'+str(i+1):result[n]['pure_had']['historical_SOURCE_CLEAN_Q_postADC'][i]for i in [2,3]}for n in result}
    return {'all_bins_and_compositions':result,'primary_pure_had_actual_input_source_Q_bins3_4':primary,
            'historical_postADC_reference_Q_bins3_4':historical,'valley_not_particle_error_truth':True,'main_population_reweighted':False}
