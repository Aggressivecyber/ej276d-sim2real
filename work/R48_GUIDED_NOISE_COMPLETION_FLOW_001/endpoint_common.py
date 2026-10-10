def psd_metrics(domains):
    tables={};values={};private={}
    for name,d in domains.items():
        p=d['features']['PSD'];q=d['features']['Qtotal_mVns'];valid=d['valid'];b=qb(q);rows=[];values[name]=[]
        b[np.log(np.maximum(q,1e-30))==EDGES[-1]]=7
        for i in range(8):
            x=np.sort(p[valid&(b==i)]);values[name].append(x);de=VALLEY['bins'][i]
            rows.append({'bin_id_1based':i+1,'N':len(x),'fraction_of_QC':len(x)/int(valid.sum())if valid.any()else None,'valley_fraction':float(np.mean((x>=de['low'])&(x<=de['high'])))if len(x)and de['valid']else None,'valley_low':de['low'],'valley_high':de['high']})
            private[name+'__bin'+str(i+1)+'__PSD']=x
        tables[name]={'source_n':len(p),'QC_n':int(valid.sum()),'outside_8bins':int(np.sum(valid&((b<0)|(b>7)))),'bins':rows}
    comparisons={}
    for name in domains:
        if name=='REAL':continue
        rows=[]
        for i in range(8):
            a=values[name][i];r=values['REAL'][i];ok=len(a)>0 and len(r)>0
            ca=tables[name]['bins'][i]['valley_fraction'];ra=tables['REAL']['bins'][i]['valley_fraction']
            rows.append({'bin_id_1based':i+1,'candidate_N':len(a),'REAL_N':len(r),'defined':ok,'W1':float(wasserstein_distance(a,r))if ok else None,'KS':float(ks_2samp(a,r).statistic)if ok else None,'valley_fraction_difference':ca-ra if ca is not None and ra is not None else None})
        comparisons[name]={'bins':rows,'all_8_supported':all(r['defined']for r in rows),'mean8_W1':float(np.mean([r['W1']for r in rows]))if all(r['defined']for r in rows)else None}
    np.savez_compressed(OUT/'PSD_SECTIONS_PRIVATE.npz',**private)
    return{'tables':tables,'vs_REAL':comparisons,'edges_logQ':EDGES,'population_weights_changed':False,'old_frozen_valley_reused':True}
def source_composition(domains,comp,hdev,lambdas):
    source=domains['SOURCE_CLEAN'];sb=qb(source['features']['Qtotal_mVns']);rows={};private={};weights=np.ones(2000);within=(sb>=0)&(sb<8);weights[within]=np.exp(lambdas[sb[within]]*hdev[within]);auxiliary={}
    for name,d in domains.items():
        if name=='REAL':continue
        ob=qb(d['features']['Qtotal_mVns']);rows[name]={}
        for category in ['pure_EE','pure_had','mixed']:
            table={}
            for mode,b in [('source_condition',sb),('output_condition',ob)]:
                tab=[]
                for i in range(8):
                    use=d['valid']&(comp==category)&(b==i);p=d['features']['PSD'][use];de=VALLEY['bins'][i]
                    population=(comp==category)&(b==i)
                    tab.append({'bin_id_1based':i+1,'source_count':int(population.sum()),'N':int(use.sum()),'invalid_count':int((population&~d['valid']).sum()),'output_PSD':ev.summary(p),'valley_fraction':float(np.mean((p>=de['low'])&(p<=de['high'])))if len(p)and de['valid']else None});private[name+'__'+category+'__'+mode+'__bin'+str(i+1)]=p
                table[mode]=tab
            rows[name][category]=table
        aux=[]
        for i in range(8):
            use=d['valid']&(sb==i);real=domains['REAL'];ruse=real['valid']&(qb(real['features']['Qtotal_mVns'])==i);p=d['features']['PSD'][use];r=real['features']['PSD'][ruse];w=weights[use];de=VALLEY['bins'][i]
            aux.append({'source_bin_id_1based':i+1,'candidate_N':len(p),'REAL_observed_bin_N':len(r),'Kish_ESS':float(w.sum()**2/np.dot(w,w))if len(w)else None,'weighted_PSD_mean':float(np.average(p,weights=w))if len(p)else None,'weighted_valley_fraction':float(np.average((p>=de['low'])&(p<=de['high']),weights=w))if len(p)and de['valid']else None,'weighted_W1_vs_REAL':float(wasserstein_distance(p,r,u_weights=w))if len(p)and len(r)else None})
        auxiliary[name]=aux
    primary={n:{'bin'+str(i+1):rows[n]['pure_had']['source_condition'][i]for i in[2,3]}for n in rows}
    for n in rows:assert [primary[n]['bin3']['source_count'],primary[n]['bin4']['source_count']]==[207,182],(n,'purehad source population changed')
    np.savez_compressed(OUT/'SOURCE_COMPOSITION_PRIVATE.npz',**private,source_Qbin=sb,auxiliary_dev_weight=weights)
    return{'all_bins_and_compositions':rows,'primary_pure_had_source_bins3_4':primary,'auxiliary_TRAIN_frozen_weight':auxiliary,'auxiliary_not_primary_population':True,'soft_h_to_high_PSD_is_assumption_not_truth':True}
def probes(model,main,mainstyle,name):
    a=load(ROOT/'PROBE_INPUTS_CORRECTED.npz');x=a['source_waveforms_V'];style=mainstyle[a['noise_origin_parent_index']];pre,signal,electronic=infer(model,x,a['z'],a['white'],style,CFG['generation_chunk']);post,ad=adc(pre);COUNTER['source_probe_rows']+=len(post)
    f=ev.features(post);sf=ev.features(x);tags=a['probe_kind'];baseline=main[a['reference_parent_index']];bf=ev.features(baseline);rows={}
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-baseline[use])**2,1))),'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-bf['features']['PSD'][use]),'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use])}
    payload={k:a[k]for k in ['source_parent_index','source_parent_ids','noise_origin_parent_index','reference_parent_index','source_waveform_modified_counterfactual','probe_kind']}
    np.savez_compressed(OUT/(name+'_SOURCE_PROBES_PRIVATE.npz'),**payload,style=style,style_origin_main_parent_index=a['noise_origin_parent_index'],signal_component_V=signal,electronic_component_V=electronic,waveforms_V=post,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'])
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return{'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),'corrected_R21_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True}
def joint_repeats(model,source,comp,name):
    a=load(ROOT/'JOINT8_INPUTS_PRIVATE.npz');parent=a['source_parent_index'];assert len(parent)==1024 and np.all(parent.reshape(128,8)==parent.reshape(128,8)[:,:1])
    pre,signal,electronic=infer(model,a['source_waveforms_V'],a['z'],a['white'],a['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['repeat_noise_rows']+=len(post)
    summary,f=summarize(post,signal,electronic,parent,8,comp,EDGES,source,VALLEY);summary['ADC']=ad
    np.savez_compressed(OUT/(name+'_joint8_REPEAT_PRIVATE.npz'),source_parent_index=parent,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,PSD=f['features']['PSD'],Q_mVns=f['features']['Qtotal_mVns'],valid=f['valid'],parent_composition=comp[parent.reshape(128,8)[:,0]])
    return{'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0,'actual_R21_joint8_input_reused':True,'style_only_and_electronic_only_new_outputs':0}
def online_auc(name):
    a=load(ROOT/name/'ONLINE_D_SCORES_PRIVATE.npz');trace=json.loads((ROOT/name/'TRAINING_TRACE.json').read_text('utf8'));result=[]
    for r in trace:
        step=r['validation']['step'];gp=a['step'+str(step)+'_gen'];rp=a['step'+str(step)+'_REAL'];y=np.r_[np.zeros(len(gp)),np.ones(len(rp))];score=np.r_[gp,rp];w=np.r_[a['source_weight'],a['REAL_weight']];use=w>0
        result.append({'step':step,'REAL_label_is1':True,'weighted_AUC':float(roc_auc_score(y[use],score[use],sample_weight=w[use])),'unweighted_AUC':float(roc_auc_score(y,score)),'scores_used_in_selection':False})
    return result
