from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent
t=(ROOT/'evaluate_endpoints.py').read_text('utf8')
def change(a,b):
    global t
    assert a in t,a;t=t.replace(a,b)
change("COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'response_updates':0}","PRIOR_FAILURE=json.loads((ROOT/'EVALUATION_FAILURE_210037_216109.json').read_text('utf8'));COUNTER=dict(PRIOR_FAILURE['counters'])")
change("'SOURCE_CLEAN':old['C2ST']['results']['SOURCE_CLEAN']","'SOURCE_CLEAN':old['C2ST']['reference_results_reused']['SOURCE_CLEAN']")
start=t.index('    generation={};repeats=')
end=t.index('    ns,nd=ev.characterize',start)
block='''    generation={};repeats={'STANDARDIZED':old['random_decomposition']['STANDARDIZED']};probe={'STANDARDIZED':old['source_probes']['STANDARDIZED']};states={'STANDARDIZED':torch.load(pathlib.Path(CFG['R23_root'])/'STANDARDIZED/TRANSLATOR.pt',map_location='cpu',weights_only=True)};alias={};aliases_evidence={}
    for name in ARMS:
        log('reuse saved main/probes/joint '+name);saved=load(ROOT/name/'generated.npz');pre=saved['pre_ADC_V'];post=saved['waveforms_V'];signal=saved['signal_component_V'];electronic=saved['electronic_component_V'];assert np.array_equal(saved['source_ids'],ids)
        states[name]=torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True);domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        previous=['STANDARDIZED']+[x for x in ARMS if x!=name and x in domains]
        for p in previous:
            if equal_states(states[name],states[p])and np.array_equal(post,domains[p]['V']):alias[name]=alias.get(p,p);aliases_evidence[name]={'canonical_endpoint':alias[name],'state_all_tensors_equal':True,'actual2000_postADC_bitwise_equal':True,'rows_generated_before_alias':2000};break
        step=(1.-(-1.))/2**14;code=np.floor((pre.astype('f8')+.5)/step);ad={'clip_low_samples':int((code<0).sum()),'clip_high_samples':int((code>=2**14).sum()),'from_saved_pre_ADC_no_new_ADC_generation':True}
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'style_reader_used':False,'signal_not_conditional_expectation':True,'weight_file':CFG['endpoint_weight_paths'][name],'alias':alias.get(name),'post_ADC_bitwise_vs_frozen_D1_selected':bool(np.array_equal(post,ow['STANDARDIZED'])),'maxabs_vs_frozen_D1_selected_V':float(np.max(np.abs(post.astype('f8')-ow['STANDARDIZED'].astype('f8')))),'generation_from_attempt1_saved_payload':True}
        if name in alias:probe[name]=probe[alias[name]];repeats[name]=repeats[alias[name]]
        else:probe[name]=saved_probes(name,post);repeats[name]=saved_joint(name,source,comp)
'''
t=t[:start]+block+t[end:]
change("width=widths(domains,old,alias)","width=unavailable_widths(old,alias)")
change("'old_control_retrained':False,'sole_training_intervention'","'old_control_retrained':False,'engineering_failure_preserved':PRIOR_FAILURE,'continuation_new_waveform_generation':0,'density_results_delivery_missing':True,'sole_training_intervention'")
defs='''
def saved_probes(name,main):
    a=load(ROOT/'PROBE_INPUTS_CORRECTED.npz');p=load(OUT/(name+'_SOURCE_PROBES_PRIVATE.npz'));post=p['waveforms_V'];f=ev.features(post);sf=ev.features(a['source_waveforms_V']);tags=a['probe_kind'];baseline=main[a['reference_parent_index']];bf=ev.features(baseline);rows={}
    assert np.array_equal(p['source_parent_index'],a['source_parent_index'])and np.array_equal(p['noise_origin_parent_index'],a['noise_origin_parent_index'])and np.array_equal(p['reference_parent_index'],a['reference_parent_index'])
    for tag in np.unique(tags):
        use=tags==tag;rows[str(tag)]={'N':int(use.sum()),'QC':int(f['valid'][use].sum()),'waveform_RMS_vs_correct_reference_V':ev.summary(np.sqrt(np.mean((post[use]-baseline[use])**2,1))),'PSD_vs_reference':ev.summary(f['features']['PSD'][use]-bf['features']['PSD'][use]),'source_PSD':ev.summary(sf['features']['PSD'][use]),'output_PSD':ev.summary(f['features']['PSD'][use]),'output_Q_mVns':ev.summary(f['features']['Qtotal_mVns'][use])}
    same=tags=='same_input_same_noise';fixed=tags=='fixed_source_changed_noise'
    return{'rows':rows,'new_generated_rows':len(post),'same_input_same_noise_bitwise':bool(np.array_equal(post[same],main[a['source_parent_index'][same]])),'fixed_source_parent_index':int(a['source_parent_index'][fixed][0]),'fixed_source_unique_actual_parents':int(len(np.unique(a['source_parent_index'][fixed]))),'corrected_R21_parent_semantics_reused':True,'nonzero_sensitivity_not_correct_physics':True,'summary_from_saved_attempt1_only':True}
def saved_joint(name,source,comp):
    p=load(OUT/(name+'_joint8_REPEAT_PRIVATE.npz'));parent=p['source_parent_index'];assert len(parent)==1024 and np.all(parent.reshape(128,8)==parent.reshape(128,8)[:,:1])
    summary,f=summarize(p['waveforms_V'],p['signal_component_V'],p['electronic_component_V'],parent,8,comp,EDGES,source,VALLEY)
    return{'groups':{'joint8':summary},'actual_rows':1024,'source_parent_union':128,'new_independent_source_parents':0,'actual_R21_joint8_input_reused':True,'style_only_and_electronic_only_new_outputs':0,'summary_from_saved_attempt1_only':True}
def unavailable_widths(old,alias):
    tables={'REAL':old['energy_conditional_width']['tables']['REAL'],'SOURCE_CLEAN':old['energy_conditional_width']['tables']['SOURCE_CLEAN'],'STANDARDIZED':old['energy_conditional_width']['tables']['STANDARDIZED']}
    for n in ARMS:
        if n in alias:tables[n]=tables[alias[n]]
        else:tables[n]={'slices':[{'bin_id_1based':i+1,'valid':False,'components':[],'fit_attempts':None,'reason':'engineering_delivery_unavailable: completed fit summary lost before persistence; not a scientific fit failure'}for i in range(8)],'scientific_fit_success_or_failure':'unknown because unsaved summary','no_repeat_fit':True}
    return{'tables':tables,'common_support_errors':{},'edges_logQ':EDGES,'old_reference_fits_reused':True,'aliases':alias,'density_components_not_particle_truth':True,'delivery_unavailable':True,'completed_EM_attempts_preserved':PRIOR_FAILURE['counters']['density_EM_attempts'],'no_new_EM_in_continuation':True}
'''
loc=t.index('\ndef run():');t=t[:loc]+defs+t[loc:]
ast.parse(t);(ROOT/'evaluate_continue.py').write_text(t,'utf8');print('continuation prepared: saved outputs only; no generation or EM repeated')
