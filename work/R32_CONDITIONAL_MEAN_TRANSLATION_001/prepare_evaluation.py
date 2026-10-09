from pathlib import Path
import ast, json
ROOT=Path(__file__).resolve().parent;R28=ROOT.parent/'R28_FRESH_PROJECTION_TRANSLATION_001'
text=(R28/'evaluate_endpoints.py').read_text('utf8')
text=text.replace("CONTROLS=['HOMO_SELECTED','HOMO_FINAL']",'CONTROLS=[]')
text=text.replace('from random_diagnostics import summarize','from random_diagnostics import summarize\nfrom mean_description import describe as describe_means')
def replace_function(source,name,new):
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    lines=source.splitlines(keepends=True);return ''.join(lines[:node.lineno-1])+new+'\n\n'+''.join(lines[node.end_lineno:])
audit='''def consumed_audit(training):
    old=load(CFG['consumed_R26_stream']);oldreceipt=json.loads(pathlib.Path(CFG['consumed_R26_receipt']).read_text('utf8'));rows={}
    for arm,t in training.items():
        new=load(ROOT/arm/'TRAINING_STREAMS_PRIVATE.npz');n=t['updates']*32
        checks={k:bool(np.array_equal(new[k],old[k][:t['updates']*3 if k=='NCE_positions' else n])) for k in new}
        if n==94400:checks.update(noise_stream=t['stream_hashes']['noise_stream']==oldreceipt['stream_hashes']['noise_stream'],style_stream=t['stream_hashes']['style_stream']==oldreceipt['stream_hashes']['style_stream'])
        fresh=json.loads((ROOT/arm/'FRESH_PROJECTION_STREAM.json').read_text('utf8'))
        assert all(checks.values()) and fresh['calls']==t['updates'] and fresh['actual_matrix_count']==t['updates']*3
        rows[arm]={'actual_rows':n,'actual_G_D_updates':t['updates'],'equal2950':t['updates']==2950,'exact_original_flow':checks,
            'fresh_calls':fresh['calls'],'same_directions_two_domains_each_G':all(v['same_draw_for_GENERATED_and_REAL'] for v in fresh['per_call_actual_float32_versions']),
            'fresh_validation_D_identity_evaluation_calls':0,'NCE_same_per_waveform_temporal_patch_positions_and_row_assignment':True}
    keys=['source','target','bin','original_row_ordinal','NCE_positions','noise_stream','style_stream']
    a,b=[training[n]for n in CFG['arms']];pair={k:a['stream_hashes'][k]==b['stream_hashes'][k]for k in keys}
    output={'arms':rows,'same_actual_steps':a['updates']==b['updates'],'same_stream_hashes':pair,'both_2950':a['updates']==b['updates']==2950,
        'same_platform':True,'migration_consistency_test':False}
    write('CONSUMED_EXPOSURE_AUDIT.json',output);return output
'''
text=replace_function(text,'consumed_audit',audit)
c2='''def c2st(domains,old,alias):
    oldpred=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in [0,1]for j in range(3)])
    selected={p:tuple(oldpred[f'phase{p}_{j}']for j in range(3))for p in [0,1]}
    gs=(selected[0][0],selected[1][0]);rs=(selected[0][1],selected[1][1]);results={};private={}
    saved_matrix=dc.matrix
    def matrix(d,view):
        key={'normalized_u480':'u','seen_reconstruction480':'seen','complement_reconstruction480':'complement'}.get(view)
        return d[key].copy()if key else saved_matrix(d,view)
    dc.matrix=matrix;views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
    for name in ARMS:
        results[name]={};d=domains[name];r=domains['REAL']
        assert not set(d['groups'][gs[0]])&set(d['groups'][gs[1]]) and not set(r['groups'][rs[0]])&set(r['groups'][rs[1]])
        assert d['valid'][np.r_[gs[0],gs[1]]].all()and r['valid'][np.r_[rs[0],rs[1]]].all()
        for view in views:
            slot=name+'__'+view
            if name in alias and alias[name]in results and view in results[alias[name]]:
                canonical=alias[name];row=dict(results[canonical][view]);row['same_arm_endpoint_identity_alias']=canonical
                p=load(OUT/(canonical+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            else:
                def persist_fit(row,p):
                    np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);progress()
                dc.PERSIST_HGB_CALLBACK=persist_fit
                row,p=dc.fit_one(d,r,gs,rs,view,CFG,COUNTER)
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);results[name][view]=row
            private.update({slot+'__'+k:p[k]for k in ['probability','label','cluster']})
            write('C2ST_RUNNING.json',{'results':results,'counters':COUNTER});log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    paired={}
    for label,mean,control in [('final_equal_updates','MEAN_FINAL','CONTROL_FINAL'),('selected','MEAN_SELECTED','CONTROL_SELECTED')]:
        paired[label]={}
        for view in views:
            a=load(OUT/(mean+'__'+view+'_PREDICTIONS_PRIVATE.npz'));b=load(OUT/(control+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            paired[label][view]=aa.paired_bootstrap(a,b,CFG)
    historical=json.loads((pathlib.Path(CFG['R28_root'])/'metrics.json').read_text('utf8'))['C2ST']['results']
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':v for p,values in selected.items()for j,v in enumerate(values)})
    return {'results':results,'paired_MEAN_minus_CONTROL':paired,'original_actual_selection_no_resplit':True,'train_each_domain':len(gs[0]),
        'holdout_each_domain':len(gs[1]),'seven_historical_and_three_pure480_views':True,'fixed_classifier_CI_not_training_uncertainty':True,
        'REAL_REAL_reference':old['C2ST']['RR_definition'],'R28_historical_C2ST_results':historical,'cross_platform_historical_not_new_loss_effect':True}
'''
text=replace_function(text,'c2st',c2)
text=text.replace("COUNTER['density_EM_attempts']+attempt<=16", "COUNTER['density_EM_attempts']+attempt<=CFG['density_EM_max']")
text=text.replace("('selected',['HOMO_SELECTED','FRESH_SELECTED']),('final_equal_updates',['HOMO_FINAL','FRESH_FINAL'])", "('selected',['CONTROL_SELECTED','MEAN_SELECTED']),('final_equal_updates',['CONTROL_FINAL','MEAN_FINAL'])")
run='''def run():
    old=json.loads((pathlib.Path(CFG['R26_root'])/'metrics.json').read_text('utf8'))
    ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
    meta=load(ROOT/'DEV_SOURCE_METADATA.npz');ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred']
    real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz')
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),'SOURCE_CLEAN':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref)}
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    generation={};repeats={};probe={};states={};alias={};evidence={}
    for name in ARMS:
        log('CPU main '+name);(ROOT/name).mkdir(exist_ok=True);model=Translator()
        states[name]=torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True)
        model.load_state_dict(states[name]);model.eval()
        previous=[p for p in ARMS if p in domains and p.split('_')[0]==name.split('_')[0]]
        canonical=next((p for p in previous if equal_states(states[name],states[p])),None)
        if canonical:
            original=load(ROOT/canonical/'generated.npz');pre=original['pre_ADC_V'];post=original['waveforms_V'];signal=original['signal_component_V'];electronic=original['electronic_component_V'];ad=generation[canonical]['ADC']
            assert np.array_equal(original['source_ids'],ids)
            alias[name]=canonical;evidence[name]={'canonical':canonical,'same_arm':True,'state_all_tensors_equal':True,'same_single_actual_input_container':True,'source_ID_order_equal':True,'no_duplicate_forward':True}
        else:
            pre,signal,electronic=infer(model,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
        np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,signal_component_V=signal,electronic_component_V=electronic,source_ids=ids)
        domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        generation[name]={'rows':2000,'all_original_source_ids_preserved':bool(np.array_equal(ids,meta['source_ids'])),
            'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
            'style_reader_used':False,'signal_not_conditional_expectation':True,'weight_file':CFG['endpoint_weight_paths'][name],'alias':alias.get(name),'CPU_generated_here':canonical is None}
        if canonical:probe[name]=probe[canonical];repeats[name]=repeats[canonical]
        else:probe[name]=common.probes(model,post,actual['style'],name);repeats[name]=common.joint_repeats(model,source,comp,name)
        write('GENERATION_PROGRESS.json',{'generation':generation,'aliases':alias,'counters':COUNTER});del model
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains);width=widths(domains,old,alias)
    for d in [*domains.values(),rr]:attach_temporal(d)
    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz');seen=maps['Pseen'];complement=maps['Pcomplement']
    for d in [*domains.values(),rr]:d['seen']=d['u']@seen;d['complement']=d['u']@complement
    cs=c2st(domains,old,alias)
    constants=load(CFG['local_training_input_paths']['FINE_CONSTANTS.npz']);tdomains={n:domains[n]for n in ['REAL',*ARMS]};tdomains['TRAIN_REFERENCE']=rr
    coverage=temporal.means_coverage(tdomains,seen,complement,domains['SOURCE_CLEAN']['b']);projection=temporal.fine_distances(tdomains,constants)
    target=load(ROOT/'MEAN_TARGETS_PRIVATE.npz',keys=['mu']);md,mp=describe_means(tdomains,ARMS,domains['SOURCE_CLEAN']['b'],target['mu'])
    np.savez_compressed(OUT/'MEAN_SD_DESCRIPTIONS_PRIVATE.npz',**mp);write('MEAN_SD_METRICS.json',md)
    frozen=json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text('utf8'));lambdas=np.asarray([r['lambda_value']for r in frozen['bins']]);h=.15*meta['edep_hadron_MeV']/(meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'])
    composition=common.source_composition(domains,comp,h,lambdas)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in ARMS}
    local={n:dc.native_statistics(d,ref)for n,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training={n:json.loads((ROOT/n/'TRAINING_RECEIPT.json').read_text('utf8'))for n in CFG['arms']};exposure=consumed_audit(training)
    online={n:common.online_auc(n)for n in CFG['arms']}
    metric={'status':'complete','training':training,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),
        'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,
        'source_composition':composition,'source_association':association,'source_probes':probe,'random_decomposition':repeats,
        'C2ST':cs,'generation':generation,'online_D_validation_AUC':online,'row_exposure':exposure,'endpoint_aliases':alias,
        'alias_evidence':evidence,'projection_coverage':coverage,'projection_distances':projection,'conditional_mean_SD':md,'counters':COUNTER,
        'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},
        'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,
        'two_current_server_arms_retrained_as_registered':True,'sole_training_intervention':'conditional mean U coefficient1',
        'endpoint_selection_uses_DEV':False,'mean_target_is_TRAIN_only':True}
    write('metrics.json',metric);write('ENDPOINT_ALIAS_RECEIPT.json',{'aliases':alias,'evidence':evidence,'counts':COUNTER})
    assert COUNTER['HGB_attempts']<=40 and COUNTER['density_EM_attempts']<=32 and COUNTER['main_generation_rows']<=8000 and COUNTER['repeat_noise_rows']<=4096 and COUNTER['source_probe_rows']<=3328
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete','threads':4})
    log('R32 evaluation complete '+json.dumps(COUNTER))
'''
text=replace_function(text,'run',run)
ast.parse(text);(ROOT/'evaluate_endpoints.py').write_text(text,'utf8')
replay=(R28/'replay_endpoints.py').read_text('utf8')
replay=replay.replace('actual=load(ROOT/\'ACTUAL_GENERATION_INPUTS.npz\');out={}',"actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');out={};cache={};states={};actual_replay_rows=0")
needle="    pre,signal,electronic=infer(model,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],128);post,ad=adc(pre);original=load(ROOT/name/'generated.npz');checks={}"
assert needle in replay
replay=replay.replace(needle,"    states[name]=model.state_dict();canonical=next((n for n in cache if n.split('_')[0]==name.split('_')[0] and all(torch.equal(states[name][k],states[n][k]) for k in states[name])),None)\n    if canonical:pre,signal,electronic,post,ad=cache[canonical]\n    else:\n        pre,signal,electronic=infer(model,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],128);post,ad=adc(pre);actual_replay_rows+=len(post);cache[name]=(pre,signal,electronic,post,ad)\n    original=load(ROOT/name/'generated.npz');checks={}")
replay=replay.replace("'rows':sum(x['rows']for x in out.values())", "'rows':actual_replay_rows,'represented_endpoint_rows':sum(x['rows']for x in out.values())")
ast.parse(replay);(ROOT/'replay_endpoints.py').write_text(replay,'utf8')
cfg=json.loads((ROOT/'config.json').read_text('utf8'));cfg['R28_root']=str(R28)
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
print('R32 four-endpoint original evaluation prepared;40HGB/32EM caps;per-slot persistence;selected/final within-arm alias only.')
