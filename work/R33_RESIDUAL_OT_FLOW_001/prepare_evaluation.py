from pathlib import Path
import json,ast,os,datetime,hashlib
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001'
src=(OLD/'evaluate_endpoints.py').read_text('utf8')
src=src.replace('from model import Translator,infer,adc','from model import adc\nfrom flow_runtime import Pipeline,infer\nfrom flow_core import correct\nimport flow_diagnostics as fd')
src=src.replace("CFG['endpoints'];CONTROLS=[]","CFG['endpoints'];CONTROLS=['BASE']")
start=src.index('def c2st(');end=src.index('def widths(',start)
src=src[:start]+'''def c2st(domains,old,alias):
    oldpred=load(pathlib.Path(CFG['R26_root'])/'results/C2ST_PRIVATE.npz',keys=[f'phase{p}_{j}'for p in [0,1]for j in range(3)])
    selected={p:tuple(oldpred[f'phase{p}_{j}']for j in range(3))for p in [0,1]}
    gs=(selected[0][0],selected[1][0]);rs=(selected[0][1],selected[1][1]);results={'BASE':old['C2ST']['results']['CONTROL_FINAL']};private={}
    saved_matrix=dc.matrix
    def matrix(d,view):
        key={'normalized_u480':'u','seen_reconstruction480':'seen','complement_reconstruction480':'complement'}.get(view)
        return d[key].copy()if key else saved_matrix(d,view)
    dc.matrix=matrix;views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
    for name in ARMS:
        results[name]={};d=domains[name];r=domains['REAL']
        assert not set(d['groups'][gs[0]])&set(d['groups'][gs[1]]) and not set(r['groups'][rs[0]])&set(r['groups'][rs[1]])
        for view in views:
            slot=name+'__'+view
            def persist_fit(row,p):
                np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);progress()
            dc.PERSIST_HGB_CALLBACK=persist_fit
            row,p=dc.fit_one(d,r,gs,rs,view,CFG,COUNTER)
            row['original_fixed_rows_retained']=True
            row['candidate_train_QC_n']=int(d['valid'][gs[0]].sum());row['candidate_holdout_QC_n']=int(d['valid'][gs[1]].sum())
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**p);write('results/C_'+slot+'.json',row);results[name][view]=row
            private.update({slot+'__'+k:p[k]for k in ['probability','label','cluster']})
            write('C2ST_RUNNING.json',{'results':results,'counters':COUNTER});log('C2ST '+slot+' BA '+str(row['balanced_accuracy']))
    paired={}
    for label,candidate,reference in [('OT_minus_INDEPENDENT','OT_CFM','INDEPENDENT_CFM'),('OT_minus_BASE','OT_CFM','BASE'),('INDEPENDENT_minus_BASE','INDEPENDENT_CFM','BASE')]:
        paired[label]={}
        for view in views:
            a=load(OUT/(candidate+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            b=load(pathlib.Path(CFG['R32_root'])/'results'/('CONTROL_FINAL__'+view+'_PREDICTIONS_PRIVATE.npz')) if reference=='BASE' else load(OUT/(reference+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
            paired[label][view]=aa.paired_bootstrap(a,b,CFG)
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{f'phase{p}_{j}':v for p,values in selected.items()for j,v in enumerate(values)})
    return {'results':results,'paired':paired,'original_actual_selection_no_resplit':True,'train_each_domain':len(gs[0]),
        'holdout_each_domain':len(gs[1]),'baseline_predictions_reused_no_fit':True,
        'REAL_REAL_reference':old['C2ST']['REAL_REAL_reference'],'fixed_classifier_CI_not_training_uncertainty':True}

''' +src[end:]
src=src.replace("tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN',*CONTROLS]};grid=", "tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN']};tables['BASE']=old['energy_conditional_width']['tables']['CONTROL_FINAL'];grid=")
src=src.replace("[('all_endpoints',[*CONTROLS,*ARMS]),('selected',['CONTROL_SELECTED','MEAN_SELECTED']),('final_equal_updates',['CONTROL_FINAL','MEAN_FINAL'])]", "[('all_final_endpoints',[*CONTROLS,*ARMS]),('OT_vs_independent',ARMS)]")
start=src.index('def run():');end=src.index('\n\n\ntry:run()',start)
src=src[:start]+'''def run():
    old=json.loads((pathlib.Path(CFG['R32_root'])/'metrics.json').read_text('utf8'))
    ref=json.loads((ROOT/'CAL_REFERENCE_METADATA.json').read_text('utf8'));actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
    meta=load(ROOT/'DEV_SOURCE_METADATA.npz');ids=actual['source_ids'];source=actual['source_waveforms_V'];comp=meta['composition'];pdg=meta['pdg_inferred']
    real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz');base_saved=load(pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz')
    assert np.array_equal(base_saved['source_ids'],ids) and np.array_equal(ids,meta['source_ids'])
    reference_source,_=adc(source)
    domains={'REAL':aa.data(real['real_V'],real['real_ids'],real['real_groups'],'REAL',ref),
        'SOURCE_CLEAN':aa.data(reference_source,ids,np.arange(2000),'G4_SOURCE',ref),
        'SOURCE_INPUT':aa.data(source,ids,np.arange(2000),'G4_SOURCE',ref),
        'BASE':aa.data(base_saved['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)}
    domains['REAL']['valid']&=real['real_QC_valid'];rr=aa.data(real['reference_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
    actual_source_bins=qb(domains['SOURCE_INPUT']['features']['Qtotal_mVns']);reference_source_bins=qb(domains['SOURCE_CLEAN']['features']['Qtotal_mVns'])
    assert [(comp=='pure_had')[actual_source_bins==b].sum()for b in [2,3]]==[213,181]
    assert [(comp=='pure_had')[reference_source_bins==b].sum()for b in [2,3]]==[207,182]
    generation={};repeats={};probe={};solver={}
    for name in ARMS:
        log('CPU complete source->base->flow '+name);(ROOT/name).mkdir(exist_ok=True);pipeline=Pipeline(CFG,ROOT,name)
        pre,signal,electronic=infer(pipeline,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);detail=pipeline.details
        post,ad=adc(pre);COUNTER['main_generation_rows']+=2000
        np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,source_ids=ids,**detail)
        base_delta=float(np.max(np.abs(detail['base_pre_ADC_V']-base_saved['pre_ADC_V'])))
        domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        generation[name]={'rows':2000,'all_original_source_ids_preserved':True,'ADC':ad,
            'flow_applied_N':int(detail['flow_applied'].sum()),'unsupported_source_Q_base_fallback_N':int((~detail['flow_applied']).sum()),
            'base_preADC_bitwise_matches_saved_R32':bool(np.array_equal(detail['base_pre_ADC_V'],base_saved['pre_ADC_V'])),
            'base_preADC_vs_saved_maxabs_V':base_delta,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),
            'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'weight_file':CFG['endpoint_weight_paths'][name],
            'fixed_base_weight_file':CFG['base_G'],'flow_steps':32,'final_ADC_once':True,'physical_final_signal_electronic_split':'unknown'}
        progress();write('GENERATION_PROGRESS.json',{'generation':generation,'counters':COUNTER})
        refined,applied,b=correct(pipeline.flow,detail['base_pre_ADC_V'][:32],source[:32],pipeline.mu,pipeline.scale,pipeline.edges,64,CFG['generation_chunk'])
        error=refined-pre[:32];f32=ev.features(pre[:32]);f64=ev.features(refined)
        solver[name]={'rows':32,'parents':'first32 fixed original DEV order','publication_steps':32,'check_steps':64,
            'maxabs_preADC_difference_V':float(np.max(np.abs(error))),'RMS_preADC_difference_V':float(np.sqrt(np.mean(error.astype('f8')**2))),
            'Q_mVns_difference':ev.summary(f64['features']['Qtotal_mVns']-f32['features']['Qtotal_mVns']),
            'PSD_difference':ev.summary(f64['features']['PSD']-f32['features']['PSD']),
            'both_finite':bool(np.isfinite(pre[:32]).all()and np.isfinite(refined).all()),'strict_limit_V':1/16383,
            'pass':bool(np.isfinite(error).all()and float(np.max(np.abs(error)))<1/16383),'solver_was_not_tuned':True}
        np.savez_compressed(OUT/(name+'_SOLVER_CHECK_PRIVATE.npz'),published32_preADC_V=pre[:32],check64_preADC_V=refined,source_ids=ids[:32],flow_applied=applied)
        write('SOLVER_CHECK.json',solver)
        probe[name]=fd.probes(pipeline,post,actual['style'],name,ROOT,OUT,CFG,EDGES,COUNTER);write('SOURCE_PROBES_SUMMARY.json',probe);progress()
        repeats[name]=fd.joint(pipeline,source,comp,name,ROOT,OUT,CFG,EDGES,VALLEY,COUNTER);write('JOINT8_SUMMARY.json',repeats);progress()
        del pipeline
    stats,distance=ev.characterize(domains,ref);psd=common.psd_metrics(domains);width=widths(domains,old,{})
    for d in [*domains.values(),rr]:attach_temporal(d)
    maps=load(pathlib.Path(CFG['R27_root'])/'results/PROJECTION_SPACE_PRIVATE.npz');seen=maps['Pseen'];complement=maps['Pcomplement']
    for d in [*domains.values(),rr]:d['seen']=d['u']@seen;d['complement']=d['u']@complement
    cs=c2st(domains,old,{})
    constants=load(CFG['local_training_input_paths']['FINE_CONSTANTS.npz']);tdomains={n:domains[n]for n in ['REAL','BASE',*ARMS]};tdomains['TRAIN_REFERENCE']=rr
    coverage=temporal.means_coverage(tdomains,seen,complement,actual_source_bins);projection=temporal.fine_distances(tdomains,constants)
    old_train_peak_mu=load(pathlib.Path(CFG['R32_root'])/'MEAN_TARGETS_PRIVATE.npz',keys=['mu'])['mu']
    md,mp=describe_means(tdomains,['BASE',*ARMS],actual_source_bins,old_train_peak_mu)
    md['source_Q_definition']='actual unADC NN input';md['TRAIN_u_mu_reference']='historical R32 TRAIN-fit peak-normalized mu; not raw-voltage flow affine mean'
    np.savez_compressed(OUT/'MEAN_SD_DESCRIPTIONS_PRIVATE.npz',**mp);write('MEAN_SD_METRICS.json',md)
    raw={};raw_payload={}
    for n in ['REAL','BASE',*ARMS]:
        d=domains[n];raw[n]=[]
        for k in [None,*range(8)]:
            use=d['valid'] if k is None else d['valid']&(d['b']==k);target=domains['REAL']['valid'] if k is None else domains['REAL']['valid']&(domains['REAL']['b']==k)
            row={'bin_id_1based':None if k is None else k+1,'N':int(use.sum()),'REAL_N':int(target.sum()),'qualified':int(use.sum())>=40 and int(target.sum())>=40}
            if use.any() and target.any():
                mv=d['V'][use].mean(0);rv=domains['REAL']['V'][target].mean(0);row['mean_full480_MSE_V2']=float(np.mean((mv-rv)**2))
                raw_payload[n+'__'+str(k)+'__mean_V']=mv
                if use.sum()>1 and target.sum()>1:
                    sv=d['V'][use].std(0,ddof=1);rsd=domains['REAL']['V'][target].std(0,ddof=1);row['SD_full480_RMS_difference_V']=float(np.sqrt(np.mean((sv-rsd)**2)));raw_payload[n+'__'+str(k)+'__SD_V']=sv
            raw[n].append(row)
    np.savez_compressed(OUT/'RAW_VOLTAGE_MOMENTS_PRIVATE.npz',**raw_payload)
    composition=fd.source_tables(domains,actual_source_bins,reference_source_bins,comp,VALLEY,EDGES)
    association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_INPUT'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in ['BASE',*ARMS]}
    historical_association={n:ds.association({'SOURCE_CLEAN':domains['SOURCE_CLEAN'],'TRANSLATED':domains[n]},comp,pdg,EDGES)for n in ['BASE',*ARMS]}
    local={n:dc.native_statistics(d,ref)for n,d in domains.items()};feat={};waves={}
    for name,d in domains.items():
        feat.update({name+'__'+key:a for key,a in {**d['features'],**d['extra']}.items()});feat[name+'__valid']=d['valid'];feat[name+'__region']=d['region'];waves[name]=d['V'].astype('f4')
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feat);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)
    training={n:json.loads((ROOT/n/'TRAINING_RECEIPT.json').read_text('utf8'))for n in CFG['arms']}
    metric={'status':'complete','training':training,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text('utf8')),
        'domains':stats,'domain_vs_REAL':distance,'local':local,'PSD_main':psd,'energy_conditional_width':width,
        'source_composition':composition,'source_association_actual_preADC_input':association,'source_association_historical_postADC_reference':historical_association,
        'source_probes':probe,'random_decomposition':repeats,'C2ST':cs,'generation':generation,'solver_check':solver,
        'same_training_input_flow':json.loads((ROOT/'PAIRED_FLOW_RECEIPT.json').read_text('utf8')),
        'projection_coverage':coverage,'projection_distances':projection,'conditional_mean_SD':md,'raw_voltage_moments':raw,'counters':COUNTER,
        'versions':{'Python':platform.python_version(),'torch':torch.__version__,'NumPy':np.__version__,'sklearn':sklearn.__version__},
        'primary_DEV_population_reweighted':False,'EVAL_read':0,'CPU_main_and_replay_same_environment':True,
        'two_current_server_flow_fits_as_registered':True,'physical_final_signal_electronic_split':'unknown','endpoint_selection_uses_DEV':False}
    write('metrics.json',metric)
    assert COUNTER['HGB_attempts']<=20 and COUNTER['density_EM_attempts']<=16 and COUNTER['main_generation_rows']<=4000 and COUNTER['repeat_noise_rows']<=2048 and COUNTER['source_probe_rows']<=1664
    write('EVALUATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counters':COUNTER,'solver_check_rows':64,'status':'complete','threads':4})
    log('R33 evaluation complete '+json.dumps(COUNTER))
''' +src[end:]
src=src.replace("log('R32", "log('R33")
ast.parse(src);(ROOT/'evaluate_endpoints.py').write_text(src,'utf8')
cfg=json.loads((ROOT/'config.json').read_text('utf8'))
# All changes above are evaluator integration; executed remote config/recipe remains frozen.
rec={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response_fitting_or_generation_here':0,
    'evaluation_recipe_unchanged':'fixed20HGB/16EM, same indices/groups, actual source preADC primary; base models/predictions reused',
    'source_AST_passed':True,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'base_no_new_HGB_EM':True}
(ROOT/'EVALUATION_SOURCE_FREEZE.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps(rec))
