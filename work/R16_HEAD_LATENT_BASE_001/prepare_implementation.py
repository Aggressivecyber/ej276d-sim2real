from __future__ import annotations
import ast,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'R15_CONDITIONAL_RESIDUAL_FLOW_001'
old=(OLD/'numeric_worker.py').read_text()
header=old[:old.index('def c2st(')]
header=header.replace('R15','R16').replace('min(2360,','min(1760,')
header=header.replace('import residual_flow as rf','import residual_flow as rf\nimport head_base as hb')
start=header.index('COUNTER=');end=header.index('\ndef log(',start)
header=header[:start]+"COUNTER={'new_head_base_fit_attempts':0,'head_base_fitting_cumulative_s':0.,'new_network_fit_attempts':0,'network_updates':0,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'new_uniform_draws':0,'new_PCA_ridge_residual_fits':0,'EVAL_read':0,'new_G4':0,'raw_waveform_decode':0};CACHE={}\n"+header[end:]
tree=ast.parse(old)
nearest=next(ast.get_source_segment(old,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nearest')
new=r'''
def c2st(domains,names,rr,sel):
    sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4')) for k,v in sel['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=sp['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g] for n in names])];r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train' if phase==0 else 'holdout']={'N_each':n,'candidate_common_valid':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    out={};pred={};paired={};private={};support=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75
    if support:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        for n in names+['REAL_REAL']:
            d=rr if n=='REAL_REAL' else domains[n];selection=ts if n=='REAL_REAL' else gs;assert not set(d['groups'][selection[0]])&set(d['groups'][selection[1]]);out[n]={};pred[n]={}
            for view in CFG['C2ST_views']:
                out[n][view],pred[n][view]=dc.fit_one(d,domains['REAL'],selection,rs,view,CFG,COUNTER);private.update({n+'__'+view+'__'+k:v for k,v in pred[n][view].items()});log(f'C2ST {n} {view} BA={out[n][view]["balanced_accuracy"]:.6f}')
        for first,second in CFG['contrasts']:
            if first in pred and second in pred:paired[first+'_minus_'+second]={v:aa.paired_bootstrap(pred[first][v],pred[second][v],CFG) for v in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'original common pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_contrasts':paired,'all_candidates_common_QC':True,'same_N_for_all':True,'original_R7_pools_only':True,'all_32_current_environment_refit_if_all_arms_finite':True,'shared_actual_head_and_tail_white':True,'RR_is_sampling_reference_not_equivalence':True,'global_new_views':{'head34':'post-ADC fixed FACTOR standardized34 head','Q_PSD_2D':'post-ADC observed Qtotal_mVns and signedPSD; raw charge, no log'},'CI_conditional_on_classifiers':True}

def widths(domains,old15,edges,names):
    tables={'REAL':old15['energy_conditional_width']['tables']['REAL'],'STANDARD':old15['energy_conditional_width']['tables']['RESIDUAL_FLOW']}
    for n in names:
        if n=='STANDARD':continue
        d=domains[n];use=d['valid'];COUNTER['density_description_slots']+=8
        tab=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in tab['slices']);tables[n]=tab
    errors=[]
    for scope,ns in [('all_candidates',names)]+[(n,['STANDARD',n]) for n in names if n!='STANDARD']:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':scope,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_and_STANDARD_reused_no_fit':True,'not_particle_truth':True}

def run():
    log(f'R16 PID={os.getpid()} device=CPUfloat64; no network training')
    train_v,val_v,domains,rfit,rr,ref,edges,sel=inputs();old14=pathlib.Path(CFG['remote_R14_root']);old15=pathlib.Path(CFG['remote_R15_root']);prior=json.loads((old15/'metrics.json').read_text());rep=npz(old14/'representation.npz');rep['kind']=str(rep['kind'].item());inp=npz(old14/'GENERATION_INPUTS.npz')
    head=cd.Flow(CFG['head_seed']).double();head.load_state_dict(torch.load(old14/'FLOW_EXPOSURES.pt',map_location='cpu',weights_only=True));head.requires_grad_(False);head.eval()
    tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old14/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.requires_grad_(False);tail.eval()
    flow=rf.ResidualFlow(CFG['seed']).double();flow.load_state_dict(torch.load(old15/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.requires_grad_(False);flow.eval()
    tx,th,_,legal=hb.encoded_heads(train_v,rep);vx,vh,_,vlegal=hb.encoded_heads(val_v,rep);assert legal.all() and vlegal.all() and len(tx)==21047 and len(vx)==599
    tz,tld=hb.transform(head,tx);vz,vld=hb.transform(head,vx);latent={'TRAIN':hb.latent_statistics(tz),'validation':hb.latent_statistics(vz),'representation_and_head_not_refitted':True};np.savez_compressed(OUT/'LATENT_PRIVATE.npz',TRAIN_z=tz,VAL_z=vz,TRAIN_logdet=tld,VAL_logdet=vld)
    params={};fits={};fitstart=time.monotonic();prior_alarm=signal.getitimer(signal.ITIMER_REAL)[0];signal.setitimer(signal.ITIMER_REAL,min(prior_alarm,CFG['fitting_limit_s']))
    try:
        for name in ['GAUSSIAN','GMM8']:
            COUNTER['new_head_base_fit_attempts']+=1;begin=time.monotonic();log('base fit '+name)
            try:
                p,row=hb.fit_gaussian(tz) if name=='GAUSSIAN' else hb.fit_gmm(tz,CFG);params[name]=p;fits[name]=row;np.savez_compressed(ROOT/(name+'_BASE.npz'),**p)
            except BaseException as ex:
                fits[name]={'status':'failed','eligible':False,'error':repr(ex),'traceback':traceback.format_exc()}
                if isinstance(ex,TimeoutError):raise
            finally:
                fits[name]['elapsed_s']=time.monotonic()-begin;COUNTER['head_base_fitting_cumulative_s']+=fits[name]['elapsed_s'];write('HEAD_BASE_FITS.json',fits)
    finally:
        remaining=prior_alarm-(time.monotonic()-fitstart)
        signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
    if 'GAUSSIAN' not in params:raise RuntimeError('Gaussian base unavailable')
    checks=hb.formula_checks(params['GAUSSIAN'],params.get('GMM8'),vz);write('BASE_DENSITY_CHECKS.json',checks)
    if not checks['pass']:raise RuntimeError('new base formula check failed')
    savedparts=npz(old15/'results/DENSITY_SCORES_PRIVATE.npz');trainparts={k.removeprefix('TRAIN_'):v for k,v in savedparts.items() if k.startswith('TRAIN_')};valparts={k.removeprefix('VAL_'):v for k,v in savedparts.items() if k.startswith('VAL_')}
    validations,vs=hb.full_scores(vz,vld,valparts,params['GAUSSIAN'],params.get('GMM8'));train_scores,ts=hb.full_scores(tz,tld,trainparts,params['GAUSSIAN'],params.get('GMM8'))
    # STANDARD is the already executed R15 complete density; no old mathematical audit is rerun.
    validations['STANDARD']=dict(prior['validation']['RESIDUAL_FLOW'],source='R15 reused frozen complete score');train_scores['STANDARD']=dict(prior['training_scores']['RESIDUAL_FLOW'],source='R15 reused frozen complete score');vs['STANDARD__complete_logdensity']=valparts['complete_logdensity'];ts['STANDARD__complete_logdensity']=trainparts['complete_logdensity']
    np.savez_compressed(OUT/'DENSITY_SCORES_PRIVATE.npz',**{'VAL_'+k:v for k,v in vs.items()},**{'TRAIN_'+k:v for k,v in ts.items()})
    eligible=['STANDARD']+[n for n in CFG['models'] if fits[n]['eligible'] and validations.get(n,{}).get('complete_NLL') is not None];order={n:i for i,n in enumerate(CFG['all_comparison_names'])};selected=min(eligible,key=lambda n:(validations[n]['complete_NLL'],order[n]));selection={'rule':'minimum complete599 validation NLL; exact tie STANDARD,GAUSSIAN,GMM8','eligible':eligible,'selected_base':selected,'DEV_selection':False}
    gaps={n:{'TRAIN_N':21047,'val_N':599,'TRAIN_NLL':train_scores[n]['complete_NLL'],'val_NLL':validations[n]['complete_NLL'],'val_minus_TRAIN_NLL':validations[n]['complete_NLL']-train_scores[n]['complete_NLL']} for n in validations if validations[n]['complete_NLL'] is not None}
    del savedparts,trainparts,valparts,vs,ts,tx,th,vx,vh,val_v
    oldgen=npz(old15/'RESIDUAL_FLOW/generated.npz');assert np.array_equal(oldgen['draw_ids'],inp['draw_ids']);domains['STANDARD']=aa.data(oldgen['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    uniform=np.random.default_rng(CFG['base_generation_uniform_seed']).uniform(size=2000);COUNTER['new_uniform_draws']=2000;actual={'head_white':inp['head_white'],'tail_white':inp['tail_white'],'draw_ids':inp['draw_ids'],'GMM8_uniform':uniform};models={'STANDARD':{'status':'reused_frozen_R15','generation':prior['models']['RESIDUAL_FLOW']['generation']}}
    for name,p in params.items():
        xx,zz,comp=hb.sample_heads(name,p,head,inp,uniform,CFG['generation_chunk']);actual[name+'__x']=xx;actual[name+'__latent']=zz
        if comp is not None:actual[name+'__component']=comp
        pre=rf.generate(flow,tail,xx,rep,inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(pre);folder=ROOT/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids']);models[name]={'status':'complete' if np.isfinite(pre).all() else 'nonfinite_generation','base_eligible':fits[name]['eligible'],'generation':{'rows':len(pre),**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}};domains[name]=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref);log('generated '+name)
    np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',**actual)
    names=[n for n in CFG['all_comparison_names'] if n in domains];contribution,contprivate=hb.psd_contribution(domains,rep);hb.psd_contribution({'TRAIN_REFERENCE':rr},rep);np.savez_compressed(OUT/'HEAD_PSD_PRIVATE.npz',**contprivate)
    stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,names,rr,sel);wd=widths(domains,prior,edges,names)
    queries={n:domains[n] for n in CFG['models'] if n in domains};nn={'original2592':nearest(queries,rfit,'ORIGINAL')};baseline=train_v[:,:100].mean(axis=1);delt=train_v-baseline[:,None];peak=delt.max(axis=1);large_ref={'V':train_v,'shape':delt/peak[:,None]};nn['expanded_fit']=nearest(queries,large_ref,'EXPANDED');nn['actual_expanded_reference_N']=21047;nn['STANDARD_and_REAL_summaries_reused']=prior['nearest_neighbours'];nn['reference_cardinality_not_memory_evidence']=True;del large_ref,delt,train_v
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete' if len(params)==2 else 'partial_base_failure','models':models,'head_base_fits':fits,'base_selection':selection,'validation':validations,'training_scores':train_scores,'train_validation_gap':gaps,'latent':latent,'head_only_full_PSD':contribution,'base_density_checks':checks,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text()),'pairing':{'same_actual_head_white_and_tail_white':True,'same_draw_ids':True,'GMM_uniform_seed':CFG['base_generation_uniform_seed'],'only_base_distribution_changed':True,'conditional_kernel_fixed_but_its_output_changes_with_head':True,'not_G4_parent_mapping':True,'STANDARD_generation_reused':True},'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'torch':torch.__version__},'environment':{'training_device':None,'final_density_and_generation':'CPUfloat64','GPU_allocated_peak_bytes':0,'GPU_reserved_peak_bytes':0,'TF32':False,'deterministic_algorithms':True},'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'environment':metrics['environment'],'threads':4});log('R16 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
'''
(ROOT/'numeric_worker.py').write_text(header+'\n'+nearest+'\n'+new)
cfg=json.loads((ROOT/'config.json').read_text())
for k in ['training_limit_s','maximum_updates','permutation_seed','batch_seed','train_batch','train_lr','train_weight_decay','train_gradient_clip','train_validation_every','training_dtype','initialization','residual_dim','head_condition_dim','flow_layers','flow_hidden','affine_logscale_bound','split','remote_R13_root']:
    cfg.pop(k,None)
cfg['frozen_models']=['R14 FLOW_EXPOSURES.pt step15700','R14 TAIL_EXPOSURES.pt step15900','R15 RESIDUAL_FLOW.pt step1400'];cfg['network_training_authorized']=False
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
supervise=(ROOT/'remote_supervise.py').read_text().replace("['UPLOAD.tar','RETURN.tar']","['UPLOAD.tar','RETURN.tar','INPUT_BINDINGS.json']")
(ROOT/'remote_supervise.py').write_text(supervise)
print(json.dumps({'prepared':['numeric_worker.py','config.json','remote_supervise.py'],'science_started':False}))
