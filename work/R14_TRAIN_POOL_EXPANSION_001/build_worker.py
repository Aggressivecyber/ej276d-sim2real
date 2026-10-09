"""Build R14 from the already executed R13 evaluation infrastructure, source-only."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'R13_FACTORIZED_CONDITIONAL_DENSITY_001/numeric_worker.py').read_text(encoding='utf-8')
s=s.replace('R13','R14').replace('import conditional_density as cd','import conditional_density as cd\nimport training as tr')
s=s.replace('max(.001,min(1760,','max(.001,min(2360,')
s=s.replace("and len(names)==5","and len(names)==6").replace("five fixed arms unavailable","six fixed arms unavailable").replace("if len(names)==5","if len(names)==6")
a=s.index('def c2st(');b=s.index('def widths(')
c=s[a:b].replace("contrasts=[('GMM_CONDITIONAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','FLOW_GLOBAL'),('FLOW_GLOBAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','GMM_CONDITIONAL')]", "contrasts=CFG['contrasts']")
s=s[:a]+c+s[b:]
a=s.index('def widths(');b=s.index('def nearest(')
s=s[:a]+'''def widths(domains,names,old13,edges):
    tables={'REAL':old13['energy_conditional_width']['tables']['REAL'],'R3B':old13['energy_conditional_width']['tables']['R3B'],'SMALL_GLOBAL':old13['energy_conditional_width']['tables']['FLOW_GLOBAL'],'SMALL_CONDITIONAL':old13['energy_conditional_width']['tables']['FLOW_CONDITIONAL']};errors=[]
    for n in CFG['models']:
        if n not in domains:continue
        COUNTER['density_description_slots']+=8
        if COUNTER['density_description_slots']>32:raise RuntimeError('width slot cap')
        d=domains[n];use=d['valid'];t=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in t['slices']);tables[n]=t
    sets=[('six_way',names)]+[(n+'_vs_REAL',[n]) for n in names]+[(a+'_minus_'+b,[a,b]) for a,b in CFG['contrasts']]
    for label,ns in sets:
        if not all(n in tables for n in ns):continue
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':label,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_R13_tables_reused_without_fitting':True,'not_particle_truth':True}

''' + s[b:]
s=s.replace('def nearest(domains,reference):','def nearest(domains,reference,label):').replace("OUT/'NN_PRIVATE.npz'","OUT/('NN_'+label+'_PRIVATE.npz')").replace("OUT/'NN_EXAMPLES.npz'","OUT/('NN_'+label+'_EXAMPLES.npz')")
a=s.index('def run():');b=s.index('try:run()',a)
s=s[:a]+'''def score_all(x,e,h,rep,flows,tails):
    logdetL=float(np.log(np.diag(rep['tail_cholesky'])).sum());jac=rep['head_log_scale']+33*h[:,0]+rep['constant_logdet'];result={};private={}
    for budget in flows:
        pieces_h=[];pieces_t=[]
        for start in range(0,len(x),512):
            sl=slice(start,start+512);xx=torch.from_numpy(x[sl]);ee=torch.from_numpy(e[sl])
            with torch.no_grad():pieces_h.append(flows[budget].log_prob(xx).numpy());pieces_t.append(cd.tail_log_prob(tails[budget],xx,ee,logdetL).numpy())
        hp=np.concatenate(pieces_h);tp=np.concatenate(pieces_t);gp=-.5*(446*np.log(2*np.pi)+(e*e).sum(axis=1))-logdetL
        for tail_name,tail_lp in [('GLOBAL',gp),('CONDITIONAL',tp)]:
            name=budget+'_'+tail_name;score=hp+tail_lp-jac
            if not np.isfinite(score).all():raise FloatingPointError('complete density nonfinite '+name)
            result[name]={'all_rows':len(x),'legal_rows':len(x),'illegal_rows':0,'complete_NLL':float(-score.mean()),'mean_head_std_logdensity':float(hp.mean()),'mean_tail_logdensity':float(tail_lp.mean()),'mean_logJacobian':float(jac.mean()),'arithmetic':'CPU float64 saved selected weights','not_new_model_selection':True};private[name]=score
    return result,private

def run():
    available=torch.cuda.is_available();free=torch.cuda.mem_get_info()[0] if available else None;device=torch.device('cuda:0' if available and free>=CFG['GPU_allocated_limit_bytes'] else 'cpu');log(f'R14 PID={os.getpid()} training device={device}')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    oldtrain_v,val_v,domains,rfit,rr,ref,edges,sel,old11,old12=inputs()
    p13=pathlib.Path(CFG['remote_R13_root']);old13=json.loads((p13/'metrics.json').read_text())
    addition=npz(ROOT/'TRAIN_ADDITION.npz');newqc=addition['QC_reason_u16']==0
    newidx=addition['original_index'][newqc];assert len(addition['original_index'])==18500 and np.array_equal(addition['artificial_group'],addition['original_index']//50)
    oldidx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200)[2000:];oldqc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200)[2000:];vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];oldmask=(oldqc==0)&~np.isin(oldidx//50,vg)
    fitidx=np.concatenate([oldidx[oldmask],newidx]);train_v=np.concatenate([oldtrain_v,addition['waveforms_V'][newqc].astype('f8')]);N=len(train_v)
    assert len(oldtrain_v)==2592 and len(fitidx)==N and len(np.unique(fitidx))==N and not np.isin(fitidx//50,vg).any() and np.all(np.diff(newidx)>0)
    fitidx.astype('<u4').tofile(ROOT/'expanded_fit_original_index.bin');(fitidx//50).astype('<u2').tofile(ROOT/'expanded_fit_group.bin')
    write('DATA_ROLES.json',{'REAL_DEV':2000,'old_fit_candidate':2600,'old_fit_QC':2592,'new_raw_candidates':18500,'new_QC_valid':int(newqc.sum()),'new_QC_invalid':int((~newqc).sum()),'expanded_fit_QC':N,'expanded_fit_candidate':21100,'fit_groups':int(len(np.unique(fitidx//50))),'unchanged_val_QC':599,'same_acquisition_artificial_groups':True,'CAL_waveform_decode':0,'EVAL_read':0,'new_G4':0,'expanded_order':'old2592 exact cached order, then new QC-valid original_index ascending'})
    del addition
    rep=npz(p13/'representation.npz');rep['kind']=str(rep['kind'].item());gmm=npz(p13/'GMM_8.npz');inp=npz(p13/'GENERATION_INPUTS.npz')
    np.savez_compressed(ROOT/'representation.npz',**rep);np.savez_compressed(ROOT/'GMM_8.npz',**gmm);np.savez_compressed(ROOT/'GENERATION_INPUTS.npz',**inp)
    tx,te,th,traw=cd.coordinates(train_v,rep);vx,ve,vh,vr=cd.coordinates(val_v,rep);del traw,vr
    training={};nets={}
    for kind in ['head','tail']:
        COUNTER['new_network_fit_attempts']+=1;begin=time.monotonic()
        try:models,rec=tr.train_expanded(kind,tx,te,vx,ve,rep,CFG,ROOT,device,COUNTER['training_cumulative_s'],log,COUNTER)
        finally:COUNTER['training_cumulative_s']+=time.monotonic()-begin
        training[kind]=rec;nets[kind]=models
    validation,vscore=score_all(vx,ve,vh,rep,nets['head'],nets['tail']);train_scores,tscore=score_all(tx,te,th,rep,nets['head'],nets['tail'])
    oldflow,oldtail=cd.load_models(p13,CFG);small_fit,small_private=score_all(tx[:2592],te[:2592],th[:2592],rep,{'SMALL':oldflow},{'SMALL':oldtail})
    gap={n:{'TRAIN_N':N,'val_N':599,'TRAIN_NLL':train_scores[n]['complete_NLL'],'val_NLL':validation[n]['complete_NLL'],'val_minus_TRAIN_NLL':validation[n]['complete_NLL']-train_scores[n]['complete_NLL']} for n in validation}
    gap['small_original_fit']={n:{'TRAIN_N':2592,'val_N':599,'TRAIN_NLL':a['complete_NLL'],'val_NLL':old13['validation']['FLOW_'+n.split('_',1)[1]]['complete_NLL'],'val_minus_TRAIN_NLL':old13['validation']['FLOW_'+n.split('_',1)[1]]['complete_NLL']-a['complete_NLL']} for n,a in small_fit.items()}
    np.savez_compressed(OUT/'SCORES_PRIVATE.npz',**{'VAL_'+n:a for n,a in vscore.items()},**{'TRAIN_'+n:a for n,a in tscore.items()},**{'OLD_TRAIN_'+n:a for n,a in small_private.items()})
    del tx,te,th,vx,ve,vh,tscore,vscore,small_private,oldflow,oldtail
    for name,oldname in [('SMALL_GLOBAL','FLOW_GLOBAL'),('SMALL_CONDITIONAL','FLOW_CONDITIONAL')]:
        old=npz(p13/oldname/'generated.npz');assert np.array_equal(old['draw_ids'],inp['draw_ids']);domains[name]=aa.data(old['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    generated_models={};heads={}
    for budget in ['UPDATES','EXPOSURES']:
        _,heads[budget]=cd.sample_heads(rep,gmm,nets['head'][budget],inp,CFG['generation_chunk'])
    for name in CFG['models']:
        budget=name.split('_')[0];pre=cd.generate(name,rep,heads[budget],nets['tail'][budget],inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(pre)
        folder=ROOT/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids'])
        generated_models[name]={'status':'complete' if np.isfinite(pre).all() else 'severe_nonfinite_generation','budget':budget,'budget_complete':all(training[k]['equal_updates_complete' if budget=='UPDATES' else 'equal_exposures_complete'] for k in ['head','tail']),'head_selected_step':training['head']['selected_steps'][budget],'tail_selected_step':training['tail']['selected_steps'][budget],'generation':{'rows':2000,**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}}
        if np.isfinite(pre).all():domains[name]=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
        write('MODEL_PROGRESS.json',generated_models)
    names=[n for n in CFG['all_comparison_names'] if n in domains];stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,names,rr,sel);wd=widths(domains,names,old13,edges)
    queries={n:domains[n] for n in CFG['models'] if n in domains};nn={'original2592':nearest(queries,rfit,'ORIGINAL')}
    baseline=train_v[:,:100].mean(axis=1);delt=train_v-baseline[:,None];peak=delt.max(axis=1);large_ref={'V':train_v,'shape':delt/peak[:,None]};nn['expanded_fit']=nearest(queries,large_ref,'EXPANDED');nn['actual_expanded_reference_N']=N;nn['old_R13_reused']=old13['nearest_neighbours'];nn['reference_size_difference_not_memory_evidence']=True
    del large_ref,delt,train_v
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete' if len(names)==6 else 'incomplete_generation','models':generated_models,'training':training,'validation':validation,'training_scores':train_scores,'train_validation_gap':gap,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text()),'pairing':{'actual_old_innovations_reused':True,'budget_and_tail_shared_draw_positions':True,'strict_tail_contrasts_same_head_and_tail_white':True,'initial_weights_from_R13':True,'minibatch_indices_change_with_N':True,'not_G4_parent_mapping':True},'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'CUDA_runtime':torch.version.cuda},'environment':{'training_device':str(device),'GPU_free_before_bytes':free,'GPU_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'final_density_and_generation':'CPU float64','GPU_allocated_peak_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,'GPU_reserved_peak_bytes':int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else 0,'TF32':False,'deterministic_algorithms':True},'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'environment':metrics['environment'],'threads':4});log('R14 complete '+json.dumps(COUNTER))
''' + s[b:]
ast.parse(s)
(ROOT/'numeric_worker.py').write_text(s,encoding='utf-8')
print('R14 worker AST ready',len(s.encode()))
