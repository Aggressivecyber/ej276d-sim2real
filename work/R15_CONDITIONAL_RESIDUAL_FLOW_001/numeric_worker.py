from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def alarm(a,b):raise TimeoutError('R15 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(2360,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np
import torch,sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import conditional_density as cd
import residual_flow as rf
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'new_network_fit_attempts':0,'residual_updates':0,'training_cumulative_s':0.,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'identity_check_rows':0,'new_random_draws':0,'head_PCA_GMM_ridge_fits':0,'EVAL_read':0,'new_G4':0,'raw_waveform_decode':0};CACHE={}
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def read(k):
    if k not in CACHE:
        b=CFG['input_files'][k];p=pathlib.Path(CFG['remote_input_root'])/k
        with p.open('rb') as f:raw=f.read(b['bytes']+1)
        if len(raw)!=b['bytes']:raise RuntimeError('input size '+k)
        CACHE[k]=raw
    return CACHE[k]
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def inputs():
    v=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy();qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200);g=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200);idx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200);roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(g,idx//50)
    vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(g[2000:],vg);val=(qc[2000:]==0)&np.isin(g[2000:],vg);assert fit.sum()==2592 and val.sum()==599
    assert not set(g[:2000])&set(g[2000:]) and not set(g[2000:][fit])&set(g[2000:][val]);ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);sel=json.loads(read('R7_selections'))
    with np.load(io.BytesIO(read('SIM_identity')),allow_pickle=False) as z:rids=z['real_event_ids'].copy()
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(i) for i in idx[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(idx[2000:][fit][pick],sel['TRAIN_reference_original_indices'])
    real=aa.data(v[:2000],rids,g[:2000],'REAL',ref,qc[:2000]);rfit=aa.data(v[2000:][fit],idsfit,g[2000:][fit],'TRAIN',ref);rr=aa.data(v[2000:][fit][pick],idsfit[pick],g[2000:][fit][pick],'TRAIN_REFERENCE',ref)
    addition=npz(pathlib.Path(CFG['remote_R14_root'])/'TRAIN_ADDITION.npz');valid=addition['QC_reason_u16']==0
    assert len(addition['original_index'])==18500 and valid.sum()==18455
    train=np.concatenate([v[2000:][fit].astype('f8'),addition['waveforms_V'][valid].astype('f8')]);fitidx=np.concatenate([idx[2000:][fit],addition['original_index'][valid]])
    assert len(train)==21047 and len(np.unique(fitidx))==21047 and not np.isin(fitidx//50,vg).any()
    assert np.array_equal(fitidx,np.fromfile(pathlib.Path(CFG['remote_R14_root'])/'expanded_fit_original_index.bin',dtype='<u4'))
    write('DATA_ROLES.json',{'TRAIN_fit':21047,'TRAIN_val':599,'REAL_DEV':2000,'old_fit_reference':2592,'TRAIN_REFERENCE':2000,'groups':422,'unchanged_R14_order':True,'same_acquisition_artificial_groups':True,'EVAL_read':0,'CAL_waveform_numeric_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'frozen_representation_fit_N':2592})
    return train,v[2000:][val].astype('f8'),{'REAL':real},rfit,rr,ref,edges,sel
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
        paired['RESIDUAL_FLOW_minus_GAUSSIAN_BASE']={view:aa.paired_bootstrap(pred['RESIDUAL_FLOW'][view],pred['GAUSSIAN_BASE'][view],CFG) for view in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'original common pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_contrasts':paired,'all_candidates_common_QC':True,'same_N_for_all':True,'original_R7_pools_only':True,'strict_paired_actual_head_and_tail_white':True,'RR_is_sampling_reference_not_equivalence':True}
def widths(domains,old14,edges):
    tables={'REAL':old14['energy_conditional_width']['tables']['REAL'],'GAUSSIAN_BASE':old14['energy_conditional_width']['tables']['EXPOSURES_CONDITIONAL']};d=domains['RESIDUAL_FLOW'];use=d['valid'];COUNTER['density_description_slots']+=8
    tab=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in tab['slices']);tables['RESIDUAL_FLOW']=tab;errors=[]
    for scope,ns in [('pair',['GAUSSIAN_BASE','RESIDUAL_FLOW']),('new',['RESIDUAL_FLOW'])]:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':scope,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_and_base_reused_no_fit':True,'not_particle_truth':True}
def nearest(domains,reference,label):
    out={'TRAIN_reference_N':len(reference['V']),'all_2000_queries_no_QC_resampling':True,'not_proof_of_no_memory':True};priv={};examples={}
    for view,key in [('V','V'),('own_shape','shape')]:
        R=np.asarray(reference[key],dtype='f8');rf=np.isfinite(R).all(axis=1);ri=np.flatnonzero(rf);R=R[rf];r2=(R*R).sum(axis=1)
        for name,d in domains.items():
            a=np.asarray(d[key],dtype='f8');ok=np.isfinite(a).all(axis=1);ix=np.flatnonzero(ok);ds=np.full(len(a),np.nan);nn=np.full(len(a),-1,dtype='i4');exact=np.zeros(len(a),bool)
            for start in range(0,len(ix),64):
                sl=ix[start:start+64];X=a[sl];sq=((X*X).sum(axis=1)[:,None]+r2[None,:]-2*X@R.T)/480;j=np.argmin(sq,axis=1);ds[sl]=np.sqrt(np.maximum(sq[np.arange(len(sl)),j],0));nn[sl]=ri[j];exact[sl]=np.array([np.array_equal(x,R[k]) for x,k in zip(X,j)])
            out.setdefault(view,{})[name]={'all_query_n':len(a),'defined_query_n':len(ix),'undefined_query_n':len(a)-len(ix),'TRAIN_reference_defined_n':len(R),'RMS_distance':ev.summary(ds),'exact_to_nearest_count':int(exact.sum()),'distance_le1e_6':int((ds<=1e-6).sum()),'distance_le1e_4':int((ds<=1e-4).sum()),'threshold_units':'V' if view=='V' else 'dimensionless','thresholds_descriptive':True}
            priv[name+'__'+view+'__distance']=ds;priv[name+'__'+view+'__neighbour_index']=nn
            chosen=ix[ev.order(d['ids'][ix],'R11-NN-example')[:3]];examples[name+'__'+view+'__query']=a[chosen];examples[name+'__'+view+'__neighbour']=reference[key][nn[chosen]];priv[name+'__'+view+'__example_query_index']=chosen
    np.savez_compressed(OUT/('NN_'+label+'_PRIVATE.npz'),**priv);np.savez_compressed(OUT/('NN_'+label+'_EXAMPLES.npz'),**examples);return out
def run():
    available=torch.cuda.is_available();free=torch.cuda.mem_get_info()[0] if available else None;device=torch.device('cuda:0' if available and free>=CFG['GPU_allocated_limit_bytes'] else 'cpu');log(f'R15 PID={os.getpid()} device={device}')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    train_v,val_v,domains,rfit,rr,ref,edges,sel=inputs();oldroot=pathlib.Path(CFG['remote_R14_root']);old14=json.loads((oldroot/'metrics.json').read_text());rep=npz(oldroot/'representation.npz');rep['kind']=str(rep['kind'].item());inp=npz(oldroot/'GENERATION_INPUTS.npz');gmm=npz(oldroot/'GMM_8.npz')
    head=cd.Flow(CFG['head_seed']).double();head.load_state_dict(torch.load(oldroot/'FLOW_EXPOSURES.pt',map_location='cpu',weights_only=True));head.requires_grad_(False);head.eval()
    tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(oldroot/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.requires_grad_(False);tail.eval()
    tx,te,th,traw=cd.coordinates(train_v,rep);vx,ve,vh,vr=cd.coordinates(val_v,rep);del traw,vr,val_v
    torch.manual_seed(CFG['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(CFG['seed'])
    model=rf.ResidualFlow(CFG['seed']).double();model.eval()
    zero_score,zero_parts=rf.full_scores(model,head,tail,vx,ve,vh,rep)
    with np.load(oldroot/'results/SCORES_PRIVATE.npz',allow_pickle=False) as z:base_val=z['VAL_EXPOSURES_CONDITIONAL'].copy()
    density_error=float(np.max(np.abs(zero_parts['complete_logdensity']-base_val)))
    _,sample_x=cd.sample_heads(rep,gmm,head,inp,CFG['generation_chunk']);np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',x=sample_x,tail_white=inp['tail_white'],draw_ids=inp['draw_ids'])
    base=npz(oldroot/'EXPOSURES_CONDITIONAL/generated.npz');assert np.array_equal(base['draw_ids'],inp['draw_ids'])
    zero=rf.generate(model,tail,sample_x,rep,inp,CFG['generation_chunk']);zero_post,_=den.adc(zero);COUNTER['identity_check_rows']=2000;zero_error=float(np.max(np.abs(zero-base['pre_ADC_V'])));zero_equal=bool(np.array_equal(zero_post,base['waveforms_V']))
    identity={'validation_rows':599,'density_maxabs_nat_event':density_error,'density_pass':density_error<=1e-7,'draw_rows':2000,'pre_ADC_maxabs_V':zero_error,'pre_ADC_pass':zero_error<=1e-10,'post_ADC_array_equal':zero_equal,'pointwise_coordinate_identity':True,'pass':density_error<=1e-7 and zero_error<=1e-10 and zero_equal};write('STEP0_IDENTITY_CHECK.json',identity)
    if not identity['pass']:raise RuntimeError('step0 identity check failed '+repr(identity))
    del zero,zero_post,zero_parts
    COUNTER['new_network_fit_attempts']+=1;begin=time.monotonic()
    try:model,training=rf.train(model,tail,tx,te,vx,ve,rep,CFG,ROOT,device,COUNTER,log)
    finally:COUNTER['training_cumulative_s']+=time.monotonic()-begin
    mathcheck=rf.mathematical_checks(model,tail,vx,ve);write('MATHEMATICAL_CHECKS.json',mathcheck)
    validation,vscore=rf.full_scores(model,head,tail,vx,ve,vh,rep);train_score,tscore=rf.full_scores(model,head,tail,tx,te,th,rep)
    np.savez_compressed(OUT/'DENSITY_SCORES_PRIVATE.npz',**{'VAL_'+k:a for k,a in vscore.items()},**{'TRAIN_'+k:a for k,a in tscore.items()})
    validations={'GAUSSIAN_BASE':old14['validation']['EXPOSURES_CONDITIONAL'],'RESIDUAL_FLOW':validation};training_scores={'GAUSSIAN_BASE':old14['training_scores']['EXPOSURES_CONDITIONAL'],'RESIDUAL_FLOW':train_score};gaps={n:{'TRAIN_N':21047,'val_N':599,'TRAIN_NLL':training_scores[n]['complete_NLL'],'val_NLL':validations[n]['complete_NLL'],'val_minus_TRAIN_NLL':validations[n]['complete_NLL']-training_scores[n]['complete_NLL']} for n in validations}
    del tx,te,th,vx,ve,vh,tscore,vscore
    domains['GAUSSIAN_BASE']=aa.data(base['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    pre=rf.generate(model,tail,sample_x,rep,inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']=len(pre);folder=ROOT/'RESIDUAL_FLOW';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids'])
    models={'RESIDUAL_FLOW':{'status':'complete' if np.isfinite(pre).all() else 'nonfinite_generation','generation':{'rows':2000,**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}}}
    domains['RESIDUAL_FLOW']=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref);stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,CFG['all_comparison_names'],rr,sel);wd=widths(domains,old14,edges)
    queries={'RESIDUAL_FLOW':domains['RESIDUAL_FLOW']};nn={'original2592':nearest(queries,rfit,'ORIGINAL')};baseline=train_v[:,:100].mean(axis=1);delt=train_v-baseline[:,None];peak=delt.max(axis=1);large_ref={'V':train_v,'shape':delt/peak[:,None]};nn['expanded_fit']=nearest(queries,large_ref,'EXPANDED');nn['actual_expanded_reference_N']=21047;nn['baseline_and_REAL_summaries_reused']=old14['nearest_neighbours'];nn['reference_cardinality_not_memory_evidence']=True;del large_ref,delt,train_v
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete','models':models,'training':training,'validation':validations,'training_scores':training_scores,'train_validation_gap':gaps,'step0':identity,'mathematical_checks':mathcheck,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text()),'pairing':{'actual_same_head_and_tail_white':True,'same_draw_ids':True,'initial_step0_is_coordinate_identity':True,'frozen_head_and_Gaussian_tail':True,'not_G4_parent_mapping':True},'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'CUDA_runtime':torch.version.cuda},'environment':{'training_device':str(device),'GPU_free_before_bytes':free,'GPU_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'final_density_and_generation':'CPUfloat64','GPU_allocated_peak_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,'GPU_reserved_peak_bytes':int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else 0,'TF32':False,'deterministic_algorithms':True},'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'environment':metrics['environment'],'threads':4});log('R15 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
