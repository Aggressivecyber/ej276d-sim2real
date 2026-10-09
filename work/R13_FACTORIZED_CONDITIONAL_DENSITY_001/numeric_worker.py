from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def alarm(a,b):raise TimeoutError('R13 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1760,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np
import torch,sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import conditional_density as cd
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'PCA_GMM_ridge_refits':0,'new_network_fit_attempts':0,'head_updates':0,'tail_updates':0,'training_cumulative_s':0.,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'new_random_draws':0,'EVAL_read':0,'new_G4':0};CACHE={}

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
    old11=json.loads((pathlib.Path(CFG['remote_R11_root'])/'metrics.json').read_text());old12=json.loads((pathlib.Path(CFG['remote_R12_root'])/'metrics.json').read_text())
    write('DATA_ROLES.json',{'REAL_DEV':2000,'TRAIN_fit':2592,'TRAIN_val':599,'TRAIN_REFERENCE':2000,'same_acquisition_artificial_groups':True,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4_parents':0,'old_R11_R12_outputs_reused':True})
    return v[2000:][fit].astype('f8'),v[2000:][val].astype('f8'),{'REAL':real},rfit,rr,ref,edges,sel,old11,old12

def c2st(domains,names,rr,sel):
    sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4')) for k,v in sel['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=sp['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g] for n in names])];r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train' if phase==0 else 'holdout']={'N_each':n,'candidate_common_valid':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    out={};pred={};paired={};private={};support=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75 and len(names)==5
    contrasts=[('GMM_CONDITIONAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','FLOW_GLOBAL'),('FLOW_GLOBAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','GMM_CONDITIONAL')]
    if support:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        for n in names+['REAL_REAL']:
            d=rr if n=='REAL_REAL' else domains[n];selection=ts if n=='REAL_REAL' else gs;assert not set(d['groups'][selection[0]])&set(d['groups'][selection[1]]);out[n]={};pred[n]={}
            for view in CFG['C2ST_views']:
                out[n][view],pred[n][view]=dc.fit_one(d,domains['REAL'],selection,rs,view,CFG,COUNTER);private.update({n+'__'+view+'__'+k:v for k,v in pred[n][view].items()});log(f'C2ST {n} {view} BA={out[n][view]["balanced_accuracy"]:.6f}')
        for a,b in contrasts:paired[a+'_minus_'+b]={view:aa.paired_bootstrap(pred[a][view],pred[b][view],CFG) for view in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'five fixed arms unavailable or original pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_contrasts':paired,'contrast_order':contrasts,'all_candidates_common_QC':True,'same_N_for_all':True,'original_R7_pools_only':True,'head_change_paired_by_draw_position_not_same_physical_head':True,'RR_is_sampling_reference_not_equivalence':True}

def widths(domains,names,old11,old12,edges):
    tables={'REAL':old11['energy_conditional_width']['tables']['REAL'],'R3B':old11['energy_conditional_width']['tables']['R3B'],'FACTOR_LW':old11['energy_conditional_width']['tables']['FACTOR_PCA_GMM'],'GMM_GLOBAL':old12['energy_conditional_width']['tables']['FACTOR_EMPIRICAL_COV']};errors=[]
    for n in CFG['models']:
        if n not in domains:continue
        COUNTER['density_description_slots']+=8
        if COUNTER['density_description_slots']>24:raise RuntimeError('widthslot limit')
        d=domains[n];use=d['valid'];t=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in t['slices']);tables[n]=t
    four=['GMM_GLOBAL','FLOW_GLOBAL','GMM_CONDITIONAL','FLOW_CONDITIONAL'];sets=[('four_way',four)]+[(n+'_vs_REAL',[n]) for n in names if n in tables]+[(a+'_minus_'+b,[a,b]) for a,b in [('GMM_CONDITIONAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','FLOW_GLOBAL'),('FLOW_GLOBAL','GMM_GLOBAL'),('FLOW_CONDITIONAL','GMM_CONDITIONAL')]]
    for label,ns in sets:
        if not all(n in tables for n in ns):continue
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':label,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_tables_reused_without_fitting':True,'not_particle_truth':True}

def nearest(domains,reference):
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
    np.savez_compressed(OUT/'NN_PRIVATE.npz',**priv);np.savez_compressed(OUT/'NN_EXAMPLES.npz',**examples);return out

def run():
    available=torch.cuda.is_available();gpu_free=torch.cuda.mem_get_info()[0] if available else None
    device=torch.device('cuda:0' if available and gpu_free>=CFG['GPU_allocated_limit_bytes'] else 'cpu');log(f'R13 PID={os.getpid()} training device={device},final CPUfloat64')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    train_v,val_v,domains,rfit,rr,ref,edges,sel,old11,old12=inputs()
    p11=pathlib.Path(CFG['remote_R11_root'])/'FACTOR_PCA_GMM';p12=pathlib.Path(CFG['remote_R12_root'])/'FACTOR_EMPIRICAL_COV'
    rep=npz(p12/'representation.npz');rep['kind']=str(rep['kind'].item());gmm=npz(p11/'GMM_8.npz');inp=npz(p11/'GENERATION_INPUTS.npz');oldglobal=npz(p12/'generated.npz');oldlw=npz(p11/'generated.npz')
    assert len(inp['draw_ids'])==2000 and np.array_equal(inp['draw_ids'],oldglobal['draw_ids']) and np.array_equal(inp['draw_ids'],oldlw['draw_ids'])
    np.savez_compressed(ROOT/'representation.npz',**rep);np.savez_compressed(ROOT/'GMM_8.npz',**gmm);np.savez_compressed(ROOT/'GENERATION_INPUTS.npz',**inp)
    train_x,train_e,_,_=cd.coordinates(train_v,rep);val_x,val_e,val_h,val_r=cd.coordinates(val_v,rep)
    training={}
    for kind in ['head','tail']:
        COUNTER['new_network_fit_attempts']+=1;begin=time.monotonic()
        try:model,record=cd.train(kind,train_x,train_e,val_x,val_e,rep,CFG,ROOT,device,COUNTER['training_cumulative_s'],log,COUNTER)
        finally:COUNTER['training_cumulative_s']+=time.monotonic()-begin
        training[kind]=record
        if kind=='head':flow=model
        else:tail=model
    flow.eval();tail.eval();init=cd.ConditionalTail().double();init.load_state_dict(torch.load(ROOT/'TAIL_INITIAL_STATE.pt',map_location='cpu',weights_only=True)['model'])
    mathcheck=cd.mathematical_checks(flow,tail,init,val_x,val_e,val_h,val_r,rep);write('MATHEMATICAL_CHECKS.json',mathcheck)
    validation,private=cd.validation_scores(val_v,rep,gmm,flow,tail);validation['FACTOR_LW']={k:old12['models']['FACTOR_EMPIRICAL_COV']['validation_old'][k] for k in ['all_validation_rows','legal_rows','illegal_rows','complete_NLL','mean_head_std_logdensity','mean_tail_logdensity','mean_logJacobian']}
    assert abs(validation['GMM_GLOBAL']['complete_NLL']-old12['models']['FACTOR_EMPIRICAL_COV']['validation_new']['complete_NLL'])<1e-7
    np.savez_compressed(OUT/'VALIDATION_SCORES_PRIVATE.npz',**private)
    domains['GMM_GLOBAL']=aa.data(oldglobal['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref);domains['FACTOR_LW']=aa.data(oldlw['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    gx,fx=cd.sample_heads(rep,gmm,flow,inp,CFG['generation_chunk']);np.savez_compressed(OUT/'HEADS_PRIVATE.npz',GMM_standardized_head=gx,FLOW_standardized_head=fx)
    models={};pairchecks={'GMM_tail_comparison_head_rule':'same frozen K8/component/head_white and frozen standardization; old GLOBAL output reused, no old generation replay','FLOW_tail_comparison_head_rule':'same computed CPUfloat64 fx array passed to both generation branches','tail_counterparts_share_head_by_construction':True,'all_actual_tail_white_reused':True,'all_actual_head_white_reused':True,'FLOW_ignores_GMM_component':True,'head_change_is_not_same_physical_head':True,'fixed_draw_positions_only_not_G4_parents':True,'old_generated_draw_ID_order_checked':True}
    for name in CFG['models']:
        x=fx if name.startswith('FLOW') else gx;pre=cd.generate(name,rep,x,tail,inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(pre)
        folder=ROOT/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids'])
        if not np.isfinite(pre).all():models[name]={'status':'severe_nonfinite_generation','generation':adc};write('MODEL_PROGRESS.json',models);continue
        models[name]={'status':'complete','head':'FLOW' if name.startswith('FLOW') else 'GMM_K8','tail':'CONDITIONAL' if name.endswith('CONDITIONAL') else 'EMPIRICAL_GLOBAL','generation':{'rows':2000,**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()},'same_head_rule_with_tail_counterpart':True}
        domains[name]=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref);write('MODEL_PROGRESS.json',models)
    names=[n for n in CFG['all_comparison_names'] if n in domains];stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,names,rr,sel);wd=widths(domains,names,old11,old12,edges)
    nn=nearest({n:domains[n] for n in CFG['models'] if n in domains},rfit);nn['old_REAL_R11_R12_reused']={'REAL_and_LW':old11['nearest_neighbours'],'GLOBAL':{view:old12['nearest_neighbours'][view]['FACTOR_EMPIRICAL_COV'] for view in ['V','own_shape']}}
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete' if len(names)==5 else 'incomplete_generation_failure','models':models,'training':training,'validation':validation,'mathematical_checks':mathcheck,'pairing':pairchecks,'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'CUDA_runtime':torch.version.cuda},'environment':{'training_device':str(device),'GPU_free_before_bytes':gpu_free,'GPU_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'final_density_and_generation':'CPU float64','GPU_allocated_peak_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,'GPU_reserved_peak_bytes':int(torch.cuda.max_memory_reserved()) if device.type=='cuda' else 0,'TF32':False,'cudnn_deterministic':True,'cudnn_benchmark':False,'deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']},'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'environment':metrics['environment'],'threads':4});log('R13 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
