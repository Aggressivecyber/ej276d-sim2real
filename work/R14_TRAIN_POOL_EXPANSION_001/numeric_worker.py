from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def alarm(a,b):raise TimeoutError('R14 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(2360,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np
import torch,sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import conditional_density as cd
import training as tr
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
    out={};pred={};paired={};private={};support=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75 and len(names)==6
    contrasts=CFG['contrasts']
    if support:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        for n in names+['REAL_REAL']:
            d=rr if n=='REAL_REAL' else domains[n];selection=ts if n=='REAL_REAL' else gs;assert not set(d['groups'][selection[0]])&set(d['groups'][selection[1]]);out[n]={};pred[n]={}
            for view in CFG['C2ST_views']:
                out[n][view],pred[n][view]=dc.fit_one(d,domains['REAL'],selection,rs,view,CFG,COUNTER);private.update({n+'__'+view+'__'+k:v for k,v in pred[n][view].items()});log(f'C2ST {n} {view} BA={out[n][view]["balanced_accuracy"]:.6f}')
        for a,b in contrasts:paired[a+'_minus_'+b]={view:aa.paired_bootstrap(pred[a][view],pred[b][view],CFG) for view in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'six fixed arms unavailable or original pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_contrasts':paired,'contrast_order':contrasts,'all_candidates_common_QC':True,'same_N_for_all':True,'original_R7_pools_only':True,'head_change_paired_by_draw_position_not_same_physical_head':True,'RR_is_sampling_reference_not_equivalence':True}

def widths(domains,names,old13,edges):
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

def score_all(x,e,h,rep,flows,tails):
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
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
