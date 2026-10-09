from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def alarm(a,b):raise TimeoutError('R17 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1760,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np
import torch,sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import conditional_density as cd
import residual_flow as rf
import head_base as hb
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'new_network_fit_attempts':0,'training_cumulative_s':0.,'head_updates_NLL_CONT':0,'head_updates_NLL_PSD':0,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'new_base_PCA_ridge_residual_fits':0,'EVAL_read':0,'new_G4':0,'raw_waveform_decode':0};CACHE={}

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
        for first,second in CFG['contrasts']:
            if first in pred and second in pred:paired[first+'_minus_'+second]={v:aa.paired_bootstrap(pred[first][v],pred[second][v],CFG) for v in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'original common pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_contrasts':paired,'all_candidates_common_QC':True,'same_N_for_all':True,'original_R7_pools_only':True,'all_32_current_environment_refit_if_all_arms_finite':True,'shared_actual_head_and_tail_white':True,'RR_is_sampling_reference_not_equivalence':True,'global_new_views':{'head34':'post-ADC fixed FACTOR standardized34 head','Q_PSD_2D':'post-ADC observed Qtotal_mVns and signedPSD; raw charge, no log'},'CI_conditional_on_classifiers':True}


def widths(domains,prior,edges,names):
    old=prior['energy_conditional_width']['tables'];tables={'REAL':old['REAL'],'STEP0':old['GMM8']}
    for n in ['NLL_CONT','NLL_PSD']:
        if n not in domains:continue
        d=domains[n];use=d['valid'];COUNTER['density_description_slots']+=8
        tab=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in tab['slices']);tables[n]=tab
    errors=[]
    for scope,ns in [('all_candidates',names),('paired_finetunes',[n for n in names if n!='STEP0'])]+[(n,['STEP0',n]) for n in names if n!='STEP0']:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':scope,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_and_STEP0_reused_no_fit':True,'not_particle_truth':True}

def psd_metrics(domains,edges):
    from scipy.stats import wasserstein_distance,ks_2samp
    valley=json.loads((ROOT/'VALLEY_DEFINITION.json').read_text());tables={};private={};ecdf_grid=np.linspace(-.1,.6,281)
    for n,d in domains.items():
        q=d['features']['Qtotal_mVns'];p=d['features']['PSD'];use=d['valid']&np.isfinite(p)&np.isfinite(q)&(q>0);lq=np.full(len(q),np.nan);lq[use]=np.log(q[use]);rows=[]
        for i in range(8):
            inside=use&(lq>=edges[i])&((lq<edges[i+1]) if i<7 else (lq<=edges[i+1]));v=np.sort(p[inside]);definition=valley['bins'][i];low=definition['low'];high=definition['high'];row={'bin_id_1based':i+1,'N':len(v),'source_QC_finite_N':int(use.sum()),'fraction_of_QC_finite':len(v)/int(use.sum()) if use.any() else None,'PSD_below_loss_grid':int((v<-.1).sum()),'PSD_above_loss_grid':int((v>.6).sum()),'valley_definition':definition,'valley_fraction':float(np.mean((v>=low)&(v<=high))) if len(v) and definition['valid'] else None,'complete_empirical_CDF_retained':True};rows.append(row);private[n+'__bin'+str(i+1)+'__PSD_sorted']=v
        tables[n]={'QC_finite_N':int(use.sum()),'all_source_N':len(q),'invalid_or_nonfinite':int(len(q)-use.sum()),'logQ_below':int(np.count_nonzero(use&(lq<edges[0]))),'logQ_above':int(np.count_nonzero(use&(lq>edges[-1]))),'PSD_below_loss_grid':int(np.count_nonzero(use&(p<-.1))),'PSD_above_loss_grid':int(np.count_nonzero(use&(p>.6))),'bins':rows};private[n+'__Qtotal_mVns']=q;private[n+'__PSD']=p;private[n+'__valid']=use
    comparisons={}
    for n in [n for n in domains if n!='REAL']:
        rows=[]
        for i in range(8):
            a=private[n+'__bin'+str(i+1)+'__PSD_sorted'];b=private['REAL__bin'+str(i+1)+'__PSD_sorted'];defined=len(a)>0 and len(b)>0
            rows.append({'bin_id_1based':i+1,'candidate_N':len(a),'REAL_N':len(b),'defined':defined,'undefined_reason':None if defined else 'an_empty_hard_charge_bin','W1':float(wasserstein_distance(a,b)) if defined else None,'KS':float(ks_2samp(a,b).statistic) if defined else None,'occupancy_abs_difference':abs(tables[n]['bins'][i]['fraction_of_QC_finite']-tables['REAL']['bins'][i]['fraction_of_QC_finite']) if defined else None,'valley_fraction_difference':tables[n]['bins'][i]['valley_fraction']-tables['REAL']['bins'][i]['valley_fraction'] if defined and tables[n]['bins'][i]['valley_fraction'] is not None and tables['REAL']['bins'][i]['valley_fraction'] is not None else None})
        comparisons[n]={'bins':rows,'all_8_bins_defined':all(r['defined'] for r in rows),'eight_equal_mean_W1':float(np.mean([r['W1'] for r in rows])) if all(r['defined'] for r in rows) else None,'eight_equal_mean_KS':float(np.mean([r['KS'] for r in rows])) if all(r['defined'] for r in rows) else None,'empty_bins_not_dropped':True}
    delta={}
    if all(n in comparisons for n in ['NLL_CONT','NLL_PSD']):
        for first,second in CFG['contrasts']:
            delta[first+'_minus_'+second]=[{'bin_id_1based':i+1,'W1_delta':comparisons[first]['bins'][i]['W1']-comparisons[second]['bins'][i]['W1'] if comparisons[first]['bins'][i]['defined'] and comparisons[second]['bins'][i]['defined'] else None,'KS_delta':comparisons[first]['bins'][i]['KS']-comparisons[second]['bins'][i]['KS'] if comparisons[first]['bins'][i]['defined'] and comparisons[second]['bins'][i]['defined'] else None}for i in range(8)]
    np.savez_compressed(OUT/'PSD_SECTIONS_PRIVATE.npz',**private)
    return {'definition':'post-ADC signed PSD, full finite empirical populations, frozen8 logQ hard bins','tables':tables,'vs_REAL':comparisons,'paired_deltas':delta,'edges_logQ':edges,'valley_is_old_TRAIN_descriptive_definition_not_loss_target':True,'CDF_loss_grid_not_a_filter':True}

def density_scores(head,kernel,tx,vx,oldparts,name):
    result={};arrays={}
    for label,x in [('TRAIN',tx),('VAL',vx)]:
        parts=[]
        for st in range(0,len(x),512):
            with torch.no_grad():parts.append(kernel.loghead(head,torch.from_numpy(x[st:st+512])).numpy())
        hp=np.concatenate(parts);tail=oldparts[label+'_GMM8__tail_logdensity'];jac=oldparts[label+'_GMM8__Jacobian'];complete=hp+tail-jac
        if not np.isfinite(complete).all():raise FloatingPointError('complete density nonfinite; no row deletion')
        result[label]={'N':len(x),'head_NLL_div34':float(-hp.mean()/34),'complete_NLL':float(-complete.mean()),'tail_logdensity_mean':float(tail.mean()),'FACTOR_Jacobian_mean':float(jac.mean()),'arithmetic':'CPUfloat64; frozen tail+Jacobian at observed standardized head coordinates'}
        arrays[label+'_'+name+'__head_logdensity']=hp;arrays[label+'_'+name+'__complete_logdensity']=complete
    result['val_minus_TRAIN_NLL']=result['VAL']['complete_NLL']-result['TRAIN']['complete_NLL'];return result,arrays

def run():
    import psd_finetune as pf
    log('R17 inputs; unchanged TRAIN21047/validation599/DEV2000')
    train_v,val_v,domains,rfit,rr,ref,edges,sel=inputs();old14=pathlib.Path(CFG['remote_R14_root']);old15=pathlib.Path(CFG['remote_R15_root']);old16=pathlib.Path(CFG['remote_R16_completion_root']);prior=json.loads((old16/'metrics.json').read_text());rep=npz(old14/'representation.npz');base=npz(old16/'GMM8_BASE.npz');inp=npz(old16/'ACTUAL_GENERATION_INPUTS.npz');oldgen=npz(old16/'GMM8/generated.npz');oldparts=npz(old16/'results/DENSITY_SCORES_PRIVATE.npz')
    assert prior['base_selection']['selected_base']=='GMM8';assert np.array_equal(oldgen['draw_ids'],inp['draw_ids']);latent=inp['GMM8__latent'];white=inp['tail_white'];assert latent.shape==(2000,34) and white.shape==(2000,446)
    torch.manual_seed(CFG['head_architecture_seed']);head=cd.Flow(CFG['head_architecture_seed']).double();head.load_state_dict(torch.load(old14/'FLOW_EXPOSURES.pt',map_location='cpu',weights_only=True));head.eval().requires_grad_(True)
    tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old14/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.eval().requires_grad_(False)
    flow=rf.ResidualFlow(CFG['seed']).double();flow.load_state_dict(torch.load(old15/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.eval().requires_grad_(False)
    tx,_,_,legal=hb.encoded_heads(train_v,rep);vx,_,_,vlegal=hb.encoded_heads(val_v,rep);assert legal.all() and vlegal.all();kernel=pf.FixedKernel(tail,flow,rep,base)
    with torch.no_grad():density=kernel.loghead(head,torch.from_numpy(vx)).numpy();pre=kernel.generate_chunks(head,torch.from_numpy(latent),torch.from_numpy(white),CFG['generation_chunk']).numpy()
    post,_=den.adc(pre);derr=float(np.max(np.abs(density-oldparts['VAL_GMM8__head_logdensity'])));verr=float(np.max(np.abs(pre-oldgen['pre_ADC_V'])));checks={'all599_head_GMM_density_maxabs_nat':derr,'density_tolerance_nat':1e-7,'old2000_pre_ADC_V_maxabs':verr,'pre_ADC_tolerance_V':1e-10,'post_ADC_array_equal':bool(np.array_equal(post,oldgen['waveforms_V'])),'source_migration_testing':False,'pass':derr<=1e-7 and verr<=1e-10 and np.array_equal(post,oldgen['waveforms_V'])};write('STEP0_CHECKS.json',checks)
    if not checks['pass']:raise RuntimeError('R17 step0 scientific formula check failed')
    del pre,post,density
    target_train=pf.targets(train_v,edges,CFG);target_val=pf.targets(val_v,edges,CFG);write('CONDITIONAL_TARGETS.json',{'TRAIN':target_train,'validation':target_val,'fit_splits_separate':True,'TRAIN_target_N':21047,'validation_target_N':599});noise=pf.sample_inputs(np.random.default_rng(CFG['validation_generation_seed']),2000,base);np.savez_compressed(ROOT/'VALIDATION_GENERATION_INPUTS.npz',**noise);np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',GMM8__latent=latent,tail_white=white,draw_ids=inp['draw_ids'],GMM8__component=inp['GMM8__component'],GMM8_uniform=inp['GMM8_uniform'])
    grad=pf.gradient_check(head,kernel,noise,target_train,edges,CFG);write('GRADIENT_CHECK.json',grad)
    if not grad['pass']:raise RuntimeError('R17 differentiable PSD gradient failed')
    device=torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4*1024**3 else 'cpu');log('paired head training device '+str(device));COUNTER['new_network_fit_attempts']=0
    step0density,_=density_scores(head,kernel,tx,vx,oldparts,'STEP0');selections={};trained={};scores={'STEP0':step0density};scoreprivate={};models={'STEP0':{'status':'reused_R16_GMM8','rows':2000,'eligible':True}}
    for name in CFG['models']:
        COUNTER['new_network_fit_attempts']+=1
        selected,rec=pf.train_arm(name,head,tail,flow,rep,base,tx,vx,noise,target_train,target_val,edges,CFG,ROOT,device,COUNTER,log);selections[name]=rec;trained[name]=selected;scores[name],arr=density_scores(selected,kernel,tx,vx,oldparts,name);scoreprivate.update(arr)
        write('PAIRED_TRAINING_RECEIPT.json',{'arms':selections,'same_initial_parameters':True,'same_TRAIN_index_seed':True,'same_generation_seed':True,'same_validation_inputs':True,'both_select_by_same_validation_J':True,'DEV_model_selection':False,'training_cumulative_s':COUNTER['training_cumulative_s']})
        if rec['error'] and 'TimeoutError' not in rec['error']:log(name+' branch stopped '+rec['error'])
    same_updates=selections['NLL_CONT']['updates']==selections['NLL_PSD']['updates'];paired={'same_initial_parameters':True,'same_optimizer_reset':True,'same_fixed_validation_inputs':True,'same_final_draws':True,'equal_actual_updates':same_updates,'indices_digest_equal':selections['NLL_CONT']['TRAIN_index_stream_sha256']==selections['NLL_PSD']['TRAIN_index_stream_sha256'],'generation_digest_equal':selections['NLL_CONT']['TRAIN_generation_stream_sha256']==selections['NLL_PSD']['TRAIN_generation_stream_sha256'],'different_updates_possible_under_per_arm450s':not same_updates,'conditional_kernel_weights_fixed_but_output_varies_with_head':True,'both_val_J_include_PSD':True,'not_G4_parent_mapping':True};write('PAIRING_RECEIPT.json',paired)
    np.savez_compressed(OUT/'DENSITY_SCORES_PRIVATE.npz',**scoreprivate);domains['STEP0']=aa.data(oldgen['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    continuous={}
    for name in CFG['models']:
        with torch.no_grad():pre=kernel.generate_chunks(trained[name],torch.from_numpy(latent),torch.from_numpy(white),CFG['generation_chunk']).numpy();q,p=pf.features_torch(torch.from_numpy(pre));cdf,occ,supp=pf.conditional(q,p,edges,CFG)
        post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(pre);np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids']);domains[name]=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref);after=domains[name]['features']['PSD'];pd=p.numpy();continuous[name]={'N':2000,'PSD_post_minus_pre_ADC':ev.summary(after-pd),'post_minus_pre_ADC_RMS':float(np.sqrt(np.mean((after-pd)**2))),'continuous_DEV_CDF_L1_vs_TRAIN':float((cdf-torch.from_numpy(target_train['CDF'])).abs().mean()),'continuous_DEV_occupancy_L1_vs_TRAIN':float((occ-torch.from_numpy(target_train['occupancy'])).abs().mean()),'DEV_not_selection':True};models[name]={'status':'complete','generation':{'rows':2000,**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}};log('generated '+name)
    for d in [*domains.values(),rr]:d['head34']=hb.encoded_heads(d['V'],rep)[0]
    names=CFG['all_comparison_names'];stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref)for n,d in domains.items()};cs=c2st(domains,names,rr,sel);wd=widths(domains,prior,edges,names);psd=psd_metrics(domains,edges)
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features);np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**{n:d['V']for n,d in domains.items()})
    environment={'training_device':str(device),'final_generation_and_validation':'CPUfloat64','GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda'else 0,'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda'else 0,'TF32':False,'deterministic_algorithms':True}
    metrics={'status':'complete','models':models,'training':selections,'complete_density_scores':scores,'step0_checks':checks,'gradient_check':grad,'data_roles':json.loads((ROOT/'DATA_ROLES.json').read_text()),'pairing':paired,'continuous_ADC_surrogate':continuous,'PSD_main':psd,'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'counters':COUNTER,'environment':environment,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'environment':environment,'threads':4});log('R17 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
