from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,traceback,copy,math,shutil
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));ARM=C['arm'];D=R/ARM
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import scipy
import torch
from network import ResponseNetwork
from noise_core import q_np,bins_for,continuous_logq,innovation
from source_flow_core import endpoint,q_torch
from distribution_loss import DistributionLoss,CATEGORIES,NAMES,fit_temporal_statistics,TEMPORAL_NAMES
from precision_order import rounding_psd_interval
from artificial_check import check,integration_check

def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):
    p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def learning_rate(step):
    if step<=C['warmup_updates']:
        return C['lr_initial']+(C['lr']-C['lr_initial'])*(step-1)/(C['warmup_updates']-1)
    fraction=(step-C['warmup_updates'])/(C['updates']-C['warmup_updates'])
    return C['lr_final']+.5*(C['lr']-C['lr_final'])*(1+math.cos(math.pi*fraction))
def raw_psd(v):
    x=np.asarray(v,dtype='f8');q=q_np(x);short=x[:,100:150].sum(1)-.5*x[:,:100].sum(1)
    return 1-short/q
def chain_update(old,value):return hashlib.sha256(old+np.ascontiguousarray(value).tobytes()).digest()

updates=0;counter={'response_fit_attempts':0,'new_response_updates':0,'SOURCE_training_rows':0,'REAL_reference_rows':0,'REAL_identity_rows':0,
    'teacher_waveforms_constructed':0,'new_HGB_fits':0,'new_density_fits':0,'DEV_read':0,'EVAL_read':0,'CAL_decode':0}
try:
    assert C['updates']==50000 and C['batch_per_region']==32 and C['identity_batch']==64
    assert C['box_projection_iterations']==64 and C['solver_steps']==128 and C['solver_check_steps']==256
    assert C['region_categories']==CATEGORIES and C['feature_names']==NAMES
    assert all(C['loss_weights'][k]==1 for k in C['loss_weights'])
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(C['CPU_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write('ARTIFICIAL_CHECK_SERVER.json',{'formula':check(),'integrated_loss_and_network':integration_check(C)})
    binding=json.loads((R/'INPUT_BINDING.json').read_text('utf8'))['inputs']
    for key,path in C['input_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==binding[key]['sha256'],key
    reviewed=json.loads((R/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
    for record in reviewed['files']:assert hashlib.sha256((R/record['name']).read_bytes()).hexdigest()==record['sha256'],record['name']
    historical=json.loads(Path(C['input_paths']['R61_training_receipt']).read_text('utf8'))
    assert historical['initial_state_SHA']==C['expected_R61_initial_state_SHA']and historical['stream_hashes']==C['expected_R61_stream_hashes']
    if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')
    dev=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    write('ARTIFICIAL_CHECK_CUDA.json',integration_check(C,dev))
    environment={'PID':os.getpid(),'UTC':utc(),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
        'device':str(dev),'GPU':torch.cuda.get_device_name(0),'CPU_threads':torch.get_num_threads(),'CPU_affinity':sorted(os.sched_getaffinity(0)),'cwd':str(R),'arm':ARM}
    assert all(environment[k]==C['expected_R33_environment'][k] for k in ['python','numpy','torch','scipy','device','GPU']),('environment drift',environment)
    D.mkdir(exist_ok=False);write(ARM+'/ENVIRONMENT.json',environment)
    # Decode only registered training fields; validation arrays in this archive stay unopened.
    with np.load(C['TRAIN_path'],allow_pickle=False) as z:
        source=np.asarray(z['source'],dtype='f4').copy();real=np.asarray(z['target'],dtype='f4').copy()
        source_ids=z['source_ids'].copy();source_h=z['h'].copy();saved_source_bins=z['source_bin'].copy();saved_REAL_bins=z['target_bin'].copy()
    assert source.shape==(C['SOURCE_N'],480) and real.shape==(C['REAL_N'],480) and len(source_ids)==len(source_h)==len(source)
    with np.load(C['constants_path'],allow_pickle=False) as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    with np.load(C['noise_calibration_path'],allow_pickle=False) as z:noise={k:z[k].copy() for k in z.files}
    valley=json.loads(Path(C['valley_definition']).read_text('utf8'))
    assert valley['source']=='R3B results/WIDTH_PRETRAIN.json REAL_TRAIN_fit2592' and not valley['new_fit']
    write('FROZEN_VALLEY_DEFINITION.json',valley)
    sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
    assert np.array_equal(sraw,saved_source_bins) and np.array_equal(rraw,saved_REAL_bins)
    assert np.all(scale>0) and np.max(np.abs(q_np(mu)-1))<1e-10
    sh=continuous_logq(sq,sb,edges);rh=continuous_logq(rq,rb,edges);sp=raw_psd(source)
    assert np.isfinite(sp).all() and np.all(np.abs(sq)<=300) and np.all(np.abs(rq)<=300)
    groupsS=[np.flatnonzero(sraw==b) for b in CATEGORIES];groupsR=[np.flatnonzero(rraw==b) for b in CATEGORIES]
    assert all(len(x)>0 for x in groupsS+groupsR)
    with np.load(C['input_paths']['distribution_statistics'],allow_pickle=False)as z:statistics={k:z[k].copy()for k in z.files}
    assert np.array_equal(statistics['source_counts'],[len(g)for g in groupsS])and np.array_equal(statistics['REAL_counts'],[len(g)for g in groupsR])
    shutil.copy2(C['input_paths']['distribution_statistics'],R/'TRAIN_DISTRIBUTION_CONSTANTS.npz')
    temporal=fit_temporal_statistics(real,rraw,C)
    assert np.array_equal(temporal['temporal_REAL_counts'],statistics['REAL_counts'])
    np.savez_compressed(R/'TRAIN_TEMPORAL_CONSTANTS.npz',**temporal)
    write('TRAIN_TEMPORAL_CONSTANTS.json',{'UTC':utc(),'fitted_REAL_TRAIN_rows':len(real),'region_categories':CATEGORIES,
        'feature_names':TEMPORAL_NAMES,'center':temporal['temporal_center'].tolist(),'scale':temporal['temporal_scale'].tolist(),
        'scale_floor':C['temporal_scale_floor'],'population_standard_deviation':True,'zero_power_counts':temporal['temporal_zero_power_counts'].tolist(),
        'power_floor':C['temporal_power_floor'],'fitted_before_first_update':True,'scientific_response_fit':False,'teacher_fit':False,
        'new_statistics_SHA256':hashlib.sha256((R/'TRAIN_TEMPORAL_CONSTANTS.npz').read_bytes()).hexdigest(),
        'old_statistics_SHA256':hashlib.sha256((R/'TRAIN_DISTRIBUTION_CONSTANTS.npz').read_bytes()).hexdigest()})
    statistics.update(temporal)
    interval_low,interval_high=rounding_psd_interval(source)
    assert np.all(interval_low<=sp)and np.all(sp<=interval_high)
    np.savez_compressed(R/'SOURCE_PSD_PRECISION_PRIVATE.npz',source_ids=source_ids,source_original_psd=sp,PSD_interval_low=interval_low,PSD_interval_high=interval_high)
    write('SOURCE_ORDER_PRECISION_DEFINITION.json',{'interval_rule':C['source_order_rule'],'source_parents':len(source),'new_threshold_fit':False,'all_original_denominator_pairs_retained':True,'interval_width_quantiles':np.quantile(interval_high-interval_low,[0,.05,.5,.95,1]).tolist(),'not_physical_measurement_uncertainty':True})
    description={'UTC':utc(),'SOURCE_parents':len(source),'REAL_parents':len(real),'SOURCE_ids_SHA256':hashlib.sha256(np.ascontiguousarray(source_ids).tobytes()).hexdigest(),
        'source_counts':statistics['source_counts'].tolist(),'REAL_counts':statistics['REAL_counts'].tolist(),'region_categories':CATEGORIES,
        'SOURCE_global_region_mass':(statistics['source_counts']/len(source)).tolist(),'REAL_global_region_mass':(statistics['REAL_counts']/len(real)).tolist(),
        'parent_sampling':'32 independent uniform-with-replacement draws in each original region per update, with explicit original-region global mass',
        'global_importance_row_weight_SOURCE':(statistics['source_counts']/len(source)/C['batch_per_region']).tolist(),
        'global_importance_row_weight_REAL':(statistics['REAL_counts']/len(real)/C['batch_per_region']).tolist(),
        'source_h_one_fraction_Q3_Q4':statistics['source_h_one_fraction'].tolist(),'source_raw_valley_fraction_Q3_Q4':statistics['source_raw_valley_fraction'].tolist(),
        'noise_mean':statistics['noise_mean'].tolist(),'noise_scale':statistics['noise_scale'].tolist(),
        'source_Q_range':[float(sq.min()),float(sq.max())],'REAL_Q_range':[float(rq.min()),float(rq.max())],
        'source_original_range_excess_parents':int(np.any(np.abs(source)>.5,axis=1).sum()),'source_val_target_val_decoded':False,
        'new_temporal_normalization_is_TRAIN_only':True,'original_seven_loss_frozen_R61_TRAIN_normalization_reused':True,'normalization_not_response_or_teacher_fit':True,'all_source_parents_eligible':True,'h_or_PDG_network_input':False}
    write('TRAIN_POPULATION_AND_COORDINATES.json',description)
    xs=torch.as_tensor(source,dtype=torch.float64,device=dev);xr=torch.as_tensor(real,dtype=torch.float64,device=dev)
    mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    torch.manual_seed(C['model_seed']);torch.cuda.manual_seed_all(C['model_seed']);model=ResponseNetwork().to(dev)
    initial_hash=state_hash(model.state_dict());parameter_count=sum(p.numel() for p in model.parameters())
    assert initial_hash==C['expected_R61_initial_state_SHA'],('initialization_changed',initial_hash)
    write(ARM+'/INITIALIZATION.json',{'UTC':utc(),'seed':C['model_seed'],'state_tensor_SHA':initial_hash,'parameter_count':parameter_count,
        'loaded_previous_trained_response':False,'zero_output_head':True,'network_teacher':False,'static_conditional_displacement':True})
    objective=DistributionLoss(statistics,C,dev)
    optimizer=torch.optim.Adam(model.parameters(),lr=C['lr_initial'],betas=tuple(C['betas']),eps=1e-8,weight_decay=0)
    streams={k:np.random.Generator(np.random.PCG64(seed)) for k,seed in C['PCG64_seeds'].items()}
    initial_states={k:copy.deepcopy(v.bit_generator.state) for k,v in streams.items()}
    write(ARM+'/INITIAL_RANDOM_STATES.json',initial_states)
    hashes={k:bytes(32) for k in C['PCG64_seeds']}
    countsS=np.zeros(len(source),dtype='i8');countsR=np.zeros(len(real),dtype='i8');countsI=np.zeros(len(real),dtype='i8')
    loss_names=list(C['loss_weights']);columns=['total_loss',*loss_names,'learning_rate','gradient_norm','max_relative_q_f64','max_relative_q_f32',
        'SOURCE_global_range_projection_fraction','REAL_range_projection_fraction','SOURCE_global_projection_RMS_V','REAL_projection_RMS_V',
        'overall_PSD_W1','Q1_Q7_mean_PSD_W1','Q3_h_one_soft_valley','Q4_h_one_soft_valley']
    dense=[];trace=[];global_max_q64=global_max_q32=0.;allfinite=True
    order_columns=['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction'];order_dense=[]
    coverage_columns=['original_seven_total','coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced',*[name+'_coordinate_RMS'for name in NAMES]];coverage_dense=[]
    counter['response_fit_attempts']=1
    write('TRAIN_STARTED.json',{'UTC':utc(),'PID':os.getpid(),'counter':counter,'sole_terminal_update':C['updates'],'teacher_waveforms_constructed':0,
        'explicit_original_parent_global_weights':True,'selection':'only final update50000','progress_log_every_updates':1000,'recovery_cache_every_updates':5000})
    for step in range(1,C['updates']+1):
        si=np.concatenate([group[streams['source_rows'].integers(0,len(group),size=C['batch_per_region'])] for group in groupsS]).astype('i8')
        ri=np.concatenate([group[streams['REAL_reference_rows'].integers(0,len(group),size=C['batch_per_region'])] for group in groupsR]).astype('i8')
        ii=streams['REAL_identity_rows'].integers(0,len(real),size=C['identity_batch']).astype('i8')
        assert np.array_equal(sraw[si],np.repeat(CATEGORIES,C['batch_per_region'])) and np.array_equal(rraw[ri],np.repeat(CATEGORIES,C['batch_per_region']))
        epsilonS=streams['source_epsilon'].standard_normal((len(si),480)).astype('<f4');epsilonI=streams['identity_epsilon'].standard_normal((len(ii),480)).astype('<f4')
        nS,_=innovation(source[si],epsilonS,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        nI,_=innovation(real[ii],epsilonI,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        bx=torch.as_tensor(np.r_[sb[si],rb[ii]],device=dev,dtype=torch.int64);q=torch.as_tensor(np.r_[sq[si],rq[ii]],device=dev)
        hc=torch.as_tensor(np.r_[sh[si],rh[ii]],device=dev);x=torch.cat((xs[si],xr[ii]));n=torch.as_tensor(np.concatenate((nS,nI)),device=dev)
        directions=objective.draw_directions(streams['projection_directions'])
        temporal_directions=objective.draw_temporal_directions(streams['baseline_projection_directions'])
        rate=learning_rate(step)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        output,raw=endpoint(model,x,n,q,bx,hc,mt,st)
        loss,pieces,diagnostic=objective(output,raw,xr[ri],xr[ii],torch.as_tensor(rraw[ii],device=dev),
            torch.as_tensor(sp[si],device=dev),torch.as_tensor(source_h[si],device=dev),directions,
            torch.as_tensor(interval_low[si],device=dev),torch.as_tensor(interval_high[si],device=dev),temporal_directions)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite direct-distribution loss')
        eq64=((q_torch(output)-q).abs()/q).max();eq32=((q_torch(output.to(torch.float32).to(torch.float64))-q).abs()/q).max()
        if max(float(eq64.detach()),float(eq32.detach()))>1e-5:raise FloatingPointError('training constrained endpoint charge error')
        if ((output<-.5)|(output>.5)).any():raise FloatingPointError('training constrained endpoint outside voltage range')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),C['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();updates=step
        counter['new_response_updates']=updates;counter['SOURCE_training_rows']+=len(si);counter['REAL_reference_rows']+=len(ri);counter['REAL_identity_rows']+=len(ii)
        np.add.at(countsS,si,1);np.add.at(countsR,ri,1);np.add.at(countsI,ii,1)
        for name,values in [('source_rows',si.astype('<u2')),('REAL_reference_rows',ri.astype('<u2')),('REAL_identity_rows',ii.astype('<u2')),
                            ('source_epsilon',epsilonS),('identity_epsilon',epsilonI)]:hashes[name]=chain_update(hashes[name],values)
        for p in directions:hashes['projection_directions']=chain_update(hashes['projection_directions'],p)
        hashes['baseline_projection_directions']=chain_update(hashes['baseline_projection_directions'],temporal_directions)
        sN=len(si);active=((raw<-.5)|(raw>.5)).any(1);correction=(output-raw).square().mean(1)
        values=[loss.detach(),*[pieces[k].detach() for k in loss_names],loss.new_tensor(rate),norm.to(loss),eq64.detach(),eq32.detach(),
            (active[:sN].to(loss)*objective.sw).sum(),active[sN:].to(loss).mean(),(correction[:sN]*objective.sw).sum().sqrt().detach(),correction[sN:].mean().sqrt().detach(),
            diagnostic['overall_PSD_W1'],diagnostic['Q1_Q7_mean_PSD_W1'],diagnostic['Q3_h_one_soft_valley'],diagnostic['Q4_h_one_soft_valley']]
        row=torch.stack(values).cpu().numpy().astype('f8');dense.append(row)
        order_row=torch.stack([diagnostic[k]for k in order_columns]).cpu().numpy().astype('f8');order_dense.append(order_row)
        assert np.isfinite(order_row).all()and np.all(order_row>=0)
        assert abs(order_row[0]-order_row[1]-row[columns.index('source_order')])<=1e-10+1e-10*abs(order_row[0])
        global_max_q64=max(global_max_q64,float(row[columns.index('max_relative_q_f64')]));global_max_q32=max(global_max_q32,float(row[columns.index('max_relative_q_f32')]))
        coverage_row=torch.stack([diagnostic[k]for k in coverage_columns]).cpu().numpy().astype('f8');coverage_dense.append(coverage_row)
        assert np.isfinite(coverage_row).all() and np.all(coverage_row>=0)
        assert abs(coverage_row[:3].sum()-row[0])<=1e-10+1e-10*abs(row[0])
        if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes'] or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:raise MemoryError('GPU allocated/reserved safety limit')
        if step%1000==0 or step in [1,C['updates']]:
            last=dict(zip(columns,row.tolist()));mean=dict(zip(columns,np.asarray(dense[-min(len(dense),1000):]).mean(0).tolist()))
            record={'UTC':utc(),'PID':os.getpid(),'arm':ARM,'global_step':step,'elapsed_s':time.monotonic()-START,'last_update':last,'last_up_to1000_mean':mean,
                'max_path_output_q_residual_f64':global_max_q64,'max_output_q_residual_f32':global_max_q32,'unique_source_parents_seen':int(np.count_nonzero(countsS)),
                'unique_REAL_reference_parents_seen':int(np.count_nonzero(countsR)),'unique_REAL_identity_parents_seen':int(np.count_nonzero(countsI)),
                'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'counter':dict(counter)}
            record['precision_order_last_up_to1000_mean']=dict(zip(order_columns,np.asarray(order_dense[-min(len(order_dense),1000):]).mean(0).tolist()))
            record['coverage_last_up_to1000_mean']=dict(zip(coverage_columns,np.asarray(coverage_dense[-min(len(coverage_dense),1000):]).mean(0).tolist()))
            trace.append(record);write(ARM+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',record);print(json.dumps(record),flush=True)
        if step%5000==0:
            # Temporary, single rolling cache for same-run recovery; never selected as a model endpoint.
            checkpoint={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'counter':dict(counter),
                'PCG64_states':{k:copy.deepcopy(v.bit_generator.state) for k,v in streams.items()},'stream_chains':hashes,
                'SOURCE_exposure_counts':countsS,'REAL_reference_exposure_counts':countsR,'REAL_identity_exposure_counts':countsI,
                'coverage_columns':coverage_columns,'coverage_values':np.asarray(coverage_dense),'loss_columns':columns,'dense_losses':np.asarray(dense),'precision_order_columns':order_columns,'precision_order_values':np.asarray(order_dense),'initial_state_SHA':initial_hash,'torch_CPU_RNG':torch.get_rng_state(),'torch_CUDA_RNG':torch.cuda.get_rng_state()}
            temp=R/'cache/RECOVERY_SAME_RUN.tmp.pt';target=R/'cache/RECOVERY_SAME_RUN.pt';torch.save(checkpoint,temp);os.replace(temp,target)
    assert updates==50000 and objective.direction_calls==50000 and objective.temporal_direction_calls==50000 and all(torch.isfinite(p).all() for p in model.parameters())
    assert np.all(countsS>0) and np.all(countsR>0) and np.all(countsI>0)
    assert countsS.sum()==counter['SOURCE_training_rows'] and countsR.sum()==counter['REAL_reference_rows'] and countsI.sum()==counter['REAL_identity_rows']
    assert {k:hashes[k].hex()for k in C['expected_R61_stream_hashes']}==C['expected_R61_stream_hashes'],('R61_stream_mismatch',hashes)
    final_states={k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()}
    assert {k:final_states[k]for k in C['expected_R61_stream_hashes']}==json.loads(Path(C['input_paths']['R61_final_random_states']).read_text('utf8'))
    with np.load(C['input_paths']['R61_parent_exposures'],allow_pickle=False)as z:
        for key,value in [('source_ids',source_ids),('source_counts',countsS),('REAL_reference_counts',countsR),('REAL_identity_counts',countsI),('source_bin0',sraw),('REAL_bin0',rraw)]:assert np.array_equal(z[key],value),key
    np.savez_compressed(D/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',columns=np.asarray(order_columns),values=np.asarray(order_dense,dtype='f8'))
    np.savez_compressed(D/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.asarray(coverage_columns),values=np.asarray(coverage_dense,dtype='f8'))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state()},D/'FINAL_TORCH_RNG.pt')
    np.savez_compressed(D/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'),columns=np.asarray(columns))
    np.savez_compressed(D/'PARENT_EXPOSURES_PRIVATE.npz',source_ids=source_ids,source_counts=countsS,REAL_reference_counts=countsR,REAL_identity_counts=countsI,
        source_bin0=sraw,REAL_bin0=rraw,source_h=source_h,source_original_psd=sp,source_original_q=sq)
    final_states={k:copy.deepcopy(v.bit_generator.state) for k,v in streams.items()};write(ARM+'/FINAL_RANDOM_STATES.json',final_states)
    receipt={'PID':os.getpid(),'UTC':utc(),'arm':ARM,'updates':updates,'elapsed_s':time.monotonic()-START,'counter':counter,'selection':'sole final50000',
        'source_parents':len(source),'REAL_parents':len(real),'actual_unique_SOURCE_REAL_reference_REAL_identity':[int(np.count_nonzero(a)) for a in [countsS,countsR,countsI]],
        'SOURCE_parent_exposure_min_max':[int(countsS.min()),int(countsS.max())],'REAL_reference_exposure_min_max':[int(countsR.min()),int(countsR.max())],
        'REAL_identity_exposure_min_max':[int(countsI.min()),int(countsI.max())],'source_global_measure':'original equal parent weights through frozen region importance correction',
        'realized_exposure_counts_are_not_uniform_claim':True,'model_seed':C['model_seed'],'initial_state_SHA':initial_hash,'final_state_SHA':state_hash(model.state_dict()),
        'stream_hashes':{k:v.hex() for k,v in hashes.items()},'stream_hash_algorithm':'SHA256(previous32byte_digest + actual_consumed_little_endian_array_bytes); initial32 zero bytes; four directions sequentially per update in V,U,DU,D2U order',
        'PCG64_seeds':C['PCG64_seeds'],'stream_reconstruction':{'SOURCE':'ten category-order integer draws,32 each; source index bytes uint16','REAL_reference':'ten category-order integer draws,32 each; REAL index bytes uint16',
            'REAL_identity':'64 uniform integer draws from full REAL; uint16','SOURCE_white':'standard_normal(320,480) defaultfloat64 thenfloat32','REAL_white':'standard_normal(64,480) defaultfloat64 thenfloat32',
            'directions':'standard_normal((d,64)) float64; unit columns normalized float64; castfloat32; d=480,480,479,478','baseline_directions':'independent seventh stream; standard_normal((31,64)) float64; normalize columns float64; castfloat32'},
        'new_response_training_runs':1,'teacher_waveforms_constructed':0,'R58_law_or_atoms_training_target':False,'h_PDG_network_inputs':False,
        'network_parameter_count':parameter_count,'max_output_relative_q_error_f64':global_max_q64,'max_output_relative_q_error_f32':global_max_q32,
        'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
        'CPU_threads':C['CPU_threads'],'source_val_target_val_decoded':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'compute_budget_enforced':False,'model_goal_complete':False,
        'objective_extension_integrity':{'same_R61_initial_state':True,'all_six_consumed_stream_hashes_equal_R61':True,'all_parent_exposures_equal_R61':True,'original_six_final_random_states_equal_R61':True,'seventh_direction_stream_recorded_independently':True,'frozen_R61_distribution_statistics_reused':True,'sole_change':C['sole_scientific_change']},
        'source_order_rule':C['source_order_rule'],'precision_order_columns':order_columns,'precision_order_trace_rows':len(order_dense),'coverage_columns':coverage_columns,'coverage_trace_rows':len(coverage_dense),'temporal_direction_calls':objective.temporal_direction_calls,'new_temporal_statistics_saved':True}
    write(ARM+'/TRAINING_RECEIPT.json',receipt)
    print(json.dumps({'training_terminal':{'updates':updates,'elapsed_s':receipt['elapsed_s'],'model_goal_complete':False}}),flush=True)
except BaseException as exc:
    write('TRAINING_FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'updates':updates,'counter':counter,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START})
    raise
