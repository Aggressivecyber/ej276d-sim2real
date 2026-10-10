from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,copy,traceback
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import scipy
import torch
from network import ResponseNetwork
from noise_core import bins_for
from residual_math import clean,array_sha,canonical,draw_directions,REGIONS,summary
from residual_analysis import path_description,baseline_description,loss_description,psd_description
from traced_response import traced_solve
from artificial_check import check

def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(R/n).write_bytes((json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
counter={'response_training_runs':0,'response_updates':0,'HGB_fits':0,'density_fits':0,'teacher_waveforms':0,'autograd_calls':0,
 'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'full_SOURCE_rows_completed':0,'full_REAL_rows_completed':0,
 'neural_forward_rows_completed':0,'neural_forward_calls_completed':0,'mathematical_Heun_field_rows':0,'saved_endpoint_batches_completed':0}
def progress(phase,**more):
    value={'UTC':utc(),'PID':os.getpid(),'phase':phase,'counter':counter,'elapsed_s':time.monotonic()-START,**more}
    write('DIAGNOSTIC_PROGRESS.json',value);print(json.dumps(clean(value)),flush=True)
def gpu_guard():
    if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes']or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:raise MemoryError('registered GPU safety limit')

try:
    assert C['resampling_batches']==32 and C['batch_per_region']==32 and C['solver_steps']==128 and C['generation_chunk']==64
    assert C['projection_directions']==64 and C['covariance_reference_replicates']==32
    assert C['response_training_runs']==C['response_updates']==C['HGB_fits']==C['density_fits']==C['DEV_read']==C['EVAL_read']==C['CAL_decode']==0
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(C['CPU_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    binding=json.loads((R/'INPUT_BINDING.json').read_text('utf8'))
    def verify_inputs():
        for key,path in C['input_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==binding['inputs'][key]['sha256'],key
    verify_inputs()
    for name,record in binding['frozen_R61_helpers'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==record['sha256'],name
    review=json.loads((R/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
    for record in review['files']:assert hashlib.sha256((R/record['name']).read_bytes()).hexdigest()==record['sha256'],record['name']
    assert torch.cuda.is_available();dev=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    env={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(dev),'GPU':torch.cuda.get_device_name(0),
     'PID':os.getpid(),'CPU_threads':torch.get_num_threads(),'CPU_affinity':sorted(os.sched_getaffinity(0)),'cwd':str(R),'UTC':utc()}
    assert all(env[k]==C['expected_R33_environment'][k]for k in ['python','numpy','torch','scipy','device','GPU'])
    write('ENVIRONMENT.json',env);progress('artificial_checks_before_TRAIN')
    write('ARTIFICIAL_CHECK_SERVER.json',check('cpu'));write('ARTIFICIAL_CHECK_CUDA.json',check(dev));gpu_guard()
    old=json.loads(Path(C['input_paths']['frozen_training_config']).read_text('utf8'))
    train_receipt=json.loads(Path(C['input_paths']['frozen_training_receipt']).read_text('utf8'))
    assert old['batch_per_region']==C['batch_per_region'] and old['projection_directions']==C['projection_directions']
    assert train_receipt['updates']==50000 and train_receipt['final_state_SHA']==binding['frozen_weight_tensor_SHA']
    with np.load(C['input_paths']['TRAIN_path'],allow_pickle=False)as z:
        source=z['source'].astype('f4');real=z['target'].astype('f4');source_ids=z['source_ids'].copy();h=z['h'].copy();saved_s=z['source_bin'].copy();saved_r=z['target_bin'].copy()
    assert source.shape==(C['SOURCE_N'],480)and real.shape==(C['REAL_N'],480)and array_sha(source_ids)==binding['source_parent_ID_SHA']
    write('DATA_ACCESS_RECEIPT.json',{'archive':C['input_paths']['TRAIN_path'],'decoded_fields':['source','target','source_ids','h','source_bin','target_bin'],
     'source_shape':list(source.shape),'REAL_shape':list(real.shape),'source_parent_ID_SHA':array_sha(source_ids),'other_data_archives_decoded':[],'DEV_read':0,'EVAL_read':0,'CAL_decode':0})
    with np.load(C['input_paths']['constants_path'],allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    with np.load(C['input_paths']['noise_calibration_path'],allow_pickle=False)as z:noise={k:z[k].copy()for k in z.files}
    with np.load(C['input_paths']['distribution_statistics'],allow_pickle=False)as z:stats={k:z[k].copy()for k in z.files}
    sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
    assert np.array_equal(sraw,saved_s)and np.array_equal(rraw,saved_r)
    assert np.array_equal(stats['source_counts'],[(sraw==b).sum()for b in REGIONS])and np.array_equal(stats['REAL_counts'],[(rraw==b).sum()for b in REGIONS])
    state=torch.load(C['input_paths']['frozen_weight'],map_location='cpu',weights_only=True);initial_weight_hash=state_hash(state)
    assert initial_weight_hash==binding['frozen_weight_tensor_SHA']
    model=ResponseNetwork().to(dev);model.load_state_dict(state,strict=True);del state;model.eval();model.requires_grad_(False)
    assert sum(p.numel()for p in model.parameters())==train_receipt['network_parameter_count']
    streams={k:np.random.Generator(np.random.PCG64(seed))for k,seed in C['PCG64_seeds'].items()}
    chains={k:bytes(32)for k in streams};consumed={k:{'calls':0,'elements':0,'bytes':0}for k in streams}
    def consume(name,value):
        arr=np.ascontiguousarray(value);chains[name]=hashlib.sha256(chains[name]+arr.tobytes()).digest()
        consumed[name]['calls']+=1;consumed[name]['elements']+=arr.size;consumed[name]['bytes']+=arr.nbytes
    write('INITIAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()})
    epsS=streams['full_SOURCE_epsilon'].standard_normal(source.shape).astype('<f4');consume('full_SOURCE_epsilon',epsS)
    epsR=streams['full_REAL_epsilon'].standard_normal(real.shape).astype('<f4');consume('full_REAL_epsilon',epsR)
    directions=draw_directions(streams['full_feature_directions'])
    for a in directions:consume('full_feature_directions',a)
    for k,v in C['expected_full_stream_hashes'].items():assert chains[k].hex()==v,k
    paths={};per_parent={};solver={}
    for name,x,eps in [('SOURCE',source,epsS),('REAL_control',real,epsR)]:
        progress(name+'_full_forward_started',attempted_rows=len(x))
        paths[name],per_parent[name],solver[name]=traced_solve(model,x,eps,mu,scale,edges,noise,64)
        counter['full_SOURCE_rows_completed'if name=='SOURCE'else'full_REAL_rows_completed']=len(x)
        counter['neural_forward_rows_completed']+=len(x);counter['neural_forward_calls_completed']+=solver[name]['actual_network_calls']
        counter['mathematical_Heun_field_rows']+=solver[name]['mathematical_Heun_field_rows'];gpu_guard()
        write('FORWARD_RECEIPT.json',solver)
    del epsS,epsR
    assert counter['neural_forward_rows_completed']==C['planned_total_network_rows']==27640
    assert counter['mathematical_Heun_field_rows']==7075840
    ref=canonical(real,False);ys=paths['SOURCE']['canonical_ADC'];yr=paths['REAL_control']['canonical_ADC']
    progress('complete_path_statistics_started')
    path,pa=path_description(paths,ref,sraw,rraw)
    write('PATH_SUMMARY.json',path);np.savez_compressed(R/'PATH_STATISTICS_PRIVATE.npz',**pa)
    del pa
    progress('baseline_joint_statistics_started')
    base,ba=baseline_description(ys,yr,ref,sraw,rraw,streams['covariance_reference_rows'],consume)
    write('BASELINE_JOINT_SUMMARY.json',base);np.savez_compressed(R/'BASELINE_JOINT_PRIVATE.npz',**ba)
    del ba
    progress('fixed_saved_endpoint_resampling_started')
    loss,la=loss_description(ys,yr,ref,sraw,rraw,stats,directions,streams,consume)
    write('LOSS_COVERAGE_SUMMARY.json',loss);np.savez_compressed(R/'LOSS_COVERAGE_PRIVATE.npz',**la,**{'directions_'+str(j):d for j,d in enumerate(directions)})
    counter['saved_endpoint_batches_completed']=32;del la
    ps,pp=psd_description(ys,yr,ref,source,source_ids,h,sraw,rraw,stats)
    write('PSD_SUMMARY.json',ps)
    for name,values in per_parent.items():
        for key,val in values.items():pp[name+'_'+key]=val
    np.savez_compressed(R/'PER_PARENT_PRIVATE.npz',**pp)
    projection={name:{'rows':len(p['range_active']),'range_active_N':int(p['range_active'].sum()),'rail_samples_total':int(p['rail_samples'].sum()),
     **{k:summary(p[k])for k in ['delta_RMS_V','projection_RMS_V','preADC_relative_Q_error_f32','preADC_relative_Q_error_f64']},
     'delta_boundaries_abs_mean_V':np.abs(p['delta_boundaries_V']).mean(0),'projection_boundaries_abs_mean_V':np.abs(p['projection_boundaries_V']).mean(0)}for name,p in per_parent.items()}
    write('PROJECTION_SUMMARY.json',projection)
    final_hash=state_hash(model.state_dict());assert final_hash==initial_weight_hash;verify_inputs()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    write('FINAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()})
    write('RANDOM_STREAM_RECEIPT.json',{'seeds':C['PCG64_seeds'],'chains':{k:v.hex()for k,v in chains.items()},'consumed':consumed,
     'full_three_chains_match_frozen_R60':True,'algorithm':'SHA256(previous32bytes + contiguous consumed bytes); initial32 zero bytes; index uint16, noise/directions little-endian float32; four directions sequential V,U,DU,D2U.',
     'full_PARENT_ORDER_preserved':True,'full_SOURCE_rows':6593,'full_REAL_rows':21047,'batch_draws_per_region_each_domain':32,'batches':32,'direction_calls':1})
    result={'UTC':utc(),'PID':os.getpid(),'status':'fixed_frozen_TRAIN_residual_diagnostic_complete','model_goal_complete':False,'original_A_F_changed':False,
     'diagnostic_markers':{'neural_boundary_increase':path['neural_boundary_increase'],'baseline_joint_residual':base['baseline_joint_residual']},
     'counter':counter,'frozen_weight_before_SHA':initial_weight_hash,'frozen_weight_after_SHA':final_hash,'inputs_hashes_unchanged_after':True,
     'full_SOURCE_PSD_W1':ps['overall_SOURCE_W1'],'full_SOURCE_common_Q1_Q7_mean_W1':ps['common_Q1_Q7_mean_W1'],
     'elapsed_s':time.monotonic()-START,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
     'artificial_network_rows_are_separate_from_project_model_counts':True,'scientific_scope':'Frozen R61 TRAIN path description and empirical reference. No new candidate or acceptance determination.'}
    write('RESULT.json',result);progress('rendering_registered_figures')
    from render_diagnostic import render
    figures=render(R);write('FIGURE_MANIFEST.json',{'figures':figures,'actual_root_visual_QA_pending':True})
    result.update(elapsed_s=time.monotonic()-START,figures=figures);write('RESULT.json',result)
    progress('registered_diagnostic_endpoint_complete',diagnostic_markers=result['diagnostic_markers'])
except BaseException as e:
    write('DIAGNOSTIC_FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'counter':counter,
     'counter_completed_rows_is_lower_bound_after_exception':True,'elapsed_s':time.monotonic()-START})
    raise
