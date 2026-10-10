from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,copy,traceback
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import scipy
import torch
from network import ResponseNetwork
from noise_core import bins_for,continuous_logq,innovation
from source_flow_core import solve,endpoint,q_torch
from distribution_loss import DistributionLoss,CATEGORIES
from diagnostic_math import COMPONENTS,psd_np,rounding_psd_interval,decompose,gradient_geometry
from full_population import describe
from artificial_check import check

def utc():return datetime.now(timezone.utc).isoformat()
def clean(v):
    if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float)and not np.isfinite(v):return None
    return v
def write(n,v):(R/n).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def hash_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
counter={'response_training_runs':0,'response_updates':0,'HGB_fits':0,'density_fits':0,'teacher_waveforms':0,
 'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'full_SOURCE_rows_completed':0,'full_REAL_rows_completed':0,
 'gradient_batches_attempted':0,'gradient_batches_completed':0,'neural_forward_rows_completed':0,'neural_forward_calls_completed':0,
 'mathematical_Heun_field_rows':0,'autograd_calls_completed':0}
def progress(phase,**more):
    v={'UTC':utc(),'PID':os.getpid(),'phase':phase,'counter':counter,'elapsed_s':time.monotonic()-START,**more}
    write('DIAGNOSTIC_PROGRESS.json',v);print(json.dumps(clean(v)),flush=True)
def gpu_guard():
    if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes']or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:raise MemoryError('registered GPU safety limit')

try:
    assert C['gradient_batches']==32 and C['gradient_components']==17 and C['planned_total_network_rows']==39928
    assert C['response_training_runs']==C['response_updates']==C['HGB_fits']==C['density_fits']==C['DEV_read']==C['EVAL_read']==0
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(C['CPU_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    binding=json.loads((R/'INPUT_BINDING.json').read_text('utf8'))
    for k,path in C['input_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==binding['inputs'][k]['sha256'],k
    for name,record in binding['frozen_R59_helpers'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==record['sha256'],name
    review=json.loads((R/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
    for record in review['files']:assert hashlib.sha256((R/record['name']).read_bytes()).hexdigest()==record['sha256'],record['name']
    assert torch.cuda.is_available();dev=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    env={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(dev),'GPU':torch.cuda.get_device_name(0),
      'PID':os.getpid(),'CPU_threads':torch.get_num_threads(),'CPU_affinity':sorted(os.sched_getaffinity(0)),'cwd':str(R),'UTC':utc()}
    assert all(env[k]==C['expected_R33_environment'][k]for k in ['python','numpy','torch','scipy','device','GPU'])
    write('ENVIRONMENT.json',env)
    write('ARTIFICIAL_CHECK_SERVER.json',check(C,'cpu'));write('ARTIFICIAL_CHECK_CUDA.json',check(C,dev));gpu_guard()
    old=json.loads(Path(C['input_paths']['frozen_training_config']).read_text('utf8'))
    train_receipt=json.loads(Path(C['input_paths']['frozen_training_receipt']).read_text('utf8'))
    assert old['batch_per_region']==C['batch_per_region']==32 and old['identity_batch']==C['identity_batch']==64
    assert train_receipt['updates']==50000 and train_receipt['final_state_SHA']==binding['frozen_weight_tensor_SHA']
    with np.load(C['input_paths']['TRAIN_path'],allow_pickle=False)as z:
        source=z['source'].astype('f4');real=z['target'].astype('f4');source_ids=z['source_ids'].copy();h=z['h'].copy();saved_s=z['source_bin'].copy();saved_r=z['target_bin'].copy()
    assert source.shape==(6593,480)and real.shape==(21047,480)and hash_array(source_ids)==binding['source_parent_ID_SHA']
    with np.load(C['input_paths']['constants_path'],allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    with np.load(C['input_paths']['noise_calibration_path'],allow_pickle=False)as z:noise={k:z[k].copy()for k in z.files}
    with np.load(C['input_paths']['distribution_statistics'],allow_pickle=False)as z:stats={k:z[k].copy()for k in z.files}
    sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
    assert np.array_equal(sraw,saved_s)and np.array_equal(rraw,saved_r)
    assert np.array_equal(stats['source_counts'],[(sraw==b).sum()for b in CATEGORIES])and np.array_equal(stats['REAL_counts'],[(rraw==b).sum()for b in CATEGORIES])
    state=torch.load(C['input_paths']['frozen_weight'],map_location='cpu',weights_only=True);initial_weight_hash=state_hash(state)
    assert initial_weight_hash==binding['frozen_weight_tensor_SHA']
    model=ResponseNetwork().to(dev);model.load_state_dict(state,strict=True);del state
    params=list(model.parameters());assert sum(p.numel()for p in params)==train_receipt['network_parameter_count']
    write('PARAMETER_ORDER.json',{'names_shapes_sizes':[{'name':n,'shape':list(p.shape),'elements':p.numel()}for n,p in model.named_parameters()],
      'total_elements':sum(p.numel()for p in params),'all_trainable_parameters_included':True})
    obj=DistributionLoss(stats,old,dev)
    streams={k:np.random.Generator(np.random.PCG64(seed))for k,seed in C['PCG64_seeds'].items()}
    initial_states={k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()};chains={k:bytes(32)for k in streams}
    def consume(name,value):chains[name]=hashlib.sha256(chains[name]+np.ascontiguousarray(value).tobytes()).digest()
    write('INITIAL_RANDOM_STATES.json',initial_states)
    full_eps_s=streams['full_SOURCE_epsilon'].standard_normal(source.shape).astype('<f4');consume('full_SOURCE_epsilon',full_eps_s)
    full_eps_r=streams['full_REAL_epsilon'].standard_normal(real.shape).astype('<f4');consume('full_REAL_epsilon',full_eps_r)
    directions=obj.draw_directions(streams['full_feature_directions'])
    for a in directions:consume('full_feature_directions',a)
    model.eval();progress('full_SOURCE_forward_started',attempted_rows=len(source))
    outS,metaS,solveS=solve(model,source,full_eps_s,mu,scale,edges,noise,True,steps=128,chunk=64)
    counter['full_SOURCE_rows_completed']=len(source);counter['neural_forward_rows_completed']+=len(source)
    counter['neural_forward_calls_completed']+=solveS['actual_network_calls'];counter['mathematical_Heun_field_rows']+=solveS['field_forward_rows'];gpu_guard()
    progress('full_REAL_forward_started',attempted_rows=len(real))
    outR,metaR,solveR=solve(model,real,full_eps_r,mu,scale,edges,noise,True,steps=128,chunk=64)
    counter['full_REAL_rows_completed']=len(real);counter['neural_forward_rows_completed']+=len(real)
    counter['neural_forward_calls_completed']+=solveR['actual_network_calls'];counter['mathematical_Heun_field_rows']+=solveR['field_forward_rows'];gpu_guard()
    progress('full_population_description_started')
    full,private=describe(source,real,source_ids,h,sraw,rraw,outS,outR,metaS,metaR,stats,edges,directions)
    full['solvers']={'SOURCE':solveS,'same_treated_REAL':solveR};write('FULL_TRAIN_SUMMARY.json',full)
    np.savez_compressed(R/'FULL_TRAIN_DIAGNOSTICS_PRIVATE.npz',**private)
    sp=private['source_original_psd'];low=private['source_PSD_interval_low'];high=private['source_PSD_interval_high']
    del full_eps_s,full_eps_r,outS,outR,metaS,metaR,private
    groupsS=[np.flatnonzero(sraw==b)for b in CATEGORIES];groupsR=[np.flatnonzero(rraw==b)for b in CATEGORIES]
    hcS=continuous_logq(sq,sb,edges);hcR=continuous_logq(rq,rb,edges)
    xs=torch.as_tensor(source,dtype=torch.float64,device=dev);xr=torch.as_tensor(real,dtype=torch.float64,device=dev)
    mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    records=[];grams=[];values_all=[];means=[];rms_all=[];raw_norms=[];indicesS=[];indicesR=[];indicesI=[]
    model.train()
    for batch in range(32):
        counter['gradient_batches_attempted']=batch+1
        si=np.concatenate([g[streams['batch_SOURCE_rows'].integers(0,len(g),32)]for g in groupsS]).astype('i8')
        ri=np.concatenate([g[streams['batch_REAL_reference_rows'].integers(0,len(g),32)]for g in groupsR]).astype('i8')
        ii=streams['batch_REAL_identity_rows'].integers(0,len(real),64).astype('i8')
        for name,a in [('batch_SOURCE_rows',si),('batch_REAL_reference_rows',ri),('batch_REAL_identity_rows',ii)]:consume(name,a.astype('<u2'))
        assert np.array_equal(sraw[si],np.repeat(CATEGORIES,32))and np.array_equal(rraw[ri],np.repeat(CATEGORIES,32))
        es=streams['batch_SOURCE_epsilon'].standard_normal((320,480)).astype('<f4');ei=streams['batch_REAL_identity_epsilon'].standard_normal((64,480)).astype('<f4')
        consume('batch_SOURCE_epsilon',es);consume('batch_REAL_identity_epsilon',ei)
        nS,_=innovation(source[si],es,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        nI,_=innovation(real[ii],ei,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        dirs=obj.draw_directions(streams['batch_projection_directions'])
        for a in dirs:consume('batch_projection_directions',a)
        bx=torch.as_tensor(np.r_[sb[si],rb[ii]],device=dev,dtype=torch.int64);q=torch.as_tensor(np.r_[sq[si],rq[ii]],device=dev)
        hc=torch.as_tensor(np.r_[hcS[si],hcR[ii]],device=dev);x=torch.cat((xs[si],xr[ii]));n=torch.as_tensor(np.concatenate((nS,nI)),device=dev)
        output,raw=endpoint(model,x,n,q,bx,hc,mt,st)
        counter['neural_forward_rows_completed']+=384;counter['neural_forward_calls_completed']+=1
        total,pieces,diagnostics,fractions=decompose(obj,output,raw,xr[ri],xr[ii],torch.as_tensor(rraw[ii],device=dev),
          torch.as_tensor(sp[si],device=dev),torch.as_tensor(h[si],device=dev),dirs,torch.as_tensor(low[si],device=dev),torch.as_tensor(high[si],device=dev))
        values=np.asarray([float(pieces[name].detach())for name in COMPONENTS]);assert np.isfinite(values).all()
        assert abs(float(total.detach())-values[:7].sum())<=C['loss_value_atol']+C['loss_value_rtol']*abs(float(total.detach()))
        vectors=[];batch_mean=[];batch_rms=[];batch_rawnorm=[]
        for j,name in enumerate(COMPONENTS):
            gradients=torch.autograd.grad(pieces[name],[*params,raw],retain_graph=j<len(COMPONENTS)-1,allow_unused=True)
            counter['autograd_calls_completed']+=1
            vector=np.concatenate([np.zeros(p.numel(),dtype='f4')if g is None else g.detach().cpu().numpy().reshape(-1)for p,g in zip(params,gradients[:-1])])
            raw_gradient=np.zeros((384,480))if gradients[-1]is None else gradients[-1].detach().cpu().numpy()
            assert np.isfinite(vector).all()and np.isfinite(raw_gradient).all()
            vectors.append(vector);batch_mean.append([raw_gradient[:320].mean(0),raw_gradient[320:].mean(0)])
            batch_rms.append([np.sqrt(np.mean(raw_gradient[:320]**2,0)),np.sqrt(np.mean(raw_gradient[320:]**2,0))])
            batch_rawnorm.append([np.linalg.norm(raw_gradient[:320]),np.linalg.norm(raw_gradient[320:])])
            del gradients,raw_gradient
        geometry=gradient_geometry(vectors,values,C)
        assert all(v['pass']for v in geometry['checks'].values()),geometry['checks']
        m=geometry['marker_inputs'];threshold=C['marker_cosine_below']
        less=lambda v:v is not None and v<threshold
        at_least=lambda v,t:v is not None and v>=t
        flags={
          'roundoff_order_conflict':fractions['roundoff_compatible_fraction_of_nonzero_mass']>=.5 and at_least(m['roundoff_norm_fraction'],.5)and less(m['roundoff_response_cosine']),
          'outside_full_window_conflict':at_least(m['outside_norm_fraction'],.5)and less(m['outside_inside_cosine']),
          'valley_PSD_conflict':at_least(m['valley_to_PSD_norm_ratio'],.5)and less(m['valley_PSD_cosine'])}
        eq=float(((q_torch(output.detach().to(torch.float32).to(torch.float64))-q).abs()/q).max())
        assert eq<=1e-5 and torch.isfinite(output).all()and not((output<-.5)|(output>.5)).any()
        record={'batch':batch+1,'losses':dict(zip(COMPONENTS,values.tolist())),'pair_masses':fractions,'marker_inputs':m,'flags':flags,
          'parameter_gradient_norms':dict(zip(COMPONENTS,geometry['norm'].tolist())),
          'total_gradient_norm':geometry['total_norm'],'component_dot_with_total':dict(zip(COMPONENTS,geometry['dot_with_total'].tolist())),
          'component_alignment_with_total':dict(zip(COMPONENTS,geometry['alignment_with_total'].tolist())),
          'decomposition_checks':geometry['checks'],'max_preADC_relative_q_error':eq,
          'unique_SOURCE_rows':int(len(np.unique(si))),'unique_REAL_reference_rows':int(len(np.unique(ri))),'unique_REAL_identity_rows':int(len(np.unique(ii)))}
        records.append(record);grams.append(geometry['gram']);values_all.append(values);means.append(batch_mean);rms_all.append(batch_rms);raw_norms.append(batch_rawnorm)
        indicesS.append(si.astype('u2'));indicesR.append(ri.astype('u2'));indicesI.append(ii.astype('u2'))
        counter['gradient_batches_completed']=batch+1;gpu_guard()
        write('GRADIENT_BATCH_RECORDS.json',records)
        progress('gradient_batch_complete',batch=batch+1,flags=flags)
        del output,raw,total,pieces,diagnostics,vectors,geometry,x,n
    assert counter['neural_forward_rows_completed']==39928 and counter['autograd_calls_completed']==544 and counter['gradient_batches_completed']==32
    final_hash=state_hash(model.state_dict());assert final_hash==initial_weight_hash
    np.savez_compressed(R/'GRADIENT_DIAGNOSTICS_PRIVATE.npz',component_names=np.asarray(COMPONENTS),loss_values=np.asarray(values_all),parameter_Gram=np.asarray(grams),
      raw_gradient_position_mean=np.asarray(means),raw_gradient_position_RMS=np.asarray(rms_all),raw_gradient_domain_norm=np.asarray(raw_norms),
      SOURCE_indices=np.asarray(indicesS),REAL_reference_indices=np.asarray(indicesR),REAL_identity_indices=np.asarray(indicesI))
    counts={k:sum(r['flags'][k]for r in records)for k in records[0]['flags']}
    markers={k:{'batches':v,'total_batches':32,'required_batches':24,'activated':v>=24}for k,v in counts.items()}
    with np.load(C['input_paths']['frozen_training_losses'],allow_pickle=False)as z:losses=z['values'].copy();columns=z['columns'].tolist()
    assert losses.shape==(50000,20)and np.isfinite(losses).all()
    gi=columns.index('gradient_norm');norm=losses[:,gi]
    trajectory={'source':'frozen R59 50000-update saved trace; no new updates','gradient_clip_norm':5.,'gradient_clipped_fraction':float(np.mean(norm>5)),
      'gradient_norm_quantiles':np.quantile(norm,[0,.05,.5,.95,1]).tolist(),'clip_factor_quantiles':np.quantile(np.minimum(1.,5/np.maximum(norm,1e-30)),[0,.05,.5,.95,1]).tolist(),
      'blocks_of_1000':[{'last_update':j+1000,**dict(zip(columns,losses[j:j+1000].mean(0).tolist())),'clip_fraction':float(np.mean(norm[j:j+1000]>5))}for j in range(0,50000,1000)]}
    write('SAVED_R59_TRAINING_TRACE_SUMMARY.json',trajectory)
    summary={'component_names':COMPONENTS,'batches':32,'diagnostic_markers':markers,'all_decompositions_pass':True,
      'mean_loss':dict(zip(COMPONENTS,np.asarray(values_all).mean(0).tolist())),
      'parameter_Gram_mean':np.asarray(grams).mean(0),'gradient_semantics':'Raw Euclidean gradients at one frozen model; not actual Adam update directions or causal proof.',
      'raw_waveform_gradient_semantics':'Derivative with respect to projection-before waveform; loss row weights already included; batch row mean/RMS is not physical waveform noise.',
      'optimizer_steps':0,'parent_samples_saved':True,'undefined_cosines_retained_as_null':True}
    write('GRADIENT_SUMMARY.json',summary)
    write('FINAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()})
    write('RANDOM_STREAM_RECEIPT.json',{'seeds':C['PCG64_seeds'],'chains':{k:v.hex()for k,v in chains.items()},
      'algorithm':'SHA256(previous32bytes + contiguous consumed bytes); initial32 zero bytes; index uint16, noise/directions little-endian float32; four directions sequential V,U,DU,D2U.',
      'full_PARENT_ORDER_preserved':True,'full_SOURCE_rows':6593,'full_REAL_rows':21047,'batch_draws_per_region_each_domain':32,'identity_draws':64,'batches':32,
      'direction_calls':obj.direction_calls,'full_stream_reconstruction_executed':False})
    assert obj.direction_calls==33
    result={'UTC':utc(),'PID':os.getpid(),'status':'fixed_frozen_TRAIN_diagnostic_complete','model_goal_complete':False,
      'original_A_F_changed':False,'numerical_decomposition_pass':True,'diagnostic_markers':markers,
      'full_TRAIN_PSD':{k:full[k]for k in ['overall_SOURCE_PSD_W1','common_Q1_Q7_mean_SOURCE_PSD_W1','overall_same_treated_REAL_PSD_W1','source_PSD_rank_rho']},
      'population_pair_order':full['source_order'],'frozen_weight_before_SHA':initial_weight_hash,'frozen_weight_after_SHA':final_hash,
      'counter':counter,'elapsed_s':time.monotonic()-START,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
      'scientific_scope':'TRAIN-only frozen diagnostic, no new candidate, no A-F acceptance determination.','teacher_waveforms_constructed':0,
      'next_model_training_authorized_by_this_result_alone':False}
    write('RESULT.json',result)
    from render_diagnostic import render
    figures=render(R);write('FIGURE_MANIFEST.json',{'figures':figures,'actual_root_visual_QA_pending':True})
    result.update(elapsed_s=time.monotonic()-START,figures=figures);write('RESULT.json',result)
    progress('registered_diagnostic_endpoint_complete',diagnostic_markers=markers)
except BaseException as e:
    write('DIAGNOSTIC_FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),
      'counter':counter,'counter_completed_rows_is_lower_bound_after_exception':True,'elapsed_s':time.monotonic()-START})
    raise
