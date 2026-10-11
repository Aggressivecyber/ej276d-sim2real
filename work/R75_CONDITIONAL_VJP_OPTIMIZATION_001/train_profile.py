"""Registered complete-parent waveform response: exact endpoints and complete VJP."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,math,os,sys,time,traceback
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));START=time.monotonic()
def write(name,value):
    p=R/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8');t.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def progress(phase,**kw):
    d=dict(UTC=utc(),phase=phase,process_elapsed_s=time.monotonic()-START,**kw)
    write('PROGRESS.json',d);print(json.dumps(d),flush=True)
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    import numpy as np
    import torch
    from public_charge_head import PublicConditionalChargeHead,window_charge,target_charge_condition
    from physical_operators import canonical_REAL_torch
    from waveform_response import ShapeResidualNetwork,prepare_static,slice_prepared,forward_prepared
    from precision_order_reference import rounding_psd_interval,psd_np
    from combined_loss import FullWaveformLoss
    from full_waveform_loss import FullWaveformLoss as OriginalFullWaveformLoss,UniformW1Cache as OriginalW1Cache
    from check_exact_uniform_w1 import run_checks as check_exact_w1
    from check_combined_loss import run as check_combined

    from artificial_checks import run_checks
    from postcheck import postcheck
    assert sorted(os.sched_getaffinity(0))==C['exact_CPU_affinity']
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,GPU=torch.cuda.get_device_name(0),CPU_affinity=sorted(os.sched_getaffinity(0)))
    for key in ['python','numpy','torch','GPU']:assert env[key]==C['expected_R33_environment'][key],(key,env[key])
    write('ENVIRONMENT.json',env)
    for key,row in json.loads((R/'INPUT_BINDING.json').read_text())['inputs'].items():assert sha(row['path'])==row['sha256'],key
    checks=run_checks(C['artificial_seed']);assert checks['pass_all']
    checks['exact_W1']={device:check_exact_w1(device,original_source=R/'full_waveform_loss.py')for device in ['cpu','cuda']}
    checks['combined']=check_combined(C['artificial_seed'],require_cuda=True)
    assert all(value['pass_all']for value in checks['exact_W1'].values())and checks['combined']['pass_all']
    write('ARTIFICIAL_CHECKS.json',checks)

    progress('artificial_CPU_CUDA_checks_passed_before_scientific_arrays')
    with np.load(C['input_paths']['TRAIN_path'],allow_pickle=False)as z:
        source=z['source'].astype('f4');real=z['target'].astype('f4');source_ids=z['source_ids'].copy();h=z['h'].copy()
        saved_source_bins=z['source_bin'].copy();saved_REAL_bins=z['target_bin'].copy();keys=list(z.files)
    nS,nR=len(source),len(real);N=nS+nR
    assert source.shape==(C['SOURCE_N'],480)and real.shape==(C['REAL_N'],480)
    assert len(np.unique(source_ids))==nS
    with np.load(C['input_paths']['constants_path'],allow_pickle=False)as z:mu,scale,edges=[z[k].copy()for k in ['mu_unit_q','scale_unit_q','edges_logQ']]
    with np.load(C['input_paths']['noise_calibration_path'],allow_pickle=False)as z:p480=z['p480'].copy();tau2=float(z['tau2'])
    valley=json.loads(Path(C['input_paths']['valley_definition']).read_text())
    qstate=torch.load(C['input_paths']['Q_head_weights'],map_location='cpu',weights_only=True)
    head=PublicConditionalChargeHead(qstate['feature_center'],qstate['feature_scale']).to('cuda').double()
    head.load_state_dict(qstate);head.eval();head.requires_grad_(False)
    x=torch.as_tensor(np.concatenate([source,real]),device='cuda',dtype=torch.float64)
    static_tensors=[torch.as_tensor(v,device='cuda',dtype=torch.float64)for v in [mu,scale,edges,p480]]
    with torch.no_grad():prepared=prepare_static(head,x,*static_tensors,tau2)
    feature=prepared['features482'][nS:];center=feature.mean(0);std=feature.std(0,unbiased=False);fscale=std.clamp_min(C['input_scale_floor'])
    torch.manual_seed(C['model_seed']);torch.cuda.manual_seed_all(C['model_seed'])
    model=ShapeResidualNetwork(center,fscale,hidden=C['hidden']).to('cuda')
    assert all(p.dtype==torch.float32 for p in model.parameters())
    realcanon=canonical_REAL_torch(x[nS:]);source_psd=psd_np(source);low,high=rounding_psd_interval(source)
    tensor=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64)
    original_bin=prepared['original_bin_raw'][:nS]
    objective=FullWaveformLoss(realcanon,tensor(edges),tensor(source_psd),tensor(low),tensor(high),original_bin,
        tensor(h),tensor([[valley['bins'][b]['low'],valley['bins'][b]['high']]for b in [2,3]]),C)
    reference_objective=copy.copy(objective);reference_objective.__class__=OriginalFullWaveformLoss
    reference_objective.w1=OriginalW1Cache(objective.device)
    initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};init_sha=state_hash(initial)

    torch.save(initial,R/'INITIAL_SHAPE.pt');torch.save(qstate,R/'FROZEN_Q_HEAD.pt')
    np.savez_compressed(R/'SHAPE_STATISTICS.npz',feature_center=center.cpu().numpy(),feature_scale=fscale.cpu().numpy(),unfloored_feature_scale=std.cpu().numpy(),
        mu=mu,scale=scale,edges=edges,p480=p480,tau2=tau2)
    write('LOSS_STATISTICS.json',objective.stats_serializable())
    np.savez_compressed(R/'PARENT_MANIFEST_PRIVATE.npz',source_ids=source_ids,source_h=h,source_psd=source_psd,
        source_PSD_interval_low=low,source_PSD_interval_high=high,SOURCE_original_row=np.arange(nS),REAL_original_TRAIN_row=np.arange(nR),
        SOURCE_original_saved_bin=saved_source_bins,REAL_original_saved_bin=saved_REAL_bins,
        SOURCE_original_Q=prepared['Q0'][:nS].cpu().numpy(),REAL_original_Q=prepared['Q0'][nS:].cpu().numpy(),
        SOURCE_Qstar=prepared['Qstar'][:nS].cpu().numpy(),REAL_Qstar=prepared['Qstar'][nS:].cpu().numpy(),
        SOURCE_original_bin=original_bin.cpu().numpy(),SOURCE_original_weight=np.full(nS,1/nS),REAL_original_weight=np.full(nR,1/nR))
    write('DATA_RECEIPT.json',dict(UTC=utc(),TRAIN_keys=keys,SOURCE_N=nS,REAL_N=nR,source_ID_SHA256=hashlib.sha256(source_ids.tobytes()).hexdigest(),
      per_update_SOURCE_rows=nS,per_update_REAL_rows=nR,all_loss_terms_full_parents=True,current_canonical_bins=True,
      original_bins_for_source_only=True,paired_REAL_original_rows=True,REAL_reference_unmodified=True,
      shape_parameters=sum(p.numel()for p in model.parameters()),shape_architecture=C['shape_network'],shape_dtype='float32',
      physical_dtype='float64',Q_head_frozen=True,Q_head_weights_SHA256=sha(C['input_paths']['Q_head_weights']),
      initial_state_SHA256=init_sha,feature_scale_floored_N=int((std<C['input_scale_floor']).sum()),heldout_arrays_read=False))
    source_rng=torch.Generator(device='cuda').manual_seed(C['training_SOURCE_epsilon_seed'])
    real_rng=torch.Generator(device='cuda').manual_seed(C['training_REAL_epsilon_seed'])
    directions_rng=np.random.Generator(np.random.PCG64(C['directions_seed']))
    temporal_rng=np.random.Generator(np.random.PCG64(C['temporal_directions_seed']))
    rngcpu=torch.get_rng_state().clone();rnggpu=[z.clone()for z in torch.cuda.get_rng_state_all()]
    rngs=source_rng.get_state().clone();rngr=real_rng.get_state().clone()
    direction_initial=copy.deepcopy(directions_rng.bit_generator.state);temporal_initial=copy.deepcopy(temporal_rng.bit_generator.state)
    params=list(model.parameters());chunk=C['neural_replay_chunk']
    names=['full_window','coordinate_RMS','PSD','temporal_joint','noise','source_order','source_valley','REAL_identity','box_correction']
    def optimizer_new():return torch.optim.Adam(model.parameters(),lr=C['lr_initial'],betas=tuple(C['betas']))
    def rate(step):
        if step<=C['warmup_updates']:return C['lr_initial']+(C['lr']-C['lr_initial'])*step/C['warmup_updates']
        r=(step-C['warmup_updates'])/(C['updates']-C['warmup_updates'])
        return C['lr_final']+(C['lr']-C['lr_final'])*.5*(1+math.cos(math.pi*r))
    def direction(rng,dimension):
        a=rng.standard_normal((dimension,C['projection_directions']));a/=np.linalg.norm(a,axis=0,keepdims=True)
        return torch.as_tensor(a.astype('f4'),device='cuda')
    def update(opt,step,use_original=False,capture=False):
        torch.cuda.synchronize();mark_time=time.perf_counter();profile={}
        def tick(label):
            nonlocal mark_time
            torch.cuda.synchronize();now=time.perf_counter();profile[label]=now-mark_time;mark_time=now
        lr=rate(step)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        eps=torch.cat([torch.randn((nS,480),dtype=torch.float32,device='cuda',generator=source_rng),
                       torch.randn((nR,480),dtype=torch.float32,device='cuda',generator=real_rng)])
        ds=[direction(directions_rng,d)for d in [480,480,479,478]];dt=direction(temporal_rng,31)
        tick('RNG_and_learning_rate')
        first=[[],[],[]];max_q64=0.;max_q32=0.;projection_n=0
        with torch.no_grad():
            for start in range(0,N,chunk):
                pre,adc,raw,meta=forward_prepared(model,slice_prepared(prepared,start,min(start+chunk,N)),eps[start:start+chunk],straight_through=True)
                for j,v in enumerate([adc,pre,raw]):first[j].append(v.detach())
                max_q64=max(max_q64,float(meta['Q_relative_error_f64'].max()));max_q32=max(max_q32,float(meta['Q_relative_error_f32'].max()))
                projection_n+=int(meta['range_projection_active'].sum())
        saved=[torch.cat(v)for v in first];del first
        tick('neural_physical_firstpass')
        # Six independent leaves avoid retaining the neural graph for full-population losses.
        leaves=[saved[j][sl].detach().requires_grad_(True)for j in range(3)for sl in [slice(0,nS),slice(nS,None)]]
        accum=[torch.zeros_like(v,dtype=torch.float64)for v in leaves];values={}
        tick('endpoint_leaves_setup')
        if use_original:
            groups=(((name,),term,{name:term},diag)for name,term,diag in reference_objective.iter_terms(*leaves,ds,dt))
        else:groups=objective.iter_gradient_groups(*leaves,ds,dt)
        for group_names,term,terms,_ in groups:
            label='+'.join(group_names);tick('forward_'+label)
            assert all(name in names and name not in values for name in group_names)
            if not bool(torch.isfinite(term)):raise FloatingPointError('nonfinite loss '+str(group_names))
            gradient=torch.autograd.grad(term,leaves,allow_unused=True)
            tick('backward_'+label)
            for j,g in enumerate(gradient):
                if g is not None:accum[j].add_(g.detach().to(torch.float64))
            values.update({name:v.detach()for name,v in terms.items()});del gradient,term
            tick('accumulate_'+label)
        assert set(values)==set(names)
        all_gradient=[torch.cat([accum[2*j],accum[2*j+1]])for j in range(3)]
        del accum,leaves
        # Every row is replayed in the same forward chunks and stochastic draw.
        tick('endpoint_gradient_concatenation')
        parameter_gradient=[torch.zeros_like(p,dtype=torch.float64)for p in params]
        replay_chunks=0
        for start in range(0,N,chunk):
            end=min(start+chunk,N)
            pre,adc,raw,meta=forward_prepared(model,slice_prepared(prepared,start,end),eps[start:end],straight_through=True)
            outputs=[adc,pre,raw]
            if not all(torch.equal(v.detach(),saved[j][start:end])for j,v in enumerate(outputs)):
                raise FloatingPointError('same-update complete-parent endpoint replay not bitwise')
            gradients=torch.autograd.grad(outputs,params,grad_outputs=[g[start:end].to(v.dtype)for g,v in zip(all_gradient,outputs)])
            for acc,g in zip(parameter_gradient,gradients):acc.add_(g.to(torch.float64))
            replay_chunks+=1
        tick('complete_neural_parameter_VJP')
        for p,g in zip(params,parameter_gradient):p.grad=g.to(p.dtype)
        captured_gradient=torch.cat([p.grad.detach().flatten().to(torch.float64)for p in params]).cpu()if capture else None
        norm=torch.nn.utils.clip_grad_norm_(params,C['gradient_clip'],error_if_nonfinite=True);opt.step()
        if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>C['GPU_limit_bytes']:raise MemoryError('registered GPU quota exceeded')
        tick('clip_and_Adam')
        result=dict(profile=profile,total=sum(values.values()),**values,gradient_norm=norm.detach(),learning_rate=lr,
            max_relative_Qstar_error_f64=max_q64,max_relative_Qstar_error_f32=max_q32,range_projection_fraction=projection_n/N,
            firstpass_rows=N,replay_rows=N,replay_chunks=replay_chunks)
        if capture:result['capture']=dict(parameter_gradient=captured_gradient,after_parameters=torch.cat([p.detach().flatten().to(torch.float64)for p in params]).cpu())
        return result
    opt=optimizer_new();rows=[];warm=C['profile_warmup_updates'];timed=C['profile_timed_updates']
    progress('complete_parent_component_profile',main_updates=0)
    for i in range(1,warm+timed+1):
        torch.cuda.synchronize();start=time.perf_counter();row=update(opt,i);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        rows.append(dict(engineering_update=i,warmup=i<=warm,seconds=elapsed,components=row['profile']))
        if i%5==0:progress('complete_parent_component_profile',engineering_updates=i,main_updates=0,last_seconds=elapsed)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=False,with_stack=False)as profiler:
        operator_row=update(opt,warm+timed+1);torch.cuda.synchronize()
    operators=[]
    for event in profiler.key_averages(group_by_input_shape=True):
        operators.append(dict(name=event.key,count=event.count,input_shapes=event.input_shapes,
            self_cpu_us=event.self_cpu_time_total,cpu_total_us=event.cpu_time_total,
            self_device_us=getattr(event,'self_device_time_total',0.),device_total_us=getattr(event,'device_time_total',0.)))
    write('OPERATOR_PROFILE.json',dict(UTC=utc(),engineering_update=warm+timed+1,main_updates=0,record_shapes=True,profile_memory=False,
       additional_instrumentation_overhead_not_cost_forecast=True,CUDA_times_present=any(v['self_device_us']>0 for v in operators),operators=operators))
    del profiler,operator_row
    del opt,row
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);torch.set_rng_state(rngcpu);torch.cuda.set_rng_state_all(rnggpu)
    source_rng.set_state(rngs);real_rng.set_state(rngr);directions_rng.bit_generator.state=copy.deepcopy(direction_initial);temporal_rng.bit_generator.state=copy.deepcopy(temporal_initial)
    opt=optimizer_new();assert state_hash(model.state_dict())==init_sha and not opt.state
    assert torch.equal(source_rng.get_state(),rngs)and torch.equal(real_rng.get_state(),rngr)
    assert torch.equal(torch.get_rng_state(),rngcpu)and all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),rnggpu))
    measured=rows[warm:];keys=list(measured[0]['components'])
    receipt=dict(UTC=utc(),scope='R74 exact complete-parent update with synchronized component boundaries; engineering only',
      engineering_updates=warm+timed+1,operator_updates=1,main_updates=0,warmup=warm,timed=timed,rows=rows,
      component_median_seconds={k:float(np.median([r['components'][k]for r in measured]))for k in keys},
      median_seconds=float(np.median([r['seconds']for r in measured])),p90_seconds=float(np.quantile([r['seconds']for r in measured],.9)),
      SOURCE_N=nS,REAL_N=nR,all_VJP_N=N,quality_metrics_recorded_or_used=False,
      initial_state_SHA256=init_sha,restored_state_SHA256=state_hash(model.state_dict()),all_RNG_restored=True,optimizer_empty=True,
      synchronization_changes_timing_only=True,instrumented_cost_not_direct_full_admission=True,model_goal_complete=False)
    write('COMPONENT_PROFILE.json',receipt)
    write('RESULT.json',dict(status='COMPLETE_PARENT_COMPONENT_PROFILE_COMPLETE',engineering_updates=warm+timed+1,operator_updates=1,main_updates=0,model_goal_complete=False))
    progress('complete_parent_component_profile_complete',engineering_updates=warm+timed+1,operator_updates=1,main_updates=0)
try:main()
except BaseException as exc:
    write('FAILURE.json',dict(UTC=utc(),error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-START,
      scientific_conclusion='engineering profile execution invalid; not an A-F negative',model_goal_complete=False));raise
