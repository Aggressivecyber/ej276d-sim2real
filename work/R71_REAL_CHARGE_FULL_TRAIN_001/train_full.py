"""Registered TRAIN-only all-parent charge component; cost gate then fixed endpoint."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,math,os,sys,time,traceback
R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text('utf8'))
START=time.monotonic()
def write(name,value):
    p=R/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8');t.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def progress(phase,**kw):
    value=dict(UTC=utc(),phase=phase,process_elapsed_s=time.monotonic()-START)
    value.update(kw)
    write('PROGRESS.json',value);print(json.dumps(value),flush=True)
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    import numpy as np
    import torch
    from public_charge_head import PublicConditionalChargeHead,window_charge
    from physical_operators import canonical_REAL_torch
    from charge_losses import FullChargeObjective
    from common_response import generate
    from artificial_checks import run_checks
    from postcheck import postcheck
    assert sorted(os.sched_getaffinity(0))==C['exact_CPU_affinity']
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,GPU=torch.cuda.get_device_name(0),CPU_affinity=sorted(os.sched_getaffinity(0)))
    for k in ['python','numpy','torch','GPU']:assert env[k]==C['expected_R33_environment'][k],(k,env[k])
    write('ENVIRONMENT.json',env)
    for key,row in json.loads((R/'INPUT_BINDING.json').read_text())['inputs'].items():
        assert sha(row['path'])==row['sha256'],key
    checks=run_checks(C['artificial_seed']);write('ARTIFICIAL_CHECKS.json',checks)
    assert checks['pass_all']
    progress('artificial_CPU_CUDA_checks_passed_before_scientific_arrays')
    with np.load(C['input_paths']['TRAIN_path'],allow_pickle=False)as z:
        source=np.asarray(z['source'],dtype='f4').copy();real=np.asarray(z['target'],dtype='f4').copy()
        source_ids=z['source_ids'].copy();source_h=z['h'].copy()
        original_source_saved_bins=z['source_bin'].copy();original_REAL_saved_bins=z['target_bin'].copy()
        train_keys=list(z.files)
    assert source.shape==(C['SOURCE_N'],480)and real.shape==(C['REAL_N'],480)
    assert len(np.unique(source_ids))==len(source_ids)==len(source)
    with np.load(C['input_paths']['constants_path'],allow_pickle=False)as z:
        mu,scale,edges=[z[k].copy()for k in ['mu_unit_q','scale_unit_q','edges_logQ']]
    with np.load(C['input_paths']['noise_calibration_path'],allow_pickle=False)as z:
        p480=z['p480'].copy();tau2=float(z['tau2'])
    valley=json.loads(Path(C['input_paths']['valley_definition']).read_text())
    x=torch.as_tensor(np.concatenate([source,real]),device='cuda',dtype=torch.float64)
    nS=len(source);qs=window_charge(x[:nS]);qr=window_charge(x[nS:])
    realcanon=canonical_REAL_torch(x[nS:]);qrc=window_charge(realcanon)
    unit=(x[nS:]-x[nS:,:100].mean(1,keepdim=True))/qr[:,None]
    center=unit.mean(0);unfloored=unit.std(0,unbiased=False);fscale=unfloored.clamp_min(C['input_scale_floor'])
    np.savez_compressed(R/'HEAD_STATISTICS.npz',feature_center=center.cpu().numpy(),feature_scale=fscale.cpu().numpy(),
        unfloored_feature_scale=unfloored.cpu().numpy(),mu=mu,scale=scale,edges=edges,p480=p480,tau2=tau2)
    torch.manual_seed(C['model_seed']);torch.cuda.manual_seed_all(C['model_seed'])
    model=PublicConditionalChargeHead(center,fscale).to('cuda').double()
    prepared=model.standardize_input(x);features=prepared['features'].detach();qinput=prepared['q_input'].detach()
    objective=FullChargeObjective(qs.detach(),qrc.detach())
    initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    init_sha=state_hash(initial);torch.save(initial,R/'INITIAL_HEAD.pt')
    rng_cpu=torch.get_rng_state().clone();rng_gpu=[v.clone()for v in torch.cuda.get_rng_state_all()]
    np.savez_compressed(R/'PARENT_MANIFEST_PRIVATE.npz',source_ids=source_ids,source_h=source_h,
        SOURCE_original_row=np.arange(nS),REAL_original_TRAIN_row=np.arange(len(real)),
        source_Q0=qs.cpu().numpy(),REAL_Q0=qr.cpu().numpy(),REAL_canonical_Q=qrc.cpu().numpy(),
        SOURCE_original_saved_bin=original_source_saved_bins,REAL_original_saved_bin=original_REAL_saved_bins,
        SOURCE_original_weight=np.full(nS,1/nS),REAL_original_weight=np.full(len(real),1/len(real)))
    write('DATA_RECEIPT.json',dict(UTC=utc(),TRAIN_SHA256=sha(C['input_paths']['TRAIN_path']),TRAIN_keys=train_keys,
      SOURCE_N=nS,REAL_N=len(real),SOURCE_ID_hash=hashlib.sha256(source_ids.tobytes()).hexdigest(),
      REAL_identity='Frozen TRAIN SHA plus original target array row index; no invented acquisition ID',
      per_update_SOURCE_rows=nS,per_update_REAL_rows=len(real),losses_use_all_rows=True,
      source_parent_weights='all1/6593',REAL_parent_weights='all1/21047',initial_state_SHA256=init_sha,
      feature_scale_floor=C['input_scale_floor'],feature_scale_floored_coordinates=int((unfloored<C['input_scale_floor']).sum()),
      REAL_charge_std=float(qrc.std(unbiased=False)),REAL_log_charge_std=float(qrc.log().std(unbiased=False)),
      heldout_arrays_read=False,shape_residual='zero frozen',Q_head_parameters=sum(p.numel()for p in model.parameters())))
    def optimizer_new():return torch.optim.Adam(model.parameters(),lr=C['lr_initial'],betas=tuple(C['betas']))
    def rate(step):
        if step<=C['warmup_updates']:return C['lr_initial']+(C['lr']-C['lr_initial'])*step/C['warmup_updates']
        ratio=(step-C['warmup_updates'])/(C['updates']-C['warmup_updates'])
        return C['lr_final']+(C['lr']-C['lr_final'])*.5*(1+math.cos(math.pi*ratio))
    def update(opt,step):
        lr=rate(step)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        out=model.forward_features(features,qinput,check_inputs=False)['q_target']
        total,terms=objective(out[:nS],out[nS:],qr)
        if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite objective')
        total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),C['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes']or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:
            raise MemoryError('registered GPU allocation or reserve exceeded')
        return torch.stack([total.detach(),*[terms[k].detach()for k in ['charge_distribution','REAL_identity','source_charge_order']],norm.detach(),out.new_tensor(lr)])
    optimizer=optimizer_new();timings=[]
    progress('engineering_full_TRAIN_cost_measurement',main_updates=0)
    for i in range(1,C['benchmark_warmup_updates']+C['benchmark_timed_updates']+1):
        torch.cuda.synchronize();t=time.perf_counter();update(optimizer,i);torch.cuda.synchronize()
        if i>C['benchmark_warmup_updates']:timings.append(time.perf_counter()-t)
    # Explicitly discard all fifty engineering optimizer updates and restore RNG.
    del optimizer
    model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_gpu)
    optimizer=optimizer_new()
    assert state_hash(model.state_dict())==init_sha and not optimizer.state
    assert torch.equal(torch.get_rng_state(),rng_cpu)and all(torch.equal(a,b)for a,b in zip(torch.cuda.get_rng_state_all(),rng_gpu))
    p90=float(np.quantile(timings,.9));estimate=C['cost_safety_multiplier']*p90*C['updates']+(time.monotonic()-START)+C['postprocess_reserve_seconds']
    admission=dict(UTC=utc(),engineering_updates=50,engineering_SOURCE_exposures=50*nS,engineering_REAL_exposures=50*len(real),
      timed_update_seconds=timings,median_update_seconds=float(np.median(timings)),p90_update_seconds=p90,
      estimate_full_run_seconds=estimate,includes_setup_and_postchecks_reserve=True,postcheck_reserve_seconds=C['postprocess_reserve_seconds'],
      execution_threshold_seconds=C['cost_limit_seconds'],pass_gate=estimate<=C['cost_limit_seconds'],
      restored_state_SHA256=state_hash(model.state_dict()),initial_state_SHA256=init_sha,optimizer_empty=True,RNG_restored=True,
      engineering_parameters_discarded=True,quality_metrics_used_for_cost_decision=False)
    write('COST_ADMISSION.json',admission)
    if not admission['pass_gate']:
        write('RESULT.json',dict(status='COST_TOO_HIGH_FULL_MAIN_NOT_STARTED',main_updates=0,engineering_updates=50,model_goal_complete=False))
        progress('cost_gate_rejected',main_updates=0,estimated_hours=estimate/3600);return
    progress('full_training_started',main_updates=0,estimated_hours=estimate/3600)
    trace=[];dense=[];train_start=time.monotonic()
    for step in range(1,C['updates']+1):
        row=update(optimizer,step);dense.append(row.cpu().numpy())
        if step%C['progress_interval_updates']==0 or step==1:
            torch.cuda.synchronize();elapsed=time.monotonic()-train_start
            item=dict(step=step,elapsed_s=elapsed,updates_per_second=step/max(elapsed,1e-12),
                ETA_seconds=(C['updates']-step)*elapsed/step,total_loss=float(row[0]),charge_distribution=float(row[1]),
                REAL_identity=float(row[2]),source_charge_order=float(row[3]),GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved())
            trace.append(item);write('TRAIN_TRACE.json',trace);progress('full_training',main_updates=step,**item)
        if step%C['recovery_interval_updates']==0:
            dest=R/('RECOVERY_UPDATE_'+str(step)+'.pt')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step,
                initial_state_SHA256=init_sha,torch_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all()),dest)
    final={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};torch.save(final,R/'FINAL_HEAD.pt')
    np.savez_compressed(R/'LOSS_BY_UPDATE_PRIVATE.npz',columns=np.array(['total','charge_distribution','REAL_identity','source_charge_order','gradient_norm','learning_rate']),values=np.asarray(dense))
    receipt=dict(UTC=utc(),completed_updates=C['updates'],unique_selected_endpoint=C['updates'],engineering_updates_discarded=50,
      training_elapsed_s=time.monotonic()-train_start,initial_state_SHA256=init_sha,final_state_SHA256=state_hash(final),
      final_weight_file_SHA256=sha(R/'FINAL_HEAD.pt'),SOURCE_main_exposures=nS*C['updates'],REAL_main_exposures=len(real)*C['updates'],
      complete_parent_per_update=True,heldout_arrays_read=False,model_goal_complete=False)
    write('TRAINING_RECEIPT.json',receipt);progress('training_endpoint_complete_postchecks_start',main_updates=C['updates'])
    # Frozen common epsilon per original parent, shared by identity and learned head.
    epsS=np.random.Generator(np.random.PCG64(C['source_epsilon_seed'])).standard_normal(source.shape).astype('f4')
    epsR=np.random.Generator(np.random.PCG64(C['real_epsilon_seed'])).standard_normal(real.shape).astype('f4')
    np.savez_compressed(R/'POSTCHECK_EPSILON_PRIVATE.npz',SOURCE=epsS,REAL=epsR)
    initial_model=PublicConditionalChargeHead(center,fscale).to('cuda').double();initial_model.load_state_dict(initial)
    post=postcheck(R,C,model,initial_model,source,real,source_ids,source_h,epsS,epsR,mu,scale,edges,p480,tau2,valley)
    write('RESULT.json',dict(status='COMPONENT_ENDPOINT_AND_POSTCHECKS_COMPLETE',training=receipt,postcheck=post,
        A_to_F='not evaluated by this TRAIN-only charge component',model_goal_complete=False))
    progress('component_endpoint_and_postchecks_complete',main_updates=C['updates'])

try:main()
except BaseException as exc:
    write('FAILURE.json',dict(UTC=utc(),error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-START,
        scientific_conclusion='execution invalid or numerical failure; not an A-F negative',model_goal_complete=False))
    raise
