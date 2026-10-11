"""One preregistered, wall-capped, shared-checkpoint local benefit comparison."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,time,json,hashlib,copy,math,ast,traceback

P=Path(__file__).resolve().parent
START=float(os.environ['PILOT_START_MONOTONIC'])
PC=json.loads((P/'config.json').read_text('utf8'))
DEADLINE=START+PC['worker_cap_s']
R65=Path(PC['R65_root'])
C=json.loads((R65/'config.json').read_text('utf8'))
assert sorted(os.sched_getaffinity(0))==PC['exact_CPU_affinity'],'R69 requires exact12-CPU allocation before scientific imports'
for path in [R65,R65/'distribution_adapter',R65/'objective_adapter',R65/'network_update_check']:
    sys.path.insert(0,str(path))

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    path=P/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(tmp,path)
def check_budget():
    if time.monotonic()>=DEADLINE-2:raise TimeoutError('worker hard budget reserve reached')
def progress(phase,**extra):
    check_budget()
    value=dict(UTC=utc(),PID=os.getpid(),phase=phase,elapsed_s=time.monotonic()-START,**extra)
    write('PROGRESS.json',value);print(json.dumps(value),flush=True)

RESULT=dict(stage=PC['stage'],status='INCOMPLETE',original_A_F_complete=False,
    original_R65_50000_complete=False,automatic_long_resume=False,estimator='uniform4_of104_full_objective_parameter_VJP',
    original_acceptance_unchanged=True,block_sampler_seed=PC['block_sampler_seed'])
try:
    progress('binding_before_scientific_imports')
    for row in json.loads((P/'FILE_BINDING.json').read_text())['files']:
        assert sha(row['path'])==row['sha256'],row['path']
    prior=json.loads((R65/'ARTIFICIAL_SUITE_RECEIPT.json').read_text())
    assert prior['pass']
    review=json.loads((R65/'ROOT_IMPLEMENTATION_REVIEW.json').read_text())
    assert len(review['files'])==115
    for row in review['files']:assert sha(R65/row['name'])==row['sha256'],row['name']
    binding=json.loads((R65/'INPUT_BINDING.json').read_text())['inputs']
    for key in ['TRAIN_path','constants_path','noise_calibration_path','distribution_statistics','temporal_statistics','valley_definition']:
        assert sha(C['input_paths'][key])==binding[key]['sha256'],key
    assert sha(R65/'cache/RECOVERY_SAME_RUN.pt')==PC['checkpoint_SHA256']
    from sampled_blocks import original_schedule,sample_blocks,conditional_variance_of_sum
    from check_sampled_blocks import run_checks
    write('ARTIFICIAL_CHECK.json',run_checks())
    import numpy as np
    import scipy
    from scipy.stats import wasserstein_distance,spearmanr
    from scipy.special import expit
    import torch
    from network import ResponseNetwork
    from noise_core import q_np,bins_for,continuous_logq,innovation
    from source_flow_core import endpoint,q_torch,canonical_adc_torch,canonical_REAL_torch
    from distribution_loss import DistributionLoss,CATEGORIES,NAMES
    from R61_distribution_loss import noise_features
    from distribution_gradient import FrozenDistributionObjective,ParentLayout
    from objective_adapter import joint_total_D,original_C_microbatch,D_PIECE_ORDER,C_NAMES
    from network_update_check import parameter_vjp_cpu_f64,scalar_parameter_gradient_cpu_f64,install_cpu_f64_clip_step
    torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.mha.set_fastpath_enabled(False)
    dev=torch.device('cuda:0')
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,device=str(dev),GPU=torch.cuda.get_device_name(0))
    assert all(env[k]==C['expected_R33_environment'][k]for k in env),env
    write('ENVIRONMENT.json',dict(env,CPU_affinity=sorted(os.sched_getaffinity(0)),PID=os.getpid(),UTC=utc()))
    # Definition-only extraction; never import or execute either old runner.
    tree=ast.parse((R65/'train_final.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['state_hash','raw_psd','charge_range','chain_update','to_cpu']]
    assert len(funcs)==5
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'bound_R65_helpers','exec'),globals())
    pk=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PHYSICAL_KEYS'for t in n.targets))
    exec(compile(ast.Module(body=[pk],type_ignores=[]),'bound_R65_physical_keys','exec'),globals())
    # Only legal TRAIN fields; no held-out archive members are accessed.
    with np.load(C['TRAIN_path'],allow_pickle=False)as z:
        source=np.asarray(z['source'],dtype='f4').copy();real=np.asarray(z['target'],dtype='f4').copy()
        source_ids=z['source_ids'].copy();source_h=z['h'].copy()
        saved_source_bins=z['source_bin'].copy();saved_REAL_bins=z['target_bin'].copy()
    assert source.shape==(6593,480)and real.shape==(21047,480)
    with np.load(C['constants_path'],allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    with np.load(C['noise_calibration_path'],allow_pickle=False)as z:noise={k:z[k].copy()for k in z.files}
    sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
    assert np.array_equal(sraw,saved_source_bins)and np.array_equal(rraw,saved_REAL_bins)
    sh=continuous_logq(sq,sb,edges);rh=continuous_logq(rq,rb,edges);sp=raw_psd(source)
    groupsS=[np.flatnonzero(sraw==b)for b in CATEGORIES];groupsR=[np.flatnonzero(rraw==b)for b in CATEGORIES]
    with np.load(C['input_paths']['distribution_statistics'],allow_pickle=False)as z:statistics={k:z[k].copy()for k in z.files}
    with np.load(C['input_paths']['temporal_statistics'],allow_pickle=False)as z:statistics.update({k:z[k].copy()for k in z.files})
    assert np.array_equal(statistics['source_counts'],[len(g)for g in groupsS])
    assert np.array_equal(statistics['REAL_counts'],[len(g)for g in groupsR])
    with np.load(R65/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False)as z:
        assert np.array_equal(z['source_ids'],source_ids)and np.array_equal(z['source_original_psd'],sp)
        interval_low=z['PSD_interval_low'].copy();interval_high=z['PSD_interval_high'].copy()
    assert np.all(interval_low<=sp)and np.all(sp<=interval_high)
    xs=torch.as_tensor(source,dtype=torch.float64,device=dev);xr=torch.as_tensor(real,dtype=torch.float64,device=dev)
    mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
    sbt,sqt,sht=(torch.as_tensor(v,device=dev)for v in [sb,sq,sh])
    layout=ParentLayout.full_reference(sraw,rraw)
    recovery=torch.load(R65/'cache/RECOVERY_SAME_RUN.pt',map_location='cpu',weights_only=False)
    assert recovery['committed_update']==recovery['step']==500
    assert state_hash(recovery['model'])==PC['checkpoint_model_SHA256']
    FIXED_LR=C['lr_final']+.5*(C['lr']-C['lr_final'])*(1+math.cos(math.pi*(501-C['warmup_updates'])/(C['updates']-C['warmup_updates'])))
    assert FIXED_LR==PC['fixed_learning_rate']
    full_code=compile((P/'sampled_step.py').read_text(),'bound_R65_update_body','exec')
    mini_code=compile((P/'mini_step.py').read_text(),'bound_R63_update_body','exec')
    reference_code=compile((P/'reference_gradient.py').read_text(),'original_full_no_update_gradient','exec')
    probe_objective=DistributionLoss(statistics,C,dev)
    reference=canonical_REAL_torch(xr)
    rp=raw_psd(reference.cpu().numpy());rn=noise_features(reference).cpu().numpy()
    initial_restore=[]

    def digest(value):
        h=hashlib.sha256()
        def visit(v):
            if isinstance(v,torch.Tensor):
                a=v.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
            elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
            elif isinstance(v,dict):
                for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
            elif isinstance(v,(list,tuple)):
                for x in v:visit(x)
            else:h.update(repr(v).encode())
        visit(value);return h.hexdigest()

    def make_arm(name):
        model=ResponseNetwork().to(dev).train();model.load_state_dict(recovery['model'],strict=True)
        params=tuple(model.parameters())
        optimizer=torch.optim.Adam(params,lr=C['lr_initial'],betas=tuple(C['betas']),eps=1e-8,weight_decay=0)
        optimizer.load_state_dict(copy.deepcopy(recovery['optimizer']))
        streams={k:np.random.Generator(np.random.PCG64(seed))for k,seed in C['PCG64_seeds'].items()}
        for k in streams:streams[k].bit_generator.state=copy.deepcopy(recovery['PCG64_states'][k])
        torch.set_rng_state(recovery['torch_CPU_RNG']);torch.cuda.set_rng_state(recovery['torch_CUDA_RNG'],0)
        restored=dict(model=state_hash(model.state_dict()),optimizer=digest(optimizer.state_dict()),
            PCG64=digest({k:v.bit_generator.state for k,v in streams.items()}),torch_CPU=digest(torch.get_rng_state()),torch_CUDA=digest(torch.cuda.get_rng_state(0)))
        expected=dict(model=state_hash(recovery['model']),optimizer=digest(recovery['optimizer']),
            PCG64=digest(recovery['PCG64_states']),torch_CPU=digest(recovery['torch_CPU_RNG']),torch_CUDA=digest(recovery['torch_CUDA_RNG']))
        assert restored==expected,(restored,expected)
        initial_restore.append(dict(arm=name,state_hashes=restored));write('COMMON_START_RESTORATION.json',initial_restore)
        objective=DistributionLoss(statistics,C,dev)
        objective.direction_calls=recovery['direction_calls'];objective.temporal_direction_calls=recovery['temporal_direction_calls']
        H=dict(globals())
        H.update(model=model,params=params,parameter_count=sum(p.numel()for p in params),optimizer=optimizer,streams=streams,objective=objective,
            named_parameter_order=[k for k,v in model.named_parameters()],frozen_buffers={k:v.detach().clone()for k,v in model.named_buffers()},
            learning_rate=lambda step:FIXED_LR,phase_write=lambda phase:check_budget(),
            physical={k:0 for k in PHYSICAL_KEYS},counter=copy.deepcopy(recovery['counter']),hashes=copy.deepcopy(recovery['stream_chains']),
            countsS=recovery['SOURCE_exposure_counts'].copy(),countsR=recovery['REAL_reference_exposure_counts'].copy(),countsI=recovery['REAL_identity_exposure_counts'].copy(),
            countsDS=recovery['full_D_SOURCE_exposure_counts'].copy(),countsDR=recovery['full_D_REAL_exposure_counts'].copy(),
            full_SOURCE_epsilon_calls=recovery['full_SOURCE_epsilon_calls'],gradient_execution_counts=copy.deepcopy(recovery['gradient_execution_counts']),
            schedule=original_schedule(len(source),64),
            block_rng=np.random.Generator(np.random.PCG64(PC['block_sampler_seed'])),sampled_block_history=[],sampled_backward_counts=np.zeros(len(source),dtype='i8'),
            columns=C['loss_columns'],order_columns=C['precision_columns'],replay_columns=C['replay_columns'],
            dense=[],order_dense=[],replay_dense=[],discarded_dense=[],trace=[],coverage_dense=[],
            discarded_columns=list(D_PIECE_ORDER)+['original_nine_total','discarded_D_total'],
            global_max_q64=0.,global_max_q32=0.,global_max_D_q64=0.,global_max_D_q32=0.,global_max_C_q64=0.,global_max_C_q32=0.,updates=500,current_update=500)
        if name=='mini':
            loss_names=list(C['loss_weights'])
            H.update(loss_names=loss_names,columns=['total_loss',*loss_names,'learning_rate','gradient_norm','max_relative_q_f64','max_relative_q_f32',
                'SOURCE_global_range_projection_fraction','REAL_range_projection_fraction','SOURCE_global_projection_RMS_V','REAL_projection_RMS_V',
                'overall_PSD_W1','Q1_Q7_mean_PSD_W1','Q3_h_one_soft_valley','Q4_h_one_soft_valley'],
                coverage_columns=['original_seven_total','coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced',*[n+'_coordinate_RMS'for n in NAMES]])
        return H

    def noise_for(x,rng):
        eps=rng.standard_normal((len(x),480)).astype('<f4')
        n,_=innovation(x,eps,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        return torch.as_tensor(n,device=dev)

    @torch.no_grad()
    def full_forward(model,domain,rng):
        x,b,q,h=(source,sb,sq,sh)if domain=='SOURCE'else(real,rb,rq,rh)
        xt=xs if domain=='SOURCE'else xr
        nt=noise_for(x,rng);bt,qt,ht=(torch.as_tensor(v,device=dev)for v in [b,q,h])
        yy=[];rr=[]
        for lo in range(0,len(x),64):
            check_budget();hi=min(lo+64,len(x))
            out,raw=endpoint(model,xt[lo:hi],nt[lo:hi],qt[lo:hi],bt[lo:hi],ht[lo:hi],mt,st)
            yy.append(out);rr.append(raw)
        y=torch.cat(yy);raw=torch.cat(rr);q64,q32=charge_range(y,qt)
        return y,raw,dict(q64=q64,q32=q32)

    @torch.no_grad()
    def metrics(y,raw,domain):
        canon=canonical_adc_torch(y,False)
        assert (q_torch(canon)>0).all()
        p=raw_psd(canon.cpu().numpy());nf=noise_features(canon).cpu().numpy()
        regions=sraw if domain=='SOURCE'else rraw
        nonnegative={};correlations={};valleys={}
        allrows=np.arange(len(p))
        scopes=[('overall',allrows,np.arange(len(real)))]+[(f'region_{b}',np.flatnonzero(regions==b),groupsR[j])for j,b in enumerate(CATEGORIES)]
        for label,si,ri in scopes:
            assert len(si)>0 and len(ri)>0
            nonnegative['PSD_W1_'+label]=float(wasserstein_distance(p[si],rp[ri]))
            for j,n in enumerate(['baseline_RMS','baseline_ACF1']):nonnegative[n+'_W1_'+label]=float(wasserstein_distance(nf[si,j],rn[ri,j]))
            if domain=='SOURCE':
                rho=float(spearmanr(sp[si],p[si]).statistic)if len(si)>=2 else float('nan')
                correlations[label]=rho if math.isfinite(rho)else None
        active=((raw<-.5)|(raw>.5)).any(1).double()
        correction=(y-raw).square().mean(1)
        for label,ix in [('overall',allrows),('support_out',np.flatnonzero((regions==-1)|(regions==8)))]:
            assert len(ix)>0
            nonnegative['projection_fraction_'+label]=float(active[ix].mean())
            nonnegative['projection_RMS_'+label]=float(correction[ix].mean().sqrt())
        if domain=='SOURCE':
            for j,b in enumerate([2,3]):
                ix=(sraw==b)&(source_h==1);assert ix.any()
                lo,hi=statistics['valley_bounds'][j]
                valleys[f'Q{b+1}_h_one']=float(np.mean((p[ix]>=lo)&(p[ix]<=hi)))
        else:
            normalized=(canon-reference)/torch.as_tensor(statistics['V480_scale'][rraw+1],device=dev)
            nonnegative['parentwise_normalized_identity_MSE']=float(normalized.square().mean())
        return dict(nonnegative=nonnegative,correlations=correlations,valleys=valleys)

    @torch.no_grad()
    def common_C(model,rng,directions,tdirs):
        si=np.concatenate([g[rng.integers(0,len(g),32)]for g in groupsS]).astype('i8')
        ri=np.concatenate([g[rng.integers(0,len(g),32)]for g in groupsR]).astype('i8')
        ii=rng.integers(0,len(real),64).astype('i8')
        n=torch.cat((noise_for(source[si],rng),noise_for(real[ii],rng)))
        x=torch.cat((xs[si],xr[ii]))
        b,q,h=(torch.as_tensor(v,device=dev)for v in [np.r_[sb[si],rb[ii]],np.r_[sq[si],rq[ii]],np.r_[sh[si],rh[ii]]])
        output,raw=endpoint(model,x,n,q,b,h,mt,st);charge_range(output,q)
        v=original_C_microbatch(probe_objective,output,raw,xr[ri],xr[ii],torch.as_tensor(rraw[ii],device=dev),
            torch.as_tensor(sp[si],device=dev),torch.as_tensor(source_h[si],device=dev),directions,
            torch.as_tensor(interval_low[si],device=dev),torch.as_tensor(interval_high[si],device=dev),tdirs)
        return {k:float(x)for k,x in v['pieces'].items()}

    @torch.no_grad()
    def evaluate(H,name):
        model=H['model'];old=state_hash(model.state_dict());old_rng=digest({k:v.bit_generator.state for k,v in H['streams'].items()})
        t=time.monotonic();rows=[]
        for j,seed in enumerate(PC['probe_seeds']):
            rng=np.random.Generator(np.random.PCG64(seed))
            y,raw,num=full_forward(model,'SOURCE',rng)
            directions=probe_objective.draw_directions(rng);tdirs=probe_objective.draw_temporal_directions(rng)
            obj=FrozenDistributionObjective(statistics,C,directions,tdirs,sraw,rraw,feature_chunk=64,device=dev,uniform_fastpath=True)
            d=obj(y,xr,layout)
            pieces={k:float(v)for k,v in d.pieces.items()};pieces.update(common_C(model,rng,directions,tdirs))
            assert set(pieces)==set(C['loss_weights'])and all(math.isfinite(v)and v>=0 for v in pieces.values())
            rows.append(dict(seed=seed,pieces=pieces,score=sum(pieces.values()),numeric=num,**metrics(y,raw,'SOURCE')))
            write(name+'/PROBES.json',rows)
            del obj,d,y,raw
            progress('common_SOURCE_probe',arm=name,completed=j+1,total=8)
        rng=np.random.Generator(np.random.PCG64(PC['REAL_control_seed']))
        y,raw,num=full_forward(model,'REAL',rng)
        directions=probe_objective.draw_directions(rng);tdirs=probe_objective.draw_temporal_directions(rng)
        control_stats=dict(statistics,source_counts=statistics['REAL_counts'].copy())
        obj=FrozenDistributionObjective(control_stats,C,directions,tdirs,rraw,rraw,feature_chunk=64,device=dev,uniform_fastpath=True)
        d=obj(y,xr,ParentLayout.full_reference(rraw,rraw))
        real_control=dict(pieces={k:float(v)for k,v in d.pieces.items()},numeric=num,**metrics(y,raw,'REAL'))
        del y,raw,obj,d
        assert old==state_hash(model.state_dict())and old_rng==digest({k:v.bit_generator.state for k,v in H['streams'].items()})
        torch.cuda.synchronize();elapsed=time.monotonic()-t
        result=dict(arm=name,model_SHA256=old,probe_count=8,SOURCE_N=6593,REAL_N=21047,probes=rows,REAL_control=real_control,evaluation_elapsed_s=elapsed,
            train_streams_and_model_unchanged=True)
        write(name+'/EVALUATION.json',result);progress('evaluation_complete',arm=name,evaluation_elapsed_s=elapsed)
        return result

    def train(H,name,threshold=None):
        progress('training_start',arm=name,candidate_updates=PC['candidate_updates'],mini_target_time_s=threshold)
        arm_deadline=time.monotonic()+PC['candidate_arm_cap_s']
        def phase_check(_):
            check_budget()
            if time.monotonic()>=arm_deadline:raise TimeoutError(name+' arm exceeded200 seconds')
        H['phase_write']=phase_check
        code=full_code if name=='candidate'else mini_code
        torch.cuda.synchronize();t=time.monotonic();n=0
        while True:
            phase_check('before_update');H['step']=501+n;exec(code,H);n+=1
            if name=='mini':
                H['counter']['scientific_neural_forward_rows']+=384
                H['counter']['scientific_neural_forward_calls']+=1
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            if elapsed>PC['candidate_arm_cap_s']:raise TimeoutError(name+' arm exceeded200 seconds')
            if name=='candidate':
                if n==PC['candidate_updates']:break
            elif elapsed>=threshold:break
        receipt=dict(arm=name,completed_updates=n,first_original_step=501,last_original_step=500+n,training_elapsed_s=elapsed,
            fixed_learning_rate=FIXED_LR,model_SHA256=state_hash(H['model'].state_dict()),
            source_occurrences=n*320,REAL_reference_occurrences=n*320,REAL_identity_occurrences=n*64,
            full_D_SOURCE_occurrences=n*6593 if name=='candidate'else 0,
            full_D_REAL_occurrences=n*21047 if name=='candidate'else 0,
            neural_forward_rows=(H['physical']['neural_forward_rows'] if name=='candidate' else n*384),
            global_max_q64=H['global_max_q64'],global_max_q32=H['global_max_q32'])
        receipt['counter_delta']={k:int(H['counter'][k]-recovery['counter'][k])for k in H['counter']}
        receipt['joint_microbatch_reverse_calls']=n if name=='mini'else 0
        if name=='candidate':
            receipt.update(block_count=104,selected_blocks_per_update=4,HT_scale=26.,
                actual_D_backward_parent_occurrences=int(H['sampled_backward_counts'].sum()),
                sampled_backward_unique_parents=int(np.count_nonzero(H['sampled_backward_counts'])),
                D_parameter_VJP_calls=H['physical']['D_parameter_VJP_calls'],
                all_parent_forward_and_distribution_exposure_preserved=True,all_parent_backward_exposure_each_step=False,
                sampler_final_state=copy.deepcopy(H['block_rng'].bit_generator.state))
            assert receipt['D_parameter_VJP_calls']==4*n
            assert len(H['sampled_block_history'])==n
            (P/name).mkdir(exist_ok=True)
            np.savez_compressed(P/name/'BLOCK_SAMPLING_PRIVATE.npz',selected_blocks=np.asarray(H['sampled_block_history'],dtype='i8'),backward_parent_counts=H['sampled_backward_counts'])
        if name=='mini':
            receipt.update(cost_match_absolute_s=abs(elapsed-threshold),cost_match_tolerance_s=max(.5,.01*threshold))
            receipt['cost_match_pass']=receipt['cost_match_absolute_s']<=receipt['cost_match_tolerance_s']
        checkpoint=dict(model=to_cpu(H['model'].state_dict()),optimizer=to_cpu(H['optimizer'].state_dict()),
            PCG64_states={k:copy.deepcopy(v.bit_generator.state)for k,v in H['streams'].items()},
            torch_CPU_RNG=torch.get_rng_state(),torch_CUDA_RNG=torch.cuda.get_rng_state(0),receipt=receipt,
            block_sampler_state=copy.deepcopy(H['block_rng'].bit_generator.state)if name=='candidate'else None)
        (P/name).mkdir(exist_ok=True)
        torch.save(checkpoint,P/name/'DIAGNOSTIC_ENDPOINT.pt')
        np.savez_compressed(P/name/'TRAIN_LOGS.npz',columns=np.asarray(H['columns']),loss=np.asarray(H['dense']),
            precision_columns=np.asarray(H['order_columns']),precision=np.asarray(H['order_dense']),
            SOURCE_counts=H['countsS'],REAL_reference_counts=H['countsR'],REAL_identity_counts=H['countsI'])
        receipt['checkpoint_SHA256']=sha(P/name/'DIAGNOSTIC_ENDPOINT.pt')
        write(name+'/TRAINING_RECEIPT.json',receipt)
        progress('training_complete',**receipt)
        return receipt

    def gradient_admission():
        progress('no_update_full_gradient_admission')
        H=make_arm('gradient_reference');before_model=state_hash(H['model'].state_dict());before_adam=digest(H['optimizer'].state_dict())
        squared=[]
        original_vjp=parameter_vjp_cpu_f64
        def counted_vjp(*args,**kwargs):
            g=original_vjp(*args,**kwargs);squared.append(float(torch.dot(g,g)));return g
        H['parameter_vjp_cpu_f64']=counted_vjp;H['step']=501
        t=time.monotonic();exec(reference_code,H);torch.cuda.synchronize()
        assert len(squared)==104
        d2=float(torch.dot(H['D_gradient'],H['D_gradient']))
        c2=float(torch.dot(H['C_gradient'],H['C_gradient']))
        dc2=float(torch.dot(H['combined'],H['combined']))
        sum2=math.fsum(squared)
        variance=conditional_variance_of_sum(104,PC['sampled_block_count'],sum2,d2)
        unchanged=before_model==state_hash(H['model'].state_dict())and before_adam==digest(H['optimizer'].state_dict())
        assert unchanged and math.isfinite(variance)and variance>=0
        ok=variance<=PC['relative_gradient_MSE_max']*d2 and variance<=PC['relative_gradient_MSE_max']*dc2
        out=dict(UTC=utc(),fixed_original_update=500,no_optimizer_update=True,model_and_Adam_unchanged=unchanged,
            source_parents=6593,REAL_parents=21047,original_block_count=104,sampled_block_count=4,HT_scale=26.,
            block_squared_norms=squared,sum_block_squared_norm=sum2,D_squared_norm=d2,C_squared_norm=c2,
            D_plus_C_squared_norm=dc2,conditional_MSE_trace=variance,
            relative_MSE_to_D=variance/d2 if d2>0 else None,
            relative_MSE_to_D_plus_C=variance/dc2 if dc2>0 else None,
            prospective_relative_MSE_limit=PC['relative_gradient_MSE_max'],pass_gate=bool(ok),
            one_frozen_theta_noise_directions_and_C_draw_only=True,clip_and_Adam_unbiased_claim=False,
            model_SHA256=before_model,Adam_SHA256=before_adam,elapsed_s=time.monotonic()-t,
            physical_counts=H['physical'],new_response_updates=0,
            numeric=dict(D_q64=H['dq64'],D_q32=H['dq32'],C_q64=H['cq64'],C_q32=H['cq32']))
        write('GRADIENT_ADMISSION.json',out)
        del H
        progress('gradient_admission_complete',pass_gate=bool(ok),relative_MSE_to_D=out['relative_MSE_to_D'],relative_MSE_to_D_plus_C=out['relative_MSE_to_D_plus_C'])
        return out

    admission=gradient_admission()
    RESULT['gradient_admission']=admission
    if not admission['pass_gate']:
        RESULT.update(status='NEGATIVE_GRADIENT_ADMISSION',new_response_updates=0,new_response_probes=0,
            decision='Close this fixed k4 estimator before training: conditional RMS gradient error exceeds a preregistered reference-gradient magnitude. No k or seed scan, no original-A-F impossibility claim.')
    else:
        progress('loaded_common_start',source_parents=6593,REAL_parents=21047)
        base=make_arm('baseline');B=evaluate(base,'baseline')
        budget_estimate=time.monotonic()-START+2*PC['candidate_arm_cap_s']+2*1.10*B['evaluation_elapsed_s']+20
        write('BASELINE_BUDGET_GATE.json',dict(UTC=utc(),elapsed_s=time.monotonic()-START,baseline_evaluation_s=B['evaluation_elapsed_s'],
            required_training_reserve_s=2*PC['candidate_arm_cap_s'],remaining_evaluation_reserve_s=2*1.10*B['evaluation_elapsed_s'],output_reserve_s=20,
            projected_total_s=budget_estimate,limit_s=PC['worker_cap_s'],pass_gate=budget_estimate<=PC['worker_cap_s']))
        if budget_estimate>PC['worker_cap_s']:
            RESULT.update(status='INCONCLUSIVE_BUDGET',reason='Complete baseline proves insufficient reserved time under the fixed600-second worker budget')
        else:
            del base
            full=make_arm('candidate');Ftrain=train(full,'candidate')
            mini=make_arm('mini');Mtrain=train(mini,'mini',Ftrain['training_elapsed_s'])
            if not Mtrain['cost_match_pass']:
                RESULT.update(status='INCONCLUSIVE_COST_MATCH',candidate_training=Ftrain,mini_training=Mtrain)
            else:
                F=evaluate(full,'candidate');M=evaluate(mini,'mini')
                values=[np.asarray([row['score']for row in e['probes']])for e in [B,F,M]]
                b,f,m=values;wins=int((f<m).sum())
                primary=dict(baseline_mean=float(b.mean()),candidate_mean=float(f.mean()),mini_mean=float(m.mean()),
                    candidate_vs_mini_relative_improvement=float(1-f.mean()/m.mean()),candidate_vs_baseline_relative_improvement=float(1-f.mean()/b.mean()),
                    candidate_beats_mini_probes=wins,pass_gate=bool(f.mean()<=.9*m.mean()and f.mean()<=.9*b.mean()and wins>=6))
                boot=np.random.Generator(np.random.PCG64(PC['bootstrap_seed'])).integers(0,8,(1000,8))
                primary['descriptive_paired_noise_MC_candidate_minus_mini_95_interval']=np.quantile((f-m)[boot].mean(1),[.025,.975]).tolist()
                harms=[]
                for group in ['pieces','nonnegative','correlations','valleys']:
                    for key in B['probes'][0][group]:
                        seq=[[row[group][key]for row in e['probes']]for e in [B,F,M]]
                        if any(v is None for s in seq for v in s):
                            harms.append(dict(group=group,key=key,pass_gate=False,status='unknown'));continue
                        bv,fv,mv=[float(np.mean(s))for s in seq]
                        if group=='correlations':bound=max(bv,mv)-.02;ok=fv>=bound
                        elif group=='valleys':bound=min(bv,mv)+.01;ok=fv<=bound
                        else:bound=min(bv,mv)+.05*max(bv,mv)+1e-10;ok=fv<=bound
                        harms.append(dict(group=group,key=key,baseline=bv,full=fv,mini=mv,bound=bound,pass_gate=ok))
                for group in ['pieces','nonnegative']:
                    for key in B['REAL_control'][group]:
                        bv,fv,mv=[e['REAL_control'][group][key]for e in [B,F,M]]
                        bound=min(bv,mv)+.05*max(bv,mv)+1e-10
                        harms.append(dict(group='REAL_'+group,key=key,baseline=bv,full=fv,mini=mv,bound=bound,pass_gate=fv<=bound))
                positive=primary['pass_gate']and all(x['pass_gate']for x in harms)
                RESULT.update(status='POSITIVE_LOCAL_ONLY'if positive else 'NEGATIVE_LOCAL_SUPPORT',primary=primary,
                    harm_gate_pass=all(x['pass_gate']for x in harms),harm_checks=harms,failed_harm_checks=[x for x in harms if not x['pass_gate']],
                    candidate_training=Ftrain,mini_training=Mtrain,decision='Only a separately registered capped confirmation may be considered; no long-run restart'if positive else 'Close this fixed sampled4 estimator; no equal-cost joint benefit; R65 stays retired')
    RESULT.update(UTC=utc(),elapsed_s=time.monotonic()-START,complete_common_evaluation=RESULT['status']in['POSITIVE_LOCAL_ONLY','NEGATIVE_LOCAL_SUPPORT'],
        GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        DEV_read=0,EVAL_read=0,CAL_decode=0,probe_population=dict(SOURCE=6593,REAL=21047),fixed_learning_rate=FIXED_LR,
        raw_SOURCE_h_one_valley_fractions=statistics['source_raw_valley_fraction'].tolist(),original_region_counts=statistics['source_counts'].tolist())
    assert RESULT['elapsed_s']<PC['worker_cap_s'] and RESULT['GPU_allocated_peak_bytes']<=C['GPU_limit_bytes']and RESULT['GPU_reserved_peak_bytes']<=C['GPU_limit_bytes']
    write('RESULT.json',RESULT);print(json.dumps({k:RESULT[k]for k in ['status','elapsed_s','complete_common_evaluation','automatic_long_resume']}),flush=True)
except BaseException as exc:
    RESULT.update(status='INCONCLUSIVE_EXECUTION',UTC=utc(),elapsed_s=time.monotonic()-START,error=repr(exc),traceback=traceback.format_exc())
    write('RESULT.json',RESULT)
    raise
