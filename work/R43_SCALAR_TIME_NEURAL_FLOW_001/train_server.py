from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,sys,copy,shutil
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from source_flow_core import FlowNetwork,TangentField,q_np,q_torch,project,initial,formula_check
from reorder_core import reconstruct_and_reorder
from psd_order_core import ordered_coupling
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
A='SCALAR_TIME_FLOW_FINAL';counter={'velocity_fit_attempts':0,'velocity_updates':0,'training_source_rows':0,'training_REAL_rows':0,'old_G_D_reader_updates_forwards':0};updates=0;step=0;trace=[];dense=[];clock=START
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def state(m):return{k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
env={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(DEVICE),'GPU':torch.cuda.get_device_name(0)if DEVICE.type=='cuda'else None}
write('SERVER_ENVIRONMENT.json',env);assert all(env[k]==CFG['expected_R33_environment'][k]for k in ['python','numpy','torch','scipy','device','GPU'])
try:
    write('FLOW_FORMULA_CHECK.json',formula_check())
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:source=z['source'].copy();real=z['target'].copy()
    shutil.copy2(CFG['frozen_R42_constants'],ROOT/'UNIT_CHARGE_CONSTANTS.npz')
    shutil.copy2(CFG['frozen_R42_constants_json'],ROOT/'UNIT_CHARGE_CONSTANTS.json')
    with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    sq=q_np(source);rq=q_np(real)
    if not np.isfinite(sq).all()or not np.isfinite(rq).all()or not(sq>0).all()or not(rq>0).all():raise ValueError('TRAIN raw source/REAL nonpositive/nonfinite q; no clipping/deletion')
    def bins(q):
        v=np.log(1000.*q);b=np.searchsorted(edges,v,side='right')-1;b[v==edges[-1]]=7;return b
    sb=bins(sq);rb=bins(rq)
    with np.load(CFG['frozen_R42_coupling'],allow_pickle=False)as z:coupled={k:z[k].copy()for k in z.files}
    write('CONSTANTS_COUPLING_REUSE.json',{'constants_refits':0,'coupling_resorts_resamples':0,'constants':CFG['frozen_R42_constants'],'coupling':CFG['frozen_R42_coupling'],'target_statistical_not_true_pairs':True})
    actual=reconstruct_and_reorder(CFG,ROOT);si=actual['source'].astype('i8');ti=actual['target'].astype('i8');bn=actual['bin'].astype('i8');ordinal=actual['original_row_ordinal'];noise=actual['original_noise'];styles=actual['original_style']
    assert np.array_equal(sb[si],bn)and np.array_equal(rb[ti],bn)
    reference=json.loads(pathlib.Path(CFG['frozen_R42_train_receipt']).read_text('utf8'))
    assert all(np.array_equal(actual[k],coupled[k])for k in ['source','target','bin','original_row_ordinal'])
    rt=np.random.Generator(np.random.PCG64(CFG['t_seed']));rt_initial=copy.deepcopy(rt.bit_generator.state)
    tt_all=coupled['t'].copy();expected_t=np.concatenate([rt.uniform(0,1,size=32).astype('f4')for _ in range(2950)]);assert np.array_equal(tt_all,expected_t)
    assert hashlib.sha256(tt_all.tobytes()).hexdigest()==reference['stream_hashes']['t']
    hashes={k:hashlib.sha256()for k in ['source','target','bin','original_row_ordinal','z_white','style','t','white']}
    counter['velocity_fit_attempts']+=1;torch.manual_seed(CFG['model_seed']);np.random.seed(CFG['model_seed']%(2**32))
    model=FlowNetwork().to(DEVICE);dest=ROOT/A;dest.mkdir(exist_ok=False)
    initial_state=state(model);torch.save(initial_state,dest/'INITIAL_FLOW.pt');torch.save(torch.get_rng_state(),dest/'INITIAL_TORCH_STATE.pt')
    if DEVICE.type=='cuda':torch.save(torch.cuda.get_rng_state(),dest/'INITIAL_CUDA_STATE.pt')
    mt=torch.tensor(mu,dtype=torch.float64,device=DEVICE);st=torch.tensor(scale,dtype=torch.float64,device=DEVICE);field=TangentField(model,mt,st)
    time_inputs=[]
    hook=model.embedding[0].register_forward_pre_hook(lambda module,args:time_inputs.append(args[0].detach().cpu().clone()))
    zero=model(torch.zeros((4,480),dtype=torch.float64,device=DEVICE),torch.tensor([0.,.5,1.,1.],device=DEVICE),torch.tensor([0,2,4,7],device=DEVICE))
    assert torch.equal(zero,torch.zeros_like(zero));hook.remove()
    vec=time_inputs[0];assert vec.shape==(4,25) and torch.count_nonzero(vec[:,1:17])==0
    np.savez_compressed(dest/'TIME_FEATURES_ARTIFICIAL_PRIVATE.npz',actual_embedding_input25=vec.numpy())
    initial_zero_columns=model.embedding[0].weight[:,1:17].detach().clone()
    zero_grad_checks=0;zero_value_checks=0;t_nonzero_steps=0;bin_nonzero_steps=0;other_nonzero_steps=0
    oldpath=pathlib.Path(CFG['historical_R42_initial_state']);initial_equal='unknown'
    if oldpath.is_file():
        oldinit=torch.load(oldpath,map_location='cpu',weights_only=True)
        initial_equal={'same_keys':set(oldinit)==set(initial_state),'each_tensor_bitwise_equal':{k:bool(torch.equal(v,oldinit[k]))for k,v in initial_state.items()}}
        assert initial_equal['same_keys'] and all(initial_equal['each_tensor_bitwise_equal'].values())
    write('INITIALIZATION_RECEIPT.json',{'actual_initial_state_saved':True,'historical_R42_state':str(oldpath),'historical_state_exists':oldpath.is_file(),'R42_actual_initial_state_comparison':initial_equal,'same_construction_seed':CFG['model_seed'],'R42_weights_loaded_as_initialization':False,'new_model_initialization_only':True,'time_embedding_input25':vec.tolist(),'zero_time_input_column_weight_count':2048,'original_frequency_buffer_unused':True})
    write('MODEL_STRUCTURE.json',{'parameter_count':sum(p.numel()for p in model.parameters()),'seed':CFG['model_seed'],'zero_initial_field_artificial_rows':4,'zero_output_exact':True,'pretrained_weights':False,'embedding_input':25,'architecture':CFG['architecture']})
    opt=torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=0)
    perms=np.empty((94400,),'u1');matched=np.empty((94400,),'u4');src_key=np.empty((94400,),'f8');tgt_key=np.empty((94400,),'f8');marginal_checks=0;max_path_charge=0.;max_field_charge=0.;sorting_s=0.
    model.train()
    for step in range(1,2951):
        if time.monotonic()-START>=CFG['training_wall_s']-15 or time.time()>=datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-600:raise TimeoutError('fixed2950 training incomplete; no prefix candidate')
        sl=slice((step-1)*32,step*32);sidx=si[sl];tidx=ti[sl];b=bn[sl];order=ordinal[sl];white=noise['white'][order];z=noise['z'][order];style=styles[order]
        p=coupled['within_batch_target_position'][sl].astype('i8');assert np.array_equal(tidx[p],coupled['matched_target'][sl])
        assert np.array_equal(np.sort(tidx[p]),np.sort(tidx))and np.array_equal(b[p],b);marginal_checks+=1
        bt=torch.from_numpy(b).to(DEVICE);x=torch.from_numpy(source[sidx]).to(DEVICE);wt=torch.from_numpy(white).to(DEVICE)
        a,_,_=initial(x,wt,mt[bt],st[bt]);target=torch.from_numpy(real[tidx[p]]).to(DEVICE,dtype=torch.float64);tq=q_torch(target)
        c=(target/tq[:,None]-mt[bt])/st[bt];ts=torch.from_numpy(tt_all[sl]).to(DEVICE,dtype=torch.float64);y=(1-ts[:,None])*a+ts[:,None]*c;v=c-a
        opt.zero_grad(set_to_none=True);pred=field(y,ts,bt);loss=(pred-v).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite velocity MSE')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
        gg=model.embedding[0].weight.grad
        assert gg is not None and torch.count_nonzero(gg[:,1:17])==0
        zero_grad_checks+=1;t_nonzero_steps+=int(torch.count_nonzero(gg[:,0])>0);bin_nonzero_steps+=int(torch.count_nonzero(gg[:,17:])>0)
        other_nonzero_steps+=int(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in model.named_parameters()if n!='embedding.0.weight'))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        assert torch.equal(model.embedding[0].weight[:,1:17],initial_zero_columns);zero_value_checks+=1
        updates=step;counter['velocity_updates']+=1;counter['training_source_rows']+=32;counter['training_REAL_rows']+=32
        path=float(q_torch(st[bt]*v).abs().max());tangent=float(q_torch(st[bt]*pred.detach()).abs().max());max_path_charge=max(max_path_charge,path);max_field_charge=max(max_field_charge,tangent)
        dense.append([float(loss.detach()),float(v.square().mean().sqrt()),float(pred.detach().square().mean().sqrt()),float(norm),path,tangent])
        perms[sl]=p;matched[sl]=tidx[p];src_key[sl]=coupled['source_raw_PSD'][sl];tgt_key[sl]=coupled['target_original_PSD'][sl]
        for k,vv in [('source',sidx),('target',tidx),('bin',b),('original_row_ordinal',order),('style',style),('t',tt_all[sl]),('white',white)]:hashes[k].update(vv.tobytes())
        hashes['z_white'].update(z.tobytes());hashes['z_white'].update(white.tobytes())
        if DEVICE.type=='cuda'and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU cap')
        if step%250==0 or step==2950:
            row={'step':step,'velocity_MSE':dense[-1][0],'target_velocity_RMS':dense[-1][1],'predicted_velocity_RMS':dense[-1][2],'preclip_gradient_norm':dense[-1][3],'max_path_q_residual':max_path_charge,'max_field_q_residual':max_field_charge,'elapsed_s_including_setup':time.monotonic()-START,'sorting_s':sorting_s}
            trace.append(row);write(A+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',{'updates':updates,'counter':counter});log(row)
    assert all(torch.isfinite(p).all()for p in model.parameters())
    torch.save(state(model),dest/'FINAL_FLOW.pt');torch.save(opt.state_dict(),dest/'FINAL_OPTIMIZER.pt')
    # Actual old z/style/NCE are retained only as audit. They were never network/loss inputs.
    np.savez_compressed(dest/'ACTUAL_CFM_STREAM_PRIVATE.npz',source=si,target=ti,bin=bn,original_row_ordinal=ordinal,t=tt_all,matched_target=matched,within_batch_target_position=perms,source_raw_PSD=src_key,target_original_PSD=tgt_key,unused_NCE_positions=actual['NCE_positions'])
    np.savez_compressed(dest/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'))
    actual_hashes={k:h.hexdigest()for k,h in hashes.items()};same={k:actual_hashes[k]==reference['stream_hashes'][k]for k in ['source','target','bin','original_row_ordinal','z_white','style','t']}
    assert all(same.values())and marginal_checks==2950
    rec={'PID':os.getpid(),'updates':updates,'actual_rows':94400,'training_s':time.monotonic()-START,'selection':'fixed final2950 only','stream_hashes':actual_hashes,'original_7_audit_streams_match':same,'new_model_no_pretrained_weights':True,'white_used_z_style_NCE_unused':True,
        'actual_unique_source_REAL':[len(np.unique(si)),len(np.unique(ti))],'loaded_source_REAL':[len(source),len(real)],'marginal_preservation_checks':marginal_checks,'PSD_stable_coupling_statistical_not_truth':True,'exact_saved_R42_coupling_reused':True,'constants_refits':0,'time_features':'scalar t plus16zeros then8bins','initial_t_RNG':rt_initial,'final_t_RNG':rt.bit_generator.state,
        'max_path_q_residual':max_path_charge,'max_field_q_residual':max_field_charge,'parameter_count':sum(p.numel()for p in model.parameters()),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0,'network_training_forward_rows':94400,'artificial_zero_check_rows':4,'old_G_D_reader_forwards_updates':0}
    parameter_checks={'zero_input_columns':list(range(1,17)),'zero_input_column_weight_count':2048,'zero_gradient_update_checks':zero_grad_checks,'unchanged_value_update_checks':zero_value_checks,'initial_final_zero_columns_bitwise_equal':bool(torch.equal(model.embedding[0].weight[:,1:17],initial_zero_columns)),'scalar_t_nonzero_gradient_steps':t_nonzero_steps,'bin_inputs_nonzero_gradient_steps':bin_nonzero_steps,'other_parameters_nonzero_gradient_steps':other_nonzero_steps,'total_parameters':rec['parameter_count'],'maximum_data_gradient_participating_parameter_count':rec['parameter_count']-2048,'all_parameters_finite':all(torch.isfinite(p).all()for p in model.parameters()),'final_zero_columns_gradient_maxabs':float(model.embedding[0].weight.grad[:,1:17].abs().max()),'actual_time17_and_embedding25_saved':True}
    assert zero_grad_checks==zero_value_checks==2950 and parameter_checks['initial_final_zero_columns_bitwise_equal'] and t_nonzero_steps>0 and bin_nonzero_steps>0
    write(A+'/SCALAR_TIME_PARTICIPATION.json',parameter_checks)
    write(A+'/TRAINING_RECEIPT.json',rec);write('MATCHED_TRAINING_FLOW_RECEIPT.json',{'all2950':True,'same_7_audit_streams':same,'original_white_sha256':actual_hashes['white'],'original_t_SHA_match':True,'marginals_preserved_each_update':True,'unused_inputs':['z','style','NCE_positions'],'old_G_initial_J20_fresh_not_applicable':True})
    write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counter':counter,'records':{A:rec},'device':str(DEVICE),'GPU_peak_allocated_bytes':rec['GPU_peak_allocated_bytes']})
    log({'complete':True,'updates':updates,'counter':counter})
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counter':counter,'attempted_step':step,'completed_updates':updates});raise
