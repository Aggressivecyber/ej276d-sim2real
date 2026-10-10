from __future__ import annotations
import os, sys, json, time, hashlib, copy, shutil, traceback
from pathlib import Path
from datetime import datetime, timezone

START = time.monotonic()
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
import numpy as np
import torch
import scipy
from source_flow_core import FlowNetwork, TangentField, initial, q_torch, q_np
from probe_core import build_probe, fixed_probe, fast

torch.set_num_threads(12); torch.set_num_interop_threads(12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
DEVICE = torch.device('cuda:0')
assert torch.cuda.is_available()
torch.cuda.set_per_process_memory_fraction(CFG['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
A = 'EXTENDED_FLOW_FINAL'
COUNTER = {'inherited_updates': 2950, 'new_updates': 0, 'new_source_rows': 0, 'new_REAL_rows': 0,
           'training_forward_rows': 0, 'fixed_probe_forward_rows': 0, 'probe_executions': 0,
           'new_response_training_runs': 0, 'DEV_read': 0, 'CAL_EVAL_G4_decode': 0}


def utc(): return datetime.now(timezone.utc).isoformat()
def write(name, data):
    p = ROOT/name; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)
def state(model): return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
def check_time():
    if time.monotonic()-START > CFG['training_wall_s']-30:
        raise TimeoutError('training science limit; prefix is not a candidate')
    if time.time() > datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-900:
        raise TimeoutError('seal margin reached; prefix is not a candidate')
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise OSError('8GiB disk reserve reached')
    if torch.cuda.max_memory_allocated()>CFG['GPU_limit_bytes'] or torch.cuda.max_memory_reserved()>CFG['GPU_limit_bytes']:
        raise MemoryError('GPU allocated/reserved cap reached')


def main():
    env = {'PID':os.getpid(),'UTC':utc(),'cwd':str(ROOT),'argv':sys.argv,'python':sys.version,
           'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
           'device':str(DEVICE),'GPU':torch.cuda.get_device_name(0),'CPU_threads':12}
    write('SERVER_ENVIRONMENT.json',env)
    expected = CFG['expected_R33_environment']
    assert all(env[k]==expected[k] for k in ['python','numpy','torch','scipy','device','GPU'])
    with np.load(CFG['TRAIN_path'],allow_pickle=False) as z:
        source=z['source'].copy(); real=z['target'].copy()
    with np.load(CFG['inherited_coupling'],allow_pickle=False) as z:
        coupled={k:z[k].copy() for k in z.files}
    old_receipt=json.loads(Path(CFG['inherited_train_receipt']).read_text())
    assert old_receipt['updates']==2950 and old_receipt['actual_rows']==94400
    assert all(len(coupled[k])==94400 for k in ['source','target','bin','matched_target','original_row_ordinal','t'])
    for key in ['source','target','bin','original_row_ordinal','t']:
        assert hashlib.sha256(coupled[key].tobytes()).hexdigest()==old_receipt['stream_hashes'][key]
    assert np.array_equal(coupled['target'].reshape(2950,32)[np.arange(2950)[:,None],coupled['within_batch_target_position'].reshape(2950,32)],coupled['matched_target'].reshape(2950,32))
    for key,name in [('constants_from','UNIT_CHARGE_CONSTANTS.npz'),('constants_json_from','UNIT_CHARGE_CONSTANTS.json')]:
        shutil.copy2(CFG[key],ROOT/name)
    with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False) as z:
        mu=z['mu_unit_q'].copy(); scale=z['scale_unit_q'].copy(); edges=z['edges_logQ'].copy()
    mt=torch.as_tensor(mu,dtype=torch.float64,device=DEVICE)
    st=torch.as_tensor(scale,dtype=torch.float64,device=DEVICE)
    torch.manual_seed(CFG['model_seed'])
    model=FlowNetwork().to(DEVICE)
    inherited=torch.load(CFG['inherited_weight'],map_location='cpu',weights_only=True)
    model.load_state_dict(inherited)
    initial_parameters={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    dest=ROOT/A; dest.mkdir(exist_ok=False)
    opt=torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=0)
    inherited_optimizer=torch.load(CFG['inherited_optimizer'],map_location='cpu',weights_only=True)
    opt.load_state_dict(inherited_optimizer)
    steps=[int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in opt.state.values()]
    assert steps and set(steps)=={2950}
    for group in opt.param_groups:
        assert group['lr']==.0002 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    zero_columns=model.embedding[0].weight[:,1:17].detach().clone()
    inherited_zero=opt.state[model.embedding[0].weight]
    assert torch.count_nonzero(inherited_zero['exp_avg'][:,1:17])==0 and torch.count_nonzero(inherited_zero['exp_avg_sq'][:,1:17])==0
    torch.save(state(model),dest/'INHERITED_FLOW.pt'); torch.save(opt.state_dict(),dest/'INHERITED_OPTIMIZER.pt')
    torch.save(torch.get_rng_state(),dest/'INITIAL_TORCH_STATE.pt'); torch.save(torch.cuda.get_rng_state(),dest/'INITIAL_CUDA_STATE.pt')
    write('INHERITANCE_RECEIPT.json',{'PID':os.getpid(),'source_model':CFG['inherited_weight'],'source_optimizer':CFG['inherited_optimizer'],
          'model_all_state_tensor_equal':all(torch.equal(v,inherited[k]) for k,v in state(model).items()),
          'inherited_optimizer_step_count':len(steps),'inherited_optimizer_steps':sorted(set(steps)),
          'zero_input_column_count':2048,'zero_column_optimizer_moments_are_zero':True,
          'historical_updates_cost_not_free':True,'historical_training_s_worker':old_receipt['training_s'],
          'new_initialization_or_old_G_D_reader':False,'parameter_count':sum(p.numel() for p in model.parameters())})
    probe,probe_info=build_probe(coupled,source,real)
    np.savez_compressed(ROOT/'FIXED_TRAIN_PROBE_PRIVATE.npz',**probe)
    write('FIXED_TRAIN_PROBE_BINDING.json',probe_info)
    rt=np.random.Generator(np.random.PCG64()); rt.bit_generator.state=copy.deepcopy(old_receipt['final_t_RNG'])
    rw=np.random.Generator(np.random.PCG64(20261010452))
    t_initial=copy.deepcopy(rt.bit_generator.state); white_initial=copy.deepcopy(rw.bit_generator.state)
    field=TangentField(model,mt,st)
    probes=[]
    def probe_at(k):
        check_time(); before=field.rows
        rec,perrow=fixed_probe(model,field,probe,mt,st,initial_parameters,k,DEVICE)
        assert field.rows-before==4096
        COUNTER['probe_executions']+=1; COUNTER['fixed_probe_forward_rows']+=4096
        probes.append(rec)
        np.savez_compressed(ROOT/'probes'/('PROBE_'+str(k)+'.npz'),metrics=perrow)
        write('FIXED_PROBE_TRAJECTORY.json',probes)
        print(json.dumps({'probe':rec['global_step'],'MSE480':rec['overall']['MSE480'],'PSD_RMSE':rec['overall']['PSD_RMSE'],'PSD_signed_mean':rec['overall']['PSD_signed_mean'],'elapsed_s':time.monotonic()-START}),flush=True)
    (ROOT/'probes').mkdir()
    probe_at(2950)
    metric_names=['velocity_MSE480','target_velocity_RMS480','predicted_velocity_RMS480','preclip_grad_norm',
                  'path_q_residual','field_q_residual','PSD_velocity_signed_residual','PSD_velocity_RMSE',
                  't_mean','t_SD_population']
    dense=np.lib.format.open_memmap(dest/'LOSS_BY_UPDATE_PRIVATE.npy',mode='w+',dtype='f8',shape=(47050,len(metric_names)))
    dtype=np.dtype([('global_step','<u4'),('original_batch0','<u2'),('t_sha256','S64'),('white_sha256','S64')])
    batch_hashes=np.lib.format.open_memmap(dest/'BATCH_HASHES_PRIVATE.npy',mode='w+',dtype=dtype,shape=(47050,))
    exposure=np.zeros(2950,'u4'); trace=[]; t_hash=hashlib.sha256(); white_hash=hashlib.sha256()
    zeros_checked=t_nonzero=bin_nonzero=other_nonzero=finite_parameters_checks=0
    max_path=max_field=0.
    counter_initial=copy.deepcopy(COUNTER)
    COUNTER['new_response_training_runs']=1
    model.train()
    for global_step in range(2951,50001):
        if (global_step-2951)%50==0:check_time()
        j=(global_step-1)%2950; sl=slice(j*32,(j+1)*32)
        si=coupled['source'][sl].astype('i8'); ri=coupled['matched_target'][sl].astype('i8'); bn=coupled['bin'][sl].astype('i8')
        times=rt.uniform(0,1,32).astype('f4'); white=rw.standard_normal((32,480)).astype('f4')
        bt=torch.as_tensor(bn,dtype=torch.int64,device=DEVICE)
        x=torch.as_tensor(source[si],device=DEVICE); w=torch.as_tensor(white,device=DEVICE)
        a,_,_=initial(x,w,mt[bt],st[bt]); target=torch.as_tensor(real[ri],dtype=torch.float64,device=DEVICE)
        c=(target/q_torch(target)[:,None]-mt[bt])/st[bt]
        ts=torch.as_tensor(times,dtype=torch.float64,device=DEVICE); y=(1-ts[:,None])*a+ts[:,None]*c; v=c-a
        opt.zero_grad(set_to_none=True); pred=field(y,ts,bt); loss=(pred-v).square().mean()
        COUNTER['training_forward_rows']+=32
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite velocity MSE')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
        gg=model.embedding[0].weight.grad
        assert gg is not None and torch.count_nonzero(gg[:,1:17])==0
        t_nonzero+=int(torch.count_nonzero(gg[:,0])>0); bin_nonzero+=int(torch.count_nonzero(gg[:,17:])>0)
        other_nonzero+=int(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in model.named_parameters() if n!='embedding.0.weight'))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        COUNTER['new_updates']+=1; COUNTER['new_source_rows']+=32; COUNTER['new_REAL_rows']+=32
        assert all(torch.isfinite(p).all() for p in model.parameters()); finite_parameters_checks+=1
        assert torch.equal(model.embedding[0].weight[:,1:17],zero_columns); zeros_checked+=1
        path=float(q_torch(st[bt]*v).abs().max()); tangent=float(q_torch(st[bt]*pred.detach()).abs().max())
        max_path=max(max_path,path);max_field=max(max_field,tangent)
        ep=-fast(st[bt]*(pred.detach()-v)); index=global_step-2951
        dense[index]=[float(loss.detach()),float(v.square().mean().sqrt()),float(pred.detach().square().mean().sqrt()),float(norm),path,tangent,float(ep.mean()),float(ep.square().mean().sqrt()),float(times.mean()),float(times.std())]
        batch_hashes[index]=(global_step,j,hashlib.sha256(times.tobytes()).hexdigest().encode(),hashlib.sha256(white.tobytes()).hexdigest().encode())
        t_hash.update(times.tobytes());white_hash.update(white.tobytes());exposure[j]+=1
        if global_step%250==0 or global_step==50000:
            dense.flush();batch_hashes.flush()
            rec={'global_step':global_step,'new_updates':COUNTER['new_updates'],'last_MSE480':float(dense[index,0]),'preclip_gradient_norm':float(norm),
                 'max_path_q_residual':max_path,'max_field_q_residual':max_field,'elapsed_s':time.monotonic()-START}
            trace.append(rec);write(A+'/TRAINING_TRACE.json',trace)
            write('TRAIN_PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'counter':COUNTER,'global_step':global_step,'last':rec})
            print(json.dumps(rec),flush=True)
        if global_step%5000==0:
            probe_at(global_step)
            cp=ROOT/'checkpoints'/str(global_step);cp.mkdir(parents=True,exist_ok=False)
            torch.save(state(model),cp/'FLOW.pt');torch.save(opt.state_dict(),cp/'OPTIMIZER.pt')
            write('checkpoints/'+str(global_step)+'/STATE.json',{'global_step':global_step,'counter':COUNTER,'t_RNG':copy.deepcopy(rt.bit_generator.state),
                  'white_RNG':copy.deepcopy(rw.bit_generator.state),'t_prefix_sha256':t_hash.hexdigest(),'white_prefix_sha256':white_hash.hexdigest(),
                  'batch_exposure':exposure.tolist(),'dense_committed_updates':COUNTER['new_updates'],'engineering_recovery_only_not_DEV_candidate':True})
    dense.flush();batch_hashes.flush();check_time()
    assert COUNTER['new_updates']==47050 and COUNTER['new_source_rows']==1505600
    assert [r['global_step'] for r in probes]==[2950]+list(range(5000,50001,5000))
    assert COUNTER['probe_executions']==11 and COUNTER['fixed_probe_forward_rows']==45056
    final_steps=[int(s['step'].item()) for s in opt.state.values()]
    assert set(final_steps)=={50000} and zeros_checked==finite_parameters_checks==47050
    torch.save(state(model),dest/'FINAL_FLOW.pt');torch.save(opt.state_dict(),dest/'FINAL_OPTIMIZER.pt')
    torch.save(torch.get_rng_state(),dest/'FINAL_TORCH_STATE.pt');torch.save(torch.cuda.get_rng_state(),dest/'FINAL_CUDA_STATE.pt')
    write(A+'/RANDOM_STATES.json',{'training_t_initial':t_initial,'training_t_final':copy.deepcopy(rt.bit_generator.state),
          'training_white_initial':white_initial,'training_white_final':copy.deepcopy(rw.bit_generator.state),
          'new_training_white_seed':20261010452,'t_sha256':t_hash.hexdigest(),'white_sha256':white_hash.hexdigest(),
          't_draw':'uniform(0,1,32).astype(float32) each update','white_draw':'standard_normal((32,480)).astype(float32) each update',
          'global_updates':[2951,50000],'original_batch0':'(global_step-1)%2950','source_random_z_style_NCE_inputs':False})
    write(A+'/EXPOSURE_SUMMARY.json',{'new_rows_per_domain':1505600,'historical_rows_per_domain':94400,'total_rows_per_domain':1600000,
          'original_batch_counts_new':exposure.tolist(),'new_exposure_row_counts':(np.repeat(exposure,32)).tolist(),
          'batch_count_min_max':[int(exposure.min()),int(exposure.max())],
          'unique_SOURCE_REAL':[len(np.unique(coupled['source'])),len(np.unique(coupled['matched_target']))],
          'loaded_SOURCE_REAL':[len(source),len(real)],'exposure_not_independent_events':True})
    write('MODEL_STRUCTURE.json',{'parameters':sum(p.numel() for p in model.parameters()),'time17':'[t,zero16]','new_parameters':0,
          'old_G_D_reader_used':False,'2048_zero_columns_gradient_checks':zeros_checked,'2048_zero_columns_value_checks':zeros_checked,
          't_nonzero_gradient_steps':t_nonzero,'bin_nonzero_gradient_steps':bin_nonzero,'other_nonzero_gradient_steps':other_nonzero,
          'all_parameter_finite_checks':finite_parameters_checks})
    rec={'PID':os.getpid(),'UTC':utc(),'training_s':time.monotonic()-START,'status':'complete','counter':COUNTER,'global_updates':50000,
         'selected_endpoint':'fixed50000 only','historical_counter_initial':counter_initial,'optimizer_final_steps':sorted(set(final_steps)),
         'loss_columns':metric_names,'fixed_probe_MSE_initial':probes[0]['overall']['MSE480'],'fixed_probe_MSE_final':probes[-1]['overall']['MSE480'],
         'probe_final_over_initial_MSE':probes[-1]['overall']['MSE480']/probes[0]['overall']['MSE480'],
         'descriptive_probe_fit_improved':probes[-1]['overall']['MSE480']<=.9*probes[0]['overall']['MSE480'],
         'original_bound_stream_hashes':old_receipt['stream_hashes'],'new_t_sha256':t_hash.hexdigest(),'new_white_sha256':white_hash.hexdigest(),
         'additional_Monte_Carlo_states_not_reuse_old94400_white':True,'unused_z_style_NCE_not_consumed':True,
         'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
         'max_path_q_residual':max_path,'max_field_q_residual':max_field,'historical_R43_training_s':old_receipt['training_s']}
    write('TRAINING_NUMERIC_RECEIPT.json',rec);write(A+'/TRAINING_RECEIPT.json',rec)
    print(json.dumps({'training_complete':rec}),flush=True)


try:
    main()
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'UTC':utc(),'error':repr(e),'traceback':traceback.format_exc(),
          'elapsed_s':time.monotonic()-START,'counter':COUNTER,'completed_global_updates':2950+COUNTER['new_updates'],
          'prefix_not_candidate':True,'recovery_requires_last_saved_engineering_checkpoint_no_unknown_resubmit':True})
    raise
