from pathlib import Path
from datetime import datetime,timezone
import json,os,time,sys,hashlib,copy,traceback
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from noise_core import q_np,bins_for,innovation
from source_flow_core import FlowNetwork,TangentField,q_torch

def write(name,v):(R/name).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
def utc():return datetime.now(timezone.utc).isoformat()
def tensors_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train(completion):
    arm=C['arms'][int(completion)];D=R/arm;D.mkdir(exist_ok=False)
    counter={'response_fit_attempts':0,'velocity_updates':0,'SOURCE_training_rows':0,'REAL_identity_rows':0,'REAL_matched_rows':0,
             'old_G_D_reader_forward_updates':0}
    updates=0;records=[];gradient_records=[];dense=[];max_q=0.
    try:
        assert (R/'NOISE_CALIBRATION.json').exists()
        torch.set_num_threads(12);torch.set_num_interop_threads(12)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        dev=torch.device('cuda:0')
        if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        env=dict(PID=os.getpid(),UTC=utc(),python=sys.version,numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,
                 device=str(dev),GPU=torch.cuda.get_device_name(0),CPU_threads=torch.get_num_threads(),cwd=str(R),arm=arm)
        write(arm+'/ENVIRONMENT.json',env)
        if any(env[k]!=C['expected_R33_environment'][k] for k in ['python','numpy','torch','scipy','device','GPU']):
            raise RuntimeError(('registered environment drift',env))
        with np.load(C['TRAIN_path'],allow_pickle=False) as z:sfull=z['source'].copy();rfull=z['target'].copy()
        with np.load(C['constants_path'],allow_pickle=False) as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
        with np.load(C['original_stream'],allow_pickle=False) as z:
            si=z['source'].astype('i8');mi=z['matched_target'].astype('i8');oldbin=z['bin'].astype('i8');ordinal=z['original_row_ordinal'].copy()
            unused_NCE_shape=list(z['unused_NCE_positions'].shape)
        with np.load(R/'NOISE_CALIBRATION.npz',allow_pickle=False) as z:noise={k:z[k].copy()for k in z.files}
        assert all(a.shape==(94400,) for a in [si,mi,oldbin,ordinal])
        assert si.min()>=0 and si.max()<len(sfull) and mi.min()>=0 and mi.max()<len(rfull)
        us,inv_s=np.unique(si,return_inverse=True);ur,inv_r=np.unique(mi,return_inverse=True)
        source=sfull[us].copy();real=rfull[ur].copy();loaded=[len(sfull),len(rfull)];del sfull,rfull
        sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
        assert np.isfinite(mu).all() and np.isfinite(scale).all() and (scale>0).all()
        assert np.max(np.abs(q_np(mu)-1))<1e-10
        uS=torch.as_tensor(source.astype('f8')/sq[:,None],device=dev)
        uR=torch.as_tensor(real.astype('f8')/rq[:,None],device=dev)
        mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev)
        torch.manual_seed(C['model_seed']);torch.cuda.manual_seed_all(C['model_seed'])
        init_rng_cpu=torch.get_rng_state().clone();init_rng_cuda=torch.cuda.get_rng_state().clone()
        model=FlowNetwork().to(dev);initial_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        ih=tensors_hash(initial_state);common=R/'COMMON_INITIAL_FLOW.pt'
        if completion:
            old=torch.load(common,map_location='cpu',weights_only=True)
            assert set(old)==set(initial_state) and all(torch.equal(v,old[k])for k,v in initial_state.items())
            common_equal=True;del old
        else:
            torch.save(initial_state,common);common_equal=True
            torch.save({'before_construction_CPU':init_rng_cpu,'before_construction_CUDA':init_rng_cuda,
                        'after_construction_CPU':torch.get_rng_state(),'after_construction_CUDA':torch.cuda.get_rng_state()},R/'COMMON_INITIAL_TORCH_RNG.pt')
        initial_zero=model.embedding[0].weight[:,1:17].detach().clone()
        write(arm+'/INITIALIZATION.json',{'PID':os.getpid(),'seed':C['model_seed'],'state_tensor_SHA':ih,'common_from_zero_tensorwise_equal':common_equal,
            'loaded_previous_response':False,'parameter_count':sum(p.numel()for p in model.parameters()),
            'three_channels':['state','fixed_full_guide','position'],'zero_output_initialization':True,'2048_inactive_time_weights':2048})
        opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        streams={k:np.random.Generator(np.random.PCG64(seed)) for k,seed in C['PCG64_seeds'].items()}
        write(arm+'/INITIAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()})
        hashes={k:hashlib.sha256()for k in ['source','matched_target','ordinal','source_t','source_epsilon','identity_t','identity_epsilon']}
        batch_counts=np.zeros(2950,dtype='u4');zero_grad_count=zero_value_count=0;t_nonzero=bin_nonzero=0
        input_amplitude={'SOURCE_triggered':0,'REAL_triggered':0,'SOURCE_total':0,'REAL_total':0}
        counter['response_fit_attempts']=1
        for step in range(1,50001):
            if time.monotonic()-START>=C['training_wall_s']-20 or time.time()>=datetime.fromisoformat(C['deadline_UTC']).timestamp()-900:
                raise TimeoutError('fixed50000 incomplete; no prefix candidate')
            batch=(step-1)%2950;sl=slice(batch*32,(batch+1)*32);ss=inv_s[sl];rr=inv_r[sl]
            sx=source[ss];rx=real[rr]
            ts=streams['source_t'].uniform(0,1,32).astype('f4')
            es=streams['source_epsilon'].standard_normal((32,480)).astype('f4')
            ti=streams['identity_t'].uniform(0,1,32).astype('f4')
            ei=streams['identity_epsilon'].standard_normal((32,480)).astype('f4')
            ns,ms=innovation(sx,es,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),completion)
            ni,mr=innovation(rx,ei,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),completion)
            bs=torch.as_tensor(sb[ss],device=dev);br=torch.as_tensor(rb[rr],device=dev)
            gS=(uS[ss]-mt[bs])/st[bs];gR=(uR[rr]-mt[br])/st[br]
            aS=gS+torch.as_tensor(ns/sq[ss,None],device=dev)/st[bs]
            aR=gR+torch.as_tensor(ni/rq[rr,None],device=dev)/st[br]
            cS=(uR[rr]-mt[bs])/st[bs]
            tt=torch.as_tensor(np.concatenate((ts,ti)),dtype=torch.float64,device=dev)
            a=torch.cat((aS,aR));c=torch.cat((cS,gR));guide=torch.cat((gS,gR));bt=torch.cat((bs,br))
            y=(1-tt[:,None])*a+tt[:,None]*c;velocity=c-a
            opt.zero_grad(set_to_none=True);pred=TangentField(model,mt,st,guide)(y,tt,bt)
            ls=(pred[:32]-velocity[:32]).square().mean();li=(pred[32:]-velocity[32:]).square().mean();loss=.5*ls+.5*li
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite two-trajectory loss')
            if step in [1,50000]:
                norms=[]
                for component in [ls,li]:
                    gs=torch.autograd.grad(component,tuple(model.parameters()),retain_graph=True,allow_unused=True)
                    norms.append(float(torch.sqrt(sum((g.double().square().sum()for g in gs if g is not None)))))
                gradient_records.append({'step':step,'unweighted_SOURCE_gradient_norm':norms[0],'unweighted_IDENTITY_gradient_norm':norms[1],
                                          'weights':[.5,.5],'extra_backward_only_no_optimizer_updates':True})
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
            gg=model.embedding[0].weight.grad
            assert gg is not None and torch.count_nonzero(gg[:,1:17])==0;zero_grad_count+=1
            t_nonzero+=int(torch.count_nonzero(gg[:,0])>0);bin_nonzero+=int(torch.count_nonzero(gg[:,17:])>0)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            updates=step;counter['velocity_updates']+=1;counter['SOURCE_training_rows']+=32
            counter['REAL_matched_rows']+=32;counter['REAL_identity_rows']+=32
            assert torch.equal(model.embedding[0].weight[:,1:17],initial_zero);zero_value_count+=1
            path_q=float(q_torch(st[bt]*velocity).abs().max());field_q=float(q_torch(st[bt]*pred.detach()).abs().max())
            max_q=max(max_q,path_q,field_q);dense.append([float(ls.detach()),float(li.detach()),float(loss.detach()),float(norm),path_q,field_q])
            input_amplitude['SOURCE_triggered']+=int(np.count_nonzero(ms['noise_completion_triggered']))
            input_amplitude['REAL_triggered']+=int(np.count_nonzero(mr['noise_completion_triggered']))
            input_amplitude['SOURCE_total']+=32;input_amplitude['REAL_total']+=32
            for k,v in [('source',si[sl]),('matched_target',mi[sl]),('ordinal',ordinal[sl]),('source_t',ts),('source_epsilon',es),('identity_t',ti),('identity_epsilon',ei)]:hashes[k].update(v.tobytes())
            batch_counts[batch]+=1
            if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes'] or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:
                raise MemoryError('GPU allocated or reserved cap')
            if step%1000==0 or step in [1,50000]:
                rec=dict(arm=arm,global_step=step,source_loss=dense[-1][0],identity_loss=dense[-1][1],total_loss=dense[-1][2],gradient_norm=dense[-1][3],
                         elapsed_s=time.monotonic()-START,max_path_field_q_residual=max_q,GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved())
                records.append(rec);write(arm+'/TRAINING_TRACE.json',records);write('TRAIN_PROGRESS.json',rec);print(json.dumps(rec),flush=True)
        assert updates==50000 and all(torch.isfinite(p).all()for p in model.parameters())
        torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt')
        # Final optimizer is needed for actual continuation evidence, not checkpoint selection.
        torch.save(opt.state_dict(),D/'FINAL_OPTIMIZER.pt')
        torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state()},D/'FINAL_TORCH_RNG.pt')
        write(arm+'/FINAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state)for k,v in streams.items()})
        np.savez_compressed(D/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'))
        rec=dict(PID=os.getpid(),UTC=utc(),arm=arm,updates=updates,elapsed_s=time.monotonic()-START,counter=counter,selection='sole50000 final',
                 original_loaded_source_REAL=loaded,actual_unique_SOURCE_REAL=[len(us),len(ur)],original_rows=94400,
                 stream_hashes={k:h.hexdigest()for k,h in hashes.items()},original_batch_occurrence_counts=batch_counts.tolist(),
                 completion=bool(completion),input_amplitudes=input_amplitude,initial_state_SHA=ih,
                 source_actual_bin_mismatch=int(np.count_nonzero(sb[inv_s]!=oldbin)),identity_actual_bin_mismatch=int(np.count_nonzero(rb[inv_r]!=oldbin)),
                 zero_input_weights_count=2048,zero_input_gradient_checks=zero_grad_count,zero_input_value_checks=zero_value_count,
                 t_nonzero_gradient_steps=t_nonzero,bin_nonzero_gradient_steps=bin_nonzero,
                 GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                 unused_original_z_style_NCE=True,original_NCE_shape=unused_NCE_shape,
                 branch_gradient_descriptions=gradient_records,unique_parent_population_increased=False)
        if completion:
            control=json.loads((R/C['arms'][0]/'TRAINING_RECEIPT.json').read_text('utf8'))
            rec['paired_stream_equality']={k:rec['stream_hashes'][k]==control['stream_hashes'][k]for k in hashes}
            assert all(rec['paired_stream_equality'].values()) and control['updates']==50000
            assert rec['initial_state_SHA']==control['initial_state_SHA']
        write(arm+'/TRAINING_RECEIPT.json',rec);print(json.dumps({'training_terminal':rec}),flush=True)
    except BaseException as e:
        write(arm+'/TRAINING_FAILURE.json',dict(PID=os.getpid(),UTC=utc(),error=repr(e),updates=updates,counter=counter,elapsed_s=time.monotonic()-START,traceback=traceback.format_exc()))
        raise

