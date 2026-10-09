from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,copy,sys
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from diffusion_core import Denoiser,schedule,physical_condition,formula_check
from flow_core import bins
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def state(m):return{k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
counter={'denoiser_fit_attempts':0,'denoiser_updates':0,'REAL_training_rows':0,'G_D_reader_flow_updates':0}
trace=[];rng_states=[];losses=[];updates=0;step=0;clock=None
env={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cwd':str(ROOT),'python':sys.version,
    'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(DEVICE),
    'GPU':torch.cuda.get_device_name(0)if DEVICE.type=='cuda'else None}
write('SERVER_ENVIRONMENT.json',env)
assert all(env[k]==CFG['expected_R33_environment'][k]for k in ['python','numpy','torch','scipy','device','GPU'])
try:
    write('DIFFUSION_FORMULA_CHECK.json',formula_check());coeff=schedule()
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:real=z['target'].copy()
    with np.load(CFG['consumed_R26_stream'],allow_pickle=False)as z:
        ti=z['target'].astype('i8');bn=z['bin'].astype('i8');ordinal=z['original_row_ordinal'].copy()
    assert len(ti)==94400 and len(real)==21047 and len(bn)==94400
    with np.load(CFG['affine_path'],allow_pickle=False)as z:mu=z['mu_V'].copy();scale=z['scale_V'].copy();edges=z['edges_logQ'].copy()
    b_all=bins(real,edges);supported=(b_all>=0)&(b_all<8)
    assert int(supported.sum())==18895 and np.array_equal(b_all[ti],bn)
    c_all=np.full((len(real),2),np.nan,'f8');c_all[supported]=physical_condition(real[supported])
    cmean=c_all[supported].mean(0);csd=c_all[supported].std(0,ddof=1)
    assert np.isfinite(cmean).all()and np.isfinite(csd).all()and (csd>0).all()
    assert np.isfinite(mu).all()and np.isfinite(scale).all()and(scale>0).all()
    np.savez_compressed(ROOT/'DIFFUSION_CONSTANTS.npz',**coeff,mu_V=mu,scale_V=scale,edges_logQ=edges,
        condition_mean=cmean,condition_sample_sd=csd)
    write('CONDITION_CONSTANTS.json',{'supported_REAL_TRAIN_N':int(supported.sum()),'mean':cmean.tolist(),
        'sample_sd_ddof1':csd.tolist(),'coordinate_order':['logQtotal_mVns','PSD'],'no_clipping':True,'TRAIN_only':True})
    old=json.loads(pathlib.Path(CFG['original_training_receipt_path']).read_text('utf8'))['records']['INDEPENDENT_CFM']
    prehash={k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in [('target',ti),('bin',bn),('original_row_ordinal',ordinal)]}
    assert all(prehash[k]==old['stream_hashes'][k]for k in prehash),'R33 target/bin/row sequence drift'
    write('TRAINING_ROW_BINDING.json',{'target_bin_ordinal_sha256':prehash,'match_R33_original':True,
        'target_rows':94400,'source_training_reads_or_forwards':0,'unique_REAL_TRAIN_N':21047,'supported_REAL_N':18895})
    torch.manual_seed(CFG['model_seed']);np.random.seed(CFG['model_seed']%(2**32))
    model=Denoiser().to(DEVICE);dest=ROOT/'CONDITIONAL_DIFFUSION_EDIT';dest.mkdir(exist_ok=False)
    torch.save(state(model),dest/'INITIAL_DENOISER.pt');torch.save(torch.get_rng_state(),dest/'INITIAL_TORCH_STATE.pt')
    write('MODEL_STRUCTURE.json',{'PID':os.getpid(),'parameter_count':sum(p.numel()for p in model.parameters()),
        'trainable_parameter_count':sum(p.numel()for p in model.parameters()if p.requires_grad),'architecture':CFG['architecture'],
        'zero_last_layer':True,'position_channel':True,'fixed_recipe_before_first_update':True})
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    xr=torch.from_numpy(real).to(DEVICE);mt=torch.tensor(mu,dtype=torch.float32,device=DEVICE);st=torch.tensor(scale,dtype=torch.float32,device=DEVICE)
    ct=torch.tensor((c_all-cmean)/csd,dtype=torch.float32,device=DEVICE)
    ab=torch.tensor(coeff['abar'],dtype=torch.float32,device=DEVICE)
    rt=np.random.Generator(np.random.PCG64(CFG['training_noise_seed']));initial_rng=copy.deepcopy(rt.bit_generator.state)
    ht=hashlib.sha256();he=hashlib.sha256();hashes={k:hashlib.sha256()for k in prehash}
    counter['denoiser_fit_attempts']+=1;clock=time.monotonic()
    try:
        model.train()
        for step in range(1,CFG['updates']+1):
            if time.monotonic()-clock>=CFG['training_wall_s']-5:break
            if time.time()>=datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-600:break
            sl=slice((step-1)*32,step*32);ids=ti[sl];b=bn[sl]
            initial=copy.deepcopy(rt.bit_generator.state)
            t=rt.integers(1,101,size=32,dtype=np.int64);eps=rt.normal(size=(32,480)).astype('f4')
            rng_states.append({'step':step,'initial':initial,'final':copy.deepcopy(rt.bit_generator.state),
                't_sha256':hashlib.sha256(t.tobytes()).hexdigest(),'epsilon_f32_sha256':hashlib.sha256(eps.tobytes()).hexdigest()})
            tt=torch.from_numpy(t).to(DEVICE);bt=torch.from_numpy(b).to(DEVICE);ee=torch.from_numpy(eps).to(DEVICE)
            y0=(xr[ids]-mt[bt])/st[bt];a=ab[tt][:,None];yt=a.sqrt()*y0+(1.-a).sqrt()*ee
            optimizer.zero_grad(set_to_none=True);pred=model(yt,tt,bt,ct[ids]);loss=(pred-ee).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite epsilon loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            updates=step;counter['denoiser_updates']+=1;counter['REAL_training_rows']+=32
            ht.update(t.tobytes());he.update(eps.tobytes())
            for k,arr in [('target',ids),('bin',b),('original_row_ordinal',ordinal[sl])]:hashes[k].update(arr.tobytes())
            losses.append([float(loss.detach()),float(norm)])
            if DEVICE.type=='cuda'and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
            if step%250==0 or step==CFG['updates']:
                row={'step':step,'epsilon_MSE':losses[-1][0],'gradient_norm_before_clip':losses[-1][1],
                    'elapsed_training_s':time.monotonic()-clock,'rows':counter['REAL_training_rows']}
                trace.append(row);write('CONDITIONAL_DIFFUSION_EDIT/TRAINING_TRACE.json',trace)
                write('TRAIN_PROGRESS.json',{'updates':updates,'counter':counter});log(row)
        torch.save(state(model),dest/'FINAL_DENOISER.pt');torch.save(optimizer.state_dict(),dest/'FINAL_OPTIMIZER.pt')
    finally:
        elapsed=time.monotonic()-clock;n=updates*32
        np.savez_compressed(dest/'TRAINING_STREAM_PRIVATE.npz',target=ti[:n],bin=bn[:n],original_row_ordinal=ordinal[:n],
            loss_by_update=np.asarray(losses,dtype='f8'))
        write('CONDITIONAL_DIFFUSION_EDIT/ACTUAL_NOISE_STREAM_PRIVATE.json',{'initial':initial_rng,'final':rt.bit_generator.state,
            'batch_states_and_hashes':rng_states,'draw_order':'integers1..100 int64[32], normal f64[32,480] cast f32',
            'NumPy':np.__version__,'PCG64_seed':CFG['training_noise_seed']})
        rec={'PID':os.getpid(),'updates':updates,'rows':n,'training_s':elapsed,'selection':'actual final only',
            'stream_hashes':{k:h.hexdigest()for k,h in hashes.items()},'t_sha256':ht.hexdigest(),'epsilon_f32_sha256':he.hexdigest(),
            'condition_statistics_TRAIN_only':True,'original_R33_stream_full_match':{k:hashes[k].hexdigest()==old['stream_hashes'][k]for k in hashes},
            'model_seed':CFG['model_seed'],'epsilon_MSE_only':True,'G_D_reader_flow_updates':0,
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0,
            'parameter_count':sum(p.numel()for p in model.parameters())}
        write('CONDITIONAL_DIFFUSION_EDIT/TRAINING_RECEIPT.json',rec)
        write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_s':time.monotonic()-START,'counter':counter,'records':{'CONDITIONAL_DIFFUSION_EDIT':rec},'device':str(DEVICE),
            'GPU_peak_allocated_bytes':rec['GPU_peak_allocated_bytes']})
        if updates==2950:assert all(rec['original_R33_stream_full_match'].values())
    log({'complete':True,'updates':updates,'counter':counter})
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),
        'elapsed_s':time.monotonic()-START,'counter':counter,'attempted_step':step,'completed_updates':updates})
    raise
