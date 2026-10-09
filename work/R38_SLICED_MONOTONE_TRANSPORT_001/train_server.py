from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,sys,gc
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from model import Translator
from flow_core import bins
from reorder_core import reconstruct_and_reorder
from sliced_core import positive_qr,learn_directions,layer,selfcheck

torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
if DEVICE.type=='cuda':
    torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
ARM='SLICED_MONOTONE_TRANSPORT';DEST=ROOT/ARM;DEST.mkdir(exist_ok=False)
COUNTER={'sliced_transform_fit_attempts':0,'direction_groups_complete':0,'scalar_maps_complete':0,'direction_gradient_steps':0,
    'direction_zero_gradient_steps':0,'fixed_G_forward_training_rows':0,'REAL_training_rows':0,'neural_fit_or_updates':0,
    'direction_subset_projection_calls':0,'full_bin_projection_calls':0}
def write(n,obj):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n','utf8')
def log(obj):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(obj),flush=True)
def check_time():
    if time.monotonic()-START>CFG['statistics_fit_wall_s']-5:raise TimeoutError('unique sliced fit900s limit')
    if time.time()>datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-300:raise TimeoutError('stage deadline')
def charge_step():
    COUNTER['direction_gradient_steps']+=1
    COUNTER['direction_subset_projection_calls']+=4 # X/Y before and after each prescribed step.
def fingerprint(d):
    h=hashlib.sha256()
    for k,v in d.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

write('SERVER_ENVIRONMENT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
    'device':str(DEVICE),'direction_device':'CPU4 float64','GPU':torch.cuda.get_device_name(0)if DEVICE.type=='cuda'else None})
try:
    write('SLICED_FORMULA_CHECK.json',selfcheck())
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    actual=reconstruct_and_reorder(CFG,ROOT)
    with np.load(CFG['affine_path'],allow_pickle=False)as z:mu=z['mu_V'].copy();scale=z['scale_V'].copy();edges=z['edges_logQ'].copy()
    si=actual['source'].astype('i8');ti=actual['target'].astype('i8');bn=actual['bin'].astype('i8');ordinal=actual['original_row_ordinal']
    assert np.array_equal(bins(data['source'],edges)[si],bn)and np.array_equal(bins(data['target'],edges)[ti],bn),'original bin conflict'
    expected=json.loads(pathlib.Path(CFG['original_training_receipt_path']).read_text('utf8'))['records']['INDEPENDENT_CFM']['stream_hashes']
    hashes={k:hashlib.sha256()for k in ['source','target','bin','original_row_ordinal','z_white','style','base_preADC']}
    base=Translator().to(DEVICE);base.load_state_dict(torch.load(CFG['base_G'],map_location=DEVICE,weights_only=True));base.eval();base.requires_grad_(False)
    before=fingerprint(base.state_dict());xs=torch.from_numpy(data['source']).to(DEVICE)
    Xall=np.empty((94400,480),'f8');Yall=np.empty_like(Xall)
    zall=actual['original_noise']['z'];wall=actual['original_noise']['white'];styleall=actual['original_style']
    COUNTER['sliced_transform_fit_attempts']=1;block_hashes=[]
    for step in range(2950):
        check_time();sl=slice(step*32,(step+1)*32);ssi=si[sl];tti=ti[sl];b=bn[sl];order=ordinal[sl]
        z=zall[order];white=wall[order];style=styleall[order]
        COUNTER['fixed_G_forward_training_rows']+=32
        with torch.no_grad():g=base(xs[ssi],torch.from_numpy(z).to(DEVICE),torch.from_numpy(white).to(DEVICE),torch.from_numpy(style).to(DEVICE))[0].cpu().numpy()
        COUNTER['REAL_training_rows']+=32
        assert np.isfinite(g).all()
        Xall[sl]=(g.astype('f8')-mu[b])/scale[b];Yall[sl]=(data['target'][tti].astype('f8')-mu[b])/scale[b]
        for k,a in [('source',ssi),('target',tti),('bin',b),('original_row_ordinal',order),('style',style),('base_preADC',g)]:hashes[k].update(a.tobytes())
        hashes['z_white'].update(z.tobytes());hashes['z_white'].update(white.tobytes());block_hashes.append(hashlib.sha256(g.tobytes()).hexdigest())
        if DEVICE.type=='cuda'and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if(step+1)%500==0 or step==2949:
            progress={'phase':'frozen_G','blocks':step+1,'counter':COUNTER,'elapsed_s':time.monotonic()-START};write('STATISTICS_PROGRESS.json',progress);log(progress)
    observed={k:v.hexdigest()for k,v in hashes.items()};same={k:observed[k]==expected[k]for k in hashes}
    write('STATISTICS_INPUT_BINDING.json',{'rows':94400,'stream_hashes':observed,'expected_R33_stream_hashes':{k:expected[k]for k in hashes},
        'R33_experimental_stream_match':same,'base_GPU_chunk':32,'source_target_edge_records_match':True,
        'z_white_style_reconstructed_from_recorded_initial_and_final_states':True,'not_migration_content_testing':True})
    write('BASE_FORWARD_HASHES.json',{'rows_per_hash':32,'actual_base_preADC_hashes':block_hashes})
    assert all(same.values()),'R33 experimental stream/G mismatch; no refit'
    after=fingerprint(base.state_dict());assert after==before
    GPUpeak=torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0
    np.savez_compressed(DEST/'ACTUAL_STATISTICS_STREAM_PRIVATE.npz',source=si,target=ti,bin=bn,original_row_ordinal=ordinal)
    XX=[Xall[bn==k].copy()for k in range(8)];YY=[Yall[bn==k].copy()for k in range(8)]
    del Xall,Yall,xs,base,data,actual,zall,wall,styleall;gc.collect()
    if DEVICE.type=='cuda':torch.cuda.empty_cache()
    counts=[{'bin_id_1based':k+1,'source_draws':len(XX[k]),'target_draws':len(YY[k]),
        'distinct_source_row_indices':int(np.unique(si[bn==k]).size),'distinct_REAL_row_indices':int(np.unique(ti[bn==k]).size),
        'draws_not_independent_acquisition_N':True}for k in range(8)]
    assert all(len(x)==len(y)and len(x)>=2 for x,y in zip(XX,YY))
    rng=np.random.Generator(np.random.PCG64(CFG['direction_seed']))
    init=np.empty((32,8,480,16),'f8');ih=[]
    for l in range(32):
        for k in range(8):
            init[l,k]=positive_qr(rng.standard_normal((480,16)))
            ih.append({'layer0':l,'bin0':k,'A_initial_sha256':hashlib.sha256(init[l,k].tobytes()).hexdigest()})
    write('DIRECTION_INITIALIZATION.json',{'seed':CFG['direction_seed'],'PCG64':True,'NumPy':np.__version__,
        'order':'layer0..31 then bin0..7, 480x16 standard normal and reduced QR, positive R diagonal',
        'all_initial_A_sha256':hashlib.sha256(init.tobytes()).hexdigest(),'groups':ih})
    aa=np.empty((8,32,480,16),'f8');xks=np.empty((8,32,257,16),'f8');yks=np.empty_like(xks)
    positions=[np.floor((np.arange(min(1024,len(x)))+.5)*len(x)/min(1024,len(x))).astype('i8')for x in XX]
    q=np.linspace(.005,.995,257);traces=[]
    for l in range(32):
        for k in range(8):
            check_time();x=XX[k];y=YY[k];pos=positions[k]
            a,trace=learn_directions(x[pos],y[pos],init[l,k],CFG,check_time,charge_step)
            orth=float(np.max(np.abs(a.T@a-np.eye(16))));assert orth<=1e-10
            u=x@a;v=y@a;COUNTER['full_bin_projection_calls']+=2
            xk=np.quantile(u,q,axis=0,method='linear');yk=np.quantile(v,q,axis=0,method='linear')
            dx=np.diff(xk,axis=0);dy=np.diff(yk,axis=0)
            valid=bool(np.isfinite(xk).all()and np.isfinite(yk).all()and np.all(dx>0)and np.all(dy>0))
            rec={'layer0':l,'bin_id_1based':k+1,'m':len(pos),'N':len(x),'direction_steps':len(trace),
                'trace':trace,'J_initial':trace[0]['J_before'],'J_final':trace[-1]['J_after'],
                'nonascending_steps':sum(t['nonascending']for t in trace),'zero_gradient_steps':sum(t['zero_gradient_step']for t in trace),
                'orthogonality_maxabs':orth,'strict_knots_valid':valid,
                'minimum_x_knot_difference':float(dx.min()),'minimum_y_knot_difference':float(dy.min())}
            np.savez_compressed(DEST/f'LAYER_{l:02d}_BIN_{k+1}.npz',A=a,xk=xk,yk=yk)
            if not valid:
                write(f'{ARM}/LAYER_{l:02d}_BIN_{k+1}.json',rec)
                raise ValueError(('non-strict knots stop entire fixed recipe',l,k))
            slopes=dy/dx;rec.update(segment_slope_min=float(slopes.min()),segment_slope_max=float(slopes.max()),
                full_projected_W2_squared_before=float(np.mean((np.sort(u,axis=0)-np.sort(v,axis=0))**2)))
            updated=layer(x,a,xk,yk,u=u);test_rows=updated[pos[:8]]
            inverse=layer(test_rows,a,yk,xk)
            inverse_err=float(np.max(np.abs(inverse-x[pos[:8]])))
            rec.update(subset_layer_inverse_maxabs=inverse_err,
                subset_layer_inverse_relative_limit=1e-8*(1+float(np.max(np.abs(x[pos[:8]])))))
            assert inverse_err<=rec['subset_layer_inverse_relative_limit'],'float64 layer inverse check'
            COUNTER['full_bin_projection_calls']+=1
            rec['full_projected_W2_squared_after']=float(np.mean((np.sort(updated@a,axis=0)-np.sort(v,axis=0))**2))
            rec['numerical_pass']=bool(np.isfinite(updated).all()and valid and orth<=1e-10)
            assert rec['numerical_pass']
            XX[k]=updated;aa[k,l]=a;xks[k,l]=xk;yks[k,l]=yk
            COUNTER['direction_groups_complete']+=1;COUNTER['scalar_maps_complete']+=16
            COUNTER['direction_zero_gradient_steps']+=rec['zero_gradient_steps'];traces.append(rec)
            write(f'{ARM}/LAYER_{l:02d}_BIN_{k+1}.json',rec)
        progress={'phase':'sliced_fit','layers_complete':l+1,'counter':COUNTER,'elapsed_s':time.monotonic()-START}
        write('STATISTICS_PROGRESS.json',progress);log(progress)
    assert COUNTER['direction_groups_complete']==256 and COUNTER['scalar_maps_complete']==4096 and COUNTER['direction_gradient_steps']==5120
    check_time()
    np.savez_compressed(DEST/'SLICED_TRANSPORT_PARAMS.npz',A=aa,xk=xks,yk=yks,mu_V=mu,scale_V=scale,edges_logQ=edges,quantiles=q)
    summary=[{key:r[key]for key in r if key!='trace'}for r in traces]
    record={'PID':os.getpid(),'rows_each_domain':94400,'G_forward_blocks':2950,'new_neural_updates':0,
        'statistical_fits':256,'maps':4096,'direction_gradient_steps':5120,'statistics':counts,'layer_summaries':summary,
        'all_numeric_checks_pass':True,'stream_hashes':observed,'original_R33_stream_full_match':same,'G_before':before,'G_after':after,
        'fit_s':time.monotonic()-START,'select_model_with_DEV':False,'version':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
        'direction_device':'CPU float64','GPU_peak_allocated_bytes':GPUpeak,'full_joint_match_not_guaranteed':True,
        'same_FLOP_as_R37':False,'nonascending_direction_steps':sum(r['nonascending_steps']for r in traces)}
    write(f'{ARM}/STATISTICS_FIT_RECEIPT.json',record)
    write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counter':COUNTER,'records':{ARM:record},'device':str(DEVICE),'GPU_peak_allocated_bytes':GPUpeak})
    log({'complete':True,'counter':COUNTER})
except BaseException as error:
    write('STATISTICS_FAILURE.json',{'PID':os.getpid(),'error':repr(error),'traceback':traceback.format_exc(),
        'elapsed_s':time.monotonic()-START,'counter':COUNTER,'unique_statistics_stream_not_retried':True,
        'complete_candidate':False,'new_generation_after_failure':False})
    raise
