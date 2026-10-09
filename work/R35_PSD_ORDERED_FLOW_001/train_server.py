from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,copy,sys
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from model import Translator
from flow_core import Velocity,bins
from reorder_core import reconstruct_and_reorder
from psd_order_core import ordered_coupling,small_check,InvalidPSDKey
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def state(m):return{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def fingerprint(d):
    h=hashlib.sha256()
    for k,v in d.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
counter={'flow_fit_attempts':0,'flow_updates':0,'fixed_G_forward_training_rows':0,'training_body_s':0}
updates=0;step=0;trace=[];dense=[];subbatch_records=[];base_hash=[]
env={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cwd':str(ROOT),
    'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
    'device':str(DEVICE),'GPU':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None}
write('SERVER_ENVIRONMENT.json',env)
expected_env=CFG['expected_R33_environment']
assert all(env[k]==expected_env[k] for k in ['python','numpy','torch','scipy','device','GPU']),'R33 environment drift; no fit'
log({'PID':os.getpid(),'phase':'one PSD-ordered CFM fit','device':str(DEVICE)})
try:
    write('PSD_ORDER_FORMULA_CHECK.json',small_check())
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    actual=reconstruct_and_reorder(CFG,ROOT)
    with np.load(CFG['affine_path'],allow_pickle=False) as z:
        mu=z['mu_V'].copy();scale=z['scale_V'].copy();edges=z['edges_logQ'].copy()
    si=actual['source'].astype('i8');ti=actual['target'].astype('i8');bn=actual['bin'].astype('i8');ordinal=actual['original_row_ordinal']
    sb=bins(data['source'],edges);tb=bins(data['target'],edges)
    binding={'rows':len(si),'source_bin_record_matches_raw_signed_charge':bool(np.array_equal(sb[si],bn)),
        'REAL_bin_record_matches_raw_signed_charge':bool(np.array_equal(tb[ti],bn)),
        'new_bin_assignment':False,'new_affine_fit':False,'source_Q_definition':'actual unADC source waveform'}
    write('TRAINING_Q_BINDING.json',binding)
    assert binding['source_bin_record_matches_raw_signed_charge'] and binding['REAL_bin_record_matches_raw_signed_charge']
    initial_state=torch.load(CFG['initial_flow_path'],map_location='cpu',weights_only=True)
    old=json.loads(pathlib.Path(CFG['original_training_receipt_path']).read_text('utf8'))['records']['INDEPENDENT_CFM']
    old_base_hashes=json.loads(pathlib.Path(CFG['original_base_hashes_path']).read_text('utf8'))['actual_base_preADC_hashes']
    assert fingerprint(initial_state)==old['initial_state_sha256'],'R33 initial state mismatch'
    base=Translator().to(DEVICE);base.load_state_dict(torch.load(CFG['base_G'],map_location=DEVICE,weights_only=True));base.eval();base.requires_grad_(False)
    base_before=fingerprint(base.state_dict());xs=torch.from_numpy(data['source']).to(DEVICE);xr=torch.from_numpy(data['target']).to(DEVICE)
    mt=torch.tensor(mu,dtype=torch.float32,device=DEVICE);st=torch.tensor(scale,dtype=torch.float32,device=DEVICE)
    original_noise=actual['original_noise'];original_style=actual['original_style']
    arm='PSD_ORDERED_CFM';dest=ROOT/arm;dest.mkdir(exist_ok=False);torch.manual_seed(CFG['model_seed'])
    flow=Velocity().to(DEVICE);flow.load_state_dict(initial_state)
    optimizer=torch.optim.Adam(flow.parameters(),lr=CFG['flow_lr'],betas=CFG['flow_betas'],eps=CFG['flow_eps'],weight_decay=0)
    rt=np.random.default_rng(CFG['t_seed']);initial_t=copy.deepcopy(rt.bit_generator.state)
    hashes={k:hashlib.sha256() for k in ['source','target','bin','original_row_ordinal','z_white','style','t','base_preADC']}
    actual_t=np.empty((94400,),'f4');matched_target=np.empty((94400,),'u4');permutations=np.empty((94400,),'u1')
    source_key=np.empty((94400,),'f8');target_key=np.empty((94400,),'f8');source_q=np.empty((94400,),'f8');target_q=np.empty((94400,),'f8')
    counter['flow_fit_attempts']+=1;clock=time.monotonic();sorting_s=0.
    try:
        for step in range(1,CFG['updates']+1):
            if time.monotonic()-clock>=CFG['training_wall_s']-5:break
            if time.time()>=datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-300:break
            sl=slice((step-1)*32,step*32);ssi=si[sl];tti=ti[sl];b=bn[sl];order=ordinal[sl]
            z=original_noise['z'][order];white=original_noise['white'][order];style=original_style[order]
            counter['fixed_G_forward_training_rows']+=32
            with torch.no_grad():
                x0=base(xs[ssi],torch.from_numpy(z).to(DEVICE),torch.from_numpy(white).to(DEVICE),torch.from_numpy(style).to(DEVICE))[0]
                btorch=torch.from_numpy(b).to(DEVICE);a=(x0-mt[btorch])/st[btorch];c=(xr[tti]-mt[btorch])/st[btorch]
            cpu_base=x0.detach().cpu().numpy();bh=hashlib.sha256(cpu_base.tobytes()).hexdigest();base_hash.append(bh)
            if bh!=old_base_hashes[step-1]:
                write('SAME_STREAM_CONFLICT.json',{'step':step,'field':'base_preADC','expected':old_base_hashes[step-1],'actual':bh})
                raise RuntimeError('R33 actual base preADC stream conflict')
            tick=time.monotonic()
            try:p,cost,groups,keys=ordered_coupling(cpu_base,data['target'][tti],b,a.cpu().numpy(),c.cpu().numpy())
            except InvalidPSDKey as e:
                write('INVALID_PSD_SORT_KEY.json',{'step':step,'bin_ids':b.tolist(),'source_indices':ssi.tolist(),'target_original_indices':tti.tolist(),**e.details})
                raise
            sorting_s+=time.monotonic()-tick
            t=rt.uniform(0,1,size=32).astype('f4');actual_t[sl]=t;permutations[sl]=p.astype('u1');matched_target[sl]=tti[p]
            source_q[sl]=keys[0];source_key[sl]=keys[2];target_q[sl]=keys[3];target_key[sl]=keys[5]
            subbatch_records.append({'step':step,'original_target_permutation':p.tolist(),'Q_subgroups':groups,
                'one_to_one_no_cross_Q':True,'original_source_order_restored':True,'all_source_target_row_marginals_preserved':True})
            cc=c[torch.from_numpy(p).to(DEVICE)];tt=torch.from_numpy(t).to(DEVICE)
            xt=(1-tt[:,None])*a+tt[:,None]*cc;velocity=cc-a
            optimizer.zero_grad(set_to_none=True);prediction=flow(xt,tt,btorch);loss=(prediction-velocity).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite CFM loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),5.);optimizer.step();updates=step;counter['flow_updates']+=1
            values=[float(loss.detach()),float(velocity.square().mean().sqrt()),float(prediction.detach().square().mean().sqrt()),float(norm),cost]
            dense.append(values)
            for key,arr in [('source',ssi),('target',tti),('bin',b),('original_row_ordinal',order),('style',style),('t',t),('base_preADC',cpu_base)]:hashes[key].update(arr.tobytes())
            hashes['z_white'].update(z.tobytes());hashes['z_white'].update(white.tobytes())
            if DEVICE.type=='cuda' and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU cap')
            if step%250==0 or step==CFG['updates']:
                row={'step':step,'CFM_loss':values[0],'velocity_label_RMS':values[1],'prediction_RMS':values[2],
                    'gradient_norm_before_clip':values[3],'coupling_mean480_squared_cost':cost,'Q_subgroups':groups,
                    'elapsed_training_s':time.monotonic()-clock,'sorting_s':sorting_s}
                trace.append(row);write(arm+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',{'arm':arm,'updates':updates,'counter':counter});log(row)
        torch.save(state(flow),dest/'FINAL_FLOW.pt');torch.save(optimizer.state_dict(),dest/'FINAL_OPTIMIZER.pt')
    finally:
        n=updates*32;elapsed=time.monotonic()-clock;counter['training_body_s']=elapsed
        np.savez_compressed(dest/'ACTUAL_CFM_STREAM_PRIVATE.npz',source=si[:n],target=ti[:n],bin=bn[:n],
            original_row_ordinal=ordinal[:n],t=actual_t[:n],matched_target=matched_target[:n],within_batch_target_position=permutations[:n],
            source_BASE_preADC_PSD=source_key[:n],target_original_PSD=target_key[:n],source_BASE_preADC_Qtotal_mVns=source_q[:n],
            target_original_Qtotal_mVns=target_q[:n],NCE_original_positions_not_used_by_CFM=actual['NCE_positions'][:updates*3])
        np.savez_compressed(dest/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'))
        write(arm+'/COUPLING_BY_UPDATE_PRIVATE.json',subbatch_records)
        write(arm+'/BASE_FORWARD_HASHES.json',{'rows_per_hash':32,'actual_base_preADC_hashes':base_hash,'base_forward_rows':len(base_hash)*32})
        record={'arm':arm,'updates':updates,'actual_rows':n,'training_s':elapsed,'sorting_s':sorting_s,
            'stream_hashes':{k:v.hexdigest() for k,v in hashes.items()},'initial_state_sha256':fingerprint(initial_state),
            'selection':'actual final only','initial_t_RNG':initial_t,'final_t_RNG':rt.bit_generator.state,
            'no_coupling_RNG_consumed':True,'G_frozen':True,'G_parameters_updated':0,
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else 0,
            'fit_loss_only_velocity_MSE':True,'PSD_used_only_for_statistical_pairing':True,'physical_pairing_claim':False}
        write(arm+'/TRAINING_RECEIPT.json',record)
        same={k:record['stream_hashes'][k]==old['stream_hashes'][k] for k in hashes}
        paired={'updates':updates,'all2950':updates==2950,'same_initial_state':record['initial_state_sha256']==old['initial_state_sha256'],
            'same_actual_source_REAL_original_bin_ordinal_z_white_style_t_base_hashes':same,
            'per_batch_base_hashes_match_R33_prefix':base_hash==old_base_hashes[:len(base_hash)],
            'only_target_permutation_changed':True,'base_state_before':base_before,'base_state_after':fingerprint(base.state_dict()),
            'physical_pairing_claim':False,'old_saved_controls_environment_match':True}
        write('PAIRED_FLOW_RECEIPT.json',paired)
        write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_s':time.monotonic()-START,'counter':counter,'records':{arm:record},'device':str(DEVICE),
            'GPU_peak_allocated_bytes':record['GPU_peak_allocated_bytes']})
        assert fingerprint(base.state_dict())==base_before
        if updates==2950:assert all(same.values()),'R33 same scientific stream conflict'
    log({'complete':True,'counter':counter,'updates':updates})
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),
        'elapsed_s':time.monotonic()-START,'counter':counter,'attempted_step':step,'completed_updates':updates})
    raise
