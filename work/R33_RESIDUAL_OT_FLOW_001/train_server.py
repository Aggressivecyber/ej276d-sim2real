from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,copy,sys
START=time.monotonic(); ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from model import Translator
from flow_core import Velocity,coupling,bins,selfcheck
from reorder_core import reconstruct_and_reorder

torch.set_num_threads(4); torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
def write(n,v):
    p=ROOT/n; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def state(m):return{k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
def fingerprint(d):
    h=hashlib.sha256()
    for k,v in d.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
counter={'flow_fit_attempts':0,'flow_updates':0,'fixed_G_forward_training_rows':0,'training_body_s':0}
records={}
write('SERVER_ENVIRONMENT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
    'device':str(DEVICE),'GPU':torch.cuda.get_device_name(0)if DEVICE.type=='cuda' else None})
log({'PID':os.getpid(),'phase':'shared_TRAIN_affine_and_two_final_CFM_fits','device':str(DEVICE)})
try:
    write('FORMULA_CHECK.json',selfcheck())
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    actual=reconstruct_and_reorder(CFG,ROOT)
    edges=np.asarray(data['edges'],dtype='f8'); sb=bins(data['source'],edges); tb=bins(data['target'],edges)
    si=actual['source'].astype('i8'); ti=actual['target'].astype('i8'); bn=actual['bin'].astype('i8')
    binding={'rows':len(si),'source_bin_record_matches_raw_signed_charge':bool(np.array_equal(sb[si],bn)),
        'REAL_bin_record_matches_raw_signed_charge':bool(np.array_equal(tb[ti],bn)),
        'stored_source_bin_matches_raw':bool(np.array_equal(sb,data['source_bin'])),
        'stored_target_bin_matches_raw':bool(np.array_equal(tb,data['target_bin'])),
        'source_disagreement_N':int(np.sum(sb[si]!=bn)),'REAL_disagreement_N':int(np.sum(tb[ti]!=bn)),
        'new_bin_assignment':False,'source_Q_definition':'raw source preADC subtract first100 baseline, signed sum100:400 in mVns'}
    write('TRAINING_Q_BINDING.json',binding)
    assert binding['source_bin_record_matches_raw_signed_charge'] and binding['REAL_bin_record_matches_raw_signed_charge'], 'actual source/REAL Q binding conflict'
    target=np.asarray(data['target'],dtype='f8'); mu=np.zeros((8,480),'f8'); raw_sd=np.zeros_like(mu); counts=[]; payload={}
    for b in range(8):
        use=np.flatnonzero(tb==b); assert len(use)>0
        mu[b]=target[use].mean(0); raw_sd[b]=target[use].std(0,ddof=0)
        counts.append({'bin_id_1based':b+1,'TRAIN_fit_N':len(use),'floor_coordinate_count':int((raw_sd[b]<1e-6).sum())})
        payload['bin'+str(b+1)+'_TRAIN_row_ordinal']=use.astype('<u4')
    scale=np.maximum(raw_sd,1e-6); assert np.isfinite(mu).all()and np.isfinite(scale).all()
    np.savez_compressed(ROOT/'AFFINE_CONSTANTS.npz',mu_V=mu,raw_population_SD_V=raw_sd,scale_V=scale,
        edges_logQ=edges,TRAIN_target_bin=tb.astype('i1'),**payload)
    write('AFFINE_METADATA.json',{'fit_role':'REAL TRAIN-fit21047; internal validation599 excluded','N':len(target),
        'rows':counts,'outside8bins_N':int(np.sum((tb<0)|(tb>=8))),'mean_and_SD_ddof':0,'SD_floor_V':1e-6,
        'identity':'existing TRAIN_PRIVATE target row ordinal; original record index not explicit in this container',
        'producer_lineage':'2592 original QC TRAIN-fit +18455 R14 QC additions; frozen split',
        'source':CFG['training_input_paths']['TRAIN_PRIVATE.npz'],'same_affine_used_for_both_domains':True,
        'DEV_or_validation_statistics':False,'source_bin_uses_actual_preADC_source':True})
    del target
    torch.manual_seed(CFG['model_seed']); initial=Velocity(); initial_state=state(initial)
    torch.save(initial_state,ROOT/'INITIAL_FLOW.pt'); torch.save(torch.get_rng_state(),ROOT/'INITIAL_TORCH_RNG.pt')
    base=Translator().to(DEVICE);base.load_state_dict(torch.load(CFG['base_G'],map_location=DEVICE,weights_only=True));base.eval();base.requires_grad_(False)
    base_before=fingerprint(base.state_dict())
    xs=torch.from_numpy(data['source']).to(DEVICE); xr=torch.from_numpy(data['target']).to(DEVICE)
    mt=torch.tensor(mu,dtype=torch.float32,device=DEVICE);st=torch.tensor(scale,dtype=torch.float32,device=DEVICE)
    original_noise=actual['original_noise'];original_style=actual['original_style'];ordinal=actual['original_row_ordinal']
    for arm in CFG['arms']:
        dest=ROOT/arm;dest.mkdir(exist_ok=False);torch.manual_seed(CFG['model_seed'])
        flow=Velocity().to(DEVICE);flow.load_state_dict(initial_state)
        optimizer=torch.optim.Adam(flow.parameters(),lr=CFG['flow_lr'],betas=CFG['flow_betas'],eps=CFG['flow_eps'],weight_decay=0)
        rt=np.random.default_rng(CFG['t_seed']);rp=np.random.default_rng(CFG['coupling_seed'])
        initial_t=copy.deepcopy(rt.bit_generator.state);initial_p=copy.deepcopy(rp.bit_generator.state)
        counter['flow_fit_attempts']+=1;clock=time.monotonic();updates=0;transport_s=0.;trace=[];dense=[]
        base_hash=[];hashes={k:hashlib.sha256()for k in ['source','target','bin','original_row_ordinal','z_white','style','t','base_preADC']}
        actual_t=np.empty((94400,),'f4');matched_target=np.empty((94400,),'u4');permutations=np.empty((94400,),'u1')
        try:
            for step in range(1,CFG['updates']+1):
                spent=time.monotonic()-clock
                if spent>=CFG['training_wall_s']-5 or counter['training_body_s']+spent>=CFG['training_total_limit_s']-5:break
                if datetime.datetime.now(datetime.timezone.utc).timestamp()>=datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-300:break
                sl=slice((step-1)*32,step*32);ssi=si[sl];tti=ti[sl];b=bn[sl];order=ordinal[sl]
                z=original_noise['z'][order];white=original_noise['white'][order];style=original_style[order]
                counter['fixed_G_forward_training_rows']+=32
                with torch.no_grad():
                    x0=base(xs[ssi],torch.from_numpy(z).to(DEVICE),torch.from_numpy(white).to(DEVICE),torch.from_numpy(style).to(DEVICE))[0]
                    btorch=torch.from_numpy(b).to(DEVICE);a=(x0-mt[btorch])/st[btorch];c=(xr[tti]-mt[btorch])/st[btorch]
                cpu_base=x0.detach().cpu().numpy();bh=hashlib.sha256(cpu_base.tobytes()).hexdigest();base_hash.append(bh)
                solve_start=time.monotonic();p,cost,subgroups=coupling(a.cpu().numpy(),c.cpu().numpy(),b,arm,rp);transport_s+=time.monotonic()-solve_start
                t=rt.uniform(0,1,size=32).astype('f4');actual_t[sl]=t;permutations[sl]=p.astype('u1');matched_target[sl]=tti[p]
                cc=c[torch.from_numpy(p).to(DEVICE)];tt=torch.from_numpy(t).to(DEVICE);xt=(1-tt[:,None])*a+tt[:,None]*cc;velocity=cc-a
                optimizer.zero_grad(set_to_none=True);prediction=flow(xt,tt,btorch);loss=(prediction-velocity).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite CFM loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),5.);optimizer.step();updates=step;counter['flow_updates']+=1
                values=[float(loss.detach()),float(velocity.square().mean().sqrt()),float(prediction.detach().square().mean().sqrt()),float(norm),cost]
                dense.append(values)
                for key,arr in [('source',ssi),('target',tti),('bin',b),('original_row_ordinal',order),('style',style),('t',t),('base_preADC',cpu_base)]:hashes[key].update(arr.tobytes())
                hashes['z_white'].update(z.tobytes());hashes['z_white'].update(white.tobytes())
                if DEVICE.type=='cuda'and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
                if step%250==0 or step==CFG['updates']:
                    row={'step':step,'CFM_loss':values[0],'velocity_label_RMS':values[1],'prediction_RMS':values[2],
                        'gradient_norm_before_clip':values[3],'gradient_clipped':values[3]>5.,'coupling_mean480_squared_cost':cost,
                        'subgroups':subgroups,'elapsed_training_s':time.monotonic()-clock,'transport_solver_s':transport_s}
                    trace.append(row);write(arm+'/TRAINING_TRACE.json',trace)
                    write('TRAIN_PROGRESS.json',{'arm':arm,'updates':updates,'counter':counter});log({'arm':arm,**row})
            torch.save(state(flow),dest/'FINAL_FLOW.pt');torch.save(optimizer.state_dict(),dest/'FINAL_OPTIMIZER.pt')
        finally:
            actual_n=updates*32;elapsed=time.monotonic()-clock;counter['training_body_s']+=elapsed
            np.savez_compressed(dest/'ACTUAL_CFM_STREAM_PRIVATE.npz',source=si[:actual_n],target=ti[:actual_n],bin=bn[:actual_n],
                original_row_ordinal=ordinal[:actual_n],t=actual_t[:actual_n],matched_target=matched_target[:actual_n],within_batch_target_position=permutations[:actual_n],
                NCE_original_positions_not_used_by_CFM=actual['NCE_positions'][:updates*3])
            np.savez_compressed(dest/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'))
            write(arm+'/BASE_FORWARD_HASHES.json',{'rows_per_hash':32,'actual_base_preADC_hashes':base_hash,'base_forward_rows':len(base_hash)*32})
            record={'arm':arm,'updates':updates,'actual_rows':actual_n,'training_s':elapsed,'transport_solver_s':transport_s,
                'stream_hashes':{k:v.hexdigest()for k,v in hashes.items()},'initial_state_sha256':fingerprint(initial_state),
                'selection':'actual final only','initial_t_RNG':initial_t,'final_t_RNG':rt.bit_generator.state,
                'initial_permutation_RNG':initial_p,'final_permutation_RNG':rp.bit_generator.state,
                'fit_singletons_from_original_batches':True,'G_frozen':True,'G_parameters_updated':0,
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0,
                'fit_loss_only_velocity_MSE':True,'legacy_NCE_not_in_CFM_loss':True}
            records[arm]=record;write(arm+'/TRAINING_RECEIPT.json',record)
            write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'elapsed_s':time.monotonic()-START,'counter':counter,'records':records,'device':str(DEVICE),
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0})
        del flow,optimizer
    same={k:records[CFG['arms'][0]]['stream_hashes'][k]==records[CFG['arms'][1]]['stream_hashes'][k]
          for k in ['source','target','bin','original_row_ordinal','z_white','style','t','base_preADC']}
    write('PAIRED_FLOW_RECEIPT.json',{'same_actual_steps':records[CFG['arms'][0]]['updates']==records[CFG['arms'][1]]['updates'],
        'both2950':all(r['updates']==2950 for r in records.values()),'same_actual_input_and_initial_response_hashes':same,
        'base_state_before':base_before,'base_state_after':fingerprint(base.state_dict()),'physical_pairing_claim':False,'only_coupling_differs':True})
    if records[CFG['arms'][0]]['updates']==records[CFG['arms'][1]]['updates']:assert all(same.values())
    assert fingerprint(base.state_dict())==base_before
    log({'complete':True,'counter':counter,'updates':{k:r['updates']for k,r in records.items()}})
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),
        'elapsed_s':time.monotonic()-START,'counter':counter,'completed_arms':list(records)})
    raise
