from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,sys
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch,scipy
from model import Translator
from flow_core import bins
from reorder_core import reconstruct_and_reorder
from covariance_core import Moments,gaussian_map,selfcheck

torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
if DEVICE.type=='cuda':
    torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
DEST=ROOT/'FULL_COVARIANCE_CALIBRATION';DEST.mkdir(exist_ok=False)
COUNTER={'statistics_fit_attempts':0,'statistics_domain_bin_estimates':0,'maps_complete':0,'fixed_G_forward_training_rows':0,
         'REAL_training_rows':0,'neural_fit_or_updates':0}
def write(n,obj):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n','utf8')
def log(obj):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(obj),flush=True)
def check_time():
    if time.monotonic()-START>CFG['statistics_fit_wall_s']-5:raise TimeoutError('unique statistics fit900s limit')
    if time.time()>datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-300:raise TimeoutError('stage deadline')
def fingerprint(d):
    h=hashlib.sha256()
    for k,v in d.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

write('SERVER_ENVIRONMENT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
    'device':str(DEVICE),'GPU':torch.cuda.get_device_name(0)if DEVICE.type=='cuda'else None})
try:
    write('COVARIANCE_FORMULA_CHECK.json',selfcheck())
    with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    actual=reconstruct_and_reorder(CFG,ROOT)
    with np.load(CFG['affine_path'],allow_pickle=False)as z:mu=z['mu_V'].copy();scale=z['scale_V'].copy();edges=z['edges_logQ'].copy()
    si=actual['source'].astype('i8');ti=actual['target'].astype('i8');bn=actual['bin'].astype('i8');ordinal=actual['original_row_ordinal']
    sb=bins(data['source'],edges);tb=bins(data['target'],edges)
    assert np.array_equal(sb[si],bn)and np.array_equal(tb[ti],bn),'original source/REAL bin conflict'
    expected=json.loads(pathlib.Path(CFG['original_training_receipt_path']).read_text('utf8'))['records']['INDEPENDENT_CFM']['stream_hashes']
    hashes={k:hashlib.sha256()for k in ['source','target','bin','original_row_ordinal','z_white','style','base_preADC']}
    base=Translator().to(DEVICE);base.load_state_dict(torch.load(CFG['base_G'],map_location=DEVICE,weights_only=True));base.eval();base.requires_grad_(False)
    before=fingerprint(base.state_dict());xs=torch.from_numpy(data['source']).to(DEVICE)
    source_mom=[Moments()for _ in range(8)];target_mom=[Moments()for _ in range(8)]
    zall=actual['original_noise']['z'];wall=actual['original_noise']['white'];styleall=actual['original_style']
    COUNTER['statistics_fit_attempts']=1;block_hashes=[]
    for step in range(2950):
        check_time();sl=slice(step*32,(step+1)*32);ssi=si[sl];tti=ti[sl];b=bn[sl];order=ordinal[sl]
        z=zall[order];white=wall[order];style=styleall[order]
        COUNTER['fixed_G_forward_training_rows']+=32
        with torch.no_grad():g=base(xs[ssi],torch.from_numpy(z).to(DEVICE),torch.from_numpy(white).to(DEVICE),torch.from_numpy(style).to(DEVICE))[0].cpu().numpy()
        COUNTER['REAL_training_rows']+=32
        assert np.isfinite(g).all();ys=(g.astype('f8')-mu[b])/scale[b];yt=(data['target'][tti].astype('f8')-mu[b])/scale[b]
        for k in np.unique(b):source_mom[k].update(ys[b==k]);target_mom[k].update(yt[b==k])
        for k,a in [('source',ssi),('target',tti),('bin',b),('original_row_ordinal',order),('style',style),('base_preADC',g)]:hashes[k].update(a.tobytes())
        hashes['z_white'].update(z.tobytes());hashes['z_white'].update(white.tobytes());block_hashes.append(hashlib.sha256(g.tobytes()).hexdigest())
        if DEVICE.type=='cuda'and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if(step+1)%250==0 or step==2949:
            progress={'blocks':step+1,'counter':COUNTER,'elapsed_s':time.monotonic()-START};write('STATISTICS_PROGRESS.json',progress);log(progress)
    observed={k:v.hexdigest()for k,v in hashes.items()};same={k:observed[k]==expected[k]for k in hashes}
    write('STATISTICS_INPUT_BINDING.json',{'rows':94400,'stream_hashes':observed,'expected_R33_stream_hashes':{k:expected[k]for k in hashes},
        'R33_experimental_stream_match':same,'base_GPU_chunk':32,'source_target_edge_records_match':True,
        'z_white_style_reconstructed_from_recorded_initial_and_final_states':True,'not_migration_content_testing':True})
    write('BASE_FORWARD_HASHES.json',{'rows_per_hash':32,'actual_base_preADC_hashes':block_hashes})
    assert all(same.values()),'R33 experimental stream/G mismatch; do not refit'
    assert fingerprint(base.state_dict())==before
    np.savez_compressed(DEST/'ACTUAL_STATISTICS_STREAM_PRIVATE.npz',source=si,target=ti,bin=bn,original_row_ordinal=ordinal)
    payload={'mu_V':mu,'scale_V':scale,'edges_logQ':edges};descriptions=[];ms=[];mt=[];tables={k:[]for k in ['A','Cs','Ct','Cs_reg','Ct_reg','Cs_eigenvalues','Ct_eigenvalues','Cs_reg_eigenvalues','Ct_reg_eigenvalues','A_eigenvalues']}
    for k in range(8):
        check_time();sm=source_mom[k];tm=target_mom[k]
        COUNTER['statistics_domain_bin_estimates']+=2
        assert sm.n==tm.n and sm.n>480,'sample count insufficient; no bin deletion'
        part,diag=gaussian_map(sm.covariance(),tm.covariance(),CFG['eigenfloor_trace_fraction'],CFG['original_covariance_negative_relative_limit'])
        record={'bin_id_1based':k+1,'source_draws':sm.n,'target_draws':tm.n,
            'distinct_source_row_indices':int(np.unique(si[bn==k]).size),'distinct_REAL_row_indices':int(np.unique(ti[bn==k]).size),
            'sample_covariance_ddof':1,'draws_not_independent_acquisition_N':True,**diag}
        write('FULL_COVARIANCE_CALIBRATION/STATISTICS_bin'+str(k+1)+'.json',record)
        np.savez_compressed(DEST/('STATISTICS_bin'+str(k+1)+'.npz'),source_mean=sm.mean,target_mean=tm.mean,**part)
        assert diag['numerical_pass'],'fixed actual covariance numeric gate failed'
        ms.append(sm.mean);mt.append(tm.mean)
        for name in tables:tables[name].append(part[name])
        descriptions.append(record);COUNTER['maps_complete']+=1;log({'bin':k+1,'A_covariance_residual':diag['regularized_covariance_relative_Frobenius_residual']})
    payload.update(ms=np.stack(ms),mt=np.stack(mt),**{n:np.stack(v)for n,v in tables.items()})
    np.savez_compressed(DEST/'CALIBRATION_MAPS.npz',**payload)
    record={'PID':os.getpid(),'rows_each_domain':94400,'G_forward_blocks':2950,'new_neural_updates':0,
        'statistical_fits':16,'maps':8,'statistics':descriptions,'all_numeric_checks_pass':True,
        'stream_hashes':observed,'original_R33_stream_full_match':same,'G_before':before,'G_after':fingerprint(base.state_dict()),
        'fit_s':time.monotonic()-START,'select_model_with_DEV':False,'version':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
        'Gaussian_standardized_coordinate_optimality_not_physical_truth':True,
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if DEVICE.type=='cuda'else 0}
    write('FULL_COVARIANCE_CALIBRATION/STATISTICS_FIT_RECEIPT.json',record)
    write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counter':COUNTER,'records':{'FULL_COVARIANCE_CALIBRATION':record},'device':str(DEVICE),
        'GPU_peak_allocated_bytes':record['GPU_peak_allocated_bytes']})
    log({'complete':True,'counter':COUNTER})
except BaseException as error:
    write('STATISTICS_FAILURE.json',{'PID':os.getpid(),'error':repr(error),'traceback':traceback.format_exc(),
        'elapsed_s':time.monotonic()-START,'counter':COUNTER,'unique_statistics_stream_not_retried':True})
    raise
