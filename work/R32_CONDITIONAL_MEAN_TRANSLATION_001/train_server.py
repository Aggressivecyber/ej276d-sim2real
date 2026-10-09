from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,copy
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch
from online_training import train
from reorder_core import reconstruct_and_reorder
from conditional_mean import selfcheck
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(min(.5,CFG['GPU_allocated_limit_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def load(n):
    with np.load(CFG['training_input_paths'][n],allow_pickle=False)as z:return{k:z[k].copy() for k in z.files}
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
counter={'response_fit_attempts':0,'G_updates':0,'D_updates':0,'style_reader_updates':0,'training_s':0}
records={}
write('SERVER_ENVIRONMENT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'torch':torch.__version__,'numpy':np.__version__,'python':__import__('sys').version,'device':device,
    'cwd':str(ROOT),'GPU':torch.cuda.get_device_name(0) if device.startswith('cuda')else None})
log({'actual_scientific_PID':os.getpid(),'phase':'mean_arithmetic_and_two_sequential_fits','device':device})
try:
    write('MEAN_FORMULA_CHECK.json',selfcheck())
    data=load('TRAIN_PRIVATE.npz');val=load('VALIDATION_INPUTS.npz');constants=load('FINE_CONSTANTS.npz');scaler=load('CRITIC_SCALER.npz')
    x=data['target'].astype('f8');s=x-x[:,:100].mean(1,keepdims=True);u=s/np.maximum(s.max(1,keepdims=True),.0025)
    bins=data['target_bin'].astype('i8');mu=np.zeros((8,480),'f8');rows=[];payload={}
    for b in range(8):
        use=np.flatnonzero(bins==b)
        if len(use):mu[b]=u[use].mean(0,dtype='f8')
        rows.append({'bin_id_1based':b+1,'TRAIN_fit_N':len(use),'valid_mu':bool(len(use))})
        payload['bin'+str(b+1)+'_target_row_ordinal']=use.astype('<u4')
    actual=reconstruct_and_reorder(CFG,ROOT)
    used=set(map(int,np.unique(actual['bin'])))
    assert all(rows[b]['valid_mu'] for b in used)
    data['mean_mu']=mu
    np.savez_compressed(ROOT/'MEAN_TARGETS_PRIVATE.npz',mu=mu,shape480_scale=constants['shape480__scale'],edges_logQ=data['edges'],TRAIN_fit_bin=bins,**payload)
    write('MEAN_TARGETS_METADATA.json',{'fit_role':'existing REAL TRAIN-fit only','N':len(x),'common_training_bin_ids_1based':[b+1 for b in sorted(used)],
        'rows':rows,'outside_frozen_bins_N':int(np.sum((bins<0)|(bins>=8))),
        'mu_definition':'float64 baseline[0:100],peak floor.0025,full480 mean; cast f32 for Torch',
        'row_identity_source':CFG['training_input_paths']['TRAIN_PRIVATE.npz'],
        'identity':'target row ordinal in frozen R23 TRAIN_PRIVATE.npz',
        'original_record_lineage':'R23 prepare_data/data_support target concatenation:2592 original TRAIN-fit QC rows then18455 R14 QC additions; validation599 excluded',
        'original_index_not_explicit_in_THIS_container':True,'row_ordinals_saved_by_bin':True,
        'scale_source':CFG['training_input_paths']['FINE_CONSTANTS.npz']+'::shape480__scale','generated_domain_statistics_fit':False,'DEV_target_statistics_used':False})
    del x,s,u
    for arm in CFG['arms']:
        torch.manual_seed(CFG['seed']);np.random.seed(0)
        record=train(arm,data,val,constants,scaler,actual.copy(),CFG,ROOT/arm,counter,log)
        records[arm]=record
        write('TRAINING_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_s':time.monotonic()-START,'counter':counter,'records':records,'torch':torch.__version__,'numpy':np.__version__,
            'device':device,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda')else 0,
            'status':'complete' if len(records)==2 else 'first_arm_terminal','same_server_comparison':True})
    keys=['source','target','bin','original_row_ordinal','NCE_positions','noise_stream','style_stream']
    a,b=[records[n] for n in CFG['arms']]
    equality={k:a['stream_hashes'][k]==b['stream_hashes'][k] for k in keys}
    write('PAIRED_FLOW_RECEIPT.json',{'same_updates':a['updates']==b['updates'],'both_2950':a['updates']==b['updates']==2950,
        'same_complete_actual_stream':equality,'fresh_seed':CFG['fresh_projection_seed'],'original_rng_untouched_by_mean_statistics':True,
        'same_initial_G_reader_D':True,'new_same_Adam_parameters':True,'cross_device_training_bitwise_claim':False})
    if a['updates']==b['updates']:assert all(equality.values())
    log({'complete':True,'PID':os.getpid(),'counter':counter,'selected_steps':{k:r['selected_step'] for k,r in records.items()}})
except BaseException as e:
    write('TRAINING_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counter':counter,'completed_arms':list(records)})
    raise
