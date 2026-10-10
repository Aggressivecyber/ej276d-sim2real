from pathlib import Path
from datetime import datetime,timezone
import os,json,time,sys
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config_post.json').read_text('utf8'))
import numpy as np,torch
from source_flow_core import FlowNetwork,solve
torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def write(n,v):(R/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
with np.load(C['train_solver_input'],allow_pickle=False) as z:
    x=z['input_V'].copy();eps=z['white'].copy();si=z['source_TRAIN_index'].copy();ri=z['REAL_TRAIN_index'].copy();ordinal=z['white_original_row_ordinal'].copy()
with np.load(C['constants_path'],allow_pickle=False) as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
with np.load(R/'NOISE_CALIBRATION.npz',allow_pickle=False) as z:noise={k:z[k].copy()for k in z.files}
assert x.shape==eps.shape==(128,480) and len(si)==len(ri)==64
historical=json.loads((R/'HISTORICAL_TRAIN_SOLVER_CHECK.json').read_text());assert historical['both_pass']
models={}
for arm in C['arms']:
    training=json.loads((Path(C['historical_training_root'])/arm/'TRAINING_RECEIPT.json').read_text('utf8'));assert training['updates']==50000
    net=FlowNetwork();net.load_state_dict(torch.load(C['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True));net.to(C['inference_device']);net.eval();net.requires_grad_(False)
    completion=True
    a,da,ra=solve(net,x,eps,mu,scale,edges,noise,completion,C['solver_steps'],C['generation_chunk'])
    b,db,rb=solve(net,x,eps,mu,scale,edges,noise,completion,C['solver_check_steps'],C['generation_chunk'])
    rms=np.sqrt(np.mean((a.astype('f8')-b.astype('f8'))**2,axis=1));lsb=1/16383.
    med=float(np.median(rms));p95=float(np.quantile(rms,.95))
    charge=max(ra['max_relative_q_error_f64'],ra['max_relative_q_error_f32'],rb['max_relative_q_error_f64'],rb['max_relative_q_error_f32'])
    np.savez_compressed(R/arm/'TRAIN_SOLVER_CHECK_PRIVATE.npz',solution128_preADC_f32_V=a,solution256_preADC_f32_V=b,RMS_difference_V=rms,
                        source_TRAIN_index=si,REAL_TRAIN_index=ri,white_original_row_ordinal=ordinal,
                        relative_q128_f32=da['preADC_charge_relative_error_f32'],relative_q256_f32=db['preADC_charge_relative_error_f32'],
                        input_variance_V2=da['input_variance100_V2'],innovation_amplitude_squared_V2=da['innovation_amplitude_squared_V2'])
    rec={'PID':os.getpid(),'arm':arm,'UTC':datetime.now(timezone.utc).isoformat(),'RMS_median_LSB':med/lsb,'RMS_p95_LSB':p95/lsb,
         'RMS_median_V':med,'RMS_p95_V':p95,'LSB_V':lsb,'median_limit_LSB':.25,'p95_limit_LSB':1.,'preADC_relative_q_max':charge,
         'relative_q_limit':1e-5,'pass':bool(med<=.25*lsb and p95<=lsb and charge<=1e-5),'solver128':ra,'solver256':rb,
         'TRAIN_SOURCE_REAL_rows':[64,64],'white_reference':C['train_solver_input'],
         'white_prefix':'first128 consumed-row white; original noise-generation ordinal retained',
         'DEV_read_or_generated':0,'candidate_steps':128,'reference_only_steps':256,'no_step_search':True}
    models[arm]=rec;write(arm+'/TRAIN_SOLVER_CHECK.json',rec);print(json.dumps({'TRAIN_gate':rec}),flush=True)
    del net,a,b,da,db
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'models':models,'both_pass':historical['both_pass'] and all(r['pass']for r in models.values()),'historical_control_pass_reused':historical['both_pass'],'new_models':1,'device':C['inference_device'],
     'elapsed_s':time.monotonic()-START,'DEV_read_or_generated':0,'new_response_updates':0,'both_required_before_DEV':True,'CPU_threads':12,'chunk':64}
write('TRAIN_SOLVER_CHECK.json',rec);print(json.dumps(rec),flush=True)
