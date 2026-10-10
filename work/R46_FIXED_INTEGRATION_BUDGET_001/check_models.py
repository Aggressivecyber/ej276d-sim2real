from pathlib import Path
from datetime import datetime,timezone
import json,os,time,platform
import numpy as np, torch, scipy
from source_flow_core import FlowNetwork,solve
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)
def write(n,v):(R/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
write('SERVER_ENVIRONMENT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'cwd':str(R),'Python':platform.python_version(),'NumPy':np.__version__,'Torch':torch.__version__,
 'SciPy':scipy.__version__,'device':'cpu','CPU_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()})
with np.load(R/'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False) as z:
 x=z['input_V'].copy();white=z['white'].copy();si=z['source_TRAIN_index'].copy();ri=z['REAL_TRAIN_index'].copy();ordinal=z['white_original_row_ordinal'].copy()
with np.load(R/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False) as z:
 mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
assert x.shape==white.shape==(128,480) and len(si)==len(ri)==64 and len(ordinal)==128
assert np.isfinite(x).all() and np.isfinite(white).all()
records={};bindings={}
for arm in C['numerical_order']:
 path=Path(C['endpoint_weight_paths'][arm]);net=FlowNetwork()
 state=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict(state);net.eval();net.requires_grad_(False)
 assert all(torch.isfinite(p).all() for p in net.parameters())
 bindings[arm]={'actual_loaded_path':str(path),'registered_original_file':C['numerical_weights_registered'][arm],
  'actual_bytes':path.stat().st_size,'loaded_state_tensor_count':len(state),'parameter_count':sum(p.numel() for p in net.parameters()),
  'weights_frozen':True,'optimizer_loaded':False,'new_updates':0,'device':'cpu',
  'registered_SHA_referenced_not_migration_hash_test':True}
 a,da,ra=solve(net,x,white,mu,scale,edges,64,64)
 b,db,rb=solve(net,x,white,mu,scale,edges,128,64)
 rms=np.sqrt(np.mean((a.astype('f8')-b.astype('f8'))**2,axis=1));lsb=1/16383.
 med=float(np.median(rms));p95=float(np.quantile(rms,.95))
 charge=max(ra['max_relative_q_error_f64'],ra['max_relative_q_error_f32'],rb['max_relative_q_error_f64'],rb['max_relative_q_error_f32'])
 d=R/arm;d.mkdir(exist_ok=True)
 np.savez_compressed(d/'TRAIN_SOLVER_CHECK_PRIVATE.npz',input_V=x,white=white,source_TRAIN_index=si,REAL_TRAIN_index=ri,
  white_original_row_ordinal=ordinal,solution64_f32_V=a,solution128_f32_V=b,RMS_difference_V=rms,
  relative_q_error64_f64=da['preADC_charge_relative_error_f64'],relative_q_error64_f32=da['preADC_charge_relative_error_f32'],
  relative_q_error128_f64=db['preADC_charge_relative_error_f64'],relative_q_error128_f32=db['preADC_charge_relative_error_f32'])
 rec={'arm':arm,'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'RMS_median_V':med,'RMS_p95_V':p95,
  'RMS_median_LSB':med/lsb,'RMS_p95_LSB':p95/lsb,'LSB_V':lsb,'median_limit_LSB':.25,'p95_limit_LSB':1.,
  'preADC_relative_q_max':charge,'relative_q_limit':1e-5,'solver64':ra,'solver128':rb,
  'pass':bool(med<=.25*lsb and p95<=lsb and charge<=1e-5),'TRAIN_source_REAL_rows':[64,64],
  'CPU_threads':12,'chunk':64,'DEV_read_or_generated':0,'additional_step_search':False,'128step_candidate':False}
 write(arm+'/TRAIN_SOLVER_CHECK.json',rec);records[arm]=rec;write('LOADED_MODEL_BINDINGS.json',bindings)
 print(json.dumps({'model_numerical_result':rec}),flush=True)
 del net,state,a,b,da,db
summary={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
 'models':records,'both_pass':all(v['pass'] for v in records.values()),'order':C['numerical_order'],
 'new_response_training_updates':0,'new_TRAIN_random_states':0,'DEV_read_or_generated':0,
 'fixed_candidate_reference_steps':[64,128],'CPU_threads':12,'chunk':64}
write('TRAIN_SOLVER_CHECK.json',summary);print(json.dumps(summary),flush=True)
