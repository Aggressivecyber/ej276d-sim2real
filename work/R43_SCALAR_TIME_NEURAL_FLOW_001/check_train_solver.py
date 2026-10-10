from __future__ import annotations
import pathlib,json,time,os,datetime
import numpy as np,torch
from source_flow_core import FlowNetwork,solve,q_np
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
with np.load(ROOT/'R42_TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False)as z:
    x=z['input_V'].copy();white=z['white'].copy();si=z['source_TRAIN_index'].copy();ri=z['REAL_TRAIN_index'].copy();wanted=z['white_original_row_ordinal'].copy();old_rms=z['RMS_difference_V'].copy()
with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
assert x.shape==white.shape==(128,480) and len(si)==len(ri)==64 and len(wanted)==128
net=FlowNetwork();net.load_state_dict(torch.load(CFG['endpoint_weight_paths']['SCALAR_TIME_FLOW_FINAL'],map_location='cpu',weights_only=True));net.eval();net.requires_grad_(False)
a,da,ra=solve(net,x,white,mu,scale,edges,32,64);b,db,rb=solve(net,x,white,mu,scale,edges,64,64)
rms=np.sqrt(np.mean((a.astype('f8')-b.astype('f8'))**2,1));lsb=1/16383.;med=float(np.median(rms));p95=float(np.quantile(rms,.95));passed=med<=.25*lsb and p95<=lsb and max(ra['max_relative_q_error_f32'],rb['max_relative_q_error_f32'])<=1e-5
np.savez_compressed(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz',source_TRAIN_index=si,REAL_TRAIN_index=ri,input_V=x,white=white,white_original_row_ordinal=wanted,solution32_f32_V=a,solution64_f32_V=b,RMS_difference_V=rms,R42_RMS_difference_V=old_rms,paired_RMS_difference_V=rms-old_rms)
rec={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'TRAIN_source_REAL_rows':[64,64],'white_prefix':'first128 consumed-row white, original noise-generation ordinal saved','DEV_read_or_generated':0,'RMS_median_V':med,'RMS_p95_V':p95,'LSB_V':lsb,'RMS_median_LSB':med/lsb,'RMS_p95_LSB':p95/lsb,'median_limit_LSB':.25,'p95_limit_LSB':1.,'solver32':ra,'solver64':rb,'pass':passed,'no_prefix_or_64step_candidate':True,'exact_R42_inputs_white_indices_order_reused':True,'R42_reference_RMS_median_LSB':float(np.median(old_rms)/lsb),'R42_reference_RMS_p95_LSB':float(np.quantile(old_rms,.95)/lsb),'paired_RMS_difference_LSB_mean':float(np.mean((rms-old_rms)/lsb))}
(ROOT/'TRAIN_SOLVER_CHECK.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n','utf8');print(json.dumps(rec),flush=True)
