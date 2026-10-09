from __future__ import annotations
import pathlib,json,time,os,datetime
import numpy as np,torch
from source_flow_core import FlowNetwork,solve,q_np
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
with np.load(CFG['training_input_paths']['TRAIN_PRIVATE.npz'],allow_pickle=False)as z:src=z['source'].copy();real=z['target'].copy()
with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
with np.load(CFG['consumed_R26_stream'],allow_pickle=False)as z:ordinal=z['original_row_ordinal'].copy()
# First128 consumed-row white values from the frozen R26/R33 trajectory; no new noise draw.
rs=json.loads(pathlib.Path(CFG['actual_training_random_states']).read_text('utf8'));rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=rs['noise_initial']
wanted=ordinal[:128];white=np.empty((128,480),'f4');filled=np.zeros(128,bool)
for step in range(int(wanted.max())//32+1):
    rng.normal(size=(32,8));block=rng.normal(size=(32,480)).astype('f4');use=(wanted>=step*32)&(wanted<(step+1)*32)
    white[use]=block[wanted[use]-step*32];filled[use]=True
assert filled.all()
def supported(x):
    q=q_np(x);l=np.log(1000*q);b=np.searchsorted(edges,l,side='right')-1;b[l==edges[-1]]=7;return np.flatnonzero((b>=0)&(b<8))[:64]
si=supported(src);ri=supported(real);assert len(si)==len(ri)==64
x=np.concatenate([src[si],real[ri]]);net=FlowNetwork();net.load_state_dict(torch.load(CFG['endpoint_weight_paths']['SOURCE_NEURAL_FLOW_FINAL'],map_location='cpu',weights_only=True));net.eval();net.requires_grad_(False)
a,da,ra=solve(net,x,white,mu,scale,edges,32,64);b,db,rb=solve(net,x,white,mu,scale,edges,64,64)
rms=np.sqrt(np.mean((a.astype('f8')-b.astype('f8'))**2,1));lsb=1/16383.;med=float(np.median(rms));p95=float(np.quantile(rms,.95));passed=med<=.25*lsb and p95<=lsb and max(ra['max_relative_q_error_f32'],rb['max_relative_q_error_f32'])<=1e-5
np.savez_compressed(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz',source_TRAIN_index=si,REAL_TRAIN_index=ri,input_V=x,white=white,white_original_row_ordinal=wanted,solution32_f32_V=a,solution64_f32_V=b,RMS_difference_V=rms)
rec={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'TRAIN_source_REAL_rows':[64,64],'white_prefix':'first128 consumed-row white, original noise-generation ordinal saved','DEV_read_or_generated':0,'RMS_median_V':med,'RMS_p95_V':p95,'LSB_V':lsb,'RMS_median_LSB':med/lsb,'RMS_p95_LSB':p95/lsb,'median_limit_LSB':.25,'p95_limit_LSB':1.,'solver32':ra,'solver64':rb,'pass':passed,'no_prefix_or_64step_candidate':True}
(ROOT/'TRAIN_SOLVER_CHECK.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n','utf8');print(json.dumps(rec),flush=True)
