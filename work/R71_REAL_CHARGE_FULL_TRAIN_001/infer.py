"""Inference CLI for the frozen R71 charge component, not an accepted full model."""
from pathlib import Path
import argparse,os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--key',default='waveforms_V')
parser.add_argument('--weights',type=Path,default=Path(__file__).resolve().parent/'FINAL_HEAD.pt')
parser.add_argument('--statistics',type=Path,default=Path(__file__).resolve().parent/'HEAD_STATISTICS.npz')
noise=parser.add_mutually_exclusive_group(required=True)
noise.add_argument('--epsilon',type=Path,help='NPZ with epsilon shape(N,480), bound to original row order')
noise.add_argument('--seed',type=int,help='PCG64 seed; one standard_normal((N,480)).astype(float32) draw')
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
parser.add_argument('--threads',type=int,default=1)
args=parser.parse_args()
if args.output.exists():raise FileExistsError('Output already exists; preserve original outputs')
if args.output.resolve()==args.input.resolve():raise ValueError('Input cannot be overwritten')
if os.name=='nt'and args.output.resolve().drive.upper()not in {'E:','D:'}:raise ValueError('Local output must be on E or D')
if args.threads<1 or args.threads>12:raise ValueError('threads must be1..12')
import numpy as np
import torch
from public_charge_head import PublicConditionalChargeHead
from common_response import generate
torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
with np.load(args.input,allow_pickle=False)as z:
    x=np.asarray(z[args.key],dtype='f4').copy()
    parent_metadata={k:z[k].copy()for k in z.files if k!=args.key and z[k].ndim==1 and len(z[k])==len(x)}
if x.ndim!=2 or x.shape[1]!=480:raise ValueError('Expected original common480 input window in volts')
if args.epsilon:
    with np.load(args.epsilon,allow_pickle=False)as z:epsilon=np.asarray(z['epsilon'],dtype='f4').copy()
else:epsilon=np.random.Generator(np.random.PCG64(args.seed)).standard_normal(x.shape).astype('f4')
with np.load(args.statistics,allow_pickle=False)as z:stats={k:z[k].copy()for k in z.files}
head=PublicConditionalChargeHead(torch.tensor(stats['feature_center'],device=args.device),torch.tensor(stats['feature_scale'],device=args.device))
head.load_state_dict(torch.load(args.weights,map_location=args.device,weights_only=True));head.eval()
pre,adc,meta=generate(head,x,epsilon,stats['mu'],stats['scale'],stats['edges'],stats['p480'],float(stats['tau2']),256)
args.output.parent.mkdir(exist_ok=True,parents=True)
np.savez_compressed(args.output,pre_ADC_V=pre,canonical_ADC_V=adc,**meta,**{'input_'+k:v for k,v in parent_metadata.items()})
print(f'R71 charge component: {len(x)} rows saved to {args.output}; full-model A-F remains unvalidated.',flush=True)
