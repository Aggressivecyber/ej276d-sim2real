"""Reproduce the saved main response using its trusted local weights; no fit.

Usage: python -B replay_response.py E:/some/new-output.npz
The caller chooses a fresh E/D output, so the R2 delivery stays immutable.
"""
import os,sys,pathlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch
from model import SmoothResponse
torch.set_num_threads(4);torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parent
target=pathlib.Path(sys.argv[1]).resolve()
if target.drive.upper() not in ('E:','D:') or target.exists():raise ValueError('fresh E/D output required')
pack=torch.load(root/'model.pt',map_location='cpu',weights_only=False) # Our own saved trusted checkpoint.
model=SmoothResponse(**pack['constructor']);model.load_state_dict(pack['state_dict']);model.eval()
with np.load(root/'generated.npz',allow_pickle=False) as z:
    c=torch.tensor(z['G4_condition']);energy=z['visible_energy_proxy_MeV']
    q=torch.tensor(((energy+.03918)/.000132506/1000).astype('f4'))
    expected=z['waveforms_V'];ids=z['sim_event_ids']
n=len(c);g=torch.Generator().manual_seed(pack['config']['generation_seeds'][0])
latent=torch.randn((n,8),generator=g);white=torch.randn((n,512),generator=g);outputs=[]
with torch.no_grad():
    for start in range(0,n,256):outputs.append(model(c[start:start+256],q[start:start+256],latent[start:start+256],white[start:start+256])[0].numpy())
out=np.concatenate(outputs);assert np.array_equal(out,expected),'saved response differs'
np.savez_compressed(target,waveforms_V=out,sim_event_ids=ids)
print('saved main response reproduced bitwise:',n,'parents; no fit')
