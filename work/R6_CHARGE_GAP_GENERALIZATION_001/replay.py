from __future__ import annotations
import os,pathlib,json,time,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from model import SmoothResponse
results={}
with np.load(ROOT/'generation_inputs.npz',allow_pickle=False) as inp:
    c=inp['G4_condition'];q=inp['Qsource_Vns'];z=inp['latent_z'];white=inp['white'];ids=inp['sim_event_ids']
for arm in ['FULL','GAP']:
    # Locally produced, owned checkpoint; only the fixed saved model is inferred.
    pack=torch.load(ROOT/arm/'R3B/model.pt',map_location='cpu',weights_only=False);model=SmoothResponse(**pack['constructor']);model.load_state_dict(pack['state_dict']);model.eval();out=[]
    with torch.no_grad():
        for i in range(0,2000,256):v,_=model(torch.tensor(c[i:i+256]),torch.tensor(q[i:i+256]),torch.tensor(z[i:i+256]),torch.tensor(white[i:i+256]));out.append(v.numpy())
    with np.load(ROOT/(arm+'_generated.npz'),allow_pickle=False) as src:expected=src['waveforms_V'];assert np.array_equal(src['sim_event_ids'],ids)
    actual=np.concatenate(out);results[arm]={'rows':2000,'compared_values':960000,'bitwise':bool(np.array_equal(actual,expected)),'maxabs':float(np.max(np.abs(actual-expected)))}
assert all(r['bitwise'] for r in results.values())
result={'status':'pass','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':results,'fits':0,'evaluations':0,'elapsed_s':time.monotonic()-START}
(ROOT/'REPLAY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
