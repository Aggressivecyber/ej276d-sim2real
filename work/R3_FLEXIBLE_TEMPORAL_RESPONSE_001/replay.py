"""Fresh saved-model/main-generation RNG replay; no optimizer or C2ST."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='4'
import pathlib,json,time,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parent
START=time.monotonic()
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
cfg=json.loads((ROOT/'config.json').read_text())
torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
from model import SmoothResponse
pack=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=True)
state=torch.load(ROOT/'GENERATION_RANDOM_STATE.pt',map_location='cpu',weights_only=True)
model=SmoothResponse(**pack['constructor']);model.load_state_dict(pack['state_dict']);model.eval()
with np.load(ROOT/'generation_inputs.npz',allow_pickle=False) as z:
    c=torch.tensor(z['G4_condition']);q=torch.tensor(z['Qsource_Vns']);ids=z['sim_event_ids']
with np.load(ROOT/'generated.npz',allow_pickle=False) as z:
    reference=z['waveforms_V'];reference_ids=z['sim_event_ids']
assert state['n']==len(c)==2000 and np.array_equal(ids,reference_ids)
gen=torch.Generator();gen.set_state(state['generator_before_draw'])
latent=torch.randn((2000,8),generator=gen);white=torch.randn((2000,512),generator=gen)
outputs=[]
with torch.no_grad():
    for first in range(0,2000,state['generation_chunk']):
        stop=first+state['generation_chunk']
        outputs.append(model(c[first:stop],q[first:stop],latent[first:stop],white[first:stop])[0].numpy())
out=np.concatenate(outputs)
equal=np.array_equal(out,reference);maxdiff=float(np.max(np.abs(out.astype('f8')-reference)))
hashfile=lambda name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'status':'pass' if equal else 'fail','from_actual_saved_weights_and_random_state':True,'main_waveform_shape':list(out.shape),'float32_values_compared':int(out.size),'bitwise_equal':bool(equal),'max_absolute_difference_V':maxdiff,'parent_ID_order_equal':True,'model_SHA256':hashfile('model.pt'),'generation_RNG_SHA256':hashfile('GENERATION_RANDOM_STATE.pt'),'generation_inputs_SHA256':hashfile('generation_inputs.npz'),'torch_version':torch.__version__,'threads':4,'optimizer_updates':0,'C2ST_refit':0,'EVAL_read':0,'SSH':0,'elapsed_s':time.monotonic()-START}
(ROOT/'REPLAY_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
assert equal,'saved model/RNG generation replay mismatch'
