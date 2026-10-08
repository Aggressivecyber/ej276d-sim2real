from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,io,hashlib,time,datetime
import numpy as np
import torch
from model import WholeResponse
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(1)
cfg=json.loads((ROOT/'config.json').read_text());reads=[]
def load(p):
    b=p.read_bytes();reads.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});return b
with np.load(io.BytesIO(load(ROOT/'GENERATION_INPUTS.npz')),allow_pickle=False) as z:
    x=z['clean_V'].copy();c=z['G4_condition'].copy();q=z['Qsource_Vns'].copy();ids=z['sim_event_ids'].copy()
    noise=[z['view1_'+k].copy() for k in ['z_pulse','z_base','white_pulse','white_base']]
assert x.shape==(2000,480)
result={};rows=0
for name in cfg['arms']:
    p=ROOT/'arms'/name;pack=torch.load(io.BytesIO(load(p/'model.pt')),map_location='cpu',weights_only=True)
    torch.manual_seed(cfg['seed']);model=WholeResponse(**pack['constructor'],pulse_enabled=pack['pulse_enabled']);model.load_state_dict(pack['state_dict']);model.eval()
    with np.load(io.BytesIO(load(p/'generated.npz')),allow_pickle=False) as z:
        target=z['waveforms_V'].copy();saved_ids=z['sim_event_ids'].copy()
    assert np.array_equal(ids,saved_ids)
    out=[]
    with torch.no_grad():
        for i in range(0,2000,cfg['generation_chunk']):
            sl=slice(i,i+cfg['generation_chunk']);args=[torch.tensor(v[sl]) for v in [x,c,q,*noise]]
            out.append(model(*args).numpy());rows+=len(out[-1])
    actual=np.concatenate(out);bitwise=np.array_equal(actual,target);delta=float(np.max(np.abs(actual.astype('f8')-target.astype('f8'))))
    result[name]={'parents':2000,'parent_order_exact':True,'bitwise_equal':bitwise,'maxabs_V':delta,
        'actual_waveform_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'saved_waveform_sha256':hashlib.sha256(target.tobytes()).hexdigest(),
        'selected_step':pack['selected_step'],'pulse_enabled':pack['pulse_enabled']}
    if not bitwise:raise RuntimeError(name+' saved model replay differs')
receipt={'status':'passed','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'elapsed_s':time.monotonic()-START,'inference_rows':rows,'waveform_values':rows*480,'all_arms':result,
    'saved_actual_inputs_not_redrawn':True,'fresh_process':True,'fit':0,'full_evaluation_rerun':0,'threads':4,'EVAL_read':0,'SSH':0,'reads':reads}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
