from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,hashlib,time,datetime,io,traceback
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
from model import Response
def read(n):
    p=ROOT/n;b=p.read_bytes();return b,{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def run():
    payload,binding=read('GENERATION_INPUTS.npz');inputs=[binding]
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        c=torch.tensor(z['G4_condition']);q=torch.tensor(z['Qsource_Vns']);ids=z['sim_event_ids'].copy()
        latent=torch.tensor(z['view1_z']);white=torch.tensor(z['view1_baseline_white']);signal=torch.tensor(z['view1_signal_white'])
    assert c.shape==(2000,4) and latent.shape==(2000,8) and white.shape==signal.shape==(2000,512)
    rows={}
    for name in cfg['arms']:
        b,r=read('arms/'+name+'/model.pt');inputs.append(r);pack=torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
        torch.manual_seed(cfg['seed']);model=Response(**pack['constructor'],signal_enabled=pack['signal_enabled']);model.load_state_dict(pack['state_dict']);model.eval()
        with torch.no_grad():out=np.concatenate([model(c[i:i+256],q[i:i+256],latent[i:i+256],white[i:i+256],signal[i:i+256])[0].numpy() for i in range(0,2000,256)])
        b,r=read('arms/'+name+'/generated.npz');inputs.append(r)
        with np.load(io.BytesIO(b),allow_pickle=False) as z:expected=z['waveforms_V'].copy();same_ids=np.array_equal(ids,z['sim_event_ids'])
        equal=np.array_equal(out,expected);diff=float(np.max(np.abs(out.astype('f8')-expected.astype('f8'))));assert out.dtype==expected.dtype==np.float32
        rows[name]={'values_compared':int(out.size),'bitwise_equal':equal,'maxabs_V':diff,'same_2000_parent_order':bool(same_ids),'reconstructed_waveform_bytes_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'saved_waveform_bytes_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'no_fit_or_metric_rerun':True}
        if not equal or not same_ids:raise RuntimeError(name+' replay differs')
    receipt={'status':'pass','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'arms':rows,'values_compared':sum(x['values_compared'] for x in rows.values()),'input_bindings':inputs,'RNG_from_actual_saved_tensors':True,'new_inference_rows':4000,'optimizer_updates':0,'HGB_fits':0,'EVAL_read':0,'SSH':0,'threads':4}
    (ROOT/'REPLAY_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
try:run()
except BaseException as e:
    (ROOT/'REPLAY_FAILURE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START},indent=2)+'\n',encoding='utf-8');raise
