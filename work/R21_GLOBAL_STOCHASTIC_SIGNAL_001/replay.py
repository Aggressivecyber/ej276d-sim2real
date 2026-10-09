from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,numpy as np,torch
from model import Translator,infer,adc
ROOT=pathlib.Path(__file__).resolve().parent;start=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
with np.load(ROOT/'ACTUAL_GENERATION_INPUTS.npz',allow_pickle=False)as a:actual={k:a[k].copy()for k in a.files}
model=Translator();model.load_state_dict(torch.load(ROOT/'GLOBAL_STYLE/TRANSLATOR.pt',map_location='cpu',weights_only=True));model.eval()
cfg=json.loads((ROOT/'config.json').read_text());pre,_,_=infer(model,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],cfg['generation_chunk']);post,_=adc(pre)
with np.load(ROOT/'GLOBAL_STYLE/generated.npz',allow_pickle=False)as a:
    r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'rows':2000,'points':960000,
       'pre_ADC_equal':bool(np.array_equal(pre,a['pre_ADC_V'])),'post_ADC_equal':bool(np.array_equal(post,a['waveforms_V'])),'source_ids_equal':bool(np.array_equal(actual['source_ids'],a['source_ids'])),
       'pre_ADC_maxabs_V':float(np.max(np.abs(pre-a['pre_ADC_V']))),'post_ADC_maxabs_V':float(np.max(np.abs(post-a['waveforms_V']))),
       'style_reader_used':False,'TRAIN_or_REAL_events_used':False,'training_updates':0}
r['pass']=r['pre_ADC_equal']and r['post_ADC_equal']and r['source_ids_equal']
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);raise SystemExit(0 if r['pass']else 1)
