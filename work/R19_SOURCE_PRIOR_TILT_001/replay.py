from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent
import numpy as np,torch
from model import Translator,infer,adc
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
cfg=json.loads((ROOT/'config.json').read_text());a=np.load(ROOT/'ACTUAL_GENERATION_INPUTS.npz',allow_pickle=False);rows={}
for arm in ['CONT','TILT']:
    model=Translator();model.load_state_dict(torch.load(ROOT/arm/'TRANSLATOR.pt',map_location='cpu',weights_only=True));pre,_,_=infer(model,a['source_waveforms_V'],a['z'],a['white'],cfg['generation_chunk']);post,_=adc(pre)
    with np.load(ROOT/arm/'generated.npz',allow_pickle=False)as b:
        rows[arm]={'pre_ADC_bitwise':bool(np.array_equal(pre,b['pre_ADC_V'])),'post_ADC_bitwise':bool(np.array_equal(post,b['waveforms_V'])),'source_ids_equal':bool(np.array_equal(a['source_ids'],b['source_ids'])),'pre_ADC_maxabs_V':float(np.max(np.abs(pre-b['pre_ADC_V']))),'post_ADC_maxabs_V':float(np.max(np.abs(post-b['waveforms_V']))),'rows':2000}
    del model
passed=all(all(v[k]for k in ['pre_ADC_bitwise','post_ADC_bitwise','source_ids_equal'])for v in rows.values());r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'arms':rows,'rows':4000,'points':1920000,'pass':passed,'TRAIN_target_or_REAL_used':False,'training_updates':0}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);raise SystemExit(0 if passed else 1)
