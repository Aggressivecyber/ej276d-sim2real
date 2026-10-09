from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import time,pathlib,json,datetime
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent
import numpy as np,torch
from model import Translator,infer,adc
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
cfg=json.loads((ROOT/'config.json').read_text());z=np.load(ROOT/'ACTUAL_GENERATION_INPUTS.npz',allow_pickle=False);model=Translator();model.load_state_dict(torch.load(ROOT/'TRANSLATOR.pt',map_location='cpu',weights_only=True));pre,_,_=infer(model,z['source_waveforms_V'],z['z'],z['white'],cfg['generation_chunk']);post,_=adc(pre)
with np.load(ROOT/'generated.npz',allow_pickle=False)as old:before=old['pre_ADC_V'];after=old['waveforms_V'];ids=old['source_ids'];result={'pre_ADC_bitwise':bool(np.array_equal(pre,before)),'post_ADC_bitwise':bool(np.array_equal(post,after)),'source_ids_equal':bool(np.array_equal(z['source_ids'],ids)),'pre_ADC_maxabs_V':float(np.max(np.abs(pre-before))),'post_ADC_maxabs_V':float(np.max(np.abs(post-after)))}
result.update(PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-START,rows=2000,points=960000,source_required=True,target_or_TRAIN_required=False,training_updates=0);result['pass']=all(result[k]for k in ['pre_ADC_bitwise','post_ADC_bitwise','source_ids_equal']);(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(0 if result['pass']else 1)
