from __future__ import annotations
import os,pathlib,json,time,datetime,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'))
import numpy as np,torch
from model import Translator,infer,adc
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def load(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');out={}
for name in CFG['endpoints']:
    model=Translator();model.load_state_dict(torch.load(CFG['endpoint_weight_paths'][name],map_location='cpu',weights_only=True));model.eval()
    pre,signal,electronic=infer(model,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],128);post,ad=adc(pre);original=load(ROOT/name/'generated.npz');checks={}
    for key,value in {'pre_ADC_V':pre,'waveforms_V':post,'signal_component_V':signal,'electronic_component_V':electronic}.items():checks[key]={'bitwise_equal':bool(np.array_equal(value,original[key])),'maxabs_V':float(np.max(np.abs(value.astype('f8')-original[key].astype('f8')))),'replay_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    out[name]={'rows':len(post),'numeric_points_pre_plus_post':int(pre.size+post.size),'checks':checks,'source_ID_order_equal':bool(np.array_equal(original['source_ids'],actual['source_ids'])),'ADC':ad};del model
passed=all(r['source_ID_order_equal']and all(v['bitwise_equal']for v in r['checks'].values())for r in out.values())
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'rows':sum(x['rows']for x in out.values()),'pass':passed,'endpoints':out,'torch':torch.__version__,'NumPy':np.__version__,'device':'cpu','REAL_or_TRAIN_waveform_read':False,'fit_or_update':0,'reader_used':False,'independent_process':True}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if passed else 1)
