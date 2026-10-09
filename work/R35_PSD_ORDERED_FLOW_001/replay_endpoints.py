from __future__ import annotations
import os,time,pathlib,json,datetime
import numpy as np,torch
from flow_runtime import Pipeline,infer
from model import adc
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def load(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');rows={};count=0
for name in CFG['endpoints']:
    pipeline=Pipeline(CFG,ROOT,name)
    pre,signal,electronic=infer(pipeline,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],CFG['generation_chunk'])
    post,ad=adc(pre);saved=load(ROOT/name/'generated.npz')
    check={'pre_ADC_V':bool(np.array_equal(pre,saved['pre_ADC_V'])),'waveforms_V':bool(np.array_equal(post,saved['waveforms_V'])),
        'source_ids':bool(np.array_equal(actual['source_ids'],saved['source_ids']))}
    for key,value in pipeline.details.items():check[key]=bool(np.array_equal(value,saved[key]))
    rows[name]={'rows':len(pre),'checks':check,'bitwise_pass':all(check.values()),'ADC':ad};count+=len(pre)
    del pipeline,saved
passed=all(r['bitwise_pass']for r in rows.values())
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'rows':count,'pass':passed,'endpoints':rows,'independent_process':True,'full_source_to_base_to_flow_replayed':True,
    'fit_or_update':0,'REAL_TRAIN_validation_read':False,'device':'cpu','torch':torch.__version__,'NumPy':np.__version__}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True)
assert passed
