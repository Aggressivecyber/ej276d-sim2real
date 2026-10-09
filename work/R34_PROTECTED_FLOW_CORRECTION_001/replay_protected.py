from __future__ import annotations
import os,time,pathlib,json,datetime
import numpy as np,torch
from flow_runtime import Pipeline,infer
from projection_core import protect
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));OLD=pathlib.Path(CFG['R33_root']);START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
actual=load(OLD/'ACTUAL_GENERATION_INPUTS.npz');rows={};count=0
for name in CFG['endpoints']:
    orig=CFG['endpoint_origins'][name];pipeline=Pipeline(CFG,OLD,orig)
    pre,signal,electronic=infer(pipeline,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],CFG['generation_chunk'])
    payload,ad,provenance=protect(pipeline.details['base_pre_ADC_V'],pre,origin='replayed');saved=load(ROOT/name/'generated.npz')
    check={k:bool(np.array_equal(v,saved[k]))for k,v in payload.items()}
    for key in ['base_pre_ADC_V','base_signal_V','base_electronic_V','flow_applied','source_Qbin_actual_preADC']:
        check[key]=bool(np.array_equal(pipeline.details[key],saved[key]))
    check['source_ids']=bool(np.array_equal(actual['source_ids'],saved['source_ids']))
    check['original_pre_ADC_f32_V']=bool(np.array_equal(pre,saved['original_pre_ADC_f32_V']))
    rows[name]={'rows':len(pre),'checks':check,'bitwise_pass':all(check.values()),'ADC':ad};count+=len(pre);del pipeline,saved,payload
passed=all(r['bitwise_pass']for r in rows.values())
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'rows':count,'pass':passed,'endpoints':rows,'independent_process':True,'full_source_to_base_to_flow_to_P_replayed':True,
    'fit_or_update':0,'REAL_TRAIN_validation_read':False,'device':'cpu','torch':torch.__version__,'NumPy':np.__version__}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True);assert passed
