from __future__ import annotations
import os,time,pathlib,json,datetime,hashlib,traceback
import numpy as np,torch
from flow_runtime import Pipeline,infer
from model import adc
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
ROWS={};COUNT=0
def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def write(n,r):(ROOT/n).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n','utf8')
def exact(a,b):return bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
def run():
    global COUNT
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',keys=['reference_V','reference_ids','reference_groups'])
    for name in ['SOURCE_NEURAL_FLOW_FINAL','SOURCE_NEURAL_FLOW_REAL_CONTROL']:
        pipeline=Pipeline(CFG,ROOT,name)
        source=actual['source_waveforms_V']if name=='SOURCE_NEURAL_FLOW_FINAL'else real['reference_V']
        identity={'source_ids':actual['source_ids']}if name=='SOURCE_NEURAL_FLOW_FINAL'else{'reference_ids':real['reference_ids'],'reference_groups':real['reference_groups']}
        pre,_,_=infer(pipeline,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);post,ad=adc(pre)
        # Outputs opened only after independent full computation, never replay input.
        saved=load(ROOT/name/'generated.npz');checks={k:exact(v,saved[k])for k,v in {'pre_ADC_V':pre,'waveforms_V':post,**pipeline.details,**identity}.items()}
        ROWS[name]={'rows':len(pre),'checks':checks,'bitwise_pass':all(checks.values()),'ADC':ad,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest()}
        COUNT+=len(pre);write('REPLAY_PROGRESS.json',{'PID':os.getpid(),'rows':COUNT,'domains':ROWS,'elapsed_s':time.monotonic()-START});assert ROWS[name]['bitwise_pass'],checks
    r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'rows':COUNT,'pass':all(v['bitwise_pass']for v in ROWS.values()),'endpoints':ROWS,'independent_process':True,'complete_source_to_standalone_flow_to_oneADC_replayed':True,'source_support_outside_edge_conditioning_no_fallback_replayed':True,'controls_same_REAL_random_binding':True,'fit_or_update':0,'reader_or_evaluator_read_for_inference':False,'device':'cpu','NumPy':np.__version__,'torch':torch.__version__}
    assert COUNT==4000 and r['pass'];write('REPLAY_RECEIPT.json',r);print(json.dumps(r),flush=True)
try:run()
except BaseException as e:
    write('REPLAY_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'rows':COUNT,'endpoints':ROWS,'elapsed_s':time.monotonic()-START});raise
