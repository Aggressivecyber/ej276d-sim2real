from __future__ import annotations
import os,time,pathlib,json,datetime,hashlib,traceback
import numpy as np,torch
from flow_runtime import Pipeline,infer
from flow_core import bins
from model import adc
ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
ROWS={};COUNT=0
def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n','utf8')
def exact(a,b):
    return bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C'))
def verify(name,pre,post,detail,saved,identity):
    checks={k:exact(a,saved[k])for k,a in {'pre_ADC_V':pre,'waveforms_V':post,**detail,**identity}.items()}
    return {'rows':len(pre),'checks':checks,'bitwise_pass':all(checks.values()),
        'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),
        'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
        'NaN_metadata_compared_by_exact_bytes':True}
def run():
    global COUNT
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
    name='FULL_COVARIANCE_CALIBRATION';pipeline=Pipeline(CFG,ROOT,name)
    pre,signal,electronic=infer(pipeline,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],CFG['generation_chunk'])
    post,ad=adc(pre);saved=load(ROOT/name/'generated.npz')
    ROWS[name]=verify(name,pre,post,pipeline.details,saved,{'source_ids':actual['source_ids']});ROWS[name]['ADC']=ad;COUNT+=len(pre)
    write('REPLAY_PROGRESS.json',{'PID':os.getpid(),'rows':COUNT,'domains':ROWS,'elapsed_s':time.monotonic()-START})
    assert ROWS[name]['bitwise_pass'],ROWS[name]['checks']
    receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
        'rows':COUNT,'pass':all(v['bitwise_pass']for v in ROWS.values()),'endpoints':ROWS,'independent_process':True,
        'complete_source_to_BASE_to_TRAIN_calibration_to_ADC_replayed':True,
        'original_source_bin_outside_support_BASE_fallback_replayed':True,
        'fit_or_update':0,'new_REAL_library_or_TRAIN_model_fit_input_read':False,
        'device':'cpu','NumPy':np.__version__,'torch':torch.__version__}
    assert COUNT==2000 and receipt['pass'];write('REPLAY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
try:run()
except BaseException as e:
    write('REPLAY_FAILURE.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'error':repr(e),'traceback':traceback.format_exc(),'completed_rows':COUNT,'domains':ROWS,'elapsed_s':time.monotonic()-START});raise
