from __future__ import annotations
import os,time,json,pathlib,datetime,traceback
os.environ['R44_THREADS']='4'
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
from diagnose_frozen import load,population,plot,DT
import numpy as np
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
def put(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
try:
    rec=json.loads((ROOT/'NUMERIC_RECEIPT.json').read_text('utf8'));assert rec['committed_rows']==rec['network_forward_rows']==94400
    arrays=load(CFG['TRAIN_path'],['source','target']);stream=load(CFG['coupling_path']);const=load(CFG['constants_path'])
    rows=np.load(ROOT/'results/PATH_ROWS_PRIVATE.npy',mmap_mode='r',allow_pickle=False)
    assert rows.dtype==DT and len(rows)==94400
    summary=json.loads((ROOT/'VELOCITY_SUMMARY.json').read_text('utf8'))
    pops,markers=population(arrays['source'],arrays['target'],stream,rows,const['edges_logQ']);hits=sum(r['hit']for r in markers)
    put('DIRECTION_MARKERS.json',{'bins':markers,'hits':hits,'required_hits':4,'supported':hits>=4,'decision':'TRAIN_path_has_directionally_aligned_mean_PSD_residual'if hits>=4 else'close_mean_TRAIN_PSD_velocity_residual_explanation','causal_claim':False,'new_training_authorized':False})
    plot(rows,summary['by_Q_time'],pops,markers)
    put('AUXILIARY_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'direction_hits':hits,'local_CPU_threads':4,'new_network_forward_rows':0,'new_ODE_generation_training_fits':0,'original_94400_rows_readonly':True})
    print(json.dumps({'PID':os.getpid(),'complete':True,'hits':hits,'markers':markers,'PSD_MSE_energy_fraction':summary['overall']['PSD_mode_energy_fraction'],'elapsed_s':time.monotonic()-START}),flush=True)
except BaseException as e:
    put('AUXILIARY_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'new_network_forward_rows':0});raise
