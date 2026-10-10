"""Portable frozen inference. Requires only packaged weights/constants and input arrays."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True
_root=Path(__file__).resolve().parent
for _name in ['tmp','torch','cuda']:
    (_root/'cache'/_name).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMP=str(_root/'cache/tmp'),TEMP=str(_root/'cache/tmp'),
                  TMPDIR=str(_root/'cache/tmp'),TORCH_HOME=str(_root/'cache/torch'),CUDA_CACHE_PATH=str(_root/'cache/cuda'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from flow_runtime import Pipeline,infer
from model import adc

def build_pipeline(root,device='cuda:0',endpoint='CONTINUOUS_Q_RESPONSE'):
    root=Path(root).resolve()
    names=['CONTINUOUS_Q_RESPONSE','CONTINUOUS_Q_REAL_CONTROL']
    assert endpoint in names
    cfg={'endpoint_weight_paths':{n:str(root/'CONTINUOUS_Q_RESPONSE/FINAL_FLOW.pt')for n in names},
         'constants_path':str(root/'UNIT_CHARGE_CONSTANTS.npz'),'endpoints':names,
         'inference_device':device,'solver_steps':128}
    return Pipeline(cfg,root,endpoint)

def main():
    import argparse,json,time
    from datetime import datetime,timezone
    p=argparse.ArgumentParser(description='Frozen R54 full480 response; explicit white input for reproducibility')
    p.add_argument('--input',type=Path,required=True,help='NPZ containing source_waveforms_V and white, both N x 480')
    p.add_argument('--output',type=Path,required=True,help='New output NPZ path (will not overwrite)')
    p.add_argument('--device',default='cuda:0',choices=['cuda:0','cpu'])
    args=p.parse_args();root=Path(__file__).resolve().parent
    assert args.output.suffix=='.npz' and not args.output.exists()
    assert not args.output.with_suffix('.receipt.json').exists()
    if os.name=='nt':assert args.output.resolve().drive.upper() in ['E:','D:'],'Windows output must be on E: or D:'
    torch.set_num_threads(4 if os.name=='nt' else 12);torch.set_num_interop_threads(4 if os.name=='nt' else 12)
    torch.use_deterministic_algorithms(True)
    with np.load(args.input,allow_pickle=False)as z:
        source=z['source_waveforms_V'].copy();white=z['white'].copy()
        identity={k:z[k].copy()for k in ['source_ids','reference_ids','reference_groups']if k in z.files}
    assert source.shape==white.shape and source.ndim==2 and source.shape[1]==480
    start=time.monotonic();model=build_pipeline(root,args.device)
    pre,_,_=infer(model,source,None,white,None,64);post,ad=adc(pre)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb')as dest:
        np.savez_compressed(dest,pre_ADC_V=pre,waveforms_V=post,**model.details,**identity)
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'rows':len(source),'device':args.device,'solver_steps':128,'chunk':64,
             'input':str(args.input),'output':str(args.output),'elapsed_s':time.monotonic()-start,'ADC':ad,
             'TRAIN_CAL_DEV_reference_read':False,'same_operator_for_SOURCE_and_REAL':True,
             'new_fit_or_update':0,'seeds_explicit_in_input_white':True,'validation_scope':'See ACCEPTANCE_A_C.json; inference execution is not a success claim.'}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
