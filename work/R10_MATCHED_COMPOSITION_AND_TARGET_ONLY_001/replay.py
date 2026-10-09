import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,time,json,hashlib,datetime
import numpy as np
import torch
from model import WholeResponse
import cuda_runtime as cu
cu.configure(2026100891)
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
pack=torch.load(ROOT/'arm/model.pt',map_location='cpu',weights_only=True);model=WholeResponse(**pack['constructor'],pulse_enabled=True);model.load_state_dict(pack['state_dict']);model.to(cu.DEVICE);model.eval()
with np.load(ROOT/'GENERATION_INPUTS.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
with np.load(ROOT/'generated.npz',allow_pickle=False) as z:target=z['waveforms_V'].copy();ids=z['draw_ids'].copy();donor=z['donor_id'].copy()
assert np.array_equal(ids,data['draw_ids']) and np.array_equal(donor,data['donor_id']);out=[]
with torch.no_grad():
    for i in range(0,2000,64):
        s=slice(i,i+64);out.append(model(*[torch.tensor(data[k][s],device=cu.DEVICE) for k in ['clean_V','c4','Qsource_Vns','z_pulse','z_base','white_pulse','white_base']]).detach().cpu().numpy())
actual=np.concatenate(out);ok=np.array_equal(actual,target);receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':'passed' if ok else 'failed','rows':2000,'values':960000,'bitwise_equal':ok,'maxabs':float(np.max(np.abs(actual.astype('f8')-target.astype('f8')))),'draw_order_equal':True,'donor_equal':True,'actual_output_array_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'saved_output_array_sha256':hashlib.sha256(target.tobytes()).hexdigest(),'fresh_process':True,'training_updates':0,'metric_rerun':0,'EVAL_read':0,'threads':4,'GPU':cu.facts()}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if ok else 1)
