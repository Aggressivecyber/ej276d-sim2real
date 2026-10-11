"""Exercise the delivered CLI on the same fixed64+64 numerical rows."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,runpy,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'cli_runtime')
assert json.loads((R/'jobs.json').read_text('utf8'))['active_scientific_jobs']==0
D=P/'inference_cli_check';D.mkdir()
reg=dict(UTC=datetime.now(timezone.utc).isoformat(),selection='same original first64 SOURCE+first64 REAL TRAIN numerical rows',
    fixed_epsilon='original saved matching SOURCE/REAL epsilon rows',new_training_updates=0,CPU_inference_rows=128,
    purpose='Verify delivered inference entry; no model/seed/checkpoint choice',
    CLI_SHA256=hashlib.sha256((S/'infer.py').read_bytes()).hexdigest(),guard=guard)
(D/'REGISTRATION.json').write_text(json.dumps(reg,indent=2)+'\n','utf8')
import numpy as np
with np.load(R/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz',allow_pickle=False)as z:x=np.concatenate([z['source'][:64],z['target'][:64]])
with np.load(S/'POSTCHECK_EPSILON_PRIVATE.npz',allow_pickle=False)as z:epsilon=np.concatenate([z['SOURCE'][:64],z['REAL'][:64]])
np.savez_compressed(D/'INPUT_PRIVATE.npz',waveforms_V=x,parent_row=np.r_[np.arange(64),np.arange(64)])
np.savez_compressed(D/'EPSILON_PRIVATE.npz',epsilon=epsilon)
expected=[]
for domain in ['SOURCE','REAL']:
    with np.load(S/('LEARNED_'+domain+'_PRIVATE.npz'),allow_pickle=False)as z:expected.append(z['pre_ADC_V'][:64].copy())
expected=np.concatenate(expected)
sys.path.insert(0,str(S));sys.argv=[str(S/'infer.py'),'--input',str(D/'INPUT_PRIVATE.npz'),'--epsilon',str(D/'EPSILON_PRIVATE.npz'),'--output',str(D/'OUTPUT_PRIVATE.npz'),'--device','cpu','--threads','1']
runpy.run_path(str(S/'infer.py'),run_name='__main__')
with np.load(D/'OUTPUT_PRIVATE.npz',allow_pickle=False)as z:
    pre=z['pre_ADC_V'];maxabs=float(np.max(np.abs(pre.astype('f8')-expected.astype('f8'))))
    err=np.sqrt(np.mean((pre.astype('f8')-expected.astype('f8'))**2,1))*16383
    assert np.median(err)<=.25 and np.quantile(err,.95)<=1
    assert np.array_equal(z['input_parent_row'],np.r_[np.arange(64),np.arange(64)])
answer=dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit=0,CPU_inference_rows=128,new_training_updates=0,
    preADC_bitwise=bool(np.array_equal(pre,expected)),max_abs_V=maxabs,median_RMS_LSB=float(np.median(err)),p95_RMS_LSB=float(np.quantile(err,.95)),
    all_input_parent_rows_preserved=True,model_goal_complete=False,guard={**guard,**memory()})
(D/'CLI_EXECUTION_RECEIPT.json').write_text(json.dumps(answer,indent=2)+'\n','utf8')
print(json.dumps(answer),flush=True)
