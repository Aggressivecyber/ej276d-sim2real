from __future__ import annotations
import os,pathlib,json,time,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic()
import numpy as np,joblib
models=joblib.load(ROOT/'task_models.joblib');differences={}
with np.load(ROOT/'REPLAY_INPUTS.npz',allow_pickle=False) as z:
    x=z['prefix_features'];assert x.shape==(64,52)
    for key,pack in models.items():
        s=pack['scaler'];p=np.column_stack([model.predict((x-s['X_mean'])/s['X_std'])*s['Y_std'][j]+s['Y_mean'][j] for j,model in enumerate(pack['models'])]);reference=z[key]
        differences[key]={'maxabs':float(np.max(np.abs(p-reference))),'allclose':bool(np.allclose(p,reference,rtol=1e-12,atol=1e-12)),'bitwise':bool(np.array_equal(p,reference))}
assert all(d['allclose'] for d in differences.values())
r={'status':'pass','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fitted_estimators_reloaded':44,'fixed_prefix_inputs':64,'values_compared':64*44,'tolerance':{'rtol':1e-12,'atol':1e-12},'fits':0,'response_generation':0,'results':differences,'elapsed_s':time.monotonic()-START}
(ROOT/'REPLAY_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
