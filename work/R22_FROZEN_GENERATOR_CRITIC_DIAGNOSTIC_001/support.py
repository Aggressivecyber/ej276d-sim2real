from pathlib import Path
import os,json,datetime,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
def safe(x):
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float):return x if np.isfinite(x) else None
    if isinstance(x,dict):return {str(k):safe(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v)for v in x]
    return x
def write(n,v):(ROOT/n).write_text(json.dumps(safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(v):print(utc(),v,flush=True)
def order(ids,prefix):return np.argsort(np.array([int.from_bytes(hashlib.sha256((prefix+'|'+str(s)).encode()).digest()[:8],'big')for s in ids],dtype=np.uint64),kind='stable')
def bins(x):
    d=x.astype('f8')-x[:,:100].astype('f8').mean(1,keepdims=True)
    q=d[:,100:400].sum(1)*1000
    return np.searchsorted(CFG['edges'],np.log(np.maximum(q,1e-30)),side='right')-1
def weights(b,h=None,lambdas=None):
    w=np.zeros(len(b),'f8')
    for j in range(8):
        m=b==j
        if not m.any():raise RuntimeError('empty required bin '+str(j+1))
        v=np.ones(m.sum())if h is None else np.exp(lambdas[j]*h[m])
        w[m]=v/v.sum()/8
    return w
def npz(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
