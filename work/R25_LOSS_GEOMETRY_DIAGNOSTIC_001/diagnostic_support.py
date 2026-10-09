from __future__ import annotations
import os,json,time,datetime,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(name,value):
    (ROOT/name).write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n','utf8')
def log(x):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(safe(x)),flush=True)
def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def check_output():
    n=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if n>CFG['output_limit_bytes']:raise MemoryError('stage output cap')
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
