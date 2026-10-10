from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,ast,ctypes
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
for name in ['tmp','torch','matplotlib']:(S/'cache'/name).mkdir(parents=True,exist_ok=True)
os.environ.update(TMP=str(S/'cache/tmp'),TEMP=str(S/'cache/tmp'),TMPDIR=str(S/'cache/tmp'),TORCH_HOME=str(S/'cache/torch'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=k.GetCurrentProcess();mask=ctypes.c_size_t();system=ctypes.c_size_t();assert k.GetProcessAffinityMask(handle,ctypes.byref(mask),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if mask.value&(1<<i)][:4];assert k.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
for p in S.glob('*.py'):ast.parse(p.read_text('utf8'),filename=str(p))
sys.path.insert(0,str(S))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(4)
from artificial_check import check
result=check(json.loads((S/'config.json').read_text('utf8')))
result.update(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),CPU_affinity=cpus,torch=torch.__version__)
(S/'ARTIFICIAL_CHECK_LOCAL.json').write_text(json.dumps(result,indent=2)+'\n','utf8')
print(json.dumps(result,indent=2))
