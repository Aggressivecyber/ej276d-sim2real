from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,ast,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001';cache=S/'cache/local_artificial_check';cache.mkdir(parents=True,exist_ok=True)
for key in ['TMP','TEMP','TMPDIR','TORCH_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH']:os.environ[key]=str(cache)
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
sys.dont_write_bytecode=True
sys.path.insert(0,str(S))
import torch,ctypes
from ctypes import wintypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
handle=kernel.GetCurrentProcess();mask=ctypes.c_size_t();systemmask=ctypes.c_size_t()
kernel.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(mask),ctypes.byref(systemmask))
affinity=[i for i in range(mask.value.bit_length()) if mask.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in affinity))
torch.set_num_threads(4);torch.set_num_interop_threads(4)
from artificial_check import check,integration_check
result=check()
integrated=integration_check(json.loads((S/'config.json').read_text('utf8')))
files=list(S.glob('*.py'))+list((R/'reviews').glob('*R59*root*.py'))
for f in files:ast.parse(f.read_text('utf8'),filename=str(f))
class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
memory=Memory();memory.cb=ctypes.sizeof(memory);psapi=ctypes.WinDLL('psapi',use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
assert psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb)
peak=memory.PeakWorkingSetSize;assert peak<4*1024**3
value={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':affinity,'worker_peak_RSS_bytes':peak,'torch':torch.__version__,'result':result,'integrated_loss_and_network':integrated,'AST_files':len(files),
    'code_binding':[{'path':f.relative_to(R).as_posix(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],'scientific_data_read':False,'response_training_updates':0}
(S/'ARTIFICIAL_CHECK_LOCAL.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
print(json.dumps(value,indent=2))
