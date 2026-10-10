from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,threading,traceback,ast,ctypes
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';start=time.monotonic()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
for n in ['tmp','torch','matplotlib']:(S/'cache'/n).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TEMP=str(S/'cache/tmp'),TMP=str(S/'cache/tmp'),TORCH_HOME=str(S/'cache/torch'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
stop=threading.Event();peak=0
def monitor():
    global peak
    while not stop.wait(.1):
        mi=PMC();mi.cb=ctypes.sizeof(mi)
        assert psapi.GetProcessMemoryInfo(handle,ctypes.byref(mi),mi.cb)
        peak=max(peak,mi.PeakWorkingSetSize)
        if peak>4*1024**3:os._exit(137)
threading.Thread(target=monitor,daemon=True).start()
sys.path.insert(0,str(S))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
try:
    from artificial_check import check
    result=check('cpu');result.update(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),CPU_affinity=cpus,CPU_threads=4,peak_RSS_bytes=peak,elapsed_s=time.monotonic()-start)
    for f in S.glob('*.py'):ast.parse(f.read_text('utf8'),filename=f.name)
    result['all_current_python_AST_pass']=True
    (S/'ARTIFICIAL_CHECK_LOCAL.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n','utf8')
    print(json.dumps(result,indent=2))
except BaseException as e:
    (S/('ARTIFICIAL_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'TRAIN_rows_read':0,'elapsed_s':time.monotonic()-start},indent=2)+'\n','utf8')
    raise
finally:stop.set()
