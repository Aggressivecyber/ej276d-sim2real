from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,threading,traceback,ast,ctypes,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';start=time.monotonic()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
for n in ['tmp','torch','matplotlib']:(S/'cache'/n).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TEMP=str(S/'cache/tmp'),TMP=str(S/'cache/tmp'),TORCH_HOME=str(S/'cache/torch'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
sys.dont_write_bytecode=True
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4];assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
class PMC(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
stop=threading.Event();peak=0
def monitor():
 global peak
 while not stop.wait(.1):
  mi=PMC();mi.cb=ctypes.sizeof(mi)
  if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(mi),mi.cb):os._exit(138)
  peak=max(peak,mi.PeakWorkingSetSize)
  if peak>4*1024**3:os._exit(137)
threading.Thread(target=monitor,daemon=True).start();sys.path.insert(0,str(S))
try:
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
 from artificial_check import check,integration_check
 formula=check();integrated=integration_check(json.loads((S/'config.json').read_text('utf8')))
 files=list(S.glob('*.py'))+list((R/'reviews').glob('*R63*root*.py'))
 for f in files:ast.parse(f.read_text('utf8'),filename=str(f))
 value={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':cpus,'CPU_threads':4,'worker_peak_RSS_bytes':peak,'elapsed_s':time.monotonic()-start,
  'torch':torch.__version__,'result':formula,'integrated_loss_and_network':integrated,'AST_files':len(files),
  'code_binding':[{'path':f.relative_to(R).as_posix(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in files],
  'scientific_data_read':False,'response_training_updates':0}
 (S/'ARTIFICIAL_CHECK_LOCAL.json').write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
 print(json.dumps({k:v for k,v in value.items()if k!='code_binding'},indent=2))
except BaseException as e:
 (S/('ARTIFICIAL_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json')).write_bytes((json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'TRAIN_rows_read':0,'elapsed_s':time.monotonic()-start},indent=2)+'\n').encode())
 raise
finally:stop.set()
