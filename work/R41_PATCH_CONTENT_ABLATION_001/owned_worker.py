from pathlib import Path
import os,sys,json,time,datetime,runpy,ctypes,traceback
from ctypes import wintypes
ROOT=Path(__file__).resolve().parent;entry=sys.argv[1];nonce=os.environ['OWNED_SCIENCE_NONCE'];start=time.monotonic()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
write(entry+'.SCIENCE_PID.json',{'PID':os.getpid(),'parent_PID':os.getppid(),'entry':entry,'nonce':nonce,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_scientific_imports':True,'python':sys.executable})
while True:
    p=ROOT/(entry+'.RSS_ACK.json')
    if p.exists():
        ack=json.loads(p.read_text('utf8'))
        if ack['nonce']==nonce and ack['actual_PID']==os.getpid()and ack['actual_PID_memory_query_success']:break
    if time.monotonic()-start>15:raise RuntimeError('actualsciencePID RSS handshake not acknowledged')
    time.sleep(.05)
exit_code=0;error=None
try:runpy.run_path(str(ROOT/entry),run_name='__main__')
except SystemExit as e:exit_code=e.code if isinstance(e.code,int)else(0 if e.code is None else 1);raise
except BaseException as e:exit_code=1;error=repr(e);raise
finally:
    class M(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;gm.argtypes=[wintypes.HANDLE,ctypes.POINTER(M),wintypes.DWORD];gm.restype=wintypes.BOOL
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE;m=M();m.cb=ctypes.sizeof(m);good=gm(kernel.GetCurrentProcess(),ctypes.byref(m),m.cb)
    write(entry+'.WORKER_RECEIPT.json',{'PID':os.getpid(),'parent_PID':os.getppid(),'entry':entry,'nonce':nonce,'native_exit':exit_code,'error':error,'elapsed_s':time.monotonic()-start,'self_peak_RSS_bytes':int(m.PeakWorkingSetSize)if good else None,'supervisor_bound_real_PID':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
