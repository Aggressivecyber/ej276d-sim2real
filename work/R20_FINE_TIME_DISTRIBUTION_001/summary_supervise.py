from __future__ import annotations
import os,pathlib,json,datetime,time,subprocess,sys
import ctypes
from ctypes import wintypes

class ProcessMemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in [
        'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

get_memory=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemoryCounters),wintypes.DWORD]
get_memory.restype=wintypes.BOOL
ROOT=pathlib.Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
start=time.monotonic();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
row={'supervisor_PID':os.getpid(),'start_UTC':utc(),'purpose':'Only existing-result waveform variation summary and validation-trace plot; no new inference or fit.',
     'allowed_s':40,'RSS_limit_bytes':4294967296,'RSS_peak_bytes':0,'RSS_probe_failures':0}
with (ROOT/'SAVED_SUMMARY.stdout.log').open('wb') as out,(ROOT/'SAVED_SUMMARY.stderr.log').open('wb') as err:
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'saved_result_summary.py')],cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row['worker_PID']=p.pid;print(json.dumps(row),flush=True)
    reason=None
    while p.poll() is None:
        try:
            mem=ProcessMemoryCounters();mem.cb=ctypes.sizeof(mem)
            if not get_memory(wintypes.HANDLE(int(p._handle)),ctypes.byref(mem),mem.cb):
                raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo')
            rss=max(mem.WorkingSetSize,mem.PeakWorkingSetSize)
            row['RSS_peak_bytes']=max(row['RSS_peak_bytes'],rss)
            if rss>row['RSS_limit_bytes']:reason='RSS_limit'
        except Exception:row['RSS_probe_failures']+=1
        if time.monotonic()-start>40:reason='wall_timeout'
        if reason:p.kill();break
        time.sleep(.05)
    p.wait()
row.update(stop_UTC=utc(),elapsed_s=time.monotonic()-start,exit_code=p.returncode,termination_reason=reason)
(ROOT/'SAVED_RESULT_SUMMARY_NATIVE_RECEIPT.json').write_text(json.dumps(row,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
print(json.dumps(row),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None else 1)
