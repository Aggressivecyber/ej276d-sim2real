from __future__ import annotations
import os,sys,time,json,datetime,pathlib,ctypes,subprocess
from ctypes import wintypes
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
for n in ['tmp','matplotlib','joblib','torch','cuda']:(ROOT/'cache'/n).mkdir(parents=True,exist_ok=True)
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(ROOT/'cache/tmp')
os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),TORCH_HOME=str(ROOT/'cache/torch'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
class M(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;gm.argtypes=[wintypes.HANDLE,ctypes.POINTER(M),wintypes.DWORD];gm.restype=wintypes.BOOL
entry=sys.argv[1];python=sys.argv[2];history=json.loads((ROOT/'RUN_HISTORY.json').read_text('utf8'))if(ROOT/'RUN_HISTORY.json').exists()else[]
if (ROOT/(entry+'.used')).exists():raise RuntimeError('owned entry already executed; retain receipt, use named engineering attempt if needed')
(ROOT/(entry+'.used')).write_text(utc(),'utf8')
remaining=CFG['science_s']-60-sum(r['elapsed_s']for r in history)
limit=min(remaining,datetime.datetime.fromisoformat(CFG['seal_deadline_UTC']).timestamp()-time.time()-120)
if limit<=0:raise TimeoutError('scientific or sealing time exhausted')
row={'entry':entry,'supervisor_PID':os.getpid(),'python':python,'start_UTC':utc(),'wall_limit_s':limit,'RSS_peak_bytes':0,'memory_probe_failures':0}
start=time.monotonic();reason=None
with (ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
    p=subprocess.Popen([python,'-B',str(ROOT/entry)],cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row['worker_PID']=p.pid;write('ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)
    try:
        while p.poll()is None:
            try:
                m=M();m.cb=ctypes.sizeof(m)
                if not gm(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb):raise OSError(ctypes.get_last_error())
                row['RSS_peak_bytes']=max(row['RSS_peak_bytes'],m.PeakWorkingSetSize,m.WorkingSetSize)
                if row['RSS_peak_bytes']>CFG['RSS_limit']:reason='RSS_limit'
            except Exception:
                row['memory_probe_failures']+=1
                if row['memory_probe_failures']>10:reason='memory_probe_unavailable'
            if time.monotonic()-start>limit:reason='wall_limit'
            if reason:p.kill();break
            time.sleep(.1)
        p.wait()
    finally:
        if p.poll()is None:p.kill();p.wait()
row.update(exit_code=p.returncode,termination_reason=reason,elapsed_s=time.monotonic()-start,stop_UTC=utc())
history.append(row);write('RUN_HISTORY.json',history);write('ACTIVE_JOB.json',row);write(entry+'.NATIVE_RECEIPT.json',row);print(json.dumps({'finished':row}),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None else 1)
