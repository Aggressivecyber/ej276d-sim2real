from __future__ import annotations
import os,sys,pathlib,json,time,datetime,subprocess,ctypes
from ctypes import wintypes
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
END=datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-90
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1'
for n in ['tmp','matplotlib','joblib']:(ROOT/'cache'/n).mkdir(parents=True,exist_ok=True)
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(ROOT/'cache/tmp')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;gm.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD];gm.restype=wintypes.BOOL
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
history=json.loads((ROOT/'failed_002/RUN_HISTORY.json').read_text('utf8'))
if (ROOT/'LOCAL_SCIENCE_TOKEN.json').exists():raise RuntimeError('local one-use science token exists; inspect existing job, do not duplicate')
write('LOCAL_SCIENCE_TOKEN.json',{'PID':os.getpid(),'UTC':utc(),'single_owned_scientific_child_at_a_time':True})
status='failed';error=None
try:
    for entry in ['numeric_worker.py','replay.py']:
        limit=min(1800-sum(r['elapsed_s']for r in history),END-time.time())
        if limit<=0:raise TimeoutError('remaining scientific/stage wall exhausted')
        start=time.monotonic();row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'allowed_s':limit,'RSS_peak_bytes':0,'RSS_probe_failures':0,'execution':'local CPU4'}
        with (ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
            p=subprocess.Popen([sys.executable,'-B',str(ROOT/entry)],cwd=ROOT,stdout=out,stderr=err,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            row['worker_PID']=p.pid;write('ACTIVE_JOB.json',row);print(json.dumps({'job_started':row}),flush=True);reason=None
            try:
                while p.poll()is None:
                    try:
                        mem=MemoryCounters();mem.cb=ctypes.sizeof(mem)
                        if not gm(wintypes.HANDLE(int(p._handle)),ctypes.byref(mem),mem.cb):raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo')
                        row['RSS_peak_bytes']=max(row['RSS_peak_bytes'],mem.WorkingSetSize,mem.PeakWorkingSetSize)
                        if row['RSS_peak_bytes']>cfg['RSS_limit_bytes']:reason='RSS_limit'
                    except Exception:
                        row['RSS_probe_failures']+=1
                        if row['RSS_probe_failures']>10:reason='RSS_probe_unavailable'
                    if time.monotonic()-start>=limit:reason='wall_limit'
                    if reason:p.kill();break
                    time.sleep(.1)
                p.wait()
            finally:
                if p.poll()is None:p.kill();p.wait()
            row.update(exit_code=p.returncode,termination_reason=reason,stop_UTC=utc(),elapsed_s=time.monotonic()-start)
        history.append(row);write('RUN_HISTORY.json',history);write('ACTIVE_JOB.json',row);print(json.dumps({'job_finished':row}),flush=True)
        if p.returncode!=0 or reason:break
    else:status='complete'
except BaseException as ex:error=repr(ex)
receipt={'supervisor_PID':os.getpid(),'status':status,'error':error,'UTC':utc(),'elapsed_s':time.monotonic()-START,'numeric_cumulative_s':sum(r['elapsed_s']for r in history),'history':history,
 'environment':{'python':sys.executable,'host':os.environ.get('COMPUTERNAME'),'CPU_threads':4,'device':'CPU'},'SSH_calls':0,'migration_integrity_test':False,'EVAL_read':0}
write('LOCAL_NATIVE_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
raise SystemExit(0 if status=='complete'else 1)
