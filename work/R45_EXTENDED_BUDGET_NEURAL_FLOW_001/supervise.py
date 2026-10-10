from pathlib import Path
import os,sys,json,time,datetime,uuid,subprocess,ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'));cfg['RSS_limit_bytes']=4294967296;entry=sys.argv[1];python=sys.argv[2]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
for n in ['tmp','torch','cuda','matplotlib','joblib']:(ROOT/'cache'/n).mkdir(parents=True,exist_ok=True)
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(ROOT/'cache/tmp')
os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(ROOT/'cache/torch'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OWNED_SCIENCE_NONCE=uuid.uuid4().hex)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
history=json.loads((ROOT/'RUN_HISTORY.json').read_text('utf8'))if(ROOT/'RUN_HISTORY.json').exists()else[]
if (ROOT/(entry+'.used')).exists():raise RuntimeError('entry already used; inspect state, do not duplicate')
(ROOT/(entry+'.used')).write_text(utc(),'utf8')
limit=min(cfg['science_budget']-sum(r['elapsed_s']for r in history)-sum(r['elapsed_s']for r in json.loads((ROOT/'REMOTE_RUN_HISTORY.json').read_text('utf8'))),datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-120)
if limit<=0:raise TimeoutError('scientific or stagewall exhausted')
class M(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE;kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;gm.argtypes=[wintypes.HANDLE,ctypes.POINTER(M),wintypes.DWORD];gm.restype=wintypes.BOOL
def memory(handle):
    m=M();m.cb=ctypes.sizeof(m)
    if not gm(handle,ctypes.byref(m),m.cb):raise OSError(ctypes.get_last_error(),'real science memory query')
    return max(int(m.WorkingSetSize),int(m.PeakWorkingSetSize))
row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'python':python,'limit_s':limit,'actual_worker_PID':None,'actual_worker_RSS_peak_bytes':None,'launcher_RSS_peak_bytes':0,'memory_probe_failures':0,'memory_probe_failure_events':[],'handshake_before_imports':False};start=time.monotonic();handle=None;reason=None
with (ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
    p=subprocess.Popen([python,'-B',str(ROOT/'owned_worker.py'),entry],cwd=ROOT,stdout=out,stderr=err,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row['launcher_PID']=p.pid;write('ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)
    try:
        while p.poll()is None:
            if handle is None:
                hp=ROOT/(entry+'.SCIENCE_PID.json')
                if hp.exists():
                    try:hs=json.loads(hp.read_text('utf8'))
                    except json.JSONDecodeError:hs=None
                    if hs and hs['nonce']==os.environ['OWNED_SCIENCE_NONCE']:
                        if hs['parent_PID']not in [p.pid,os.getpid()]:raise RuntimeError('scientific child is not owned launcher descendant')
                        handle=kernel.OpenProcess(0x0400|0x0010|0x0001|0x00100000,False,hs['PID'])
                        if not handle:raise OSError(ctypes.get_last_error(),'OpenProcess actualPID')
                        peak=memory(handle);row.update(actual_worker_PID=hs['PID'],actual_worker_parent_PID=hs['parent_PID'],actual_worker_RSS_peak_bytes=peak,handshake_before_imports=hs['before_scientific_imports'])
                        write(entry+'.RSS_ACK.json',{'actual_PID':hs['PID'],'nonce':hs['nonce'],'actual_PID_memory_query_success':True,'supervisor_PID':os.getpid(),'RSS_limit_bytes':cfg['RSS_limit_bytes'],'UTC':utc()});write('ACTIVE_JOB.json',row);print(json.dumps({'bound_actual_science':row}),flush=True)
                if time.monotonic()-start>20 and handle is None:reason='actual_science_PID_handshake_unavailable'
            else:
                try:
                    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],memory(handle))
                    if row['actual_worker_RSS_peak_bytes']>cfg['RSS_limit_bytes']:reason='actual_worker_RSS_limit'
                except OSError:
                    row['memory_probe_failures']+=1;row['memory_probe_failure_events'].append({'UTC':utc(),'PID':row['actual_worker_PID'],'launcher_poll':p.poll(),'context':'active_actual_PID_query'})
                    if row['memory_probe_failures']>10:reason='actual_worker_memory_query_unavailable'
            row['launcher_RSS_peak_bytes']=max(row['launcher_RSS_peak_bytes'],memory(wintypes.HANDLE(int(p._handle))))
            if time.monotonic()-start>limit:reason='science_or_stagewall_limit'
            if reason:
                if handle:kernel.TerminateProcess(handle,1)
                p.kill();break
            time.sleep(.1)
        p.wait()
        if handle:
            try:row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],memory(handle))
            except OSError:pass
            code=wintypes.DWORD();row['actual_worker_native_exit']=int(code.value)if kernel.GetExitCodeProcess(handle,ctypes.byref(code))else None
    except BaseException as e:
        reason=repr(e)
        if handle:kernel.TerminateProcess(handle,1)
        if p.poll()is None:p.kill()
        p.wait()
    finally:
        if p.poll()is None:
            if handle:kernel.TerminateProcess(handle,1)
            p.kill();p.wait()
        if handle:kernel.CloseHandle(handle)
row.update(launcher_native_exit=p.returncode,termination_reason=reason,elapsed_s=time.monotonic()-start,stop_UTC=utc())
wp=ROOT/(entry+'.WORKER_RECEIPT.json')
if wp.exists():row['worker_self_receipt']=json.loads(wp.read_text('utf8'))
history.append(row);write('RUN_HISTORY.json',history);write(entry+'.NATIVE_RECEIPT.json',row);write('ACTIVE_JOB.json',row);print(json.dumps({'finished':row}),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None and row['actual_worker_PID']is not None else 1)
