from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,uuid,subprocess,ctypes
from ctypes import wintypes
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
entry=sys.argv[1];python=CFG['CPU_python']
if entry!='render_boundary_layout.py':raise RuntimeError('unregistered display-only entry')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
for name in ['tmp','torch','matplotlib','joblib']:(ROOT/'cache'/name).mkdir(exist_ok=True,parents=True)
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(ROOT/'cache/tmp')
os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(ROOT/'cache/torch'),
    MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),
    OWNED_SCIENCE_NONCE=uuid.uuid4().hex)
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
class M(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in
      ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
       'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
class MS(ctypes.Structure):
    _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(n,ctypes.c_ulonglong)for n in
      ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
class PE(ctypes.Structure):
    _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),
      ('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),
      ('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(PE)];kernel.Process32FirstW.restype=wintypes.BOOL
kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(PE)];kernel.Process32NextW.restype=wintypes.BOOL
kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MS)];kernel.GlobalMemoryStatusEx.restype=wintypes.BOOL
gm=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
gm.argtypes=[wintypes.HANDLE,ctypes.POINTER(M),wintypes.DWORD];gm.restype=wintypes.BOOL
def memory(h):
    m=M();m.cb=ctypes.sizeof(m)
    if not gm(h,ctypes.byref(m),m.cb):raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo actual process')
    return int(m.WorkingSetSize),int(m.PeakWorkingSetSize)
def available():
    m=MS();m.dwLength=ctypes.sizeof(m)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError(ctypes.get_last_error(),'GlobalMemoryStatusEx')
    return int(m.ullAvailPhys)
def descendants(root):
    snap=kernel.CreateToolhelp32Snapshot(2,0)
    if snap in [None,ctypes.c_void_p(-1).value]:raise OSError(ctypes.get_last_error(),'process snapshot')
    pairs=[];pe=PE();pe.dwSize=ctypes.sizeof(pe)
    try:
        good=kernel.Process32FirstW(snap,ctypes.byref(pe))
        while good:
            pairs.append((int(pe.th32ProcessID),int(pe.th32ParentProcessID)))
            good=kernel.Process32NextW(snap,ctypes.byref(pe))
    finally:kernel.CloseHandle(snap)
    owned={root}
    while True:
        new={pid for pid,parent in pairs if parent in owned}
        if new<=owned:return owned
        owned|=new
def tree_memory(root,main_handle):
    total=0;ids=descendants(root)
    for pid in ids:
        h=main_handle if pid==root else kernel.OpenProcess(0x0400|0x0010,False,pid)
        if not h:raise OSError(ctypes.get_last_error(),'open owned child memory')
        try:total+=memory(h)[0]
        finally:
            if pid!=root:kernel.CloseHandle(h)
    return total,sorted(ids)
def stop_owned(pid):
    ids=descendants(pid)
    for child in sorted(ids-{pid},reverse=True):
        h=kernel.OpenProcess(1,False,child)
        if h:
            kernel.TerminateProcess(h,1);kernel.CloseHandle(h)
    h=kernel.OpenProcess(1,False,pid)
    if h:kernel.TerminateProcess(h,1);kernel.CloseHandle(h)

if (ROOT/(entry+'.used')).exists():raise RuntimeError('entry already used: inspect its state, no duplicate submission')
assert CFG['science_budget']is None and CFG['deadline_UTC']is None and CFG['output_limit_bytes']is None
expected={'check_models.py':850*1024**2,'generate_all.py':1100*1024**2,'evaluate_saved.py':1550*1024**2,
    'replay_all.py':850*1024**2,'summarize_and_plot.py':600*1024**2,'render_train.py':500*1024**2}
expected['render_boundary_layout.py']=500*1024**2
minimum=expected[entry]+400*1024**2
av=available()
admission={'PID':os.getpid(),'UTC':utc(),'entry':entry,'available_physical_bytes':av,
    'expected_stage_RSS_bytes':expected[entry],'system_headroom_bytes':400*1024**2,'minimum_available_bytes':minimum,
    'pass':av>=minimum,'CPU_threads':4,'RSS_safe_limit_bytes':CFG['RSS_limit_bytes'],
    'CPU_not_gated_on_GPU':True,'no_user_apps_or_system_settings_changed':True}
write(entry+'.RESOURCE_ADMISSION.json',admission)
if not admission['pass']:
    print(json.dumps({'resource_wait':admission}),flush=True);raise SystemExit(2)
history=json.loads((ROOT/'RUN_HISTORY.json').read_text('utf8'))if(ROOT/'RUN_HISTORY.json').exists()else[]
(ROOT/(entry+'.used')).write_bytes(utc().encode())
start=time.monotonic();handle=None;reason=None
row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'python':python,'science_budget_s':None,
    'wall_budget_s':None,'deadline_UTC':None,'compute_budget_enforced':False,'actual_worker_PID':None,
    'actual_worker_RSS_peak_bytes':None,'job_tree_RSS_peak_bytes':None,'launcher_RSS_peak_bytes':0,
    'memory_probe_failures':0,'memory_probe_failure_events':[],'handshake_before_imports':False,
    'minimum_available_during_job_bytes':av,'resource_admission':admission,'remote_previous_replay_terminal':'unknown',
    'global_single_physical_job_confirmed':False}
with(ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
    p=subprocess.Popen([python,'-B',str(ROOT/'owned_worker.py'),entry],cwd=ROOT,stdout=out,stderr=err,
        env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row['launcher_PID']=p.pid;write('ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)
    try:
        while p.poll()is None:
            if handle is None:
                hp=ROOT/(entry+'.SCIENCE_PID.json')
                if hp.exists():
                    try:hs=json.loads(hp.read_text('utf8'))
                    except json.JSONDecodeError:hs=None
                    if hs and hs['nonce']==os.environ['OWNED_SCIENCE_NONCE']:
                        if hs['parent_PID']!=p.pid and hs['PID']!=p.pid:raise RuntimeError('worker ownership differs from launcher')
                        handle=kernel.OpenProcess(0x0400|0x0010|0x0001|0x00100000,False,hs['PID'])
                        if not handle:raise OSError(ctypes.get_last_error(),'OpenProcess actual PID')
                        current,peak=memory(handle)
                        row.update(actual_worker_PID=hs['PID'],actual_worker_parent_PID=hs['parent_PID'],
                            actual_worker_RSS_peak_bytes=peak,job_tree_RSS_peak_bytes=current,
                            nonce=hs['nonce'],handshake_before_imports=hs['before_scientific_imports'])
                        write(entry+'.RSS_ACK.json',{'actual_PID':hs['PID'],'nonce':hs['nonce'],
                            'actual_PID_memory_query_success':True,'supervisor_PID':os.getpid(),'RSS_limit_bytes':CFG['RSS_limit_bytes'],'UTC':utc()})
                        write('ACTIVE_JOB.json',row);print(json.dumps({'bound_actual_science':row}),flush=True)
                if time.monotonic()-start>20 and handle is None:reason='actual_science_PID_handshake_unavailable'
            else:
                try:
                    current,peak=memory(handle);tree,ids=tree_memory(row['actual_worker_PID'],handle)
                    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],peak)
                    row['job_tree_RSS_peak_bytes']=max(row['job_tree_RSS_peak_bytes'],tree)
                    row['last_owned_tree_PIDs']=ids
                    if max(peak,tree)>CFG['RSS_limit_bytes']:reason='actual_worker_or_owned_tree_RSS_limit'
                except OSError as e:
                    row['memory_probe_failures']+=1;row['memory_probe_failure_events'].append({'UTC':utc(),'reason':repr(e),'cause':'unknown'})
                    if row['memory_probe_failures']>10:reason='actual_science_memory_query_unavailable'
            row['minimum_available_during_job_bytes']=min(row['minimum_available_during_job_bytes'],available())
            row['launcher_RSS_peak_bytes']=max(row['launcher_RSS_peak_bytes'],memory(wintypes.HANDLE(int(p._handle)))[1])
            if reason:
                stop_owned(row['actual_worker_PID']or p.pid);p.kill();break
            time.sleep(.25)
        p.wait()
        if handle:
            try:row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],memory(handle)[1])
            except OSError as e:row['memory_probe_failure_events'].append({'UTC':utc(),'reason':repr(e),'cause':'unknown','at':'final_memory_query'})
            code=wintypes.DWORD();row['actual_worker_native_exit']=int(code.value)if kernel.GetExitCodeProcess(handle,ctypes.byref(code))else None
    except BaseException as e:
        reason=repr(e)
        if p.poll()is None:stop_owned(row['actual_worker_PID']or p.pid);p.kill()
        p.wait()
    finally:
        if handle:kernel.CloseHandle(handle)
row.update(launcher_native_exit=p.returncode,termination_reason=reason,elapsed_s=time.monotonic()-start,stop_UTC=utc())
wp=ROOT/(entry+'.WORKER_RECEIPT.json')
if wp.exists():row['worker_self_receipt']=json.loads(wp.read_text('utf8'))
history.append(row);write('RUN_HISTORY.json',history);write(entry+'.NATIVE_RECEIPT.json',row);write('ACTIVE_JOB.json',row)
print(json.dumps({'finished':row}),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None and row['actual_worker_PID']is not None else 1)
