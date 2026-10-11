from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, time, uuid, subprocess, signal, shutil, threading

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
cfg=json.loads((ROOT/'config.json').read_text('utf8'))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['CPU_threads']])
entry=sys.argv[1]
use_gpu=entry in cfg['GPU_scientific_entries']
if entry not in cfg['allowed_scientific_entries']:raise ValueError('unregistered scientific entry')
if cfg['science_budget'] != 600 or cfg['CPU_threads'] != 12:
    raise ValueError('R66 requires preregistered600-second budget')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[k]=str(cfg['CPU_threads'])
for n in ['tmp','torch','cuda','matplotlib','joblib']:(ROOT/'cache'/n).mkdir(exist_ok=True,parents=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(ROOT/'cache/tmp'),
    TEMP=str(ROOT/'cache/tmp'),TMP=str(ROOT/'cache/tmp'),TORCH_HOME=str(ROOT/'cache/torch'),
    CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
    JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OWNED_SCIENCE_NONCE=uuid.uuid4().hex)
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
history=json.loads((ROOT/'REMOTE_RUN_HISTORY.json').read_text())if(ROOT/'REMOTE_RUN_HISTORY.json').exists()else[]
if(ROOT/(entry+'.used')).exists():raise RuntimeError('already submitted; resolve existing identity and native receipt')
meminfo={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
own=[]
base='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
for d in Path('/proc').iterdir():
    if not d.name.isdigit()or int(d.name)==os.getpid():continue
    try:
        a=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace');c=str((d/'cwd').resolve())
        if 'python' in a and(base in a or c.startswith(base)):own.append({'PID':int(d.name),'argv':a,'cwd':c})
    except(OSError,PermissionError):pass
resource={'UTC':utc(),'MemAvailable_bytes':meminfo['MemAvailable'],
          'RSS_quota_bytes':cfg['RSS_limit_bytes'],'CPU_threads':cfg['CPU_threads'],
          'disk_free_bytes':shutil.disk_usage(ROOT).free,'other_own_project_jobs':own,
          'execution_device':'cpu','GPU_admission_applied':False,
          'GPU_allocated_reserved_safety_limits_bytes':cfg['GPU_limit_bytes'],
          'compute_budget_enforced':True,'limit_s':600,'stage_output_limit_bytes':None}
if use_gpu:
    free_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
    other_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    resource.update(execution_device='cuda:0',GPU_admission_applied=True,GPU_free_bytes=int(free_gpu[0])*1048576,other_GPU_compute_jobs=other_gpu)
    if resource['GPU_free_bytes']<11811160064 or other_gpu:raise RuntimeError(('GPU admission unavailable',resource))
write(entry+'.START_RESOURCE.json',resource)
if meminfo['MemAvailable']<cfg['MemAvailable_min_bytes']:raise RuntimeError('registered machine memory admission unavailable')
if resource['disk_free_bytes']<=0:raise RuntimeError('filesystem full')
if own:raise RuntimeError(('other project job must be resolved',own))
(ROOT/(entry+'.used')).write_bytes((utc()+'\n').encode())
def memory(pid):
    vals={}
    for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):vals[line.split(':',1)[0]]=int(line.split()[1])*1024
    if 'VmRSS'not in vals:raise OSError('actual PID RSS unavailable')
    return max(vals.values())
def process_tree(pid):
    links={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:links[int(d.name)]=int((d/'stat').read_text().rsplit(')',1)[1].split()[1])
        except(OSError,ValueError,IndexError):pass
    seen={pid}
    while True:
        new={p for p,parent in links.items()if parent in seen}
        if new.issubset(seen):return seen
        seen.update(new)
def tree_rss(pid):
    total=0;members=[]
    for p0 in process_tree(pid):
        try:
            value=next(int(l.split()[1])*1024 for l in Path('/proc/'+str(p0)+'/status').read_text().splitlines()if l.startswith('VmRSS:'))
            total+=value;members.append(p0)
        except(OSError,StopIteration):
            if p0==pid:raise OSError('owned tree root RSS unavailable')
    return total,members
start=time.monotonic()
os.environ['PILOT_START_MONOTONIC']=str(start)
row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'python':sys.executable,
     'limit_s':600,'compute_budget_enforced':True,'wall_deadline_UTC':None,
     'stage_output_limit_bytes':None,'actual_worker_PID':None,'actual_worker_RSS_peak_bytes':None,
     'memory_probe_failures':0,'memory_probe_failure_events':[],'handshake_before_imports':False,
     'job_tree_RSS_peak_bytes':0,'execution_device':'cuda:0' if use_gpu else 'cpu','GPU_admission_applied':use_gpu,
     'GPU_process_memory_peak_bytes':0 if use_gpu else None,'GPU_query_failures':[],'RSS_limit_bytes':cfg['RSS_limit_bytes'],'start_resource':resource}
reason=None
with(ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'linux_owned_worker.py'),entry],cwd=ROOT,
        stdout=out,stderr=err,env=os.environ.copy(),start_new_session=True)
    row['launcher_PID']=p.pid;write('REMOTE_ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)
    deadline_fired=threading.Event()
    def hard_deadline():
        if p.poll() is None:
            deadline_fired.set()
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    watchdog=threading.Timer(max(0.,start+600-time.monotonic()),hard_deadline)
    watchdog.daemon=True;watchdog.start()
    def stop_owned():
        if p.poll()is None:
            if os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
            else:p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    last_disk=0.;last_progress=0.;last_gpu=0.
    try:
        while p.poll()is None:
            if row['actual_worker_PID']is None:
                hp=ROOT/(entry+'.SCIENCE_PID.json')
                if hp.exists():
                    try:hs=json.loads(hp.read_text())
                    except json.JSONDecodeError:hs=None
                    if hs and hs['nonce']==os.environ['OWNED_SCIENCE_NONCE']:
                        if hs['PID']!=p.pid or hs['parent_PID']!=os.getpid()or hs['cwd']!=str(ROOT):
                            raise RuntimeError('owned science identity mismatch')
                        peak=memory(p.pid);row.update(actual_worker_PID=p.pid,actual_worker_parent_PID=hs['parent_PID'],
                            actual_worker_RSS_peak_bytes=peak,handshake_before_imports=hs['before_scientific_imports'])
                        write(entry+'.RSS_ACK.json',{'actual_PID':p.pid,'nonce':hs['nonce'],
                            'actual_PID_memory_query_success':True,'supervisor_PID':os.getpid(),
                            'RSS_limit_bytes':cfg['RSS_limit_bytes'],'UTC':utc()})
                        write('REMOTE_ACTIVE_JOB.json',row);print(json.dumps({'bound_actual_science':row}),flush=True)
                if time.monotonic()-start>20 and row['actual_worker_PID']is None:
                    reason='startup_identity_handshake_failed_before_imports'
            else:
                try:
                    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],memory(p.pid))
                    total,members=tree_rss(p.pid)
                    row['job_tree_RSS_peak_bytes']=max(row['job_tree_RSS_peak_bytes'],total)
                    row['last_job_tree_PIDs']=members
                    if total>cfg['RSS_limit_bytes']:reason='machine_job_tree_RSS_limit'
                except OSError as exc:
                    if p.poll()is None:
                        row['memory_probe_failures']+=1
                        row['memory_probe_failure_events'].append({'UTC':utc(),'PID':p.pid,'poll':p.poll(),
                            'context':'actual_PID_query','error':repr(exc),'cause':'unknown'})
                        if row['memory_probe_failures']>5:reason='machine_RSS_monitor_unavailable'
            if use_gpu and row['actual_worker_PID'] is not None and time.monotonic()-last_gpu>=2:
                last_gpu=time.monotonic()
                try:
                    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                    members=process_tree(p.pid);used=0
                    for line in output.splitlines():
                        if line.strip():
                            parts=line.split(',')
                            if int(parts[0]) in members:used+=int(parts[1])*1048576
                    row['GPU_process_memory_peak_bytes']=max(row['GPU_process_memory_peak_bytes'],used)
                    if used>cfg['GPU_limit_bytes']:reason='machine_GPU_process_memory_limit'
                except BaseException as exc:
                    row['GPU_query_failures'].append({'UTC':utc(),'error':repr(exc)})
                    if len(row['GPU_query_failures'])>5:reason='machine_GPU_monitor_unavailable'
            if time.monotonic()-last_disk>20:
                last_disk=time.monotonic()
                row['last_disk_free_bytes']=shutil.disk_usage(ROOT).free
                if row['last_disk_free_bytes']<=0:reason='filesystem_full'
            if time.monotonic()-last_progress>30:
                last_progress=time.monotonic()
                print(json.dumps({'running':{'entry':entry,'actual_PID':row['actual_worker_PID'],
                    'elapsed_s':time.monotonic()-start,'job_tree_RSS_peak_bytes':row['job_tree_RSS_peak_bytes'],
                    'limit_s':600}}),flush=True)
            if reason:stop_owned();break
            time.sleep(.1)
        p.wait()
    except BaseException as exc:reason=repr(exc);stop_owned()
watchdog.cancel()
if deadline_fired.is_set():reason='hard600_second_worker_budget'
row.update(actual_worker_native_exit=p.returncode,launcher_native_exit=p.returncode,
    termination_reason=reason,elapsed_s=time.monotonic()-start,stop_UTC=utc())
wp=ROOT/(entry+'.WORKER_RECEIPT.json')
if wp.exists():
    row['worker_self_receipt']=json.loads(wp.read_text())
    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes']or 0,row['worker_self_receipt']['self_peak_RSS_bytes'])
history.append(row)
write('REMOTE_RUN_HISTORY.json',history);write(entry+'.NATIVE_RECEIPT.json',row);write('REMOTE_ACTIVE_JOB.json',row)
print(json.dumps({'finished':row}),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None and row['handshake_before_imports']else 1)
