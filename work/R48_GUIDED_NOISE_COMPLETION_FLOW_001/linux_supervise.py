from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,uuid,subprocess,signal
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT);cfg=json.loads((ROOT/'config.json').read_text('utf8'));entry=sys.argv[1]
if entry not in ['calibrate.py','train_control.py','train_completion.py','check_models.py','generate_all.py','evaluate_saved.py','replay_all.py']:raise ValueError('unregistered entry')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]=str(cfg['CPU_threads'])
for n in ['tmp','torch','cuda','matplotlib','joblib']:(ROOT/'cache'/n).mkdir(exist_ok=True,parents=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(ROOT/'cache/tmp'),TEMP=str(ROOT/'cache/tmp'),TMP=str(ROOT/'cache/tmp'),
    TORCH_HOME=str(ROOT/'cache/torch'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
    JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OWNED_SCIENCE_NONCE=uuid.uuid4().hex)
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
history=json.loads((ROOT/'REMOTE_RUN_HISTORY.json').read_text('utf8'))if(ROOT/'REMOTE_RUN_HISTORY.json').exists()else[]
if (ROOT/(entry+'.used')).exists():raise RuntimeError('already submitted; inspect existing identity/receipt')
remaining=cfg['science_budget']-sum(r['elapsed_s']for r in history)
training_used=sum(r['elapsed_s']for r in history if r['entry'].startswith('train_'))
entry_limit=min(cfg['training_wall_s'],cfg['training_total_limit_s']-training_used) if entry.startswith('train_')else remaining
limit=min(remaining,entry_limit,datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-600)
if limit<=0:raise TimeoutError('stage/science exhausted')
meminfo={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
gpuinfo=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
otherGPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
startResource={'UTC':utc(),'MemAvailable_bytes':meminfo['MemAvailable'],'GPU_free_bytes':int(gpuinfo)*1048576,'other_GPU_compute_jobs':otherGPU,'RSS_quota_bytes':cfg['RSS_limit_bytes'],'GPU_quota_bytes':cfg['GPU_limit_bytes'],'CPU_threads':cfg['CPU_threads']}
write('SERVER_START_RESOURCE.json',startResource)
if meminfo['MemAvailable']<60129542144:raise RuntimeError('R48 startup MemAvailable below registered56GiB; no implicit lower quota')
if entry.startswith('train_')and(int(gpuinfo)*1048576<11811160064 or otherGPU):raise RuntimeError('GPU admission not satisfied; no science submitted')
import shutil
if shutil.disk_usage(ROOT).free<cfg['disk_free_reserve_bytes']:raise RuntimeError('disk reserve admission failed')
own=[]
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name)==os.getpid():continue
    try:
        a=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace');c=str((d/'cwd').resolve())
        if 'python' in a and ('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305' in a or c.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305')):own.append({'PID':int(d.name),'argv':a,'cwd':c})
    except(OSError,PermissionError):pass
startResource['other_own_project_jobs']=own;write('SERVER_START_RESOURCE.json',startResource)
if own:raise RuntimeError(('other project science status must be resolved',own))
(ROOT/(entry+'.used')).write_text(utc(),'utf8')
def memory(pid):
    values={}
    for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':',1)[0]]=int(line.split()[1])*1024
    if 'VmRSS' not in values:raise OSError('actual PID RSS not available')
    return max(values.values())
def process_tree(pid):
    links={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            fields=(d/'stat').read_text().rsplit(')',1)[1].split();links[int(d.name)]=int(fields[1])
        except(OSError,ValueError,IndexError):pass
    seen={pid};changed=True
    while changed:
        new={p for p,parent in links.items()if parent in seen};changed=not new.issubset(seen);seen.update(new)
    return seen
def tree_rss(pid):
    total=0;members=[]
    for p0 in process_tree(pid):
        try:
            lines=Path('/proc/'+str(p0)+'/status').read_text().splitlines()
            value=next(int(l.split()[1])*1024 for l in lines if l.startswith('VmRSS:'))
            total+=value;members.append(p0)
        except(OSError,StopIteration):
            if p0==pid:raise
    return total,members
last_GPU_query=0.;last_progress_query=0.;last_progress_step=None;last_disk_query=0.
def gpu_used(pid):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
    members=process_tree(pid);total=0
    for l in text.splitlines():
        if not l.strip():continue
        parts=l.split(',')
        if int(parts[0])in members:total+=int(parts[1])*1048576
    return total

start=time.monotonic();row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'python':sys.executable,
    'limit_s':limit,'actual_worker_PID':None,'actual_worker_RSS_peak_bytes':None,'memory_probe_failures':0,'memory_probe_failure_events':[],'handshake_before_imports':False,'job_tree_RSS_peak_bytes':0,'GPU_process_memory_peak_bytes':0,'GPU_query_failures':[]};reason=None
with (ROOT/(entry+'.stdout.log')).open('wb')as out,(ROOT/(entry+'.stderr.log')).open('wb')as err:
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'linux_owned_worker.py'),entry],cwd=ROOT,stdout=out,stderr=err,
        env=os.environ.copy(),start_new_session=True)
    row['launcher_PID']=p.pid;write('REMOTE_ACTIVE_JOB.json',row);print(json.dumps({'started':row}),flush=True)
    def stop_owned():
        if p.poll()is None:
            if os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
            else:p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait()
    try:
        while p.poll()is None:
            if time.monotonic()-last_progress_query>=30:
                last_progress_query=time.monotonic()
                progress=ROOT/'TRAIN_PROGRESS.json'
                if progress.exists():
                    try:
                        item=json.loads(progress.read_text())
                        if (item.get('arm'),item.get('global_step'))!=last_progress_step:
                            last_progress_step=(item.get('arm'),item.get('global_step'));print(json.dumps({'actual_training_progress':item}),flush=True)
                    except(json.JSONDecodeError,OSError):pass
            if time.monotonic()-last_disk_query>=20:
                last_disk_query=time.monotonic()
                if shutil.disk_usage(ROOT).free<cfg['disk_free_reserve_bytes']:reason='disk_free_reserve_limit'
                if sum(q.stat().st_size for q in ROOT.rglob('*')if q.is_file())>cfg['output_limit_bytes']:reason='stage_output_limit'
            if row['actual_worker_PID']is None:
                hp=ROOT/(entry+'.SCIENCE_PID.json')
                if hp.exists():
                    try:hs=json.loads(hp.read_text('utf8'))
                    except json.JSONDecodeError:hs=None
                    if hs and hs['nonce']==os.environ['OWNED_SCIENCE_NONCE']:
                        if hs['PID']!=p.pid or hs['parent_PID']!=os.getpid()or hs['cwd']!=str(ROOT):raise RuntimeError('owned actual science identity mismatch')
                        peak=memory(hs['PID']);row.update(actual_worker_PID=hs['PID'],actual_worker_parent_PID=hs['parent_PID'],
                            actual_worker_RSS_peak_bytes=peak,handshake_before_imports=hs['before_scientific_imports'])
                        write(entry+'.RSS_ACK.json',{'actual_PID':p.pid,'nonce':hs['nonce'],'actual_PID_memory_query_success':True,
                            'supervisor_PID':os.getpid(),'RSS_limit_bytes':cfg['RSS_limit_bytes'],'UTC':utc()})
                        write('REMOTE_ACTIVE_JOB.json',row);print(json.dumps({'bound_actual_science':row}),flush=True)
                if time.monotonic()-start>20 and row['actual_worker_PID']is None:reason='handshake_unavailable'
            else:
                try:
                    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes'],memory(p.pid))
                    tree,treeids=tree_rss(p.pid);row['job_tree_RSS_peak_bytes']=max(row['job_tree_RSS_peak_bytes'],tree);row['last_job_tree_PIDs']=treeids
                    if tree>cfg['RSS_limit_bytes']:reason='job_tree_RSS_limit'
                    if time.monotonic()-last_GPU_query>=2:
                        last_GPU_query=time.monotonic()
                        try:
                            used=gpu_used(p.pid);row['GPU_process_memory_peak_bytes']=max(row['GPU_process_memory_peak_bytes'],used)
                            if used>cfg['GPU_limit_bytes']:reason='GPU_process_memory_limit'
                        except BaseException as e:row['GPU_query_failures'].append({'UTC':utc(),'error':repr(e)})
                except OSError:
                    if p.poll()is None:
                        row['memory_probe_failures']+=1;row['memory_probe_failure_events'].append({'UTC':utc(),'PID':p.pid,'poll':p.poll(),'context':'actual_PID_query'})
                        if row['memory_probe_failures']>5:reason='actual_worker_RSS_unavailable'
            if time.monotonic()-start>limit:reason='science_or_stagewall_limit'
            if reason:stop_owned();break
            time.sleep(.1)
        p.wait()
    except BaseException as e:reason=repr(e);stop_owned()
row.update(actual_worker_native_exit=p.returncode,launcher_native_exit=p.returncode,termination_reason=reason,
    elapsed_s=time.monotonic()-start,stop_UTC=utc())
wp=ROOT/(entry+'.WORKER_RECEIPT.json')
if wp.exists():
    row['worker_self_receipt']=json.loads(wp.read_text('utf8'))
    row['actual_worker_RSS_peak_bytes']=max(row['actual_worker_RSS_peak_bytes']or 0,row['worker_self_receipt']['self_peak_RSS_bytes'])
history.append(row);write('REMOTE_RUN_HISTORY.json',history);write(entry+'.NATIVE_RECEIPT.json',row);write('REMOTE_ACTIVE_JOB.json',row)
print(json.dumps({'finished':row}),flush=True)
raise SystemExit(0 if p.returncode==0 and reason is None and row['handshake_before_imports']else 1)
