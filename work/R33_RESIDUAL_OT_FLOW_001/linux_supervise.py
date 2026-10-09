from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,uuid,subprocess,signal
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT);cfg=json.loads((ROOT/'config.json').read_text('utf8'));entry=sys.argv[1]
if entry not in ['train_server.py']:raise ValueError('unregistered entry')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
for n in ['tmp','torch','cuda','matplotlib','joblib']:(ROOT/'cache'/n).mkdir(exist_ok=True,parents=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(ROOT/'cache/tmp'),TEMP=str(ROOT/'cache/tmp'),TMP=str(ROOT/'cache/tmp'),
    TORCH_HOME=str(ROOT/'cache/torch'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
    JOBLIB_TEMP_FOLDER=str(ROOT/'cache/joblib'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OWNED_SCIENCE_NONCE=uuid.uuid4().hex)
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
history=json.loads((ROOT/'REMOTE_RUN_HISTORY.json').read_text('utf8'))if(ROOT/'REMOTE_RUN_HISTORY.json').exists()else[]
if (ROOT/(entry+'.used')).exists():raise RuntimeError('already submitted; inspect existing identity/receipt')
remaining=cfg['science_budget']-sum(r['elapsed_s']for r in history)
limit=min(remaining,datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-300,30 if entry=='pid_probe.py' else 1850)
if limit<=0:raise TimeoutError('stage/science exhausted')
(ROOT/(entry+'.used')).write_text(utc(),'utf8')
def memory(pid):
    values={}
    for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':',1)[0]]=int(line.split()[1])*1024
    if 'VmRSS' not in values:raise OSError('actual PID RSS not available')
    return max(values.values())
start=time.monotonic();row={'entry':entry,'supervisor_PID':os.getpid(),'start_UTC':utc(),'python':sys.executable,
    'limit_s':limit,'actual_worker_PID':None,'actual_worker_RSS_peak_bytes':None,'memory_probe_failures':0,'memory_probe_failure_events':[],'handshake_before_imports':False};reason=None
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
                    if row['actual_worker_RSS_peak_bytes']>cfg['RSS_limit_bytes']:reason='actual_worker_RSS_limit'
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
