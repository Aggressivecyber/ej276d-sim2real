from __future__ import annotations
import os,sys,pathlib,json,time,datetime,subprocess,signal,tarfile
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());END=datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-90;START=time.monotonic();HISTORY=[]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1'
for k in ['TMP','TEMP','TMPDIR']:os.environ[k]=str(ROOT/'cache/tmp')
for k in ['tmp','matplotlib','joblib']:(ROOT/'cache'/k).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def launch(entry):
    left=min(cfg['numeric_wall_limit_s']-sum(r['elapsed_s'] for r in HISTORY),END-time.time(),cfg['numeric_wall_limit_s'])
    if left<=0:raise TimeoutError('R18 remaining numeric/stage budget exhausted')
    start=time.monotonic();out=(ROOT/(entry+'.stdout.log')).open('wb');err=(ROOT/(entry+'.stderr.log')).open('wb')
    child=subprocess.Popen([sys.executable,'-B',str(ROOT/entry)],cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True)
    row={'entry':entry,'supervisor_PID':os.getpid(),'worker_PID':child.pid,'start_UTC':utc(),'allowed_s':left,'RSS_peak_bytes':0,'RSS_probe_failures':0,'exit_code':None,'termination_reason':None}
    emit('ACTIVE_JOB.json',row);print(json.dumps({'job_started':row}),flush=True)
    try:
        while child.poll() is None:
            try:
                text=pathlib.Path('/proc/'+str(child.pid)+'/status').read_text();rss={x.split(':')[0]:int(x.split()[1])*1024 for x in text.splitlines() if x.startswith(('VmRSS:','VmHWM:'))};row['RSS_peak_bytes']=max(row['RSS_peak_bytes'],rss.get('VmRSS',0),rss.get('VmHWM',0))
                if row['RSS_peak_bytes']>cfg['RSS_limit_bytes']:row['termination_reason']='RSS_limit'
            except (FileNotFoundError,ProcessLookupError):pass
            except BaseException:row['RSS_probe_failures']+=1
            if time.monotonic()-start>=left:row['termination_reason']='wall_limit'
            if row['termination_reason']:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
                break
            time.sleep(.2)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        out.close();err.close();row.update(exit_code=child.returncode,stop_UTC=utc(),elapsed_s=time.monotonic()-start);HISTORY.append(row);emit('RUN_HISTORY.json',HISTORY);emit('ACTIVE_JOB.json',row);print(json.dumps({'job_finished':row}),flush=True)
    return child.returncode
status='failed';error=None
try:
    if (ROOT/'SCIENCE_LAUNCH_TOKEN.json').exists():raise RuntimeError('R18 token consumed; no duplicate submission without resolving original status')
    emit('SCIENCE_LAUNCH_TOKEN.json',{'PID':os.getpid(),'UTC':utc(),'unique':True})
    for entry in ['numeric_worker.py','replay.py']:
        if launch(entry)!=0:break
    else:status='complete'
except BaseException as ex:error=repr(ex)
receipt={'supervisor_PID':os.getpid(),'status':status,'error':error,'UTC':utc(),'elapsed_s':time.monotonic()-START,'numeric_cumulative_s':sum(r['elapsed_s'] for r in HISTORY),'history':HISTORY,'environment':{'python':sys.executable,'host':os.uname().nodename,'CPU_threads':4},'migration_integrity_test':False,'EVAL_read':0}
emit('REMOTE_NATIVE_RECEIPT.json',receipt)
selected=[f for f in ROOT.rglob('*') if f.is_file() and 'cache' not in f.relative_to(ROOT).parts and f.name not in ['UPLOAD.tar','RETURN.tar','INPUT_BINDINGS.json','VALLEY_DEFINITION.json'] and f.suffix!='.py' and f.name not in ['config.json','PROTOCOL.md','READINESS.json','START_RECEIPT.json','new_original_index.bin','new_groups.bin','candidate_original_index.bin','SELECTION.json']]
size=sum(f.stat().st_size for f in selected)
if 2*size+8*1024*1024>cfg['output_limit_bytes']:raise RuntimeError('R18 output cap')
emit('REMOTE_OUTPUT_MANIFEST.json',{'files':[{'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size} for f in selected],'total_bytes':size,'hashes_not_migration_checked':True})
with tarfile.open(ROOT/'RETURN.tar','w') as t:
    for f in selected:t.add(f,arcname=f.relative_to(ROOT).as_posix(),recursive=False)
    t.add(ROOT/'REMOTE_OUTPUT_MANIFEST.json',arcname='REMOTE_OUTPUT_MANIFEST.json',recursive=False)
print(json.dumps({'remote_complete':receipt,'return_tar_bytes':(ROOT/'RETURN.tar').stat().st_size}),flush=True)
raise SystemExit(0 if status=='complete' else 1)
