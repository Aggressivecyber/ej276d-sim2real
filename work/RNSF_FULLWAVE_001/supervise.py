"""One-shot native process receipts and cumulative budget; trial-local only."""
from __future__ import annotations
import pathlib, os, sys, json, time, datetime, signal, subprocess, fcntl
ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())

def write(path,obj):
    p=ROOT/path;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def other_jobs():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
            cwd=str((p/'cwd').resolve())
            if p.stat().st_uid!=os.getuid() or str(ROOT) in cmd or cwd==str(ROOT):continue
            if (str(ROOT.parent) in cmd or cwd.startswith(str(ROOT.parent)+'/')) and any(x in cmd for x in ['run.py','numeric_worker.py','supervise.py']):
                if 'python' in cmd:found.append({'PID':int(p.name),'command':cmd})
        except (FileNotFoundError,PermissionError,ProcessLookupError,OSError):pass
    return found

def main():
    mode=sys.argv[1]
    assert mode in ['smoke','train','evaluate','replay']
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:CFG['threads']])
    lock=(ROOT.parent/'.rnfs-science.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=other_jobs()
    if jobs:
        write('LAUNCH_DEFERRED.json',{'UTC':now(),'other_project_jobs':jobs,'submitted':False})
        raise RuntimeError('another project scientific job is active')
    ledgerfile=ROOT/'NATIVE_LEDGER.json'
    ledger=json.loads(ledgerfile.read_text()) if ledgerfile.exists() else []
    if any(r['mode']==mode and r['native_exit']==0 for r in ledger):
        raise RuntimeError('completed mode cannot be repeated')
    active=ROOT/'ACTIVE_JOB.json'
    if active.exists():
        previous=json.loads(active.read_text())
        if previous.get('status')=='running':raise RuntimeError('previous job state unresolved; do not resubmit')
    spent=sum(r['wall_s'] for r in ledger)
    remaining=CFG['numeric_wall_limit_s']-spent
    if remaining<30:raise RuntimeError('cumulative numerical budget exhausted')
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
    env.update(PYTHONPATH=str(ROOT/'deps'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    (ROOT/'tmp').mkdir(exist_ok=True)
    attempt=len(ledger)+1
    with (ROOT/f'{attempt:02d}_{mode}.stdout.log').open('w') as out,(ROOT/f'{attempt:02d}_{mode}.stderr.log').open('w') as err:
        started=time.monotonic();utc=now()
        proc=subprocess.Popen([CFG['python'],'-B','run.py',mode],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        record={'mode':mode,'attempt':attempt,'PID':proc.pid,'supervisor_PID':os.getpid(),'start_UTC':utc,
                'status':'running','cumulative_before_s':spent,'remaining_s':remaining,'root':str(ROOT)}
        write('ACTIVE_JOB.json',record)
        print(json.dumps(record),flush=True)
        stopped=None;peak=0;check_other=time.monotonic()
        while proc.poll() is None:
            time.sleep(1)
            try:
                status=pathlib.Path(f'/proc/{proc.pid}/status').read_text()
                rss=int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                peak=max(peak,rss)
                if rss>CFG['rss_limit_bytes']:stopped='RSS_limit'
            except (FileNotFoundError,StopIteration):pass
            if time.monotonic()-started>remaining:stopped='cumulative_wall_limit'
            if time.monotonic()-check_other>30:
                jobs=other_jobs();check_other=time.monotonic()
                if jobs:stopped='other_project_job_appeared'
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and 'deps' not in p.relative_to(ROOT).parts and 'tmp' not in p.relative_to(ROOT).parts)
                if size>CFG['output_limit_bytes']:stopped='output_limit'
            if stopped:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                break
        code=proc.wait();elapsed=time.monotonic()-started
    record.update(status='complete' if code==0 else 'failed',native_exit=code,stop_UTC=now(),wall_s=elapsed,
                  stop_reason=stopped,peak_RSS_polled_bytes=peak,cumulative_after_s=spent+elapsed)
    ledger.append(record);write('NATIVE_LEDGER.json',ledger);write('ACTIVE_JOB.json',record)
    print(json.dumps(record),flush=True)
    sys.exit(0 if code==0 else 1)

if __name__=='__main__':main()
