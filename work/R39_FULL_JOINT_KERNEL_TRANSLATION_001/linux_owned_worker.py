from pathlib import Path
from datetime import datetime, timezone
import os,sys,json,time,runpy,resource
ROOT=Path(__file__).resolve().parent;entry=sys.argv[1];nonce=os.environ['OWNED_SCIENCE_NONCE'];start=time.monotonic()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
write(entry+'.SCIENCE_PID.json',{'PID':os.getpid(),'parent_PID':os.getppid(),'nonce':nonce,'entry':entry,
    'UTC':datetime.now(timezone.utc).isoformat(),'cwd':str(ROOT),'argv':sys.argv,'before_scientific_imports':True,'python':sys.executable})
while True:
    p=ROOT/(entry+'.RSS_ACK.json')
    if p.exists():
        ack=json.loads(p.read_text('utf8'))
        if ack['nonce']==nonce and ack['actual_PID']==os.getpid() and ack['actual_PID_memory_query_success']:break
    if time.monotonic()-start>15:raise RuntimeError('actual Linux science PID was not acknowledged')
    time.sleep(.03)
exit_code=0;error=None
try:runpy.run_path(str(ROOT/entry),run_name='__main__')
except SystemExit as e:
    exit_code=e.code if isinstance(e.code,int)else(0 if e.code is None else 1)
    raise
except BaseException as e:exit_code=1;error=repr(e);raise
finally:
    write(entry+'.WORKER_RECEIPT.json',{'PID':os.getpid(),'parent_PID':os.getppid(),'nonce':nonce,'entry':entry,
        'native_exit':exit_code,'error':error,'elapsed_s':time.monotonic()-start,
        'self_peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        'supervisor_bound_real_PID':True,'UTC':datetime.now(timezone.utc).isoformat()})
