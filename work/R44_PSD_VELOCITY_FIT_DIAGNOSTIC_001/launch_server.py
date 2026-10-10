from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,time
r=Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text('utf8'));start=time.monotonic()
def utc():return datetime.now(timezone.utc).isoformat()
def put(n,v):(r/n).write_text(json.dumps(v,indent=2)+'\n','utf8')
args=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',f"cd {c['remote_root']} && exec {c['server_python']} -B linux_supervise.py diagnose_frozen.py"]
rec={'local_PID':os.getpid(),'start_UTC':utc(),'remote_root':c['remote_root'],'argv':args,'scientific_recipe':'unique frozen R43 path forward; no updates or ODE'}
with(r/'SSH.stdout.log').open('wb')as out,(r/'SSH.stderr.log').open('wb')as err:
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));rec['SSH_PID']=p.pid;put('LOCAL_ACTIVE_JOB.json',rec);print(json.dumps({'local_started':rec}),flush=True)
    for line in p.stdout:
        out.write(line);out.flush();print(line.decode('utf8',errors='replace').rstrip(),flush=True)
        try:
            item=json.loads(line)
            if 'bound_actual_science'in item:rec['actual_remote_handshake']=item['bound_actual_science'];put('LOCAL_ACTIVE_JOB.json',rec)
        except(json.JSONDecodeError,UnicodeDecodeError):pass
    p.wait()
rec.update(SSH_native_exit=p.returncode,stop_UTC=utc(),local_transport_wall_s=time.monotonic()-start);put('LOCAL_NATIVE_RECEIPT.json',rec);put('LOCAL_ACTIVE_JOB.json',rec);print(json.dumps({'local_finished':rec}),flush=True)
raise SystemExit(p.returncode)
