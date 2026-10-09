from pathlib import Path
from datetime import datetime, timezone
import json,sys,os,subprocess,time
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'));entry=sys.argv[1]
if entry not in ['train_server.py','train_recover.py']:raise ValueError('entry')
token=ROOT/(entry+'.SSH_SUBMITTED')
if token.exists():raise RuntimeError('already submitted; inspect original job, never duplicate unknown')
token.write_text(datetime.now(timezone.utc).isoformat(),'utf8')
command=[
    'C:/Windows/System32/OpenSSH/ssh.exe','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'rylan@100.82.231.102',cfg['server_python']+' -B '+cfg['remote_root']+'/linux_supervise.py '+entry]
start=time.monotonic();record={'local_launch_PID':os.getpid(),'entry':entry,'UTC':datetime.now(timezone.utc).isoformat(),'command':command}
with (ROOT/(entry+'.SSH.stderr.log')).open('wb')as err,(ROOT/(entry+'.SSH.stdout.log')).open('w',encoding='utf8')as out:
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    record['SSH_PID']=p.pid;(ROOT/(entry+'.LOCAL_ACTIVE.json')).write_text(json.dumps(record,indent=2)+'\n','utf8')
    print(json.dumps(record),flush=True)
    for line in p.stdout:
        out.write(line);out.flush();print(line.rstrip(),flush=True)
        try:
            obj=json.loads(line)
            if 'bound_actual_science' in obj:
                record['remote_bound_actual_science']=obj['bound_actual_science']
                (ROOT/(entry+'.LOCAL_ACTIVE.json')).write_text(json.dumps(record,indent=2)+'\n','utf8')
        except json.JSONDecodeError:pass
    p.wait();record.update(SSH_native_exit=p.returncode,elapsed_ssh_s=time.monotonic()-start,stop_UTC=datetime.now(timezone.utc).isoformat())
(ROOT/(entry+'.LOCAL_NATIVE_RECEIPT.json')).write_text(json.dumps(record,indent=2)+'\n','utf8')
print(json.dumps({'local_finished':record}),flush=True)
raise SystemExit(p.returncode)
