from pathlib import Path
from datetime import datetime, timezone
import json,os,subprocess,time,sys
R=Path(__file__).resolve().parent;cfg=json.loads((R/'config.json').read_text());entry=sys.argv[1]
assert entry in ['calibrate.py','train_control.py','train_completion.py','check_models.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_train.py']
def utc():return datetime.now(timezone.utc).isoformat()
def put(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
start=time.monotonic()
args=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',
      f"cd {cfg['remote_root']} && exec {cfg['server_python']} -B linux_supervise_post.py {entry}"]
record={'entry':entry,'local_PID':os.getpid(),'start_UTC':utc(),'remote_root':cfg['remote_root'],'argv':args}
with(R/(entry+'.SSH.stdout.log')).open('wb')as out,(R/(entry+'.SSH.stderr.log')).open('wb')as err:
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    record['SSH_PID']=p.pid;put(entry+'.LOCAL_ACTIVE_JOB.json',record);print(json.dumps({'local_started':record}),flush=True)
    for line in p.stdout:
        out.write(line);out.flush();print(line.decode('utf8','replace').rstrip(),flush=True)
        try:
            item=json.loads(line)
            if 'bound_actual_science'in item:record['actual_remote_handshake']=item['bound_actual_science'];put(entry+'.LOCAL_ACTIVE_JOB.json',record)
        except json.JSONDecodeError:pass
    p.wait()
record.update(SSH_native_exit=p.returncode,stop_UTC=utc(),transport_wall_s=time.monotonic()-start)
put(entry+'.LOCAL_NATIVE_RECEIPT.json',record);put(entry+'.LOCAL_ACTIVE_JOB.json',record)
print(json.dumps({'local_finished':record}),flush=True)
raise SystemExit(p.returncode)
