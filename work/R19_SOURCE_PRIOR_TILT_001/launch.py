import os,pathlib,json,subprocess,threading,time,datetime,sys
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());start=time.monotonic();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
argv=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',cfg['execution']['location'],'cd '+cfg['execution']['remote_root']+' && '+cfg['execution']['python']+' -B remote_supervise.py']
p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));row={'local_bridge_PID':os.getpid(),'SSH_PID':p.pid,'start_UTC':utc(),'argv':argv,'exit_code':None,'unknown_remote_state':False}
(ROOT/'LOCAL_LAUNCH_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
def pump(stream,name):
    with (ROOT/name).open('wb') as f:
        for line in iter(stream.readline,b''):f.write(line);f.flush();print(line.decode(errors='replace').rstrip(),flush=True)
threads=[threading.Thread(target=pump,args=(p.stdout,'SSH.stdout.log')),threading.Thread(target=pump,args=(p.stderr,'SSH.stderr.log'))]
for t in threads:t.start()
try:p.wait(timeout=min(2500,datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()-time.time()-60))
except subprocess.TimeoutExpired:
    row['unknown_remote_state']=True;p.kill();p.wait()
for t in threads:t.join(timeout=3)
row.update({'exit_code':p.returncode,'stop_UTC':utc(),'elapsed_s':time.monotonic()-start});(ROOT/'LOCAL_NATIVE_RECEIPT.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
raise SystemExit(0 if p.returncode==0 else 1)
