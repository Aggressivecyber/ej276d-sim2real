from __future__ import annotations
import pathlib,sys,json,hashlib,datetime,time,subprocess,threading,shlex,os
ROOT=pathlib.Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text());ready=json.loads((ROOT/'READINESS.json').read_text())
for s in ready['execution_sources']:
    b=pathlib.Path(s['path']).read_bytes()
    if len(b)!=s['bytes'] or hashlib.sha256(b).hexdigest()!=s['sha256']:raise RuntimeError('READINESS execution source drift')
if (ROOT/'EXTRACTION_STARTED.json').exists():raise RuntimeError('one source extraction already consumed; no retry')
identity=json.loads((ROOT/'EVAL_IDENTITIES.json').read_text());indices=identity['original_index']
interface=(ROOT/'interface_core.py').read_text();worker=(ROOT/'remote_extract.py').read_text()
bootstrap='from __future__ import annotations\n'+ 'PLAN='+repr(cfg['remote_source'])+'\nINDICES='+repr(indices)+'\nREMOTE_NUMPY_VERSION='+repr(cfg['remote_numpy_version'])+'\nINTERFACE_SOURCE='+repr(interface)+'\n'+worker.replace('from __future__ import annotations\n','',1)
payload=bootstrap.encode('utf-8');(ROOT/'UPLOADED_SOURCE.py').write_bytes(payload)
cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','rylan@100.82.231.102',shlex.quote(cfg['remote_python'])+' -B -']
start=datetime.datetime.now(datetime.timezone.utc);clock=time.monotonic()
remaining=(datetime.datetime.fromisoformat(cfg['deadline_UTC'])-start).total_seconds()-60
timeout=min(630,remaining,cfg['numeric_wall_limit_s'])
if timeout<=0:raise RuntimeError('stage budget')
(ROOT/'EXTRACTION_STARTED.json').write_text(json.dumps({'PID':os.getpid(),'UTC':start.isoformat(),'command':cmd,'upload_bytes':len(payload),'upload_sha256':hashlib.sha256(payload).hexdigest()}))
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
print(json.dumps({'kind':'R4_extract_started','supervisor_PID':os.getpid(),'SSH_PID':p.pid,'UTC':start.isoformat(),'timeout_s':timeout}),flush=True)
errors=[];bytes_out=[0];bytes_in=[0];stderr=bytearray();h=hashlib.sha256()
def feed():
    try:bytes_in[0]=p.stdin.write(payload);p.stdin.close()
    except BaseException as e:errors.append('stdin '+repr(e))
def drain_out():
    try:
        with (ROOT/'EVAL_EXTRACT.npz').open('wb') as f:
            while True:
                b=p.stdout.read(65536)
                if not b:break
                if bytes_out[0]+len(b)>33554432:raise RuntimeError('stdout32MiB cap')
                f.write(b);bytes_out[0]+=len(b);h.update(b)
    except BaseException as e:errors.append('stdout '+repr(e))
def drain_err():
    try:
        while True:
            b=p.stderr.readline()
            if not b:break
            if len(stderr)+len(b)>262144:raise RuntimeError('stderr256KiB cap')
            stderr.extend(b);print(b.decode('utf-8',errors='replace').strip(),flush=True)
    except BaseException as e:errors.append('stderr '+repr(e))
threads=[threading.Thread(target=f,daemon=True) for f in [feed,drain_out,drain_err]]
for t in threads:t.start()
reason=None
while p.poll() is None:
    if errors or time.monotonic()-clock>=timeout:
        reason='own SSH timeout/stream failure; remote state unknown; no repeat';p.kill();break
    time.sleep(.2)
p.wait(timeout=5)
for t in threads:t.join(timeout=5)
(ROOT/'SSH_STDERR.log').write_bytes(stderr)
events=[]
for line in stderr.decode('utf-8',errors='replace').splitlines():
    try:events.append(json.loads(line))
    except ValueError:pass
receipt={'kind':'EVAL_extraction','supervisor_PID':os.getpid(),'SSH_PID':p.pid,'SSH_exit':p.returncode,
         'start_UTC':start.isoformat(),'stop_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'wall_s':time.monotonic()-clock,'errors':errors,'reason':reason,'upload_bytes':bytes_in[0],
         'stdout_bytes':bytes_out[0],'stdout_SHA256':h.hexdigest(),'remote_events':events,
         'remote_state':'native success' if p.returncode==0 and reason is None and not errors else 'unknown unless returned receipt',
         'retries':0,'source_hash_repeated':0}
if p.returncode==0 and reason is None and not errors:
    remote=[e['receipt'] for e in events if e.get('kind')=='R4_remote_result']
    if len(remote)!=1 or remote[0]['output_bytes']!=bytes_out[0] or remote[0]['output_sha256']!=h.hexdigest():receipt['errors'].append('transport result binding')
(ROOT/'EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
history=ROOT/'RUN_HISTORY.json';past=json.loads(history.read_text()) if history.exists() else []
past.append(receipt);history.write_text(json.dumps(past,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
sys.exit(0 if p.returncode==0 and reason is None and not receipt['errors'] else 1)
