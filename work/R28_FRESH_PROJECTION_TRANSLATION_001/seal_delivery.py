from pathlib import Path
import sys, os, json, datetime, time, hashlib, subprocess

ROOT=Path(__file__).resolve().parent
DEADLINE=datetime.datetime.fromisoformat('2026-10-09T15:31:09.766899+00:00')
EXCLUDE={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def now_ok():
    if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('R28 executor seal deadline reached')
def encode(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def write(n,v):
    now_ok();(ROOT/n).write_bytes(encode(v))
def load(n):return json.loads((ROOT/n).read_text('utf8'))
def members():
    result=[]
    for p in ROOT.rglob('*'):
        if p.is_symlink():raise RuntimeError('stage symlink not permitted in seal')
        if p.is_file() and p.name not in EXCLUDE:result.append(p)
    return sorted(result,key=lambda p:p.relative_to(ROOT).as_posix())

if '--child' in sys.argv:
    now_ok();started=utc();start=time.monotonic()
    hist=load('RUN_HISTORY.json');resource=load('RESOURCE_RECEIPT.json')
    assert len(hist)==5 and all(r['actual_worker_native_exit']==0 and r['launcher_native_exit']==0 and r['handshake_before_imports'] for r in hist)
    assert resource['science_elapsed_s']<1800 and resource['training_s']<900
    assert resource['G_updates']==resource['D_updates']==resource['reader_updates']==2950
    assert resource['HGB_attempts']==20 and resource['density_EM_attempts']==14 and resource['density_description_slots']==16
    assert resource['actual_worker_RSS_peak_bytes']<=resource['RSS_limit_bytes'] and resource['GPU_peak_allocated_bytes']<=resource['GPU_allocated_limit_bytes']
    assert load('REPLAY_RECEIPT.json')['pass'] is True
    resource['finalization']={'PID':os.getpid(),'start_UTC':started,'source':'pure control inventory and byte SHA only; no array decoding or scientific recomputation'}
    resource['stage_output_bytes']='final total including manifest/receipt is recorded in FINAL_RECEIPT; member bytes below explicitly exclude those two files'
    write('RESOURCE_RECEIPT.json',resource)
    # Stabilize integer byte/count accounting, with no self-hash or manifest cycle.
    for _ in range(5):
        paths=members();count=len(paths);size=sum(p.stat().st_size for p in paths)
        if resource.get('sealed_member_count')==count and resource.get('sealed_member_bytes')==size:break
        resource.update(sealed_member_count=count,sealed_member_bytes=size,accounting_excluded_files=sorted(EXCLUDE))
        write('RESOURCE_RECEIPT.json',resource)
    else:raise RuntimeError('resource inventory byte count did not stabilize')
    rows=[]
    for p in members():
        now_ok();st=p.stat();h=hashlib.sha256();actual=0
        with p.open('rb') as handle:
            while True:
                b=handle.read(1024*1024)
                if not b:break
                h.update(b);actual+=len(b)
        if actual!=st.st_size or p.stat().st_mtime_ns!=st.st_mtime_ns:raise RuntimeError('stage member changed during seal')
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':actual,'sha256':h.hexdigest()})
    total=sum(r['bytes'] for r in rows)
    assert total==resource['sealed_member_bytes']
    manifest={'stage':ROOT.name,'seal_child_PID':os.getpid(),'UTC':utc(),'status':'sealed_members','hash_method':'each stage member stream-read once SHA256; no scientific decode','member_count':len(rows),'member_bytes':total,'excluded_self_reference':sorted(EXCLUDE),'files':rows}
    write('FINAL_MANIFEST.json',manifest)
    manifest_bytes=(ROOT/'FINAL_MANIFEST.json').read_bytes()
    print(json.dumps({'PID':os.getpid(),'UTC':utc(),'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'manifest_bytes':len(manifest_bytes),'member_count':len(rows),'member_bytes':total,'elapsed_s':time.monotonic()-start}),flush=True)
else:
    now_ok()
    if (ROOT/'FINAL_MANIFEST.json').exists() or (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('already sealed; do not rerun')
    chain=load('NATIVE_CHAIN.json')
    chain['assembly_native_tools']=[{'tool':'309b55','PID':36208,'native_exit':0,'pure_control':True},{'tool':'0c5b16','PID':41736,'native_exit':0,'pure_control':True}]
    chain['seal_parent_PID']=os.getpid()
    write('NATIVE_CHAIN.json',chain)
    start=time.monotonic();start_utc=utc()
    proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'seal_delivery.py'),'--child'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:out,err=proc.communicate(timeout=min(55,max(.1,DEADLINE.timestamp()-time.time())))
    except subprocess.TimeoutExpired:
        proc.kill();out,err=proc.communicate()
        print(json.dumps({'parent_PID':os.getpid(),'child_PID':proc.pid,'native_exit':proc.returncode,'error':'seal child timeout','stdout':out.decode('utf8','replace'),'stderr':err.decode('utf8','replace')}),flush=True)
        raise
    if proc.returncode!=0:
        print(json.dumps({'parent_PID':os.getpid(),'child_PID':proc.pid,'native_exit':proc.returncode,'stdout':out.decode('utf8','replace'),'stderr':err.decode('utf8','replace')}),flush=True)
        raise SystemExit(proc.returncode)
    result=json.loads(out.decode('utf8'))
    receipt={'stage':ROOT.name,'status':'sealed','no_further_stage_writes':True,'seal_parent_PID':os.getpid(),'seal_child_PID':proc.pid,'seal_child_native_exit':proc.returncode,'start_UTC':start_utc,'receipt_write_UTC':utc(),'seal_elapsed_s':time.monotonic()-start,'executor_deadline_UTC':DEADLINE.isoformat(),'manifest_sha256':result['manifest_sha256'],'manifest_bytes':result['manifest_bytes'],'member_count':result['member_count'],'member_bytes':result['member_bytes'],'total_file_count':result['member_count']+2,'total_output_bytes':0,'output_cap_bytes':268435456,'self_hash_excluded':True,'science_elapsed_s':load('RESOURCE_RECEIPT.json')['science_elapsed_s'],'science_or_array_decode_during_seal':0,'child_console_receipt':result,'stderr':err.decode('utf8','replace'),'parent_native_exit_evidence':'calling exec tool; not inferred from receipt'}
    for _ in range(5):
        actual=result['member_bytes']+result['manifest_bytes']+len(encode(receipt))
        if receipt['total_output_bytes']==actual:break
        receipt['total_output_bytes']=actual
    assert receipt['total_output_bytes']<receipt['output_cap_bytes']
    write('FINAL_RECEIPT.json',receipt)
    rp=ROOT/'FINAL_RECEIPT.json';last_write=datetime.datetime.fromtimestamp(rp.stat().st_mtime,datetime.timezone.utc).isoformat()
    print(json.dumps({'seal_parent_PID':os.getpid(),'seal_child_PID':proc.pid,'child_native_exit':proc.returncode,'last_filesystem_write_UTC':last_write,'stop_UTC':utc(),'manifest_sha256':result['manifest_sha256'],'receipt_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'total_files':receipt['total_file_count'],'total_output_bytes':receipt['total_output_bytes'],'no_further_stage_writes':True}),flush=True)
