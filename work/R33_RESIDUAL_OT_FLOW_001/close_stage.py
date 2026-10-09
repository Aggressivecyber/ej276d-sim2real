from __future__ import annotations
import pathlib, subprocess, sys, os, json, datetime, time, hashlib
ROOT=pathlib.Path(__file__).resolve().parent
DEADLINE=datetime.datetime.fromisoformat('2026-10-09T19:42:39.460916+00:00').timestamp()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('stage already sealed; no second seal')
remaining=DEADLINE-time.time()
if remaining<=60:raise TimeoutError('insufficient seal time')
start=utc();tic=time.monotonic()
with(ROOT/'DELIVERY_ASSEMBLY.stdout.log').open('wb')as out,(ROOT/'DELIVERY_ASSEMBLY.stderr.log').open('wb')as err:
    child=subprocess.Popen([sys.executable,'-B',str(ROOT/'assemble_delivery.py')],cwd=ROOT,stdout=out,stderr=err,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:code=child.wait(timeout=min(40,remaining-30))
    except subprocess.TimeoutExpired:
        child.kill();code=child.wait()
        dump(ROOT/'DELIVERY_ASSEMBLY_NATIVE_RECEIPT.json',{'supervisor_PID':os.getpid(),'child_PID':child.pid,'native_exit':code,'timeout':True,'start_UTC':start,'stop_UTC':utc()})
        raise
dump(ROOT/'DELIVERY_ASSEMBLY_NATIVE_RECEIPT.json',{'supervisor_PID':os.getpid(),'child_PID':child.pid,'native_exit':code,'timeout':False,'start_UTC':start,'stop_UTC':utc(),'elapsed_s':time.monotonic()-tic,'role':'control assembly only; no model/evaluator calls'})
if code:raise RuntimeError(f'assembly failed native {code}; preserve original failure')
active=json.loads((ROOT/'ACTIVE_JOB.json').read_text('utf8'))
assert active['actual_worker_native_exit']==0 and active['entry']=='summarize_and_plot.py'
excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
rows=[]
for f in sorted(ROOT.rglob('*')):
    if not f.is_file()or f.relative_to(ROOT).as_posix()in excluded:continue
    if time.time()>=DEADLINE-10:raise TimeoutError('seal deadline')
    st=f.stat();sha=hashlib.sha256()
    with f.open('rb')as stream:
        for b in iter(lambda:stream.read(1048576),b''):sha.update(b)
    st2=f.stat()
    if(st.st_size,st.st_mtime_ns)!=(st2.st_size,st2.st_mtime_ns):raise RuntimeError('concurrent stage file modification')
    rows.append({'path':f.relative_to(ROOT).as_posix(),'bytes':st.st_size,'sha256':sha.hexdigest()})
assert sum(r['bytes']for r in rows)<536870912
manifest={'stage':'R33_RESIDUAL_OT_FLOW_001','status':'sealed fixed recipe negative result','creator_PID':os.getpid(),'UTC':utc(),
    'members':rows,'member_count':len(rows),'member_bytes':sum(r['bytes']for r in rows),
    'self_reference_excluded':sorted(excluded),'no_science_reexecution':True,'input_originals_not_modified':True,
    'hash_scope':'new local stage only; byte hashing for final manifest, not a migration comparison'}
dump(ROOT/'FINAL_MANIFEST.json',manifest)
binding={'bytes':(ROOT/'FINAL_MANIFEST.json').stat().st_size,'sha256':hashlib.sha256((ROOT/'FINAL_MANIFEST.json').read_bytes()).hexdigest()}
resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'))
receipt={'status':'sealed','no_further_stage_writes':True,'seal_PID':os.getpid(),'assembly_PID':child.pid,'assembly_native_exit':code,
    'start_UTC':start,'last_write_UTC':utc(),'executor_seal_deadline_UTC':'2026-10-09T19:42:39.460916+00:00',
    'deadline_pass':time.time()<DEADLINE,'manifest':binding,'manifest_member_count':len(rows),'manifest_member_bytes':manifest['member_bytes'],
    'final_total_files':len(rows)+2,'final_total_bytes':0,'science_actual_s':resource['science_actual_s_including_failure_and_plots'],
    'gate_retained':False,'key_outputs':['REPORT.md','METRICS_COMPACT.json','GATE_METRICS.json','REPRODUCTION.md','RESOURCE_RECEIPT.json','results/figures'],
    'assembly_and_manifest_role':'pure control; scientific jobs had already terminated',
    'seal_native_exit_evidence':'actual caller tool native exit is returned after this last stage write'}
for _ in range(5):
    encoded=(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    total=manifest['member_bytes']+binding['bytes']+len(encoded)
    if receipt['final_total_bytes']==total:break
    receipt['final_total_bytes']=total
assert receipt['deadline_pass'] and receipt['final_total_bytes']<536870912
(ROOT/'FINAL_RECEIPT.json').write_bytes(encoded)
actual_mtime=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps({'sealed':receipt,'FINAL_RECEIPT_sha256':hashlib.sha256(encoded).hexdigest(),
    'actual_last_stage_file_mtime_UTC':actual_mtime,'stop_after_last_write_UTC':utc()},ensure_ascii=False),flush=True)
