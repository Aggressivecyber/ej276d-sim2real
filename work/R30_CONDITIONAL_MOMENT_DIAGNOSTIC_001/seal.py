from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
if '--child' in sys.argv:
    cfg=json.loads((ROOT/'config.json').read_text('utf8'))
    if datetime.now(timezone.utc)>=datetime.fromisoformat(cfg['deadline_UTC']):raise TimeoutError('seal deadline')
    res=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'))
    res.update(final_control_assembly_UTC=now(),output_bytes_before_manifest_receipt=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
               completion='science_and_all_eight_visual_QA_complete; single final seal',science_rerun=False)
    write(ROOT/'RESOURCE_RECEIPT.json',res)
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name in ('FINAL_MANIFEST.json','FINAL_RECEIPT.json'):continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat(cfg['deadline_UTC']):raise TimeoutError('seal hash deadline')
        b=p.read_bytes();entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    total=sum(e['bytes'] for e in entries)
    if total>cfg['output_limit_bytes']:raise RuntimeError('output cap')
    manifest={'stage':ROOT.name,'UTC':now(),'seal_child_PID':os.getpid(),'files':entries,'file_count':len(entries),'listed_bytes':total,
              'excluded_self_reference':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'old_stages_not_written':True}
    write(ROOT/'FINAL_MANIFEST.json',manifest)
    print(json.dumps({'child_PID':os.getpid(),'UTC':now(),'manifest_sha256':hashlib.sha256((ROOT/'FINAL_MANIFEST.json').read_bytes()).hexdigest(),
                      'manifest_bytes':(ROOT/'FINAL_MANIFEST.json').stat().st_size,'listed_file_count':len(entries),'listed_bytes':total}),flush=True)
else:
    if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('already sealed; no additional writes')
    start=time.monotonic()
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'seal.py'),'--child'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    out,err=p.communicate(timeout=20)
    if p.returncode!=0:
        print(json.dumps({'seal_parent_PID':os.getpid(),'seal_child_PID':p.pid,'native_exit':p.returncode,'stdout':out,'stderr':err}),flush=True)
        raise SystemExit(p.returncode)
    child=json.loads(out)
    receipt={'stage':ROOT.name,'seal_parent_PID':os.getpid(),'seal_child_PID':p.pid,'observed_child_native_exit':p.returncode,
             'parent_native_exit_confirmed_by_calling_tool':True,'child_result':child,'seal_elapsed_s':time.monotonic()-start,
             'actual_last_write_UTC':now(),'state':'sealed','no_further_stage_writes':True,
             'output_cap_bytes':128*1024**2,'files_count_including_manifest_receipt':child['listed_file_count']+2}
    before=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    count=before
    for _ in range(5):
        receipt['output_total_bytes_including_manifest_receipt']=count
        data=(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
        newcount=before+len(data)
        if newcount==count:break
        count=newcount
    if count>receipt['output_cap_bytes']:raise RuntimeError('final output cap')
    (ROOT/'FINAL_RECEIPT.json').write_bytes(data)
    stopped=now()
    print(json.dumps({'sealed':True,'seal_parent_PID':os.getpid(),'seal_child_PID':p.pid,'observed_child_native_exit':p.returncode,
                      'actual_last_write_UTC':receipt['actual_last_write_UTC'],'stopped_UTC':stopped,
                      'manifest_sha256':child['manifest_sha256'],'receipt_sha256':hashlib.sha256(data).hexdigest(),
                      'files':receipt['files_count_including_manifest_receipt'],'total_bytes':receipt['output_total_bytes_including_manifest_receipt']}),flush=True)
