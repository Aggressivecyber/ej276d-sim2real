from __future__ import annotations
import pathlib,json,hashlib,datetime,os,time
ROOT=pathlib.Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-10-09T02:10:58.545151+00:00'):raise RuntimeError('executor deadline passed')
if (ROOT/'FINAL_MANIFEST.json').exists() or (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('already sealed')
start=now();records=[];hashed=0
for f in sorted(ROOT.rglob('*')):
    if not f.is_file():continue
    name=f.relative_to(ROOT).as_posix();n=f.stat().st_size
    if name in ['UPLOAD.tar','REMOTE_RETURN.tar']:
        records.append({'path':name,'bytes':n,'sha256':None,'status':'normal transport artifact; no migration content hash audit requested'})
    else:
        h=hashlib.sha256()
        with f.open('rb') as s:
            while True:
                b=s.read(1024*1024)
                if not b:break
                h.update(b);hashed+=len(b)
        records.append({'path':name,'bytes':n,'sha256':h.hexdigest(),'status':'new stage artifact byte binding, no source/destination comparison'})
manifest={'stage':'R10','PID':os.getpid(),'UTC':start,'files':records,'file_count':len(records),'total_recorded_bytes':sum(x['bytes'] for x in records),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'transport_archive_hash_omission':'explicit user migration waiver; original input version bindings in config are not new migration proof','seal_hash_bytes':hashed,'original_stages_modified':False}
raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode();(ROOT/'FINAL_MANIFEST.json').write_bytes(raw)
base=[f for f in ROOT.rglob('*') if f.is_file()];bytesbase=sum(f.stat().st_size for f in base)
receipt={'stage':'R10','status':'sealed_negative_result','PID':os.getpid(),'start_UTC':start,'seal_UTC':now(),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'report_sha256':hashlib.sha256((ROOT/'REPORT.md').read_bytes()).hexdigest(),'compact_sha256':hashlib.sha256((ROOT/'METRICS_COMPACT.json').read_bytes()).hexdigest(),'resource_sha256':hashlib.sha256((ROOT/'RESOURCE_RECEIPT.json').read_bytes()).hexdigest(),'reproduction_sha256':hashlib.sha256((ROOT/'REPRODUCTION.md').read_bytes()).hexdigest(),'final_file_count':len(base)+1,'final_stored_bytes':bytesbase,'file_cap_bytes':512*1024**2,'new_scientific_run_in_seal':False,'response_training_ended':True,'EVAL_read':0,'archive_migration_SHA_comparison':False,'future_stage_started':False,'native_exit_evidence':'actual exec tool output, receipt successful only after this command native0'}
for _ in range(8):
    out=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode();size=bytesbase+len(out)
    if size==receipt['final_stored_bytes']:break
    receipt['final_stored_bytes']=size
out=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
if bytesbase+len(out)>512*1024**2:raise RuntimeError('stage output cap')
(ROOT/'FINAL_RECEIPT.json').write_bytes(out)
files=[f for f in ROOT.rglob('*') if f.is_file()];last=max(f.stat().st_mtime for f in files)
print(json.dumps({'PID':os.getpid(),'last_write_UTC':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),'manifest_sha256':receipt['manifest_sha256'],'receipt_sha256':hashlib.sha256(out).hexdigest(),'report_sha256':receipt['report_sha256'],'compact_sha256':receipt['compact_sha256'],'reproduction_sha256':receipt['reproduction_sha256'],'files':len(files),'total_bytes':sum(f.stat().st_size for f in files),'seal_hash_bytes':hashed,'science_runs_in_seal':0,'stop_writing':True}),flush=True)
