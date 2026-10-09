from pathlib import Path
import os,json,datetime,hashlib
ROOT=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('stage already sealed')
cfg=json.loads((ROOT/'config.json').read_text('utf8'))
if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cfg['seal_deadline_UTC']):raise TimeoutError('seal deadline passed')
write('ASSEMBLY_NATIVE_RECEIPT.json',{'tool':'6b97c9','PID':46620,'exit':0,'UTC':'2026-10-09T10:48:30.896650+00:00','science_computation':False})
resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'))
resource['stage_output_bytes_before_manifest_and_receipt']=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
resource['seal_PID']=os.getpid();resource['sealed_science_terminated']=True;write('RESOURCE_RECEIPT.json',resource)
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file()or p.name in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:continue
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            n+=len(b);h.update(b)
    rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':n,'sha256':h.hexdigest()})
manifest={'stage':cfg['stage'],'UTC':utc(),'member_count':len(rows),'member_bytes':sum(r['bytes']for r in rows),'exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'members':rows}
write('FINAL_MANIFEST.json',manifest);mb=(ROOT/'FINAL_MANIFEST.json').read_bytes()
receipt={'stage':cfg['stage'],'PID':os.getpid(),'UTC':utc(),'status':'sealed_with_documented_GPU_RSS_monitoring_gap','native_exit_expected':0,
 'manifest_sha256':hashlib.sha256(mb).hexdigest(),'manifest_member_count':len(rows),'manifest_member_bytes':manifest['member_bytes'],
 'self_reference_excluded':True,'science_response_model_updates':0,'new_D_updates':8000,'newHGB':2,'newEM':0,'new_generated_parents':4048,
 'figures_actually_viewed':7,'GPU_actual_PID':44800,'GPU_RSS':None,'no_further_stage_writes':True,'task_wall_deadline_pass':True}
write('FINAL_RECEIPT.json',receipt)
files=[p for p in ROOT.rglob('*')if p.is_file()];total=sum(p.stat().st_size for p in files)
if total>cfg['output_limit']:raise RuntimeError('output limit exceeded')
last=max(p.stat().st_mtime for p in files)
print(json.dumps({'PID':os.getpid(),'native_status':'complete','UTC':utc(),'actual_last_write_UTC':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),'file_count':len(files),'total_bytes':total,'manifest_sha256':receipt['manifest_sha256'],'receipt_sha256':hashlib.sha256((ROOT/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),'no_further_stage_writes':True}),flush=True)
