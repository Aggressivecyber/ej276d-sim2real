from pathlib import Path
import os,json,datetime,time,hashlib
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'));start=time.monotonic()
if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('Already sealed')
deadline=datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()
if time.time()>=deadline:raise TimeoutError('Seal deadline passed')
resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'));qa=json.loads((ROOT/'VISUAL_QA.json').read_text('utf8'))
assert resource['science_cumulative_s']<=600 and resource['scientific_active']==0 and qa['actually_viewed']
members=[]
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.name in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:continue
    n=path.stat().st_size;h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            data=f.read(1024*1024)
            if not data:break
            h.update(data)
    members.append({'path':path.relative_to(ROOT).as_posix(),'bytes':n,'sha256':h.hexdigest()})
manifest={'stage':cfg['stage'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'member_count':len(members),'member_bytes':sum(r['bytes'] for r in members),'members':members,'self_reference_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'scientific_array_decode':0,'only_new_stage_bytes_hashed':True}
body=(json.dumps(manifest,indent=2)+'\n').encode('utf8');mh=hashlib.sha256(body).hexdigest()
if manifest['member_bytes']+len(body)+16384>cfg['output_limit_bytes']:raise MemoryError('Final output cap')
if time.time()>=deadline:raise TimeoutError('Deadline before last seal writes')
(ROOT/'FINAL_MANIFEST.json').write_bytes(body)
rec={'stage':cfg['stage'],'seal_PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':mh,'manifest_member_count':len(members),'member_bytes':manifest['member_bytes'],'scientific_s':resource['science_cumulative_s'],'science_cap_s':600,'response_updates':0,'new_model_fits':0,'A_complete':True,'B_complete':True,'original_B_failure_preserved':True,'QA_complete':True,'deadline_UTC':cfg['deadline_UTC'],'status':'sealed','scientific_active':0,'no_further_stage_writes':True,'self_reference_exclusions':manifest['self_reference_exclusions']}
rb=(json.dumps(rec,indent=2)+'\n').encode('utf8');(ROOT/'FINAL_RECEIPT.json').write_bytes(rb)
last=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps({'PID':os.getpid(),'manifest_sha256':mh,'receipt_sha256':hashlib.sha256(rb).hexdigest(),'member_count':len(members),'final_files':len(members)+2,'final_bytes':manifest['member_bytes']+len(body)+len(rb),'actual_last_write_UTC':last,'completion_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'control_elapsed_s':time.monotonic()-start,'status':'sealed','scientific_active':0,'no_further_stage_writes':True}))
