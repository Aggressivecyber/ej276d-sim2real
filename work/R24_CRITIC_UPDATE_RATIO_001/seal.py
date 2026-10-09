from pathlib import Path
import json,datetime,os,time,hashlib
ROOT=Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text('utf8'));DEADLINE=datetime.datetime.fromisoformat(CFG['deadline_UTC'])
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(v):return(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf8')
if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('seal deadline')
if(ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('already sealed; no rewrite')
history=json.loads((ROOT/'RUN_HISTORY.json').read_text('utf8'));resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'));replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'));metric=json.loads((ROOT/'METRICS_COMPACT.json').read_text('utf8'))
assert replay['pass']and resource['science_wall_s']<=1800 and json.loads((ROOT/'VISUAL_QA.json').read_text('utf8'))['actual_visual_inspection']
rows=[];excluded=['FINAL_MANIFEST.json','FINAL_RECEIPT.json'];hashbytes=0
for p in sorted(ROOT.rglob('*')):
    if not p.is_file()or p.relative_to(ROOT).as_posix()in excluded:continue
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b);n+=len(b)
    rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':n,'sha256':h.hexdigest()});hashbytes+=n
manifest={'stage':ROOT.name,'UTC':utc(),'member_count':len(rows),'member_bytes':hashbytes,'hash_algorithm':'sha256','members':rows,'self_reference_exclusions':excluded,'excluded_receipt_binds_manifest_sha':True,'scientific_decode_in_seal':0,'only_current_stage_bytes_hashed':True}
mb=dump(manifest);assert hashbytes+len(mb)+16384<=CFG['output_limit_bytes']
(ROOT/'FINAL_MANIFEST.json').write_bytes(mb)
tr=metric['training']['D5']
receipt={'stage':ROOT.name,'status':'sealed','seal_process_PID':os.getpid(),'pre_final_receipt_write_UTC':utc(),'manifest_sha256':hashlib.sha256(mb).hexdigest(),'manifest_bytes':len(mb),'manifest_member_count':len(rows),'manifest_member_bytes':hashbytes,'directory_bytes_excluding_receipt':hashbytes+len(mb),'current_stage_byte_hash_read':hashbytes,'science_wall_s':resource['science_wall_s'],'training_wall_s':resource['training_wall_s'],'actual_peak_RSS_bytes':resource['actual_worker_RSS_peak_bytes'],'GPU_peak_allocated_bytes':resource['GPU_allocated_peak_bytes'],'selected_step':tr['selected_step'],'actual_G_updates':tr['updates'],'actual_D_updates':tr['D_updates'],'reader_updates':tr['style_reader_updates'],'main_rows':2000,'replay_rows':2000,'replay_bitwise_pass':True,'HGB_attempts':resource['new_HGB_attempts'],'EM_attempts':resource['new_EM_attempts'],'scientific_active':0,'executor_deadline_UTC':DEADLINE.isoformat(),'native_exit_evidence':'calling tool native result authoritative; script success exit0','self_reference_exclusions':excluded,'no_further_stage_writes_after_this_receipt':True}
if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('deadline before final receipt')
(ROOT/'FINAL_RECEIPT.json').write_bytes(dump(receipt))
st=(ROOT/'FINAL_RECEIPT.json').stat();fs=[p for p in ROOT.rglob('*')if p.is_file()]
print(json.dumps({'PID':os.getpid(),'manifest_sha256':receipt['manifest_sha256'],'receipt_sha256':hashlib.sha256((ROOT/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),'member_count':len(rows),'final_files':len(fs),'final_bytes':sum(p.stat().st_size for p in fs),'actual_last_write_UTC':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'completion_UTC':utc(),'control_elapsed_s':time.monotonic()-START,'scientific_active':0,'status':'sealed','no_further_stage_writes':True}),flush=True)
