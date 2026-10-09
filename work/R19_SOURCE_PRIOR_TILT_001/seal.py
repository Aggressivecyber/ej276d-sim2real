from __future__ import annotations
import pathlib,json,hashlib,datetime,os,time
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();DEADLINE=datetime.datetime.fromisoformat('2026-10-09T09:24:33.361272+00:00')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,a):p.write_bytes((json.dumps(a,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('seal deadline; no late output')
if (ROOT/'FINAL_MANIFEST.json').exists()or(ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('immutable seal already exists')
metrics=json.loads((ROOT/'metrics.json').read_bytes());native=json.loads((ROOT/'REMOTE_NATIVE_RECEIPT.json').read_bytes());replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_bytes());qa=json.loads((ROOT/'FIGURE_QA.json').read_bytes());resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_bytes())
assert native['status']=='complete'and all(x['exit_code']==0 for x in native['history'])and replay['pass']and qa['all_eight_actually_viewed']
assert metrics['counters']['G_updates']==8000 and metrics['counters']['D_updates']==8000
assert all(metrics['pairing'].values())and metrics['counters']['HGB_attempts']==14 and metrics['counters']['density_EM_attempts']<=16
assert resource['science_cumulative_s']<1800 and resource['RSS_peak_bytes']<=4294967296 and resource['GPU_allocated_peak_bytes']<=4294967296
assert all(t['training_s']<600 and t['updates']==4000 and t['D_updates']==4000 for t in metrics['training'].values())
before=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
resource.update(output_existing_bytes_before_seal=before,final_control_reserved_bytes=65536,output_upper_bound_at_seal=before+65536,final_hash_is_pure_bytes_no_array_decode=True,science_new_after_terminal=0,stage_stop_rule='immutable after FINAL_RECEIPT; next task only by ROOT',science_status='implemented/replay_passed; overall_response_not_matched',final_seal_PID=os.getpid(),final_seal_UTC=utc())
if before+65536>268435456:raise MemoryError('stage output cap')
write(ROOT/'RESOURCE_RECEIPT.json',resource)
exclude={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};files=[p for p in sorted(ROOT.rglob('*'))if p.is_file()and p.name not in exclude];entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files]
manifest={'stage':'R19_SOURCE_PRIOR_TILT_001','UTC':utc(),'PID':os.getpid(),'self_reference_exclusions':sorted(exclude),'member_count':len(entries),'member_bytes':sum(x['bytes']for x in entries),'files':entries,'hash_purpose':'single final artifact manifest; not migration test or source-input reaudit'}
write(ROOT/'FINAL_MANIFEST.json',manifest);manifest_sha=sha(ROOT/'FINAL_MANIFEST.json');receipt={'stage':'R19','status':'sealed','PID':os.getpid(),'UTC':utc(),'seal_elapsed_s_before_final_write':time.monotonic()-START,'science_native_exit_codes':[x['exit_code']for x in native['history']],'main_worker_PID':native['history'][0]['worker_PID'],'replay_worker_PID':native['history'][1]['worker_PID'],'science_cumulative_s':native['numeric_cumulative_s'],'manifest_sha256':manifest_sha,'manifest_members':len(entries),'manifest_member_bytes':manifest['member_bytes'],'key_SHA256':{p.name:sha(p)for p in [ROOT/'REPORT.md',ROOT/'METRICS_COMPACT.json',ROOT/'RESOURCE_RECEIPT.json',ROOT/'REPLAY_RECEIPT.json',ROOT/'REPRODUCTION.md',ROOT/'FIGURE_QA.json']},'self_hash_excluded':True,'native_seal_process_exit':'confirm from calling exec tool; not self-predicted','no_more_stage_writes':True,'stage_deadline_UTC':DEADLINE.isoformat(),'scientific_success_claim':False}
write(ROOT/'FINAL_RECEIPT.json',receipt)
# No output-file mutation after receipt. Native tool proves actual process exit.
receipt_sha=sha(ROOT/'FINAL_RECEIPT.json');allfiles=[p for p in ROOT.rglob('*')if p.is_file()];last=max(p.stat().st_mtime for p in allfiles)
print(json.dumps({'PID':os.getpid(),'manifest_SHA256':manifest_sha,'receipt_SHA256':receipt_sha,'manifest_members':len(entries),'stage_file_count':len(allfiles),'stage_bytes':sum(p.stat().st_size for p in allfiles),'last_write_UTC':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'stop_writing':True,'key_SHA256':receipt['key_SHA256']}),flush=True)
