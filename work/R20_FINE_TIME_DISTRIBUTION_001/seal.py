from __future__ import annotations
import pathlib,json,hashlib,datetime,os,time
ROOT=pathlib.Path(__file__).resolve().parent
START=time.monotonic()
DEADLINE=datetime.datetime.fromisoformat('2026-10-09T10:07:53.627396+00:00')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,d):(ROOT/n).write_bytes((json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('seal deadline; no late output')
if (ROOT/'FINAL_MANIFEST.json').exists()or(ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('stage already sealed')
m=read('metrics.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json')
qa=read('FIGURE_QA.json');resource=read('RESOURCE_RECEIPT.json');pair=read('PAIRING_RECEIPT.json')
assert native['status']=='complete' and all(r['exit_code']==0 for r in native['history'])
assert replay['pass']and replay['rows']==4000 and not replay['TRAIN_target_or_REAL_used']
assert qa['all_nine_actually_viewed']and all(r['layout_pass']for r in qa['figures'])
assert all(pair['common_stream_hashes_equal'].values())and all(pair['same_old_R19_TILT_streams'].values())
assert m['counters']['G_updates']==8000 and m['counters']['D_updates']==8000
assert m['counters']['HGB_attempts']==14 and m['counters']['density_EM_attempts']<=16
assert m['counters']['main_generation_rows']==4000 and m['counters']['repeat_noise_rows']==2048 and m['counters']['source_probe_rows']<=2048
assert all(t['updates']==4000 and t['D_updates']==4000 and t['training_s']<600 for t in m['training'].values())
assert sum(t['training_s']for t in m['training'].values())<1200
assert resource['science_cumulative_s']<1800 and resource['RSS_peak_bytes']<=4294967296 and resource['GPU_allocated_peak_bytes']<=4294967296
assert m['EVAL_read']==0 and m['counters']['new_G4']==0 and m['counters']['CAL_waveform_decode']==0
before=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
if before+65536>268435456:raise MemoryError('output cap')
resource.update(output_existing_bytes_before_seal=before,final_control_reserved_bytes=65536,
    output_upper_bound_at_seal=before+65536,final_hash_is_pure_bytes_no_array_decode=True,
    model_inference_after_remote_terminal=0,fit_after_remote_terminal=0,
    final_seal_PID=os.getpid(),final_seal_UTC=utc(),science_status='implemented/replay_passed; overall_response_not_matched',
    no_more_stage_writes_after_FINAL_RECEIPT=True,
    closure_preparation_native={'tool_chunk_id':'2080b5','PID':7268,'native_exit':0,'elapsed_s':.498558})
write('RESOURCE_RECEIPT.json',resource)
exclude={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
paths=[p for p in sorted(ROOT.rglob('*'))if p.is_file()and p.relative_to(ROOT).as_posix()not in exclude]
entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in paths]
if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('deadline before manifest write')
write('FINAL_MANIFEST.json',{'stage':'R20_FINE_TIME_DISTRIBUTION_001','UTC':utc(),'PID':os.getpid(),
    'self_reference_exclusions':sorted(exclude),'member_count':len(entries),
    'member_bytes':sum(e['bytes']for e in entries),'files':entries,
    'hash_purpose':'single final artifact seal; no migration comparison or source-input re-audit'})
manifest_sha=sha(ROOT/'FINAL_MANIFEST.json')
keys=['REPORT.md','METRICS_COMPACT.json','RESOURCE_RECEIPT.json','REPLAY_RECEIPT.json',
      'REPRODUCTION.md','FIGURE_QA.json','REPEAT_WAVEFORM_SUMMARY.json']
receipt={'stage':'R20','status':'sealed','PID':os.getpid(),'UTC':utc(),
    'science_native_exit_codes':[r['exit_code']for r in native['history']],
    'main_worker_PID':native['history'][0]['worker_PID'],'replay_worker_PID':native['history'][1]['worker_PID'],
    'science_cumulative_s_including_helper_failures':resource['science_cumulative_s'],
    'manifest_SHA256':manifest_sha,'manifest_members':len(entries),'manifest_member_bytes':sum(e['bytes']for e in entries),
    'key_SHA256':{n:sha(ROOT/n)for n in keys},'self_hash_excluded':True,
    'native_seal_process_exit':'confirm from calling exec tool; not self-predicted',
    'seal_elapsed_s_before_final_write':time.monotonic()-START,'stage_seal_deadline_UTC':DEADLINE.isoformat(),
    'scientific_success_claim':False,'no_more_stage_writes':True}
write('FINAL_RECEIPT.json',receipt)
# No further file mutations after this receipt. The native tool confirms process exit.
files=[p for p in ROOT.rglob('*')if p.is_file()]
print(json.dumps({'PID':os.getpid(),'manifest_SHA256':manifest_sha,'receipt_SHA256':sha(ROOT/'FINAL_RECEIPT.json'),
    'manifest_members':len(entries),'stage_file_count':len(files),'stage_bytes':sum(p.stat().st_size for p in files),
    'last_write_UTC':datetime.datetime.fromtimestamp(max(p.stat().st_mtime for p in files),datetime.timezone.utc).isoformat(),
    'elapsed_s':time.monotonic()-START,'stop_writing':True,'key_SHA256':receipt['key_SHA256']}),flush=True)
