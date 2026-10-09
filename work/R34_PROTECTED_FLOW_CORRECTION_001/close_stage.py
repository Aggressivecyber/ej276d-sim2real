from __future__ import annotations
import pathlib, os, json, datetime, time, hashlib

ROOT=pathlib.Path(__file__).resolve().parent
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def read(name): return json.loads((ROOT/name).read_text('utf8'))

if (ROOT/'FINAL_RECEIPT.json').exists():
    raise RuntimeError('stage already sealed; do not write or seal again')
cfg=read('config.json')
DEADLINE=datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp()
start=utc(); tic=time.monotonic()
if DEADLINE-time.time()<60:
    raise TimeoutError('insufficient seal time')
history=read('RUN_HISTORY.json'); resource=read('RESOURCE_RECEIPT.json')
active=read('ACTIVE_JOB.json'); gate=read('GATE_METRICS.json')
assert len(history)==3 and all(r['actual_worker_native_exit']==0 for r in history)
assert active['actual_worker_native_exit']==0 and active['entry']=='summarize_and_plot.py'
assert resource['science_actual_s_including_guard_handshakes_statistics_recovery_replay_and_plots']<600
assert resource['HGB_actual']==20 and resource['density_actual_EM_attempts']==14
assert all(not row['retain'] for row in gate['endpoints'].values())
assert read('REPLAY_RECEIPT.json')['pass']
assert read('AUXILIARY_PRECISION_RECOVERY_RECEIPT.json')['exact_recovery_pass']
assert read('FIGURE_QA.json')['final_png_count']==13
for name in ['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','DEPENDENCY_INDEX.json']:
    assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0

# The preceding direct command's real tool result has already been observed.
dump(ROOT/'DELIVERY_ASSEMBLY_NATIVE_RECEIPT.json',{
    'role':'pure control/report assembly; no model/evaluator calls',
    'actual_PID':55408,'actual_native_exit':0,'actual_tool_chunk_id':'a03722',
    'worker_reported_stop_UTC':'2026-10-09T19:04:24.915988+00:00',
    'tool_wall_time_s':0.4432831,'scientific_reexecution':False,
    'receipt_recorded_by_seal_PID':os.getpid(),'receipt_recorded_UTC':utc()})
tool_index=read('NATIVE_TOOL_INDEX.json')
tool_index['delivery_assembly_native0']='a03722'
tool_index['report_readback_native0']='8b6e0f'
dump(ROOT/'NATIVE_TOOL_INDEX.json',tool_index)

excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
rows=[]
for f in sorted(ROOT.rglob('*')):
    if not f.is_file() or f.relative_to(ROOT).as_posix() in excluded:
        continue
    if time.time()>=DEADLINE-10:
        raise TimeoutError('seal deadline')
    st=f.stat(); sha=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            sha.update(block)
    st2=f.stat()
    if (st.st_size,st.st_mtime_ns)!=(st2.st_size,st2.st_mtime_ns):
        raise RuntimeError('concurrent stage file modification')
    rows.append({'path':f.relative_to(ROOT).as_posix(),'bytes':st.st_size,'sha256':sha.hexdigest()})
cap=402653184
assert sum(row['bytes'] for row in rows)<cap
manifest={
    'stage':'R34_PROTECTED_FLOW_CORRECTION_001','status':'sealed fixed-projection negative result',
    'creator_PID':os.getpid(),'UTC':utc(),'members':rows,'member_count':len(rows),
    'member_bytes':sum(row['bytes'] for row in rows),'self_reference_excluded':sorted(excluded),
    'no_science_reexecution_during_seal':True,'input_originals_not_modified':True,
    'hash_scope':'new local stage only; final pure-byte manifest, not a migration comparison'}
dump(ROOT/'FINAL_MANIFEST.json',manifest)
manifest_bytes=(ROOT/'FINAL_MANIFEST.json').read_bytes()
binding={'bytes':len(manifest_bytes),'sha256':hashlib.sha256(manifest_bytes).hexdigest()}
receipt={
    'status':'sealed','no_further_stage_writes':True,'seal_PID':os.getpid(),
    'assembly_PID':55408,'assembly_native_exit':0,'assembly_native_tool':'a03722',
    'start_UTC':start,'last_write_UTC':utc(),'executor_seal_deadline_UTC':cfg['deadline_UTC'],
    'deadline_pass':time.time()<DEADLINE,'manifest':binding,
    'manifest_member_count':len(rows),'manifest_member_bytes':manifest['member_bytes'],
    'final_total_files':len(rows)+2,'final_total_bytes':0,'unique_new_output_cap_bytes':cap,
    'science_actual_s':resource['science_actual_s_including_guard_handshakes_statistics_recovery_replay_and_plots'],
    'gate_retained':False,'key_outputs':['REPORT.md','METRICS_COMPACT.json','GATE_METRICS.json',
        'REPRODUCTION.md','RESOURCE_RECEIPT.json','FIGURE_QA.json','results/figures'],
    'assembly_and_manifest_role':'pure control; all scientific jobs were already terminal',
    'seal_native_exit_evidence':'actual caller tool exit returned after this last stage write'}
for _ in range(10):
    encoded=(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    total=manifest['member_bytes']+binding['bytes']+len(encoded)
    if receipt['final_total_bytes']==total:
        break
    receipt['final_total_bytes']=total
else:
    raise RuntimeError('receipt byte count did not converge')
assert receipt['deadline_pass'] and receipt['final_total_bytes']<cap
(ROOT/'FINAL_RECEIPT.json').write_bytes(encoded)
actual_mtime=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps({'sealed':receipt,'FINAL_RECEIPT_sha256':hashlib.sha256(encoded).hexdigest(),
    'actual_last_stage_file_mtime_UTC':actual_mtime,'stop_after_last_write_UTC':utc(),
    'seal_elapsed_s':time.monotonic()-tic},ensure_ascii=False),flush=True)
