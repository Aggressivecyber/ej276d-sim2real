from __future__ import annotations
from pathlib import Path
import datetime,hashlib,json,os,time
ROOT=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def dump(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('sealed; no repeat seal or stage write')
cfg=read('config.json');deadline=datetime.datetime.fromisoformat(cfg['deadline_UTC']).timestamp();start=utc();tic=time.monotonic()
if time.time()>deadline-60:raise TimeoutError('insufficient seal time')
history=read('RUN_HISTORY.json');resource=read('RESOURCE_ACCOUNTING.json');gate=read('GATE_METRICS.json');active=read('ACTIVE_JOB.json')
assert len(history)==6 and sum(r['actual_worker_native_exit']!=0 for r in history)==1
assert active['entry']=='fix_CDF_layout.py' and active['actual_worker_native_exit']==0
assert history[1]['entry']=='evaluate_endpoints.py' and history[1]['actual_worker_native_exit']==1
assert resource['conservative_charge_including_preflight_s']<1800
assert resource['flow_fit_attempts']==1 and resource['flow_updates']==2950
assert resource['HGB_attempts']==10 and resource['density_EM_attempts']==8
assert resource['resume_new_HGB_EM_generation']==0 and not gate['retain']
assert read('REPLAY_RECEIPT.json')['pass'] and read('REPLAY_RECEIPT.json')['rows']==2000
assert read('FIGURE_QA.json')['final_figure_count']==13
for n in ['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','RESOURCE_ACCOUNTING.json','FAILURES_AND_REPAIRS.json','COUNTER_SEMANTICS.json']:
 assert (ROOT/n).is_file()and (ROOT/n).stat().st_size>0
dump('DELIVERY_ASSEMBLY_NATIVE_RECEIPT.json',{'role':'pure control/report assembly; scientific array decode, model or evaluator calls0','actual_PID':41220,'actual_native_exit':0,'actual_tool_chunk_id':'587f08','worker_reported_stop_UTC':'2026-10-09T19:41:12.344320+00:00','tool_wall_time_s':.5548061,'receipt_recorded_by_seal_PID':os.getpid(),'UTC':utc(),'earlier_assembly_native0':'5a45af/PID50284; subsequent explicit flow-counter semantics added without science','initial_assembly_failure':'0a532e/PID62060, preserved in ASSEMBLY_FAILURE_001.json and control_failures/assembly_001'})
excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};rows=[]
for p in sorted(ROOT.rglob('*')):
 if not p.is_file()or p.relative_to(ROOT).as_posix()in excluded:continue
 if time.time()>deadline-10:raise TimeoutError('seal deadline')
 st=p.stat();h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 end=p.stat()
 if (st.st_size,st.st_mtime_ns)!=(end.st_size,end.st_mtime_ns):raise RuntimeError('concurrent stage file modification')
 rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':st.st_size,'sha256':h.hexdigest()})
cap=402653184;member_bytes=sum(r['bytes']for r in rows);assert member_bytes<cap
manifest={'stage':cfg['stage'],'status':'sealed PSD-ordered CFM negative joint result','creator_PID':os.getpid(),'UTC':utc(),'members':rows,'member_count':len(rows),'member_bytes':member_bytes,'self_reference_excluded':sorted(excluded),'no_science_reexecution_during_seal':True,'originals_not_modified':True,'hash_scope':'new local stage only; final pure-byte manifest, not migration comparison'}
dump('FINAL_MANIFEST.json',manifest);payload=(ROOT/'FINAL_MANIFEST.json').read_bytes();binding={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
receipt={'status':'sealed','no_further_stage_writes':True,'seal_PID':os.getpid(),'assembly_PID':41220,'assembly_native_exit':0,'assembly_native_tool':'587f08','start_UTC':start,'last_write_UTC':utc(),'executor_seal_deadline_UTC':cfg['deadline_UTC'],'deadline_pass':time.time()<deadline,'manifest':binding,'manifest_member_count':len(rows),'manifest_member_bytes':member_bytes,'final_total_files':len(rows)+2,'final_total_bytes':0,'output_cap_bytes':cap,'science_guarded_elapsed_s':resource['science_guarded_elapsed_s_including_failure_plots'],'science_conservatively_charged_including_preflight_s':resource['conservative_charge_including_preflight_s'],'actual_worker_RSS_peak_bytes':resource['actual_worker_RSS_peak_bytes'],'GPU_peak_allocated_bytes':resource['GPU_peak_allocated_bytes'],'RSS_missing_events':1,'flow_fit_attempts':1,'flow_updates':2950,'HGB':10,'density_slots_and_EM':8,'main_joint_probe_replay_rows':[2000,1024,832,2000],'gate_passes':[gate[k]['pass']for k in ['gate1_complete_response','gate2_PSD_width_noise','gate3_source_preservation','gate4_boundary_solver_replay']],'retained':False,'decision':gate['decision'],'project_goal_not_marked_complete':True,'key_outputs':['REPORT.md','METRICS_COMPACT.json','GATE_METRICS.json','REPRODUCTION.md','RESOURCE_ACCOUNTING.json','FIGURE_QA.json','results/figures','PSD_ORDERED_CFM/FINAL_FLOW.pt'],'seal_role':'pure control; all science terminal','seal_native_exit_evidence':'actual caller tool result returned after last write, not predicted here'}
for _ in range(10):
 encoded=(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8');total=member_bytes+binding['bytes']+len(encoded)
 if receipt['final_total_bytes']==total:break
 receipt['final_total_bytes']=total
else:raise RuntimeError('self-excluded receipt byte count did not converge')
assert receipt['deadline_pass']and total<cap
(ROOT/'FINAL_RECEIPT.json').write_bytes(encoded)
mtime=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps({'sealed':receipt,'FINAL_RECEIPT_sha256':hashlib.sha256(encoded).hexdigest(),'actual_last_stage_file_mtime_UTC':mtime,'stop_after_last_write_UTC':utc(),'seal_elapsed_s':time.monotonic()-tic},ensure_ascii=False),flush=True)
