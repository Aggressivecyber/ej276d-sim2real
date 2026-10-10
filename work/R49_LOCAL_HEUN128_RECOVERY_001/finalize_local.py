from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
R=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf-8-sig'))
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
if(R/'FINAL_RECEIPT.json').exists():raise RuntimeError('stage already sealed; no further writes')
C=read('config.json');H=read('RUN_HISTORY.json');N=read('TRAIN_SOLVER_CHECK.json')
assert N['both_pass']
G=read('GATE_METRICS.json');A=read('ACCEPTANCE_A_C.json');RP=read('REPLAY_RECEIPT.json')
E=read('EVALUATION_NUMERIC_RECEIPT.json');GN=read('GENERATION_NUMERIC_RECEIPT.json')
assert RP['rows']==8000
assert E['counters']['HGB_attempts']==40 and E['counters']['density_description_slots']==32
assert [GN['counters'][k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows']]==[8000,2048,1664,256]
QA=read('FIGURE_QA.json');filesPNG={p.name for p in(R/'results/figures').glob('*.png')}
assert filesPNG=={x['file']for x in QA['figures']if x['actually_opened']}
for x in QA['figures']:
    assert hashlib.sha256((R/'results/figures'/x['file']).read_bytes()).hexdigest()==x['SHA256']
for n in ['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json']:assert(R/n).is_file()
remote={'attempt':'R49_FROZEN_HEUN128_RESPONSE_001','directory':C['parent_remote_stage'],
    'completed_native_science_s':2107.3375221949973,'completed_counts':{'generated_rows':11968,'HGB':40,'density_slots':32,'EM':32},
    'replay_worker_PID':218727,'replay_supervisor_PID':218726,'replay_terminal':'unknown',
    'replay_observed_elapsed_lower_bound_s':120.15029214299284,'full_replay_elapsed_s':None,
    'outage_UTC':'2026-10-10T10:22:45.059596+00:00','server_user_reported_status':'maintenance',
    'user_explicit_local_failover':True,'SSH_queries_stopped':True,
    'remote_results_used_as_local_array_metrics':False,'global_single_physical_job_confirmed':False}
account={'stage':R.name,'UTC':utc(),'compute_budget_enforced':False,'science_budget_s':None,
    'wall_budget_s':None,'deadline_UTC':None,'stage_output_budget_bytes':None,
    'actual_local_scientific_elapsed_s':sum(r['elapsed_s']for r in H),'actual_local_scientific_jobs':len(H),
    'actual_scientific_elapsed_s':sum(r['elapsed_s']for r in H),'actual_scientific_jobs':len(H),
    'actual_scientific_elapsed_scope':'this independent local recovery only; historical remote separate',
    'native_jobs':H,'all_local_native_exits_zero':all(r.get('actual_worker_native_exit')==0 for r in H),
    'all_native_exits_zero':all(r.get('actual_worker_native_exit')==0 for r in H),
    'actual_local_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in H),
    'local_job_tree_RSS_peak_bytes':max(r['job_tree_RSS_peak_bytes']for r in H),
    'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in H),
    'job_tree_RSS_peak_bytes':max(r['job_tree_RSS_peak_bytes']for r in H),
    'RSS_query_missing_events':[e for r in H for e in r['memory_probe_failure_events']],
    'new_response_training_runs':0,'new_response_updates':0,'new_calibration_fits':0,
    'historical_R48_training_each_updates':50000,'historical_training_science_recharged':False,
    'counters':E['counters'],'generation':GN['counters'],'replay_rows':RP['rows'],'replay_pass':RP['pass'],
    'EVAL_read':0,'CAL_waveform_decode':0,'new_source_acquisition':0,'CPU_threads':4,
    'RSS_safety_limit_bytes':4294967296,'GPU_used':False,'remote_attempt_preserved':remote,
    'preparation_control_failures':[read(n)for n in ['ENVIRONMENT_IMPORT_FAILURE_001.json','PREPARATION_FAILURE_001.json','CONTROL_READ_FAILURE_001.json','CONTROL_OPTIONAL_STATUS_READ_001.json','CONTROL_REPORT_PATCH_FAILURE_001.json']if(R/n).exists()],
    'scientific_failure_files':[str(p.relative_to(R))for p in R.glob('*FAILURE*.json')if p.name not in ['ENVIRONMENT_IMPORT_FAILURE_001.json','PREPARATION_FAILURE_001.json','CONTROL_READ_FAILURE_001.json','CONTROL_REPORT_PATCH_FAILURE_001.json']],
    'local_all_new_files_E_D':True,'migration_content_tests':0,'original_stage_files_modified':False}
write('RESOURCE_RECEIPT.json',account)
write('NATIVE_CHAIN.json',{'local_science':H,'remote_attempt_preserved':remote,
    'scope':'local native elapsed counted once; remote terminal unknown is not invented or zeroed'})
write('FIGURE_QA_SUMMARY.json',{'actually_opened_count':len(filesPNG),'files':sorted(filesPNG),'all_final_images_opened':True})
items=[]
for p in sorted(R.rglob('*')):
    if not p.is_file()or p.name in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:continue
    with p.open('rb')as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    items.append({'path':str(p.relative_to(R)).replace('\\','/'),'bytes':p.stat().st_size,'SHA256':h.hexdigest()})
manifest={'stage':R.name,'UTC':utc(),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],
    'files':items,'listed_files':len(items),'listed_bytes':sum(x['bytes']for x in items)}
write('FINAL_MANIFEST.json',manifest);mh=hashlib.sha256((R/'FINAL_MANIFEST.json').read_bytes()).hexdigest()
write('FINAL_RECEIPT.json',{'stage':R.name,'status':'sealed','PID':os.getpid(),'UTC':utc(),
    'new_response_training_updates':0,'new_calibration_fits':0,'candidate_steps':128,'reference_steps':256,
    'both_local_numerical_pass':True,'both_numerical_pass':True,'primary_A_C_pass':G['primary_A_C_pass'],'primary_decision':G['decision'],
    'project_A_F_target_complete':False,'model_project_target_complete':False,'D_E_not_run':True,
    'actual_local_scientific_elapsed_s':account['actual_local_scientific_elapsed_s'],
    'actual_scientific_elapsed_s':account['actual_local_scientific_elapsed_s'],
    'remote_replay_terminal':'unknown','remote_completed_science_s':remote['completed_native_science_s'],
    'remote_replay_observed_elapsed_lower_bound_s':remote['replay_observed_elapsed_lower_bound_s'],
    'global_single_physical_job_confirmed':False,
    'main_generated_rows':8000,'total_generated_rows':11968,'independent_main_replay_rows':RP['rows'],'replay_pass':RP['pass'],
    'independent_main_replay_pass':RP['pass'],
    'actual_HGB_fits':40,'density_slots':32,'actual_EM_fits':E['counters']['density_EM_attempts'],
    'listed_files':len(items),'listed_bytes':manifest['listed_bytes'],'manifest_SHA256':mh,
    'compute_budget_enforced':False,'no_further_stage_writes':True,'outer_native_exit_actual_tool_to_bind':True})
allfiles=[p for p in R.rglob('*')if p.is_file()]
print(json.dumps({'actual_seal_PID':os.getpid(),'last_stage_write_UTC':datetime.fromtimestamp((R/'FINAL_RECEIPT.json').stat().st_mtime,timezone.utc).isoformat(),
    'FINAL_RECEIPT_SHA256':hashlib.sha256((R/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),'FINAL_MANIFEST_SHA256':mh,
    'actual_files':len(allfiles),'actual_bytes':sum(p.stat().st_size for p in allfiles),
    'actual_local_science_s':account['actual_local_scientific_elapsed_s'],'primary_A_C_pass':G['primary_A_C_pass'],
    'remote_replay_terminal':'unknown','no_further_stage_writes':True}),flush=True)
