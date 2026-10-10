from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib
R=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((R/n).read_text('utf8'))
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
if(R/'FINAL_RECEIPT.json').exists():raise RuntimeError('stage already sealed; no additional writes')
C=read('config.json');history=read('REMOTE_RUN_HISTORY.json');num=read('TRAIN_SOLVER_CHECK.json')
assert num['both_pass']
gate=read('GATE_METRICS.json');acceptance=read('ACCEPTANCE_A_C.json');compact=read('METRICS_COMPACT.json')
replay=read('REPLAY_RECEIPT.json');evalrec=read('EVALUATION_NUMERIC_RECEIPT.json');gen=read('GENERATION_NUMERIC_RECEIPT.json')
assert replay['rows']==8000
assert evalrec['counters']['HGB_attempts']==40 and evalrec['counters']['density_description_slots']==32
assert [gen['counters'][k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows']]==[8000,2048,1664,256]
qa=read('FIGURE_QA.json');actual={p.name for p in (R/'results/figures').glob('*.png')}
assert actual=={x['file']for x in qa['figures']if x['actually_opened']}
for x in qa['figures']:
    assert hashlib.sha256((R/'results/figures'/x['file']).read_bytes()).hexdigest()==x['SHA256']
account={'stage':R.name,'UTC':utc(),'compute_budget_enforced':False,'science_budget_s':None,
         'wall_budget_s':None,'deadline_UTC':None,'stage_output_budget_bytes':None,
         'actual_scientific_elapsed_s':sum(r['elapsed_s']for r in history),
         'actual_scientific_jobs':len(history),'native_jobs':history,
         'all_native_exits_zero':all(r['actual_worker_native_exit']==0 for r in history),
         'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in history),
         'job_tree_RSS_peak_bytes':max(r['job_tree_RSS_peak_bytes']for r in history),
         'RSS_query_missing_events':[e for r in history for e in r['memory_probe_failure_events']],
         'new_response_training_runs':0,'new_response_updates':0,'new_calibration_fits':0,
         'historical_R48_training_each_updates':50000,'historical_training_elapsed_charged_to_R49':False,
         'counters':evalrec['counters'],'generation':gen['counters'],'replay_rows':replay['rows'],'replay_pass':replay['pass'],
         'EVAL_read':0,'CAL_waveform_decode':0,'new_source_acquisition':0,
         'engineer_preexec_repairs':[read('DOWNSTREAM_PREEXEC_REPAIR.json'),read('REPLAY_NEGATIVE_SEAL_PREEXEC_CORRECTION.json')],
         'new_scientific_failure_receipts':[str(p.relative_to(R))for p in R.glob('*FAILURE*.json')],
         'local_only_paths':'E/D','normal_migration_not_equivalence_test':True}
write('RESOURCE_RECEIPT.json',account)
write('NATIVE_CHAIN.json',{'science':history,'local_transport':[read(p.name)for p in R.glob('*.LOCAL_NATIVE_RECEIPT.json')],
                         'scope':'actual science native elapsed counted once; transport wall not added twice'})
write('FIGURE_QA_SUMMARY.json',{'actually_opened_count':len(actual),'files':sorted(actual),'all_final_images_opened':True})
files=[]
for p in sorted(R.rglob('*')):
    if not p.is_file()or p.name in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:continue
    data=p.read_bytes();files.append({'path':str(p.relative_to(R)).replace('\\','/'),'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest()})
manifest={'stage':R.name,'UTC':utc(),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],
          'files':files,'listed_files':len(files),'listed_bytes':sum(x['bytes']for x in files)}
write('FINAL_MANIFEST.json',manifest)
mh=hashlib.sha256((R/'FINAL_MANIFEST.json').read_bytes()).hexdigest()
write('FINAL_RECEIPT.json',{'stage':R.name,'status':'sealed','PID':os.getpid(),'UTC':utc(),
    'new_training_updates':0,'candidate_steps':128,'reference_steps':256,'both_numerical_pass':True,
    'primary_A_C_pass':gate['primary_A_C_pass'],'primary_decision':gate['decision'],
    'model_project_target_complete':False,'D_E_not_run':True,'actual_scientific_elapsed_s':account['actual_scientific_elapsed_s'],
    'main_generated_rows':8000,'total_generated_rows':11968,'independent_main_replay_rows':replay['rows'],'independent_main_replay_pass':replay['pass'],
    'actual_HGB_fits':40,'density_slots':32,'actual_EM_fits':evalrec['counters']['density_EM_attempts'],
    'listed_files':len(files),'listed_bytes':manifest['listed_bytes'],'manifest_SHA256':mh,
    'compute_budget_enforced':False,'no_further_stage_writes':True,
    'outer_native_exit_to_be_bound_from_actual_tool':True})
allfiles=[p for p in R.rglob('*')if p.is_file()]
print(json.dumps({'seal_PID':os.getpid(),'last_stage_write_UTC':datetime.fromtimestamp((R/'FINAL_RECEIPT.json').stat().st_mtime,timezone.utc).isoformat(),
    'FINAL_RECEIPT_SHA256':hashlib.sha256((R/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),
    'FINAL_MANIFEST_SHA256':mh,'actual_files':len(allfiles),'actual_bytes':sum(p.stat().st_size for p in allfiles),
    'science_s':account['actual_scientific_elapsed_s'],'primary_A_C_pass':gate['primary_A_C_pass'],
    'no_further_stage_writes':True}),flush=True)
