from pathlib import Path
import json,datetime,os
ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'metrics.json').read_text('utf8'));H=json.loads((ROOT/'RUN_HISTORY.json').read_text('utf8'));R=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'));CFG=json.loads((ROOT/'config.json').read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
tr=M['training']['D5'];old=M['control_training_reused']['STANDARDIZED']
fields=['updates','D_updates','style_reader_updates','training_s','selected_step','initial_validation','selected_validation','final_validation','validation_generation_rows','validation_support','same_time_selected_G_reader_D','selected_using_DEV_or_D','actually_sampled_unique_source','GPU_peak_allocated_bytes','device','initial_G_reader','initial_D']
trainagg={n:{k:r[k]for k in fields}for n,r in [('D5',tr),('R23_STANDARDIZED_D1',old)]}
trainagg['D5'].update(original_D_updates=tr['original_D_updates'],extra_D_updates=tr['extra_D_updates'],stop_reason=tr['stop_reason'],extra_D_exposure=tr['extra_D_exposure'])
compact={k:M[k]for k in ['status','data_roles','domains','domain_vs_REAL','local','PSD_main','energy_conditional_width','source_composition','source_association','source_probes','generation','counters','versions','online_D_validation_AUC','pairing','paired_common_prefix','primary_DEV_population_reweighted','EVAL_read','CPU_main_and_replay_same_environment','old_control_retrained','sole_training_intervention','equal_compute_or_D_exposure']}
compact['training']=trainagg
compact['source_probes']={n:{k:v for k,v in d.items()if k!='fixed_source_parent_index'}for n,d in compact['source_probes'].items()}
compact['training']['D5']['trajectory']=[r['validation']for r in tr['validation_trace']]
compact['C2ST']=M['C2ST']
compact['random_decomposition']={n:{'groups':{'joint8':{k:v for k,v in d['groups']['joint8'].items()if k!='rows'}},'actual_rows':d['actual_rows'],'source_parent_union':d['source_parent_union'],'new_independent_source_parents':0}for n,d in M['random_decomposition'].items()}
compact['publication_scope']={'aggregate_only':True,'per_event_ID_index_prediction_actual_streams_repeat_parent_lists':'complete local metrics/private NPZ only','old_R23_RAW':'background contrast, not sole ratio control'}
write('METRICS_COMPACT.json',compact)
toolmap=json.loads((ROOT/'NATIVE_TOOL_MAP.json').read_text('utf8'))if(ROOT/'NATIVE_TOOL_MAP.json').exists()else{}
wall=sum(a['elapsed_s']for a in H);peak=max(a['actual_worker_RSS_peak_bytes']for a in H);training_native=json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text('utf8'))
resource={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'science_wall_s':wall,'science_budget_s':1800,'all_prepare_failures_training_generation_evaluation_plots_replay_charged':True,'scientific_failures':[a for a in H if a['launcher_native_exit']!=0 or a.get('termination_reason')],
 'training_wall_s':tr['training_s'],'training_limit_total_s':900,'training_by_arm':{'D5':{'G_updates':tr['updates'],'D_updates':tr['D_updates'],'reader_updates':tr['style_reader_updates'],'original_D_updates':tr['original_D_updates'],'extra_D_updates':tr['extra_D_updates'],'training_s':tr['training_s'],'G_cap':2950,'D_cap':14750}},
 'control_R23_D1':{'training_repeated_here':False,'G_D_reader_each':old['updates'],'training_s':old['training_s'],'selected_step':old['selected_step']},'equal_compute_or_D_exposure':False,'extra_D_fake_rows':tr['extra_D_exposure']['extra_D_generated_fake_rows'],'original_D_fake_rows':tr['original_D_updates']*32,'original_G_source_rows':tr['updates']*32,'identity_REAL_rows':tr['updates']*32,'validation_generated_rows':tr['validation_generation_rows'],
 'actual_worker_RSS_peak_bytes':peak,'actual_worker_RSS_limit_bytes':4294967296,'actual_PID_monitored_in_every_job':all(a['handshake_before_imports']and a['actual_worker_PID']is not None for a in H),'memory_probe_failures':sum(a['memory_probe_failures']for a in H),'repeated_64MiB_probe':False,
 'GPU_allocated_peak_bytes':training_native['GPU_peak_allocated_bytes'],'GPU_allocated_limit_bytes':4294967296,'CPU_threads':4,'single_scientific_process_at_a_time':True,'scientific_jobs':[{**a,'tool_chunks':toolmap.get(a['entry'],[])}for a in H],
 'new_HGB_attempts':M['counters']['HGB_attempts'],'HGB_limit':7,'new_EM_attempts':M['counters']['density_EM_attempts'],'EM_limit':8,'density_description_slots':M['counters']['density_description_slots'],'main_rows':M['counters']['main_generation_rows'],'replay_rows':R['rows'],'joint8_rows':M['counters']['repeat_noise_rows'],'source_probe_rows':M['counters']['source_probe_rows'],
 'EVAL_waveform_read':0,'CAL_waveform_read':0,'raw_long_waveform_decode':0,'new_G4':0,'SSH':0,'package_installs':0,'child_agents':0,'Git_actions':0,'migration_consistency_test':False,'only_new_stage_files_hashed_at_seal':True,
 'output_limit_bytes':268435456,'pre_final_manifest_bytes':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file()),'first_command_UTC':CFG['first_actual_command_UTC'],'first_command_PID':CFG['first_actual_PID'],'registration_UTC':CFG['registered_UTC'],'executor_seal_deadline_UTC':CFG['deadline_UTC'],'root_stage_deadline_UTC':CFG['absolute_deadline_UTC'],
 'engineering_notes':['R23 proven true-PID handshake reused without repeat probe','no old classifier/width/reference/scaler refit','complete original G-stream prefix compared scientifically; not a migration test','Torch train scalar-conversion warning retained, not a failed update','GPU2.12 training and CPU2.10 main/replay reported separately','C2ST legend-only layout repair charged; initial figure retained; no metric changes']}
assert wall<=1800 and tr['training_s']<=900 and tr['updates']<=2950 and tr['D_updates']<=14750 and peak<=4294967296 and resource['GPU_allocated_peak_bytes']<=4294967296 and R['pass']
write('RESOURCE_RECEIPT.json',resource)
print(json.dumps({'PID':os.getpid(),'UTC':resource['UTC'],'science_wall_s':wall,'actual_peak_RSS_bytes':peak,'GPU_peak_allocated_bytes':resource['GPU_allocated_peak_bytes'],'response_updates_in_control':0}))
