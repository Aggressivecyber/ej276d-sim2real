"""One final control/report/manifest seal after scientific jobs and visual QA terminate."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib
R=Path(__file__).resolve().parent;START=time.monotonic()
def read(n):return json.loads((R/n).read_text('utf-8-sig'))
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
c=read('config_cpu.json');m=read('METRICS_COMPACT.json');gate=read('GATE_METRICS.json');num=read('TRAIN_SOLVER_CHECK.json');replay=read('REPLAY_RECEIPT.json');qa=read('FIGURE_QA.json')
assert all(x['actual_opened']and x['legible']for x in qa['figures'])
remote=read('REMOTE_RUN_HISTORY.json');local=read('RUN_HISTORY.json');fail=read('ENGINEERING_FAILURES.json')
assert not (R/'FINAL_RECEIPT.json').exists()
assert datetime.now(timezone.utc)<datetime.fromisoformat(c['deadline_UTC'])
science=sum(r['elapsed_s']for r in remote+local)
assert science<=14400 and replay['rows']==8000 and replay['pass']
resource={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'science_cumulative_s':science,'science_limit_s':14400,
 'remote_scientific_jobs':remote,'local_scientific_jobs':local,'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']or 0 for r in remote+local),
 'remote_job_tree_RSS_peak_bytes':max(r.get('job_tree_RSS_peak_bytes',0)for r in remote),
 'memory_probe_failures':sum(r['memory_probe_failures']for r in remote+local),
 'memory_probe_failure_events':[e for r in remote+local for e in r['memory_probe_failure_events']],
 'unknown_gap_reason_not_fabricated':True,'server_CPU_threads':12,'server_RSS_limit_bytes':c['RSS_limit_bytes'],'GPU_science_jobs':0,
 'local_CPU_threads':4,'local_RSS_limit_bytes':4*1024**3,'output_limit_bytes':4*1024**3,'EVAL_read':0,'CAL_waveform_decode':0,
 'new_response_training_runs_updates_optimizer_steps':0,'new_TRAIN_const_or_random_fits':0,'HGB_attempts':m['counters']['HGB_attempts'],
 'density_slots':m['counters']['density_description_slots'],'actual_density_EM_attempts':m['counters']['density_EM_attempts'],
 'main_generation_rows':m['counters']['main_generation_rows'],'repeat_noise_rows':m['counters']['repeat_noise_rows'],
 'source_probe_rows':m['counters']['source_probe_rows'],'source_perturbation_rows':m['counters']['extra_guide_probe_rows'],'independent_replay_rows':replay['rows'],
 'normal_copy_only_no_migration_consistency_test':True,'failures':fail,'sealed_stage_originals_modified':False}
write('RESOURCE_RECEIPT.json',resource)
write('NATIVE_CHAIN.json',{'remote':remote,'local':local,'control_receipts':[p.name for p in R.glob('*NATIVE_RECEIPT.json')if not p.name.endswith('.NATIVE_RECEIPT.json')],
 'native_tool_handles_reported_separately_after_actual_return':True,'unknown_control_PIDs':True})
s,l=c['arms'];cmp=m['long_short_common64_comparison']
psd_cmp=cmp['eight_prespecified_distance_differences']['PSD_overall_W1']
rms_cmp=cmp['eight_prespecified_distance_differences']['baseline_RMS_V_overall_W1']
science_summary=(f"At the common64-step inference budget, extended training lowered source shape493 BA from {m['C2ST']['results'][s]['full_historical_shape']['balanced_accuracy']:.3f} to {m['C2ST']['results'][l]['full_historical_shape']['balanced_accuracy']:.3f}, and overall PSD W1 from {psd_cmp['short']:.8f} to {psd_cmp['long']:.8f}. These are real descriptive gains, while the baseline RMS W1 increased by {100*rms_cmp['relative_change_long_minus_short_over_short']:.3f}%. LONG cannot be promoted on these marginal gains because required density support, source-conditioned PSD preservation and same-treated REAL controls also matter.")
lines=['# R46 frozen short/long integration comparison','',
 'Both frozen models passed the prospectively fixed64-vs128 TRAIN integration check. The sole LONG50000 candidate is '+('retained for development; final project acceptance remains separate.'if gate['LONG_primary_retain']else'closed by the unchanged joint scientific gates. No SHORT fallback was selected.'),'',
 science_summary,'',
 'The historical R45 fixed32 recipe remains a numerical failure. R46 changes only inference budget to64 Heun steps, with128 used as a TRAIN reference;128 is not an exact solution or candidate. Frozen SHORT and LONG have2950 and50000 historical updates respectively, with94400 and1600000 Monte Carlo row appearances per domain. Their unique parent populations are unchanged. No model, optimizer, calibration, initialization, coupling or TRAIN random state was fitted in R46.','',
 '| Model | TRAIN RMS median / p95 (LSB) | preADC relative charge max | source shape493 / u480 BA | gates1–6 |','|---|---:|---:|---:|---|']
for n in [s,l]:
 r=num['models'][n];scores=m['C2ST']['results'][n];gg=gate['models'][n]
 lines.append('| '+('SHORT2950'if n==s else'LONG50000')+f" | {r['RMS_median_LSB']:.6f} / {r['RMS_p95_LSB']:.6f} | {r['preADC_relative_q_max']:.3g} | {scores['full_historical_shape']['balanced_accuracy']:.3f} / {scores['normalized_u480']['balanced_accuracy']:.3f} | "+'/'.join('T'if gg[k]['pass']else'F'for k in gg if k.startswith('gate'))+' |')
lines+=['','LONG minus SHORT uses the new common64-step outputs, common fixed600/250 partitions, and500 group bootstrap draws. The interval is conditional on the fitted classifiers; it does not estimate response-training or seed uncertainty.','',
 '| View | paired BA difference | 95% interval |','|---|---:|---:|']
for v,r in cmp['paired_BA_long_minus_short'].items():lines.append(f"| {v} | {r['BA_delta_first_minus_second']:.6f} | [{r['BA_delta_95CI'][0]:.6f}, {r['BA_delta_95CI'][1]:.6f}] |")
lines+=['',f"The prespecified complete-response improvement description is supported: **{cmp['complete_response_improvement_description_supported']}**. Both full-shape paired intervals must have upper bounds below0 and all8 distances must be no more than10% worse. This comparison does not replace LONG-versus-BASE gates or project A–F acceptance. The observed two-view shape and PSD gains do not complete the full response-model objective.",'',
 '| Distance | SHORT | LONG | LONG−SHORT relative | required missing bins |','|---|---:|---:|---:|---|']
def fmt(x):return 'undefined'if x is None else f'{x:.7g}'
for key,r in cmp['eight_prespecified_distance_differences'].items():
 missing=r.get('unsupported_required_bins_by_domain',r.get('failed_required_bins_by_domain',{}))
 lines.append(f"| {key} | {fmt(r['short'])} | {fmt(r['long'])} | {fmt(r['relative_change_long_minus_short_over_short'])} | {json.dumps(missing)} |")
lines+=['','PSD conditional distance uses fixedQ1–7. Component center/raw width uses the original fixedQ2–7. Missing or failed required bins are retained as undefined and fail the comparison; no candidate-valid intersection is selected. Component fits are descriptive density components, not particle truth. Failed convergence or separation does not establish that physical particle peaks disappeared. All support gaps and original8-bin tables remain in metrics.json.','']
for n in [s,l]:
 gg=gate['models'][n];bad=[k for k,r in gg.items()if k.startswith('gate')and not r['pass']]
 lines += [('SHORT'if n==s else'LONG')+' scientific failures: '+', '.join(bad)+'.',
  'Required density failures: '+str(gg['gate2_PSD_width_centers_noise']['new_required_density_failures'])+'.',
  'Density reasons in those required bins: '+json.dumps([row for row in gg['gate2_PSD_width_centers_noise']['all_density_support_gaps'][n]if row['bin_id_1based']in gg['gate2_PSD_width_centers_noise']['new_required_density_failures']])+'.',
  'Actual-input pure-had source bins3/4 valley changes (fraction): '+str([r['delta']for r in gg['gate3_source_preservation']['actual_source_pure_had_valleys']])+'.','']
valley=m['source_primary_actual_Q']
lines += [f"For the unchanged213/181 actual-input pure-had source groups in bins3/4, SHORT valley fractions are {valley[s]['bin3']['valley_fraction']:.6f}/{valley[s]['bin4']['valley_fraction']:.6f}; LONG fractions are {valley[l]['bin3']['valley_fraction']:.6f}/{valley[l]['bin4']['valley_fraction']:.6f}. This is a source-conditioned preservation risk, not a statement that the REAL density components are particle truth. LONG-versus-BASE_REAL baseline BA difference is {m['C2ST']['same_treated_REAL_control_paired']['LONG_50000_REAL_CONTROL']['baseline_local']['BA_delta_first_minus_second']:.6f}; the normalized full-shape difference is {m['C2ST']['same_treated_REAL_control_paired']['LONG_50000_REAL_CONTROL']['normalized_u480']['BA_delta_first_minus_second']:.6f}. The fixed control gates remain unchanged.",'']
lines+=['Source2000 retains417 pureEE,1518 pure-had and65 mixed parents. Actual unquantized NN-input source-Q bins3/4 have213/181 pure-had events; historical ADC source-reference bins have207/182 and are separately labelled. All417 outside-TRAIN-support rows traverse each new model using the fixed nearest-edge conditioning index; there is no BASE fallback, clipping of source charge or reweighting. Nonzero source sensitivity does not prove correct event-wise detector mapping.','',
 'Same-treated REAL controls apply each stochastic model to the same2000 existing TRAIN-reference waveforms and original random arrays. The three gates compare new controls to frozen BASE_REAL, not to an assumed RAW identity. Absolute effects against RAW are still reported. The missing historical RAW Q–PSD2D paired reference remains unknown, without a replacement fit. REAL controls receive no G4 composition or PDG labels.','',
 f"Generation covered11968 rows (8000 main,2048 repeats,1664 corrected source probes,256 source perturbations). {replay['rows']} main SOURCE+REAL rows independently replayed bitwise from original input/white through the full64-step chain and oneADC. Shared functional components retain cross covariance; they are not independent physical signal/electronics. Finite preADC charge conservation is distinct from ADC charge rounding.",'',
 f"Actual science cost was{science:.3f}s; new HGB fits{resource['HGB_attempts']}, density slots{resource['density_slots']}, actual EM attempts{resource['actual_density_EM_attempts']}. Server inference/evaluation/replay usedCPU12; local plots usedCPU4. Peak actual-worker RSS{resource['actual_worker_RSS_peak_bytes']}B; RSS probe failures{resource['memory_probe_failures']}. No GPU science, new training, new REAL acquisition, EVAL or CAL waveform decoding occurred.",'',
 'These are historically exposed same-acquisition development results. The short/long comparison follows one inherited optimization trajectory and unequal training costs, not independent seeds or a blinded external validation. Charge-population mismatch is preserved; no overall-spectrum acceptance rule was relaxed. Numerical success and exact replay do not establish detector-response validity.','',
 'Ordinary control failures and the pre-evaluation fixed-support correction are retained in ENGINEERING_FAILURES.json and engineering_evaluation_definition_001. The correction added no fit, generation or scientific retry. All final PNGs were actually opened; FIGURE_QA.json records the final files.','',
 'Artifacts: METRICS_COMPACT.json contains aggregate publishable evidence; full metrics/identities/random inputs/predictions/waveforms stay on E/D and the project server. GATE_METRICS.json retains each original gate and failure. REPRODUCTION.md describes the exact server64-step pipeline and8000-row replay.']
(R/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode())
repro=f'''# R46 reproduction

Actual server root: {c['remote_root']}
Python: {c['server_python']}; CPU12, chunk64, no GPU scientific job. See SERVER_ENVIRONMENT.json, Linux PID/nonce/RSS acknowledgements and remote native receipts. All new local caches/temp/data are E/D.

In a new independent directory, copy the registered frozen SHORT R43 and LONG R45 weights/constants and exact128 TRAIN check payload using normal transfer. Reuse R46 network.py/source_flow_core.py and configuration with updated explicit paths. Do not copy old .used markers or receipts. Run check_models.py under linux_supervise.py once. Only if both fixed64/128 checks pass, copy the explicit inputs/reference files from CPU_PAYLOAD_PACKAGE_RECEIPT.json. Config_cpu.json lists the frozen model and reference paths. The copied historical R7/R26/R27/R32/R39 predictions and density tables are reused without refits.

Run generate_all.py, then evaluate_saved.py and replay_all.py sequentially, one scientific child at a time. Each launch must acknowledge actual PID+nonce before imports. Generate/replay both useCPU12/chunk64 and exactly64 Heun steps, f64 coordinates/projection and f32 neural network. No optimizer is loaded, no response weight changes and no target recoupling/random training state is created. Independent replay recomputes from actual source/reference input and saved white, before opening saved generated results; REPLAY_RECEIPT.json records8000 bitwise matches. It is a scientific reproducibility check, not a migration consistency test.

After normal result transfer, run summarize_and_plot.py with the local4-thread supervisor and copied exact inputs/metadata. This performs only prescribed saved-output descriptions and plots, no inference or evaluator fit. Then run plot_display_fix.py for the final comparison-figure title; DISPLAY_CORRECTION_RECEIPT.json retains the original PNG and records this display-only correction. FixedQ1–7/Q2–7 aggregate support behavior is bound by FIXED_SUPPORT_CORRECTION_RECEIPT.json, preserving the original pre-correction source/freezes. The originalR45 fixed32 failure is not rewritten. SHORT is not a fallback when LONG fails. All4000 candidate outputs and controls use the same frozen CPU environment; the historical2950/50000 training budgets differ and are explicit.

Restore only in a new directory. OldR43–R46 sealed stages remain immutable. No extra integration steps, solver/window/floor/bin/seed search, historical reference refit, training or EVAL use is authorized by these reproduction notes.
'''
(R/'REPRODUCTION.md').write_bytes(repro.encode())
excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};rows=[]
for p in sorted(R.rglob('*')):
 if p.is_file()and p.name not in excluded:
  rows.append({'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)})
total=sum(x['bytes']for x in rows);assert total<4*1024**3
write('FINAL_MANIFEST.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'files':rows,'file_count':len(rows),
 'total_bytes':total,'self_reference_exclusions':sorted(excluded),'byte_manifest_not_extra_migration_test':True})
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'status':'sealed','no_further_stage_writes':True,
 'manifest_sha256':digest(R/'FINAL_MANIFEST.json'),'manifest_files':len(rows),'manifest_bytes':total,
 'LONG_primary_retain':gate['LONG_primary_retain'],'LONG_SHORT_improvement_supported':cmp['complete_response_improvement_description_supported'],
 'science_cumulative_s':science,'last_write_UTC_from_final_native_stdout':True,
 'actual_native_exit_and_tool_reference_available_after_process_returns':True,'executor_deadline_UTC':c['deadline_UTC']}
write('FINAL_RECEIPT.json',receipt)
print(json.dumps({'PID':os.getpid(),'last_write_UTC':datetime.fromtimestamp((R/'FINAL_RECEIPT.json').stat().st_mtime,timezone.utc).isoformat(),
 'FINAL_MANIFEST_sha256':digest(R/'FINAL_MANIFEST.json'),'FINAL_RECEIPT_sha256':digest(R/'FINAL_RECEIPT.json'),
 'manifest_files':len(rows),'manifest_bytes':total,'science_cumulative_s':science,'control_seal_elapsed_s':time.monotonic()-START,'no_further_stage_writes':True}),flush=True)
