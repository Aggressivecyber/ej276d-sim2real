"""Control-only R15 report and inventory from saved JSON, without science rerun."""
from pathlib import Path
import json,os,hashlib,datetime
ROOT=Path(__file__).resolve().parent
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
start=utc();cfg=read('config.json')
if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cfg['deadline_UTC']):raise TimeoutError('R15 seal deadline')
m=read('metrics.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');num=read('NUMERIC_RECEIPT.json');graphics=read('PLOT_LABEL_FIX_RECEIPT.json');qa=read('FIGURE_QA.json');chargeplot=read('CHARGE_PLOT_METADATA.json')
if qa['status']!='eight_key_figures_visually_inspected':raise RuntimeError('Missing actual figure QA')
tr=m['training'];w=m['energy_conditional_width'];names=['GAUSSIAN_BASE','RESIDUAL_FLOW'];cs=m['C2ST']
compact={k:v for k,v in m.items() if k not in ['training','energy_conditional_width','nearest_neighbours']}
compact['training']={k:v for k,v in tr.items() if k!='trace'}
compact['training'].update(initial_validation=tr['trace'][0],selected_validation=next(r for r in tr['trace'] if r['step']==tr['selected_step']),final_validation=tr['trace'][-1],full_trace_file='TRAINING.json')
compact['energy_conditional_width']={k:v for k,v in w.items() if k!='tables'}
compact['energy_conditional_width']['tables']={name:{k:[{kk:vv for kk,vv in row.items() if kk not in ['updates','trace']} for row in val] if k=='slices' else val for k,val in tab.items()} for name,tab in w['tables'].items()}
nn=m['nearest_neighbours'];compact['nearest_neighbours']={k:nn[k] for k in ['original2592','expanded_fit','actual_expanded_reference_N','reference_cardinality_not_memory_evidence']}
oldnn=nn['baseline_and_REAL_summaries_reused'];compact['nearest_neighbours']['baseline_reused']={ref:{view:{'GAUSSIAN_BASE':oldnn[ref][view]['EXPOSURES_CONDITIONAL']} for view in ['V','own_shape']} for ref in ['original2592','expanded_fit']}
compact['nearest_neighbours']['historic_REAL_summary_scope']='Existing R14/R13 summaries retained in full local metrics; no new REAL nearest query.'
compact['publication_scope']='Aggregate statistics only. Event IDs, waveforms, predictions, neighbours and full training/EM trajectories remain private.'
compact['key_figures']=[r['file'] for r in qa['figures']]
write('METRICS_COMPACT.json',compact)
summary=read('RESULT_SUMMARY.json');summary['width']['reasons']={name:[{'bin':r['bin']+1,'valid':r['valid'],'failure_reasons':r['failure_reasons']} for r in tab['slices']] for name,tab in w['tables'].items()};write('RESULT_SUMMARY.json',summary)
lines=['# R15 conditional residual flow','',
'The residual flow improves held-out density and baseline discrimination but does not close the complete response discrepancy. Complete validation NLL changes from -2420.615 to -2436.294 nat/event. Full voltage BA drops0.970 to0.950 and baseline BA0.686 to0.604, while full shape remains0.902 and time-tail BA remains0.930. Charge-PSD double-band structure and narrow-bin support are essentially unchanged.','',
'The experiment fixes R14 expanded21047 TRAIN-fit records, original599 internal validation and2000 DEV, plus the EXPOSURES head/conditional-Gaussian checkpoints selected at15700/15900. Representation/scales/m0 and empirical L still come from original2592. No original long waveform, CAL waveform, EVAL or new G4 is decoded. One acquisition and artificial groups remain the independence limit. These are target-density development results, not a verified G4-to-REAL conditional mapping.','',
'## Fixed scientific comparison','',
'The new eight-layer446-dimensional conditional affine RealNVP uses34 head coordinates,223/223 splits and257->128->128->446 SiLU networks, bounded logscale3*tanh(raw/3). All final matrices/biases start at zero; each layer restores original coordinate order. Only this network is fitted. Frozen m/ell remain unchanged, but the inverse flow need not have zero mean/unit variance: the contrast permits additional conditional mean and covariance changes as well as non-Gaussian structure, so it is not a pure higher-moment or variance-only mechanism test.','',
'Complete density includes frozen head density, flow logdensity/logdet, -sum(ell), -logdetL, head standardization,33logQ and the FACTOR constant Jacobian. Tail coordinates span the full waveform; they are different from the time-tail220:400 classification window. The density gain cannot be assumed to repair that temporal region.','',
'## Training and selection','',
f'Single seed20261009151, Adam3e-4/weightdecay1e-5/batch128/clip5, float32 cuda:0. Actual{tr["actual_updates"]}/{tr["planned_updates"]} updates in{m["counters"]["training_cumulative_s"]:.6f}s. All599 validation rows at step0/every100 select the strictly smallest corresponding conditional NLL; exact ties retain the earliest checkpoint. No restarts, schedule, seed or structure changes and no DEV selection.','',
'| Checkpoint | Step | Conditional599 val NLL[nat/event] |','|---|---:|---:|',
f'| Initial | 0 | {tr["trace"][0]["conditional_NLL_nat_event"]:.6f} |',
f'| Selected | {tr["selected_step"]} | {tr["selected_conditional_NLL"]:.6f} |',
f'| Final | {tr["trace"][-1]["step"]} | {tr["trace"][-1]["conditional_NLL_nat_event"]:.6f} |','',
'Later validation worsens substantially and fluctuates; the full trace, selected/initial/final weights and final optimizer/random states are preserved. Extra updates are not hidden or claimed as improvement. CPUfloat64 final scores do not reselect the checkpoint.','',
'| Model | Full599 val NLL | Full21047 TRAIN NLL | Val minus TRAIN |','|---|---:|---:|---:|']
for n in names:
    g=m['train_validation_gap'][n];lines.append(f'| {n} | {g["val_NLL"]:.6f} | {g["TRAIN_NLL"]:.6f} | {g["val_minus_TRAIN_NLL"]:.6f} |')
lines+=['','Both models use identical data/coordinates and complete density terms. Validation improves by15.678913nat/event but the gap increases from8.19235 to20.04283. This is descriptive evidence of improved selected density with a larger generalization gap; it does not establish response equivalence.','',
'## Common grouped classification and paired effects','',
'All2000 candidate and REAL rows pass inherited QC. The same original R7 common pools supply600 classifier-training and250 held-out rows per domain, six frozen views,18 HGB fits. REAL-REAL is same-N TRAIN-reference versus DEV sampling, not a physical equivalence threshold.','',
'| View | Gaussian BA | Flow BA | REAL-REAL BA | Flow minus Gaussian BA | 500-group95%CI |','|---|---:|---:|---:|---:|---|']
for view in cfg['C2ST_views']:
    a=cs['results']['GAUSSIAN_BASE'][view];b=cs['results']['RESIDUAL_FLOW'][view];r=cs['results']['REAL_REAL'][view];d=cs['paired_contrasts']['RESIDUAL_FLOW_minus_GAUSSIAN_BASE'][view]
    lines.append(f'| {view} | {a["balanced_accuracy"]:.3f} | {b["balanced_accuracy"]:.3f} | {r["balanced_accuracy"]:.3f} | {d["BA_delta_first_minus_second"]:.3f} | {d["BA_delta_95CI"]} |')
lines+=['','Baseline and voltage BA intervals show improvement in this fixed contrast; shape/prompt/postpeak/tail intervals include zero. No familywise significance or equivalence claim is made. AUC/intervals and every view are retained in full/compact metrics. Intervals condition on these classifiers and artificial groups, not response retraining or independent acquisitions.','',
'## PSD support, observable changes and tails','',
'Both Gaussian and flow have qualified two-component fits only in bins1,2,6; REAL has all8. Pairwise common support is exactly[1,2,6] for peak means, raw widths and detrended widths. Bins3,4,5,7,8 fail the frozen convergence/support rules and remain undefined. Do not compare averages across different supports or promote fitted components into particle truth. The frozen q4 CAL-reference failure and q1 HC-neutron absence remain unmodified.','',
'| PSD component | Field | Common bins | Gaussian MAE | Flow MAE |','|---|---|---|---:|---:|']
for row in w['common_support_errors']:
    if row['scope']=='pair':lines.append(f'| {row["component"]} | {row["field"]} | {row["common_bin_ids_1based"]} | {row["MAE"]["GAUSSIAN_BASE"]:.8f} | {row["MAE"]["RESIDUAL_FLOW"]:.8f} |')
lines+=['','These errors change only slightly; the five unsupported regions and displaced/blurred charge-PSD structure remain. All new2000 outputs are finite, unclipped by ADC and QC-valid; no redraw, truncation or CFD recrop. Three negative-PSD events in each candidate remain outside the shared0..0.6 body view and visible in the full-range plot; REAL has zero outside. All panels share the complete charge range.','',
'Existing local waveform summaries (no added statistic calculation):','',
'| Domain | Baseline RMS median/max[mV] | cross50 mean/SD[index] | peak mean/SD[index] | Prompt curvature median[mV] | Postpeak curvature median[mV] |','|---|---|---|---|---:|---:|']
for n in ['REAL']+names:
    f=m['local'][n]['regions']['overall']['features'];a=f['baseline_RMS_V'];c=f['cross50_index'];p=f['peak_index'];pc=f['prompt_curvature_RMS_V'];tc=f['postpeak_curvature_RMS_V']
    lines.append(f'| {n} | {1000*a["quantiles"][3]:.6f}/{1000*a["quantiles"][-1]:.6f} | {c["mean"]:.6f}/{c["sample_sd"]:.6f} | {p["mean"]:.6f}/{p["sample_sd"]:.6f} | {1000*pc["quantiles"][3]:.6f} | {1000*tc["quantiles"][3]:.6f} |')
lines+=['','Baseline RMS median moves toward REAL, with a remaining extreme upper tail (flow max5.303mV versus REAL1.593mV). Baseline spectra decrease toward REAL but remain elevated, and ACF structure remains. Prompt curvature increases especially at q3/q4; prompt100:150 contains peak and early fall, not only the upstroke. Postpeak curvature is closer in point summaries without a clear postpeak classifier gain. Complete waveform quantile bands may look close while full/tail joint discrimination remains high.','',
'REAL cross50 spans118.9995..120.0049; flow spans114.5043..127.0849, similar to Gaussian114.4951..127.1213. Its narrow observed alignment support is not restored. Full cross50/phase/peak quantiles and quartile curve/RMS/ACF/rise summaries remain in metrics. These residuals do not by themselves prove a processing-chain or physical cause.','',
'Every new draw is queried in voltage and own-shape views against original2592 and expanded21047 references, with baseline/historic REAL summaries reused. Exact-copy counts are zero. Larger reference cardinality changes nearest distances; neither a shorter distance nor no exact copy proves or excludes memory. No retraining or reference selection is added.','',
'## Mathematics, replay and execution','',
'Step0 restores all599 complete validation scores and all2000 baseline pre-ADC waveforms with maximum difference0, post-ADC array_equal. Selected inverse error1.33e-15 and forward/inverse logdet error7.11e-15 at val0/199/399/598; the full446x446 fixed-head autodiff Jacobian logdet error at val0 is3.55e-15. These pass the predefined bounds without repeating the old representation audit.','',
f'Independent replay PID{replay["PID"]} reproduces2000 rows/960000 points: pre-ADC maxabs0 and bitwise true; post-ADC array_equal true. It loads actual saved head/innovations and selected weights, with no fit, new draws or complete evaluation rerun.','',
f'Remote numeric worker PID{native["history"][0]["worker_PID"]} and replay both exited0. Remote cumulative{native["numeric_cumulative_s"]:.6f}s; graphics-only label repair used native4c999a/PID53100/exit0,{6.6250923:.6f}s including launcher overhead, so conservative cumulative{native["numeric_cumulative_s"]+6.6250923:.6f}s/2400. Peak supervised remote RSS{max(r["RSS_peak_bytes"] for r in native["history"])}B/4GiB, GPU allocated{num["environment"]["GPU_allocated_peak_bytes"]}B/4GiB. Local plot-fix RSS was not sampled; no science is rerun to fill this unknown.','',
'One source-only read failure requested nonexistent old plots.py; correct plotting.py was then read. Normal return SCP succeeded, but safe extraction rejected pre-uploaded INPUT_BINDINGS.json before any extraction; the already transferred tar was unpacked while skipping this control, with no retransfer/science retry. Original failure/entry/tar and native codes are preserved. An inherited contrast legend in the original C2ST figure was corrected from saved aggregate JSON only; use c2st_paired_corrected.png. Original source/image are preserved. All eight key figures were actually viewed; none is claimed as distributional success from median curves.','',
'## Implication and reproduction','',
'Relaxing the conditional residual density helps baseline and selected complete likelihood, but this single architecture/budget leaves the joint time-tail and overall distributions far from REAL. The fixed head and alignment-support model remain candidate limits; this experiment does not isolate them as the sole cause. Do not respond by extending this fit or changing its selected point from DEV. ROOT will register any next controlled problem. The full project remains active, and target-density learning still precedes a validated G4 response mapping.','',
'See REPRODUCTION.md, config.json/PROTOCOL.md/READINESS.json/INPUT_BINDINGS.json, RESIDUAL_INITIAL_STATE/RESIDUAL_FLOW/RESIDUAL_FINAL_STATE, saved permutations/random/optimizer states, ACTUAL_GENERATION_INPUTS.npz, generated.npz, density/C2ST/NN private payloads, complete metrics/trace, mathematical/replay/native/resource/QA receipts and the final inventory. Only aggregate compact metrics are intended for Git.']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(ROOT/'REPRODUCTION.md').write_text(f'''# R15 reproduction

Actual remote entry: {cfg['remote_root']}/remote_supervise.py, interpreter {cfg['python']} on rylan@100.82.231.102. It ran numeric_worker.py then replay.py with one child at a time; actual PIDs/native statuses/timing are in REMOTE_NATIVE_RECEIPT and LOCAL_NATIVE_RECEIPT. Existing Python/NumPy/torch/CUDA versions and device are in metrics. No installation or migration-content equality test was used.

Inputs are readonly R14 {cfg['remote_R14_root']} and R10 {cfg['remote_input_root']}; INPUT_BINDINGS inherits exact prior versions without a repeated source hash. R14 TRAIN_ADDITION plus original processed2592 reconstruct the same21047 order; original599 val/2000 DEV are unchanged. R14 representation.npz/FLOW_EXPOSURES.pt/TAIL_EXPOSURES.pt/GENERATION_INPUTS/GMM_8 and saved baseline generated/scores remain frozen. No raw long waveform or EVAL is necessary.

For inference in a new authorized folder, copy current residual_flow.py/conditional_density.py/density.py/replay.py/config.json, selected RESIDUAL_FLOW.pt and ACTUAL_GENERATION_INPUTS.npz; point config remote_R14_root to the existing representation and frozen Gaussian tail. The saved actual x/tail_white/draw_ids suffice; replay never resamples a head. Independent2000-row replay already passed bitwise and is recorded in REPLAY_RECEIPT. A new machine/environment is not promised bitwise-identical without that scientific check.

From-scratch training uses seed20261009151 and saved actual permutations/initial state, fixed expanded rows and Gaussian preprocessing, single Adam path with step0/every100 selection. TRAINING.json retains all164 checkpoints. The optimizer and RESIDUAL_FINAL_STATE correspond to final16300, not selected1400; do not combine that optimizer state with selected weights as though it were a selected checkpoint. The initial/final random states are retained. Float32 GPU selection and CPUfloat64 report arithmetic are explicitly distinct; no score-based CPU reselection.

Figure-label recovery is plot_label_fix.py from saved aggregate metrics only. It is not part of fitting/replay; the initially executed plotting.py and original image remain available. The normal SCP return tar and control failure are preserved. Do not write into closed stages or rerun their entries for delivery.
''',encoding='utf8')
failure={'scientific_native_history':native['history'],'scientific_retries':0,'preparation_failures':cfg['preparation_failures'],'retrieval_failure':read('FAILURE_RETRIEVAL_CONTROL_001.json'),'retrieval_recovery_tool':'315bfc','retrieval_recovery_exit':0,'plot_label_failure_and_recovery':graphics,'plot_fix_native_tool':'4c999a','plot_fix_native_exit':0,'all_original_failures_kept':True,'new_metric_fit_for_recovery':0,'model_or_width_failures_not_retried':True}
write('FAILURE_RECEIPT.json',failure)
files=[f for f in ROOT.rglob('*') if f.is_file()];before=sum(f.stat().st_size for f in files)
resource={'stage':'R15','UTC':utc(),'closure_PID':os.getpid(),'executor_deadline_UTC':cfg['deadline_UTC'],'remote_numeric_cumulative_s':native['numeric_cumulative_s'],'graphics_receipt_elapsed_s':graphics['elapsed_s'],'graphics_conservative_native_wall_s':6.6250923,'numeric_cumulative_conservative_s':native['numeric_cumulative_s']+6.6250923,'numeric_cap_s':2400,'history':native['history'],'training_cumulative_s':m['counters']['training_cumulative_s'],'training_cap_s':900,'updates':16300,'updates_cap':16300,'CPU_threads':4,'remote_RSS_peak_bytes':max(r['RSS_peak_bytes'] for r in native['history']),'local_graphics_RSS_peak':'unknown/not sampled','RSS_cap_bytes':4294967296,'GPU':num['environment'],'GPU_allocated_cap_bytes':4294967296,'counters':m['counters'],'replay':replay['result'],'data_roles':m['data_roles'],'new_raw_long_waveform_read':0,'normal_upload_tar_bytes':(ROOT/'UPLOAD.tar').stat().st_size,'normal_return_tar_bytes':(ROOT/'REMOTE_RETURN.tar').stat().st_size,'migration_content_test':False,'source_input_rehash':0,'observed_local_files_before_accounting':len(files),'observed_local_bytes_before_accounting':before,'output_cap_bytes':268435456,'control_sources_added_after_science':['unpack_return.py','plot_label_fix.py','finalize.py'],'exact_system_dependency_IO':'unknown/not per-read audited','closure_scientific_decode_fit_generate_replay':0,'EVAL_read':0,'new_G4':0,'failure_costs_preserved':True}
if before+1024*1024>resource['output_cap_bytes']:raise RuntimeError('Insufficient reserved output for closure')
write('RESOURCE_RECEIPT.json',resource)
members=[{'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,'SHA256':sha(f)} for f in sorted(ROOT.rglob('*')) if f.is_file() and f.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
write('FINAL_MANIFEST.json',{'stage':'R15','UTC':utc(),'members':members,'member_count':len(members),'member_bytes':sum(r['bytes'] for r in members),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'purpose':'Own-stage inventory only, no source-destination migration equality test'})
write('FINAL_RECEIPT.json',{'stage':'R15','PID':os.getpid(),'start_UTC':start,'final_write_started_UTC':utc(),'status':'complete_scientific_and_control_delivery','manifest_SHA256':sha(ROOT/'FINAL_MANIFEST.json'),'manifest_members':len(members),'manifest_member_bytes':sum(r['bytes'] for r in members),'native_exit_code':'Actual caller tool result','numeric_native_exit_codes':[r['exit_code'] for r in native['history']],'no_further_stage_writes':True})
last=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps({'PID':os.getpid(),'UTC':utc(),'actual_last_write_UTC':last,'manifest_SHA256':sha(ROOT/'FINAL_MANIFEST.json'),'receipt_SHA256':sha(ROOT/'FINAL_RECEIPT.json'),'members':len(members),'member_bytes':sum(r['bytes'] for r in members)}))
