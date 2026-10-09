from __future__ import annotations
import datetime,hashlib,json,os,pathlib,sys,time,re
ROOT=pathlib.Path(__file__).resolve().parent
DEADLINE=datetime.datetime.fromisoformat('2026-10-09T10:41:48.905481+00:00')
START=time.monotonic()
def utc():return datetime.datetime.now(datetime.timezone.utc)
def check():
    if utc()>=DEADLINE:raise TimeoutError('R21 executor seal deadline reached; no further writes')
def write(name,value):
    check();(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
def text(name,value):
    check();(ROOT/name).write_text(value,'utf8')
def read(name):return json.loads((ROOT/name).read_text('utf8'))
def fs(value):return 'undefined' if value is None else f'{value:.8g}'
def compact(metrics):
    out={k:metrics[k]for k in ['status','reference_training_reused','data_roles','domains','domain_vs_REAL','local','PSD_main','energy_conditional_width','source_composition','source_association','C2ST','generation','pairing','counters','versions']}
    tr=metrics['training']['GLOBAL_STYLE']
    out['training']={'GLOBAL_STYLE':{k:v for k,v in tr.items()if k!='validation_trace'}}
    out['training_trace_reference']='GLOBAL_STYLE/TRAINING_TRACE.json (aggregated per-step validation, no event list)'
    rep=metrics['random_decomposition']
    out['random_decomposition']={k:v for k,v in rep.items()if k!='groups'}
    out['random_decomposition']['groups']={k:{a:b for a,b in v.items()if a!='rows'}for k,v in rep['groups'].items()}
    out['source_probes']={k:v for k,v in metrics['source_probes'].items()if k!='fixed_source_parent_index'}
    out['publication_scope']='Aggregated summaries only; actual identities, predictions, streams and per-parent repeats stay in local private artifacts.'
    return out
def assemble():
    m=read('metrics.json');native=read('LOCAL_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json')
    assert native['status']=='complete' and replay['pass']
    tr=m['training']['GLOBAL_STYLE'];counter=dict(m['counters'])
    label_qa=read('PLOT_LABEL_QA_RECEIPT.json')
    numerical_total=native['numeric_cumulative_s']+2.3245094
    failures=[]
    for folder in ['failed_001','failed_002']:
        receipts=[read(str(p.relative_to(ROOT)))for p in (ROOT/folder).glob('FAILURE_*.json')]
        unique={r['PID']:r for r in receipts}
        for r in unique.values():
            if any(f['PID']==r['PID']for f in failures):continue
            failures.append(r)
    combined=dict(counter)
    for r in failures:
        for k,v in r['counters'].items():
            if isinstance(v,(int,float)):combined[k]=combined.get(k,0)+v
    assert combined['D_updates']<=4000 and combined['G_updates']<=4000 and combined['structural_check_rows']<=128
    assert combined['HGB_attempts']<=12 and combined['density_EM_attempts']<=8
    assert numerical_total<=1800
    peak=max(r['RSS_peak_bytes']for r in native['history']);assert peak<=4*1024**3
    write('METRICS_COMPACT.json',compact(m))
    details={'PID':os.getpid(),'UTC':utc().isoformat(),'scientific_native_history':native['history'],
      'numeric_cumulative_s':numerical_total,'numeric_limit_s':1800,
      'main_supervised_cumulative_s':native['numeric_cumulative_s'],
      'saved_plot_label_QA_native_wall_s':2.3245094,'saved_plot_label_QA_native_exit_code':1,
      'saved_plot_label_QA_RSS_peak_bytes':None,
      'saved_plot_label_QA_failure':'Figure completed; post-render GetProcessMemoryInfo structure omitted one field, making the RSS probe unavailable and final assertion fail. Native1 preserved; no rerender or model rerun to fill RSS.',
      'training_success_s':tr['training_s'],'failed_training_conservative_upper_s':10.956509799987543,
      'training_total_conservative_upper_s':tr['training_s']+10.956509799987543,'training_limit_s':600,
      'counters_successful_run':counter,'counters_including_failures':combined,
      'failed_scientific_attempts':[{'PID':r['PID'],'error':r['error'],'counters':r['counters']}for r in failures],
      'RSS_peak_bytes':peak,'RSS_probe_failures':sum(r['RSS_probe_failures']for r in native['history']),
      'GPU_allocated_peak_bytes':0,'CPU_threads':4,'environment':native['environment'],
      'versions':m['versions'],'SSH_calls':0,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4':0,
      'installation':0,'migration_integrity_test':False,'source_hash_reaudit':False,
      'post_scientific_closure':'JSON summaries and pure-byte final artifact hashes only; no response fitting/inference/data-array decoding.',
      'config_metadata_notes':{'python':'Retained server path is historical; actual interpreter is in LOCAL_NATIVE_RECEIPT.',
        'absolute_deadline_UTC':'Retained old R20 field is unused by executed local clocks; deadline_UTC and local addendum set the actual R21 deadline.'}}
    write('RESOURCE_RECEIPT.json',details)
    cs=m['C2ST'];sc=m['source_composition']['primary_pure_had_source_bins3_4']
    d=m['domain_vs_REAL'];body=[]
    body.append('# R21: global stochastic signal response\n')
    body.append('GLOBAL_STYLE produces measurable stochastic signal changes and restores the prespecified middle-charge pure-had centers relative to the saved R20/FINE result. Baseline ACF1 W1 falls from 0.338123 to 0.215922, and Q/PSD HGB BA falls from 0.844 to 0.788. The complete response remains unsuccessful: full shape BA is 0.992, all temporal/local voltage views remain highly separable, and both density-component conditional widths remain too narrow. Different hardware and update counts limit attribution to the new package.\n')
    body.append('This stage keeps the complete G4-derived 480-point source waveform and all original 2000 DEV parents. It adds one eight-dimensional style modulation plus latent-recovery package to the R19/TILT starting translator, while reusing the R20/FINE frozen source sampling, fine-time losses and actual old random streams. Signal and electronic labels identify network branches; neither is established physical noise truth or a conditional expectation.\n')
    body.append(f"Local CPU execution completed {tr['updates']} G/D/reader updates; selected step {tr['selected_step']}. J20 initial/selected/final = {fs(tr['initial_validation']['J'])} / {fs(tr['selected_validation']['J'])} / {fs(tr['final_validation']['J'])}. Validation latent L1 initial/selected/final = {fs(tr['initial_validation']['style_reconstruction_L1'])} / {fs(tr['selected_validation']['style_reconstruction_L1'])} / {fs(tr['final_validation']['style_reconstruction_L1'])}. Model selection used only fixed internal TRAIN validation. Latent error reduction alone is not response realism.\n")
    body.append('The saved R20/FINE reference used 4000 GPU updates with different Torch/NumPy/sklearn versions. The new local completed old-stream prefix matches are '+json.dumps(tr['old_R20_FINE_completed_prefix_match'])+'. Full equal-update matching is '+str(tr['updates']==4000)+'. Initial J20 difference from that historical environment = '+fs(tr['initial_J20_abs_difference'])+'. This is an environment/update-limited historical comparison, not a strictly isolated equal-compute ablation.\n')
    body.append('## Complete waveform and PSD\n')
    body.append('| Fixed HGB view | R20/FINE BA | GLOBAL_STYLE BA |\n|---|---:|---:|\n')
    for view,new in cs['results']['GLOBAL_STYLE'].items():body.append(f"| {view} | {fs(cs['reference_results_reused']['R20_FINE'][view]['balanced_accuracy'])} | {fs(new['balanced_accuracy'])} |\n")
    body.append('\nThe classifiers use the same saved DEV holdout parents/artificial REAL groups. Paired bootstrap intervals condition on the fitted classifiers and do not include response-training or acquisition uncertainty. A score near one retains a large domain gap; a lower training loss or attractive mean curve does not establish success.\n')
    body.append('| Domain | overall PSD W1 | mean of all 8 supported PSD W1 | baseline ACF1 W1 |\n|---|---:|---:|---:|\n')
    for name in ['SOURCE_CLEAN','R19_TILT','R20_FINE','GLOBAL_STYLE']:
        acf=d[name]['overall']['ACF1']['W1']
        body.append(f"| {name} | {fs(d[name]['overall']['PSD']['W1'])} | {fs(m['PSD_main']['vs_REAL'][name]['mean8_W1'])} | {fs(acf)} |\n")
    body.append('\nEvery eight-bin count, center/raw width/detrended width and failure remains in METRICS_COMPACT. Density components are descriptions, not incoming-particle ground truth. Width errors only use the same mutually qualified bins; unsupported bins cannot be counted as passing. The old q4 reference nonconvergence remains unchanged; no fallback or refit repairs it.\n')
    body.append('On bins1-4, jointly qualified across REAL/SOURCE_CLEAN/R19_TILT/R20_FINE/GLOBAL_STYLE, upper-component center MAE is 0.009086 versus FINE 0.028157, and raw-sigma MAE is 0.038965 versus 0.042385. Lower-component raw-sigma MAE is 0.018849 versus 0.018497, so width improvement is not uniform. GLOBAL_STYLE and both historical translated references fail bin8 two-component qualification; REAL remains qualified in all8. The Q/PSD BA paired delta is -0.056, 95% cluster-bootstrap interval [-0.091159,-0.022025]; full-shape delta -0.004 has interval [-0.013918,0.005849]. These are fixed-classifier descriptive comparisons, not an acquisition-independent equivalence test.\n')
    body.append('## Prespecified pure-had source bins\n\n| Domain | bin 3 output PSD mean | bin 3 valley fraction | bin 4 output PSD mean | bin 4 valley fraction |\n|---|---:|---:|---:|---:|\n')
    for name,row in sc.items():body.append(f"| {name} | {fs(row['bin3']['output_PSD']['mean'])} | {fs(row['bin3']['valley_fraction'])} | {fs(row['bin4']['output_PSD']['mean'])} | {fs(row['bin4']['valley_fraction'])} |\n")
    body.append('\nThese are source-charge-conditioned pure-deposition groups, not a classification of REAL or incoming PDG. Output-conditioned bins, original source mixture, PDG inference and all source/output charge migration are reported separately. Source conditioning is required to avoid hiding displacement by output-charge selection.\n')
    body.append('## Actual random response and source controls\n\n')
    body.append('The original 128 parents ×8 joint draws plus 32 fixed parents ×32 style-only draws and the same 32×32 electronic-only draws total 3072 outputs. First decomposition draws share an actual joint reference. Variance uses n−1; waveform RMS SD is the square root of mean point variance. Repeated outputs do not increase independent G4 parents.\n')
    body.append('| Mode | composition | parents | mean PSD sample SD | output RMS SD [mV] | signal RMS SD [mV] | electronic RMS SD [mV] | parents straddling frozen valley |\n|---|---|---:|---:|---:|---:|---:|---:|\n')
    for mode,row in m['random_decomposition']['groups'].items():
        for kind,s in row['by_composition'].items():
            wave=s['within_parent_waveform_RMS_V']
            body.append(f"| {mode} | {kind} | {s['parents']} | {fs(s['within_parent_PSD_SD']['mean'])} | {fs(1000*wave['output_post_ADC']['full480']['mean'])} | {fs(1000*wave['signal_pre_ADC']['full480']['mean'])} | {fs(1000*wave['electronic_pre_ADC']['full480']['mean'])} | {s['parents_straddling_valley']} |\n")
    body.append('\nStyle affects the signal branch and can indirectly change the old electronic branch through modulated features. Random variation or recoverable latent codes may still be texture or class splitting. Conversely, zero same-parent valley straddling does not validate the pure-group centers. There is no repeated-measurement REAL truth for these same G4 parents.\n')
    body.append('The existing corrected 832 source probes keep actual source/noise/reference parents distinct. Fixed-source changes are measured relative to that fixed source reference. Same-input/same-all-noise bitwise check = '+str(m['source_probes']['same_input_same_noise_bitwise'])+'. Nonzero source sensitivity is not calibrated physical correspondence. The new modulation gradient was finite and nonzero; 96 pre-update structural rows passed and were reused after the indexing-only engineering repair.\n')
    body.append('## Execution, failures and reproducibility\n\n')
    body.append(f"Two terminal engineering failures occurred before successful training: frozen-table schema KeyError (G/D 0/0; {native['history'][0]['elapsed_s']:.6f}s) and uint16 NCE index incompatibility (G/D 0/1; {native['history'][1]['elapsed_s']:.6f}s). Their source/log/native artifacts remain in failed_001 and failed_002. The fixes preserve lambda values and actual NCE positions. One failed D update and the completed structural rows remain charged; the new update cap is 3999 and training time deducts the second failure's conservative wall. Numerical cumulative wall including saved-result title QA = {numerical_total:.6f}s; monitored main/replay/failed-worker peak RSS = {peak} bytes; CPU4/GPU0.\n")
    body.append('A later saved-PSD-only rendering clarified the gray valley caption from ambiguous own-Q to source-Q, retaining the original image. Its figure completed correctly, but an incomplete Windows memory-counter structure made the post-render RSS check fail (native1). The original receipt and 2.3245094s conservative tool wall remain charged; RSS for that small helper is unknown. It performed no model generation, feature computation or fit and was not rerun to repair the instrumentation.\n')
    body.append('Independent saved-model replay used only the selected translator and actual source/style/z/white: 2000 rows, 960000 points, pre/post ADC and parent identities '+('bitwise equal.'if replay['pass']else'FAILED.')+' No latent reader, REAL/TRAIN or fitting was used. All execution/cache/data writes stay E/D. SSH, installation, EVAL/new G4/raw-long decoding and new CAL-waveform decoding are zero. Config retains historical server Python/old absolute-deadline metadata; actual local native receipt and R21 local addendum determine execution.\n')
    body.append('REAL comes from one acquisition; groups of 50 are artificial bookkeeping, not independent runs. This historically exposed DEV diagnostic is not an external blind test. G4 clean waveforms encode total EE/had deposition via the existing analytic recipe, not microscopic deposition timing. The learned unpaired mapping and soft source sampling assumptions do not identify a unique per-event detector response.\n')
    body.append('Necessary weights, optimizer/end states, actual inputs/streams, full metrics, decomposed waveforms, source controls and classifier predictions remain in this local stage. REPRODUCTION.md gives the entry and required existing inputs; FIGURE_QA.json records actual visual inspection. The next scientific direction is for ROOT to register after reviewing the residuals; no extra arm, sweep or fit was added.\n')
    text('REPORT.md',re.sub(r'\|\n{2,}(?=\|)','|\n','\n'.join(body)))
    text('REPRODUCTION.md',f"""# R21 local reproduction

Actual interpreter: `{native['environment']['python']}`. Torch/NumPy/sklearn versions are in RESOURCE_RECEIPT. The owned numerical entry was `local_supervise.py`, cwd this stage, with E-drive TEMP/TMP and bytecode disabled; it launches numeric_worker then replay sequentially. Existing server preparation helpers were never deployed or run.

For saved-weight inference in a **new E/D directory**, copy model.py/legacy_model.py, replay.py, config.json, GLOBAL_STYLE/TRANSLATOR.pt, GLOBAL_STYLE/generated.npz and ACTUAL_GENERATION_INPUTS.npz preserving paths. Set TEMP/TMP/TMPDIR and library caches to that new E/D directory; set OMP/MKL/OPENBLAS/NUMEXPR/BLIS threads=4, PYTHONDONTWRITEBYTECODE=1 and PYTHONUTF8=1. Run this existing Python with `-B replay.py`. This exact standalone replay was already executed: PID {replay['PID']}, rows 2000/points 960000, pre/post ADC bitwise pass={replay['pass']}. Inference needs neither StyleReader nor REAL/TRAIN.

Full retraining also needs the readonly local R2 G4 cache, R14 TRAIN addition/index map, P1K first5200 waveform/metadata payload, R3B generation metadata, frozen R18 initial critic/validation tensors, R19 TILT checkpoint and R20 actual training/fine constants. Exact paths are retained in config.json and LOCAL_PREPARATION_RECEIPT. No EVAL or SSH is needed. The current admission token belongs to a completed job; do not rerun it in the closed directory. Preserve and adapt token/failed-cost handling in a new stage if a future registered task authorizes training.

This is not a strict CPU/GPU equal-update comparison with R20/FINE. Actual completed prefixes and platform/update limitations are in TRAINING_RECEIPT and PAIRING_RECEIPT. The original science and both engineering failures remain unchanged.
""")
    write('ASSEMBLY_NATIVE_RECEIPT.json',{'PID':os.getpid(),'UTC':utc().isoformat(),'elapsed_s':time.monotonic()-START,'operation':'pure JSON report/compact/resource/reproduction','scientific_array_decode':0,'fit':0,'generation':0})
    print(json.dumps({'PID':os.getpid(),'UTC':utc().isoformat(),'status':'assembled','updates':tr['updates'],'selected_step':tr['selected_step'],'numeric_cumulative_s':numerical_total}))
def seal():
    assert read('FIGURE_QA.json')['all_inspected']
    resource=read('RESOURCE_RECEIPT.json')
    excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
    files=[p for p in sorted(ROOT.rglob('*'))if p.is_file()and p.name not in excluded]
    current=sum(p.stat().st_size for p in files)
    assert current<268435456-65536
    resource['stage_bytes_before_manifest_and_seal']=current
    resource['output_limit_bytes']=268435456
    resource['seal_hash_is_pure_bytes_no_arrays']=True
    write('RESOURCE_RECEIPT.json',resource)
    rows=[]
    for p in files:
        check();h=hashlib.sha256();n=0
        with p.open('rb')as f:
            while chunk:=f.read(1024*1024):h.update(chunk);n+=len(chunk)
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':n,'sha256':h.hexdigest()})
    manifest={'UTC':utc().isoformat(),'PID':os.getpid(),'members':rows,'member_count':len(rows),'member_bytes':sum(r['bytes']for r in rows),'self_reference_exclusions':sorted(excluded),'scientific_arrays_decoded_for_seal':0}
    write('FINAL_MANIFEST.json',manifest)
    b=(ROOT/'FINAL_MANIFEST.json').read_bytes()
    receipt={'PID':os.getpid(),'UTC':utc().isoformat(),'status':'sealed','manifest_sha256':hashlib.sha256(b).hexdigest(),'member_count':len(rows),'member_bytes':manifest['member_bytes'],'numeric_cumulative_s':resource['numeric_cumulative_s'],'entry':'close_stage.py seal','scientific_array_decode':0,'fits':0,'generation':0,'elapsed_s':time.monotonic()-START,'deadline_UTC':DEADLINE.isoformat(),'no_further_stage_writes':True}
    write('FINAL_RECEIPT.json',receipt)
    receipt['last_write_UTC']=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat()
    receipt['total_stage_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
    print(json.dumps(receipt));assert receipt['total_stage_bytes']<=268435456
if __name__=='__main__':
    check()
    if sys.argv[1]=='assemble':assemble()
    elif sys.argv[1]=='seal':seal()
    else:raise ValueError('assemble or seal')
