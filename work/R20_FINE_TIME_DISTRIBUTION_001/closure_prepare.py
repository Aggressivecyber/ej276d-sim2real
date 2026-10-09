from __future__ import annotations
import pathlib,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def load(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
figures={
 'sampling_prior.png':'Both arms use the same frozen R19 prior; no new sampling intervention. Axes, labels and ESS visible.',
 'charge_psd_source_composition.png':'Five domains share coordinates. Thin generated bands remain; FINE pure-had center shifts downward at middle charge. No cropped points or overlapping labels.',
 'conditional_psd_CDF.png':'All eight observed-charge bins retained, same axes. Full empirical CDF and counts readable; generated distributions remain narrow.',
 'conditional_psd_by_composition.png':'All 3x8 source-condition panels retained, fixed valley shown. Source composition is not a REAL particle label.',
 'energy_peak_width.png':'Density means/sigmas on common physical axes. Failed bin8 remains a gap; both new high components are too narrow.',
 'conditional_waveforms.png':'Median and 10–90% bands readable. Both new arms reduce some R19 postpeak repetitive spikes; source bins versus REAL own bins are explicitly different conditioning.',
 'noise_source_controls.png':'Correct source/noise reference semantics shown. New ACF distributions are farther from REAL despite similar baseline RMS.',
 'training_c2st.png':'Common J20 and old J19 trajectories, selection and seven C2ST views displayed; near-unity scores retained.',
 'fine_loss_trajectory.png':'Existing validation JSON only. All three fine terms plus J20/J19 shown; selected stars agree with CONT500/FINE750 and final rebound retained.'}
qa={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'method':'Actual view_image inspection of all nine saved figures, after rendering; no numerical rerun.',
    'all_eight_actually_viewed':True,'all_nine_actually_viewed':True,
    'figures':[{'path':'results/figures/'+n,'actual_viewed':True,'layout_pass':True,'observation':s}for n,s in figures.items()],
    'scientific_success_inferred_from_layout':False}
assert all((ROOT/r['path']).exists()for r in qa['figures'])
write('FIGURE_QA.json',qa)
native={'PID':os.getpid(),'UTC':qa['UTC'],'tools':[
 {'tool':'0dd4c7','PID':17888,'native_exit':0,'role':'first source-only preparation'},
 {'tool':'b55e96','PID':62232,'native_exit':0,'role':'normal deployment'},
 {'tool':'e6d20a -> 4194c8','local_bridge_PID':56356,'SSH_PID':28500,'remote_supervisor_PID':211442,'remote_main_PID':211443,'remote_replay_PID':212360,'native_exit':0,'role':'main and standalone replay, sequential'},
 {'tool':'9bb3b0','PID':36040,'native_exit':0,'role':'normal result retrieval'},
 {'tool':'8c55a7','PID':'unknown','native_exit':1,'role':'control inventory wrong plots path','elapsed_s':.6614258},
 {'tool':'8bcef5','PID':'unknown','native_exit':1,'role':'helper import psutil failed before worker spawn','elapsed_s':.4821916},
 {'tool':'d4c466','supervisor_PID':10700,'worker_PID':38260,'native_exit':1,'role':'existing-result helper failed at validation-trace plot; native/logs/source preserved','elapsed_s':2.427184499974828},
 {'tool':'3cd998','supervisor_PID':35996,'worker_PID':36420,'native_exit':0,'role':'same-result summary and plot; no inference/fit/generation','elapsed_s':2.0745500000193715},
 {'tool':'8f83b2','PID':29816,'native_exit':0,'role':'pure JSON report assembly','elapsed_s':.5521434}],
 'no_science_process_still_running':'Main and standalone replay exited0, local summary children exited; no duplicate submissions.',
 'tool_output_exit_source':'Actual exec_command tool receipts, not predicted by scripts.'}
write('KNOWN_NATIVE_TOOL_RECEIPTS.json',native)
res=load('RESOURCE_RECEIPT.json')
res['post_result_scope_note']='The frozen PROTOCOL short closure sentence says small-JSON summary. The task explicitly also requires same-parent waveform variation, omitted by original main summary. This was completed using its saved repeat outputs (983040 values each attempt), no new source/model/fit, failed and successful helper time counted. Primary metrics.json remains unchanged.'
res['report_assembly_native']={'tool_chunk_id':'8f83b2','PID':29816,'native_exit':0,'elapsed_s':.5521434}
res['independent_replay_output_rows']=4000
res['pure_byte_manifest_is_not_a_transfer_test']=True
write('RESOURCE_RECEIPT.json',res)
repeat=load('REPEAT_WAVEFORM_SUMMARY.json')
for a,r in repeat['arms'].items():
    assert sum(v['full480']['N']for v in r['by_composition'].values())==128
(ROOT/'REPRODUCTION.md').write_text('''# R20 reproduction

Executed server directory: `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R20_REMOTE_20261009_084753`.
Interpreter: `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`. The actual version block is in `metrics.json`; training used cuda:0, CPU numerical threads4. Libraries/environment were not changed.

Actual entry: `python -B remote_supervise.py` from that directory. The supervisor started `numeric_worker.py` (PID211443/native0) and then `replay.py` (PID212360/native0), recorded in `REMOTE_NATIVE_RECEIPT.json`. Selected models: `CONT/TRANSLATOR.pt` step500 and `FINE/TRANSLATOR.pt` step750. Final step4000 models and optimizer/random states are retained separately. Whole science was one sequential supervised job, not concurrent fits.

Inputs are referenced by `config.json`: readonly R10 inputs, R14 TRAIN_ADDITION, R18 initial critic/internal validation and corrected repeat/probe definitions, and R19/TILT selected G/frozen sampling lambdas/actual generation inputs. Full480 source waveforms, REAL TRAIN21047 and val599, original2000 DEV parents are kept distinct. CAL/EVAL waveforms were not used. TRAIN fine constants/projections were saved before updates as `FINE_CONSTANTS.npz`; seed20261009201 is isolated from all source/target/noise/NCE streams. Complete actual streams/RNG states remain private in each arm. Initial G/D and six actual stream hashes agree across arms and the R19 TILT streams.

For a fresh inference-only replay, copy `model.py`, `replay.py`, `config.json`, `ACTUAL_GENERATION_INPUTS.npz`, both selected `TRANSLATOR.pt` and each arm's saved `generated.npz` into a NEW directory with the same two arm subdirectories. Set E/D temp/cache locally, or a fresh server directory, and use the recorded environment to run `python -B replay.py`. Do not overwrite the sealed stage or copy its one-use execution token into a new training launch. The replay only loads saved source/z/white and selected weights; it does not read REAL/TRAIN or fit again. The already executed fresh replay compared4000 rows/1920000 samples pre- and post-ADC, both bitwise identical, maxabs0, IDs equal. This is scientific saved-model replay, not migration equality testing.

Full training reproduction needs the readonly dependencies at their config paths, the frozen protocol/config and the listed executed code from READINESS.json, with one new runtime directory and an owned supervisor. The actual inference and old reference results were compared in the recorded environment; another GPU/software environment need not be bitwise identical. SOURCE/G4 clean curves contain only total deposits, no recovered microscopic timing. REAL density components and tilted soft prior are not particle truth. This is one historical acquisition's exposed TRAIN/DEV diagnostic, not external validation.

`saved_result_summary.py` only summarizes already-saved128x8 waveforms and plots existing validation JSON; failed and repaired control/helper runs are retained. `postprocess.py` and `closure_prepare.py` assemble existing JSON without new inference/fit. `METRICS_COMPACT.json` excludes event identities, predictions and training/repeated-parent lists; the full private arrays, models and inputs remain on E/D. FINAL_MANIFEST is one artifact seal, not a transfer equality test.
''','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'control_only':True,'nine_figures_actually_inspected':True,'new_fit':0,'new_generation':0}),flush=True)
