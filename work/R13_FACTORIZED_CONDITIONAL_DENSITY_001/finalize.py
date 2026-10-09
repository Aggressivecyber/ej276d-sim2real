"""Control-only R13 closure. No array decoding, inference, fitting or transfer tests."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def now():
    return datetime.now(timezone.utc).isoformat()
def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def dump(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

start = now()
m = load('metrics.json')
native = load('REMOTE_NATIVE_RECEIPT.json')
numeric = load('NUMERIC_RECEIPT.json')
replay = load('REPLAY_RECEIPT.json')
transfer = load('RETURN_TRANSFER_RECEIPT.json')
compact = {k:m[k] for k in ['status','validation','mathematical_checks','pairing','domains','domain_vs_REAL','local','C2ST','nearest_neighbours','counters','versions','environment','elapsed_s','EVAL_read','same_acquisition_development','target_density_not_G4_mapping']}
compact['training'] = {}
for kind,t in m['training'].items():
    summary = {k:v for k,v in t.items() if k not in ('trace','random_index_stream')}
    summary['initial_validation_NLL'] = t['trace'][0]['NLL_nat_event']
    summary['final_validation_NLL'] = t['trace'][-1]['NLL_nat_event']
    compact['training'][kind] = summary
width = m['energy_conditional_width']
compact['energy_conditional_width'] = {k:v for k,v in width.items() if k != 'tables'}
compact['energy_conditional_width']['tables'] = {}
for name,table in width['tables'].items():
    compact['energy_conditional_width']['tables'][name] = {
        k:([{kk:vv for kk,vv in row.items() if kk not in ('updates','trace')} for row in v] if k=='slices' else v)
        for k,v in table.items()
    }
compact['publication_scope'] = 'Aggregate metrics only. Event IDs, predictions, nearest indices, full fit/training trajectories and model payloads remain in local full artifacts.'
dump('METRICS_COMPACT.json', compact)

qa = {
    'status':'eight_figures_visually_inspected', 'UTC':now(), 'reviewer':'sole executor',
    'figures':[
        {'path':'results/figures/charge_psd_full.png','finding':'Shared full limits preserve negative PSD outliers; six domains and QC counts readable.'},
        {'path':'results/figures/charge_psd_main_body.png','finding':'Shared PSD 0 to 0.6 zoom, all six outside counts present; retains original full-range companion.'},
        {'path':'results/figures/c2st_paired.png','finding':'Six views, RR sampling reference, four paired differences and 95% intervals readable.'},
        {'path':'results/figures/training_likelihood.png','finding':'Both early selected checkpoints and later validation deterioration visible; all complete NLL values shown.'},
        {'path':'results/figures/noise_and_curvature.png','finding':'Baseline PSD/ACF/RMS and conditional curvature contrasts visible; prompt includes peak/early fall.'},
        {'path':'results/figures/conditional_waveforms.png','finding':'All four output charge regions, V and own-shape bands, common time axes and counts readable.'},
        {'path':'results/figures/energy_peak_width.png','finding':'All eight frozen bins retained; unsupported components marked with crosses, no gaps hidden.'},
        {'path':'results/figures/nearest_neighbours.png','finding':'All 2000 new queries and old references readable; no-copy versus no-memory distinction labeled.'}
    ], 'redraws':0, 'new_scientific_calculation_for_QA':False
}
dump('FIGURE_QA.json', qa)
dump('FAILURE_RECEIPT.json', {
    'scientific_jobs':[{'PID':h['worker_PID'],'entry':h['entry'],'exit_code':h['exit_code'],'elapsed_s':h['elapsed_s'],'termination_reason':h['termination_reason']} for h in native['history']],
    'scientific_failures':0, 'scientific_retries':0,
    'retained_limitations':['Two-component descriptions often fail convergence/separation/support. These are scientific failures, not grounds to rerun.', 'Both network validation trajectories worsen after their selected early checkpoints.'],
    'control_failure_status':'No failed preparatory command identified in recorded R13 tool receipts; this finalizer is not a scientific job.'
})

report = '''# R13 factorized conditional density

The conditional residual network improves held-out continuous likelihood substantially, but none of the four factorial arms reproduces the complete REAL waveform distribution. Increasing density flexibility at the head does not resolve the remaining domain difference. This is a same-acquisition TRAIN/DEV diagnostic of target density; it is not a validated G4-to-REAL event response.

## Fixed design and actual execution

The frozen FACTOR representation, PCA basis/scales, ridge conditional mean and R12 empirical residual Cholesky remain unchanged. There are 2592 TRAIN-fit records, 599 complete internal validation records, and the original 2000 DEV records. CAL boundaries/references are reused as control metadata; CAL waveform numeric decoding and EVAL reading are zero. No new PCA, GMM or ridge fitting occurred.

An eight-coupling RealNVP head and a head-conditioned residual mean/log-scale network each ran exactly 2000 Adam updates, batch128, lr0.0003, weight decay0.00001 and clip norm5, with fixed seeds 20261009131/20261009132. Initial states, random states, all 21 validation checkpoints and selected weights are saved. RealNVP is an established invertible density construction, not an algorithmic novelty claimed here ([original paper](https://arxiv.org/abs/1605.08803)).

The two networks were trained on RTX5070 cuda:0; final likelihood and generation use saved selected weights on CPU float64. Worker161998 ran from 04:37:00.204663 to 04:37:41.259889 UTC with native exit0. Independent replay162395 also exited0. Total numeric time is 42.658897 s, training counter20.894185 s; four CPU threads, peak RSS2590203904 B, GPU allocated peak80701952 B. No installation or migration content consistency test was performed.

## Training and complete validation

| Network | Initial val NLL | Best val NLL | Selected step | Final step2000 NLL |
|---|---:|---:|---:|---:|
| Head | 47.403652 | 34.034595 | 900 | 40.007610 |
| Conditional residual | −2260.954346 | −2287.472656 | 200 | −2276.621582 |

Both validation trajectories deteriorate after the selected early checkpoint. The head is particularly worse by the final checkpoint. These trajectories do not establish training insufficiency; the following larger-TRAIN experiment can test sample-size sensitivity without altering this baseline.

All599 validation events are legal. Complete480-dimensional NLL includes the invertible-representation Jacobian and was recomputed in CPU float64 from selected weights:

| Arm | Complete NLL, nat/event | DEV full-shape BA | DEV full-voltage BA |
|---|---:|---:|---:|
| GMM_GLOBAL | −2371.774569 | 0.924 | 0.976 |
| FLOW_GLOBAL | −2371.917438 | 0.900 | 0.980 |
| GMM_CONDITIONAL | −2398.293033 | 0.926 | 0.978 |
| FLOW_CONDITIONAL | −2398.435901 | 0.932 | 0.980 |
| Historical FACTOR_LW | −2373.852796 | 0.944 | 0.986 |
| Same-N REAL–REAL | — | 0.512 | 0.546 |

The selected flow head adds only0.142869 nat/event, while conditioning the residual adds26.518464 nat/event. Better likelihood does not translate into full waveform agreement. All six domains retain2000 QC-valid records. C2ST uses the same600/250 grouped training/test positions across all candidates. The REAL–REAL result is a sampling reference, not an equivalence threshold.

## Paired effects and remaining differences

For complete normalized shape, paired BA differences (new minus comparator) are: conditional residual on GMM +0.002, 95%CI[−0.015128,0.020191]; conditional residual on FLOW +0.032, CI[0.007963,0.057066]; FLOW replacing GMM with GLOBAL residual −0.024, CI[−0.052968,0.005124]; FLOW replacing GMM with CONDITIONAL residual +0.006, CI[−0.019173,0.030308]. The intervals use500 artificial-group bootstrap replicates conditional on fitted classifiers; they do not include network retraining uncertainty or independent acquisition uncertainty. No multiplicity-adjusted mechanism claim is made.

Conditional residual changes improve high-charge postpeak curvature relative to the GLOBAL residual, but the q4 prompt second-difference median rises to about8.5 mV versus REAL6.7 mV. The prompt100:150 window includes the peak and early fall, so this is not exclusively a rising-edge statement. Baseline RMS means are REAL0.6464 mV, GMM_GLOBAL0.7253, FLOW_GLOBAL0.7239, GMM_CONDITIONAL0.7138 and FLOW_CONDITIONAL0.7112; baseline spectral/ACF discrepancies remain. Similar conditional median curves are insufficient to establish distributional equality.

Two-component descriptions have no qualified bins for GMM_GLOBAL or GMM_CONDITIONAL; FLOW_GLOBAL and FLOW_CONDITIONAL qualify only bin7. Four-arm common support is empty, so four-way peak-position/width errors are undefined. Unsupported bins remain on the figure/table with failure reasons, including nonconvergence, nonseparation, support below20 and variance-floor contact. No fallback fit or cut change is used. Density components and high-confidence reference classes are not particle truth. The inherited q4 reference failure and q1 high-confidence neutron absence are preserved.

The shared innovation protocol keeps the actual R11 white draws/component positions. Residual contrasts share the same head by construction; FLOW versus GMM contrasts are paired by draw position, not the same physical head or an independent G4 parent. All2000 queries of each new arm were compared with2592 TRAIN-fit records; no exact nearest-waveform copy was found. This is not proof of no memorization.

## Mathematical checks and reproducibility

Flow inverse maximum error4.44e−16, forward/inverse logdet cancellation7.11e−15, and full34-dimensional autodiff logdet check0. Initial conditional residual identity matches the old Gaussian density within4.55e−13 nat; final full446-dimensional covariance-form comparison agrees within2.28e−12 nat. Frozen conditional covariance orientation is still L diag(exp(2ell)) Lᵀ; it cannot express arbitrary head-dependent covariance rotations.

The fresh replay process restored saved weights and actual innovations and reproduced6000 generated rows /2880000 points for all three new arms exactly, before and after ADC (maxabs0, bitwise pass). It did not refit or resample. Raw waveforms, private features/predictions/nearest indices and full traces remain local; METRICS_COMPACT.json contains aggregate publication results. All eight delivered figures were visually inspected, without additional fitting or metric evaluation.

The full invertible representation does not guarantee adequate head/tail density. This target-density experiment does not attribute the remaining discrepancy to G4, establish an event-level response, repair missing independent labels/runs, or create a new blind test. The next registered sample-size comparison keeps these results immutable.

## Artifacts

See METRICS_COMPACT.json and full metrics.json; selected FLOW.pt/TAIL.pt; three arm generated.npz files; complete HEAD_TRAINING.json/TAIL_TRAINING.json; MATHEMATICAL_CHECKS.json; REMOTE_NATIVE_RECEIPT.json/REPLAY_RECEIPT.json; results/figures; RESOURCE_RECEIPT.json; REPRODUCTION.md; and FINAL_MANIFEST.json. No further scientific computation was performed for closure.
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
(ROOT/'REPRODUCTION.md').write_text('''# R13 reproduction

Executed server directory: /home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R13_REMOTE_20261009_042246. Environment interpreter: /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python. Main entry was remote_supervise.py, which ran numeric_worker.py once and then replay.py once. CPU4, CUDA training, final CPU float64. Native job identity and actual versions are in REMOTE_NATIVE_RECEIPT.json, NUMERIC_RECEIPT.json and metrics.json.

Read-only dependencies are R10_REMOTE_20261009_012338/inputs (TRAIN-fit/internal val/DEV), R11_REMOTE_20261009_020915 (FACTOR representation, K8 GMM and actual innovations) and R12_REMOTE_20261009_034933 (empirical covariance and historical outputs). Exact paths/configuration and recorded input identities are in config.json, READINESS.json and PROTOCOL.md. Transfer used normal copy/scp exit status; no content equality test was performed.

For an authorized new-directory inference reproduction, retain central representation.npz, GMM_8.npz, GENERATION_INPUTS.npz, FLOW.pt, TAIL.pt, conditional_density.py/density.py and replay.py. Adjust ROOT/config only in that new directory, preserve parameters and draw positions. The executed independent replay162395 restored these saved inputs and reproduced all6000 pre-ADC/post-ADC rows bitwise (REPLAY_RECEIPT.json). No current closure command reruns it.

From-scratch training uses saved head/tail initial states, fixed seeds and checkpoint grid, initial/final random states and full training traces. CUDA training selection arithmetic is float32, final inference CPU float64; do not claim bitwise portability across other hardware/library environments. Complete whole-study scientific evaluation requires the recorded read-only datasets/references; old stages must remain untouched.
''',encoding='utf-8')
files_before = [f for f in ROOT.rglob('*') if f.is_file()]
resource = {
    'stage':'R13_FACTORIZED_CONDITIONAL_DENSITY_001','UTC':now(), 'control_closure_PID':os.getpid(),
    'stage_deadline_UTC':'2026-10-09T05:52:46.629848+00:00','executor_stop_UTC':'2026-10-09T05:42:46.629848+00:00',
    'numeric_cumulative_s':native['numeric_cumulative_s'],'numeric_cap_s':1800,
    'training_cumulative_s':numeric['counters']['training_cumulative_s'],'training_cap_s':1200,
    'history':native['history'],'CPU_threads':4,'RSS_peak_bytes':max(h['RSS_peak_bytes'] for h in native['history']),
    'RSS_cap_bytes':4*1024**3,'RSS_probe_failures':sum(h['RSS_probe_failures'] for h in native['history']),
    'GPU':numeric['environment'],'GPU_allocated_cap_bytes':4*1024**3,
    'counters':numeric['counters'],'replay_rows':6000,'replay_points':2880000,
    'normal_upload_tar_bytes':174080,'normal_return_tar_bytes':transfer['return_tar_bytes'],
    'transfer_content_equality_test':False,'new_G4':0,'EVAL_read':0,'CAL_numeric_decode':0,
    'observed_local_files_before_final_accounting':len(files_before),'observed_local_bytes_before_final_accounting':sum(f.stat().st_size for f in files_before),
    'new_output_cap_bytes':512*1024**2,'exact_full_data_IO_bytes':'unknown; simplified project accounting does not meter library or every byte read',
    'closure':'JSON/control summaries and packaging-only byte hashes; no arrays decoded, inference, fit, replay or migration test.'
}
dump('RESOURCE_RECEIPT.json',resource)
members=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_file() and path.name not in ('FINAL_MANIFEST.json','FINAL_RECEIPT.json'):
        members.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'SHA256':digest(path)})
dump('FINAL_MANIFEST.json',{'stage':resource['stage'],'UTC':now(),'members':members,'member_count':len(members),'member_bytes':sum(f['bytes'] for f in members),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'purpose':'Final package inventory, not transfer source/destination equality verification.'})
dump('FINAL_RECEIPT.json',{'stage':resource['stage'],'PID':os.getpid(),'start_UTC':start,'last_write_UTC':now(),'status':'complete_scientific_and_control_delivery','numeric_native_exit_codes':[h['exit_code'] for h in native['history']],'manifest_SHA256':digest(ROOT/'FINAL_MANIFEST.json'),'manifest_members':len(members),'manifest_member_bytes':sum(f['bytes'] for f in members),'own_native_exit_code':'supplied by caller tool result after process exits','no_further_stage_writes':True})
print(json.dumps({'PID':os.getpid(),'UTC':now(),'manifest_SHA256':digest(ROOT/'FINAL_MANIFEST.json'),'receipt_SHA256':digest(ROOT/'FINAL_RECEIPT.json'),'members':len(members),'member_bytes':sum(f['bytes'] for f in members)}))
