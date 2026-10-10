"""One control-only R45 negative-result report and seal; no scientific imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, time

R = Path(__file__).resolve().parent
START = time.monotonic()
def utc(): return datetime.now(timezone.utc).isoformat()
def read(name): return json.loads((R / name).read_text('utf8'))
def write(name, value):
    (R / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', 'utf8')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

assert not (R / 'FINAL_RECEIPT.json').exists(), 'stage already sealed'
t = read('TRAINING_NUMERIC_RECEIPT.json')
s = read('TRAIN_SOLVER_CHECK.json')
assert t['global_updates'] == 50000 and t['counter']['new_updates'] == 47050
assert s['pass'] is False
assert not (R / 'EXTENDED_FLOW_FINAL/generated.npz').exists()
assert not (R / 'ACTUAL_GENERATION_INPUTS.npz').exists()
assert not (R / 'generate_all.py.NATIVE_RECEIPT.json').exists()
assert not (R / 'evaluate_saved.py.NATIVE_RECEIPT.json').exists()
assert not (R / 'replay_all.py.NATIVE_RECEIPT.json').exists()
remote = read('REMOTE_RUN_HISTORY.json')
local = read('RUN_HISTORY.json')
assert [x['entry'] for x in remote] == ['train_extended.py', 'check_train_solver.py']
assert [x['entry'] for x in local] == ['plot_training.py']
assert all(x['actual_worker_native_exit'] == 0 for x in remote + local)
probes = read('FIXED_PROBE_TRAJECTORY.json')
assert [x['global_step'] for x in probes] == [2950,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]
exposure = read('EXTENDED_FLOW_FINAL/EXPOSURE_SUMMARY.json')
events = read('ENGINEERING_EVENTS.json')
if not any(e['tool'] == '1b9235' for e in events):
    events.append({'tool':'1b9235','native_exit':1,'PID':'unknown','kind':'source/control query',
      'error':'Queried nonexistent TRAINING_EXPOSURE_PRIVATE.json; actual saved name is EXPOSURE_SUMMARY.json',
      'scientific_compute_s':0,'response_updates_or_forwards':0,'resolved':'read actual saved exposure summary; no scientific rerun'})
write('ENGINEERING_EVENTS.json', events)

gates = {f'gate{i}': {'status':'not_evaluated','pass':None,
    'reason':'Fixed TRAIN 32/64 numerical prerequisite failed; DEV evaluation forbidden'} for i in range(1,6)}
gates['gate6'] = {'status':'fail','pass':False,'numerical_check':s,
    'training_and_inheritance_complete':True,'main_replay_status':'not_executed_by_stop_rule'}
write('GATE_METRICS.json', {'stage':R.name,'gates':gates,'retain_recipe':False,
    'verdict':'close_fixed_32step_long_budget_recipe','complete_project_acceptance':False,
    'no_64step_rescue_or_intermediate_selection':True})
compact = {'stage':R.name,'status':'closed_numerical_failure','training':t,
    'exposure_summary':{k:exposure[k] for k in ['new_rows_per_domain','historical_rows_per_domain',
      'total_rows_per_domain','batch_count_min_max','unique_SOURCE_REAL','loaded_SOURCE_REAL','exposure_not_independent_events']},
    'fixed_TRAIN_probe':probes,'numeric':s,'gates':gates,
    'new_DEV_main_rows':0,'new_HGB_fits':0,'new_EM_fits':0,'main_replay_rows':0,
    'full_response_and_PSD_distribution_conclusion':'not_evaluated',
    'excludes_per_event_identity_white_hash_and_prediction_arrays':True}
write('METRICS_COMPACT.json', compact)

science = sum(x['elapsed_s'] for x in remote + local)
resource = {'stage':R.name,'UTC':utc(),'scientific_elapsed_s':science,
    'GPU_training_including_import_probes_and_saving_s':remote[0]['elapsed_s'],
    'CPU12_TRAIN_numerical_check_s':remote[1]['elapsed_s'],
    'local_CPU4_saved_TRAIN_plot_s':local[0]['elapsed_s'],
    'training_program_s':t['training_s'],'historical_R43_training_program_s_separate':t['historical_R43_training_s'],
    'historical_R43_cost_not_reexecuted_or_free':True,
    'science_limit_s':14400,'training_limit_s':10800,
    'server_tree_RSS_limit_bytes':48*1024**3,'server_GPU_limit_bytes':10*1024**3,
    'server_worker_RSS_HWM_peak_bytes':max(x['actual_worker_RSS_peak_bytes'] for x in remote),
    'server_simultaneous_sampled_tree_RSS_peak_bytes':max(x['job_tree_RSS_peak_bytes'] for x in remote),
    'GPU_process_peak_bytes':max(x['GPU_process_memory_peak_bytes'] for x in remote),
    'GPU_allocated_peak_bytes':t['GPU_peak_allocated_bytes'],'GPU_reserved_peak_bytes':t['GPU_peak_reserved_bytes'],
    'allocator_containment_not_summed':True,'local_worker_RSS_peak_bytes':local[0]['actual_worker_RSS_peak_bytes'],
    'RSS_query_gap_events':[e for x in remote+local for e in x['memory_probe_failure_events']],
    'RSS_query_gap_cause':'unknown; timestamps near terminal receipt do not establish cause',
    'local_CPU_threads':4,'server_CPU_threads':12,'new_response_training_runs':1,
    'counter':t['counter'],'TRAIN_numerical_field_forward_rows':sum(s[n]['field_forward_rows'] for n in ['solver32','solver64']),
    'TRAIN_numerical_final_solution_rows':256,
    'DEV_body_copy_or_array_decode':0,'DEV_generation':0,'new_HGB':0,'new_EM':0,'main_replay':0,
    'new_CAL_EVAL_G4_decode':0,'migration_content_consistency_tests':0,
    'ordinary_control_failures':events,'scientific_native_failures':0,
    'engineering_and_transfer_wall_separate_from_science':True,
    'ordinary_control_wall_comprehensive_metering':'not instrumented; native logs/receipts preserved',
    'output_limit_bytes':4*1024**3,'executor_deadline_UTC':'2026-10-10T06:00:04.557894+00:00',
    'budget_pass':science<14400 and remote[0]['elapsed_s']<10800}
write('RESOURCE_RECEIPT.json', resource)
tools = {'train_extended.py':{'start_tool':'68ae81','completion_tool':'aaae29','session':59497},
         'check_train_solver.py':{'start_tool':'529b8e','completion_tool':'1c5124','session':64405},
         'plot_training.py':{'start_completion_tool':'b5572e'}}
write('NATIVE_CHAIN.json', {'science':[dict(x, tool_reference=tools[x['entry']]) for x in remote+local],
    'first_control':read('FIRST_CONTROL_RECEIPT.json'),
    'normal_transfers':[read('FETCH_TRAIN_NATIVE_RECEIPT.json'),read('FETCH_CHECK_NATIVE_RECEIPT.json')],
    'transfer_native_tools':{'train':'d2e806','check':'c3ba36'},'ordinary_control_failures':events,
    'completed_before_seal':True,'no_active_scientific_process':True})
figures = [R/'results/figures'/n for n in ['fixed_TRAIN_probe.png','probe_by_charge.png','probe_charge_time.png','extended_training_trace.png','TRAIN_solver_gate.png']]
assert all(p.is_file() for p in figures)
write('FIGURE_QA.json', {'UTC':utc(),'all_final_figures_actually_opened':True,
    'inspection':'All five opened using view_image; axes, 11 checkpoints, 8 Q bins, limits and negative numerical result readable; no clipping or overlap requiring repair.',
    'ROOT_also_confirmed_five_final_images':True,'scientific_recompute_for_QA':0,
    'figures':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in figures]})

table = '\n'.join(f"| {p['global_step']} | {p['overall']['MSE480']:.9f} | {p['overall']['PSD_RMSE']:.9f} | {p['overall']['PSD_signed_mean']:+.9f} |" for p in probes)
report = f'''# R45: longer fixed neural-flow training, numerical prerequisite failed

The only registered 50,000-update endpoint was trained successfully, but its fixed 32-step Heun recipe failed the TRAIN numerical prerequisite. Close this recipe. All five distribution/scientific gates are **not evaluated**; numerical/replay gate 6 fails. No DEV arrays were decoded or generated, no HGB/EM was fitted, and no 4,000-row main replay was run. This provides no new claim about waveform or PSD distribution similarity.

R43 model and Adam at 2,950 were inherited exactly. New 47,050 updates consumed 1,505,600 rows per domain; including history, 1,600,000 exposures per domain. The unique sampled SOURCE/REAL populations are {exposure['unique_SOURCE_REAL']}; loaded pools are {exposure['loaded_SOURCE_REAL']}. These repeated Monte Carlo exposures do not add independent parents. Original empirical coupling, constants, scalar-time architecture, loss and optimizer rules stayed fixed. Additional t continued R43's final PCG64 state; white used independent PCG64(20261010452). Statistical PSD-order coupling is not physical event pairing.

The fixed 4,096 TRAIN states use 512 occurrence positions per Q bin, one PCG64(20261010453) iterated bins 0–7, with independent probe t/white. They reuse TRAIN parents and are not unseen-parent validation or model selection. MSE fell {100*(1-t['probe_final_over_initial_MSE']):.3f}% (final/initial {t['probe_final_over_initial_MSE']:.9f}); the registered 10% descriptive marker is false. PSD velocity RMSE improved overall but signed residual changed sign and the series fluctuates. None of these path-velocity quantities is generated-waveform PSD acceptance or proof of convergence.

| Total update | Fixed-probe MSE480 | PSD velocity RMSE | Signed PSD residual |
|---:|---:|---:|---:|
{table}

On the unchanged 64 SOURCE + 64 REAL TRAIN inputs/white, CPU12/chunk64 preADC 32-vs-64 RMS median is **{s['RMS_median_LSB']:.9f} LSB > 0.25**, p95 **{s['RMS_p95_LSB']:.9f} LSB <= 1**. Maximum relative q errors are 32-step f32 {s['solver32']['max_relative_q_error_f32']:.9g}, 64-step f32 {s['solver64']['max_relative_q_error_f32']:.9g}; f64 errors are about 2.6e-16. Charge preservation passes, median integration tolerance fails. The 64-step result is a numerical reference, not a rescue candidate. No solver/representation/floor was changed and no intermediate checkpoint was selected.

Training actual worker107155/super107150 ended 01:34:34.348301 UTC, native0; CPU check113870/super113865 ended 01:39:30.691453 UTC, native0; local plotting25148/super59088 ended 01:40:46.605670 UTC, native0. Total scientific charge **{science:.9f} s**, comprising GPU training {remote[0]['elapsed_s']:.9f}, CPU numerical check {remote[1]['elapsed_s']:.9f}, plotting {local[0]['elapsed_s']:.9f}. Historical R43 worker training {t['historical_R43_training_s']:.9f}s is separately retained, not repeated here or treated as free.

Server worker RSS HWM 2,042,970,112 B; sampled tree peak 2,037,903,360 B. GPU process peak 629,145,600 B; allocator peak 313,275,392 B, reserved 343,932,928 B, with containment rather than addition. Two terminal RSS query gaps remain recorded with cause **unknown**; self HWM is available. CPU check and local plotting had zero gaps. There were three ordinary source/control query errors and no failed scientific process; their original tools and unknown PIDs remain in ENGINEERING_EVENTS/NATIVE_CHAIN. Normal transfer exits were successful; migration content consistency tests were not performed.

All five figures were actually opened and checked. Weights, Adam states, every-5,000 engineering checkpoints, dense loss/exposure records, t/white RNG states and per-batch hashes remain local/server research artifacts. Prepared generation/evaluation/replay code and config_cpu are **unexecuted** because the numerical stop fired. Only FINAL_FLOW/FINAL_OPTIMIZER at total50,000 represent this failed candidate. The project remains incomplete under PROJECT_ACCEPTANCE; this stage closes only its registered recipe and starts no new experiment.
'''
(R/'REPORT.md').write_text(report,'utf8')
repro = '''# R45 reproduction and scope

Actual server root: `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R45_REMOTE_20261010_010004_UTC`.
Actual server Python: `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python` (3.12.3, NumPy1.26.4, Torch2.12.0+cu130, SciPy1.11.4). GPU training used RTX5070, CPU12; numerical check used the same environment CPU12/chunk64. Local plotting used Python3.13.3/CPU4. Executed training config is EXECUTED_TRAIN_CONFIG.json/config_remote.json; inherited keys in local/config_cpu are not evidence of running those planned stages.

In a **new independent directory**, retain network.py/source_flow_core.py/probe_core.py/train_extended.py and actual executed config from PRETRAIN_FREEZE, plus input paths/bindings. TRAIN_PRIVATE is the existing R33 server input; frozen R43 final model/Adam/coupling/constants are named in INPUT_BINDINGS. Completed R45 inherited/final model and Adam are in EXTENDED_FLOW_FINAL. Engineering checkpoints every5,000 are not selected candidates. Re-running the trajectory would be a new separately budgeted computation, not an authorized continuation of this sealed directory.

`reconstruct_training_random.py --global-step <2951..50000>` streams draws from saved RANDOM_STATES and PCG64 white seed20261010452 and compares the requested original_batch0/t/white hash with BATCH_HASHES_PRIVATE.npy. This helper was provided, not run as an extra stochastic test. Use original NumPy1.26.4 for reproduction; all optional output paths must stay E/D or this project's server root. Original old94400 white is not used as new training white.

`check_train_solver.py` loads final50,000 weights and exact R42_TRAIN_SOLVER_CHECK_PRIVATE.npz. It was run once under Linux PID/nonce/RSS supervision and failed the .25LSB median threshold. TRAIN_SOLVER_CHECK_PRIVATE saves input, white, indices and both solutions. This closes the fixed32 recipe; no new DEV scientific job or independent main replay is authorized by this document. generate_all/evaluate_saved/replay_all are unexecuted preparation files, not successful replay evidence.

Five TRAIN PNGs are reproducible solely from saved probe/loss/check arrays using plot_training.py. Original native receipts, logs, transfers and environment are preserved. Cross-platform CPU/GPU bitwise equivalence and source/destination migration validation are not claimed. No EVAL/CAL/newG4 exposure occurred.
'''
(R/'REPRODUCTION.md').write_text(repro,'utf8')

# One terminal manifest; its own bytes and the final receipt are explicitly excluded.
excluded = {'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}
paths = sorted(p for p in R.rglob('*') if p.is_file() and p.relative_to(R).as_posix() not in excluded)
assert all(not p.is_symlink() for p in paths)
rows = [{'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
payload_bytes = sum(x['bytes'] for x in rows)
assert payload_bytes < 4*1024**3
manifest = {'stage':R.name,'UTC':utc(),'files':rows,'file_count':len(rows),'payload_bytes':payload_bytes,
    'self_reference_exclusions':sorted(excluded),'hash_scope':'this new stage only; plain byte hashing, no scientific decode or migration comparison'}
write('FINAL_MANIFEST.json', manifest)
mp = R/'FINAL_MANIFEST.json'
receipt = {'stage':R.name,'PID':os.getpid(),'UTC':utc(),'status':'sealed','scientific_status':'closed_numerical_failure',
    'five_scientific_gates':'not_evaluated','numerical_gate':'fail','project_acceptance':'not_complete',
    'manifest_sha256':sha(mp),'manifest_bytes':mp.stat().st_size,
    'manifest_file_count':len(rows),'manifest_payload_bytes':payload_bytes,'stage_file_count_including_manifest_and_receipt':len(rows)+2,
    'scientific_elapsed_s':science,'control_report_and_plain_hash_elapsed_s':time.monotonic()-START,
    'DEV_generation_HGB_EM_main_replay':0,'science_processes_terminal_native0':True,
    'no_further_stage_writes':True,'final_native_exit_confirmed_by_caller_tool':True,
    'self_reference_exclusion':'FINAL_RECEIPT.json is last write, excluded from manifest; terminal hash returned to parent',
    'deadline_UTC':'2026-10-10T06:00:04.557894+00:00'}
write('FINAL_RECEIPT.json', receipt)
rp = R/'FINAL_RECEIPT.json'
print(json.dumps({'PID':os.getpid(),'native_exit':0,'status':'sealed','manifest':str(mp),
    'manifest_sha256':receipt['manifest_sha256'],'receipt':str(rp),'receipt_sha256':sha(rp),
    'receipt_bytes':rp.stat().st_size,'stage_file_count':len(rows)+2,
    'stage_bytes':payload_bytes+mp.stat().st_size+rp.stat().st_size,
    'actual_last_write_UTC':datetime.fromtimestamp(rp.stat().st_mtime,timezone.utc).isoformat(),
    'science_elapsed_s':science,'no_further_stage_writes':True}),flush=True)
