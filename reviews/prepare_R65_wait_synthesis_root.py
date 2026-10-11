"""Bind static compute accounting and conditional GAP applicability evidence.

Only saved JSON/text is read. No science arrays, model, fit, or SSH.
"""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os

R = Path(__file__).resolve().parents[1]
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
O = R/'reviews/R65_WAIT_ANALYSIS_001/root_synthesis'
K = ctypes.WinDLL('kernel32', use_last_error=True)
K.GetCurrentProcess.restype = ctypes.c_void_p
K.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
K.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
H = K.GetCurrentProcess()
assert K.SetProcessAffinityMask(H,8)
assert K.SetProcessWorkingSetSizeEx(H,16*1024**2,1024**3,6)

def j(name):
    return json.loads((R/name).read_text('utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

files = [
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/config.json',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/train_final.py',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/noise_core.py',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/distribution_adapter/distribution_gradient.py',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/network_update_check/network_update_check.py',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/ROOT_IMPLEMENTATION_REVIEW.json',
 'work/R6_CHARGE_GAP_GENERALIZATION_001/config.json',
 'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/DIAGNOSTIC_DEFINITION.json',
 'work/R23_ONLINE_STANDARDIZED_CRITIC_001/config.json',
 'work/R23_ONLINE_STANDARDIZED_CRITIC_001/data_support.py',
 'work/R23_ONLINE_STANDARDIZED_CRITIC_001/prepare_data.py',
 'work/R33_RESIDUAL_OT_FLOW_001/config.json',
 'work/R33_RESIDUAL_OT_FLOW_001/train_server.py',
 'work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/config.json',
 'work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/train_server.py',
 'work/R43_SCALAR_TIME_NEURAL_FLOW_001/prepare_stage.py',
 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/TRAIN_POPULATION_AND_COORDINATES.json',
 'reviews/R65_RUNTIME_OBSERVATIONS_001/20261011_033454.json',
 'reviews/R65_RUNTIME_OBSERVATIONS_001/20261011_034119_395195.json',
 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001/LATEST_REMOTE_OBSERVATION.json',
]
cfg = j(files[0])
full = j('work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/TRAIN_POPULATION_AND_COORDINATES.json')
definition = j('work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/DIAGNOSTIC_DEFINITION.json')
gap = j('work/R6_CHARGE_GAP_GENERALIZATION_001/config.json')['gap_charge_mVns']
edges = definition['edges_mVns']
covered = [i+1 for i in range(8) if gap[0] <= edges[i] and edges[i+1] <= gap[1]]
assert covered == [6]
assert full['region_categories'].index(5) == 6
observations = [
 j(files[-3])['PHYSICAL_PROGRESS.json']['value'],
 j(files[-2])['samples'][-1]['physical_progress'],
 j(files[-1])['PHYSICAL_PROGRESS.json'],
]
intervals = []
for a,b in zip(observations, observations[1:]):
    assert a['PID'] == b['PID'] == 291399 and a['native_attempt_id'] == b['native_attempt_id']
    assert a['phase'] == b['phase'] == 'full_D_parameter_replay'
    ds = b['current_update']-a['current_update']
    dt = (datetime.fromisoformat(b['UTC'])-datetime.fromisoformat(a['UTC'])).total_seconds()
    assert ds > 0 and dt > 0
    intervals.append({'from_update':a['current_update'],'to_update':b['current_update'],
        'phase':a['phase'],'from_UTC':a['UTC'],'to_UTC':b['UTC'],
        'elapsed_s':dt,'completed_update_difference':ds,'mean_s_per_update_over_interval':dt/ds})
per = {'new_response_updates':1,'SOURCE_training_rows':320,'REAL_reference_rows':320,
       'REAL_identity_rows':64,'full_D_SOURCE_law_rows':6593,'full_D_REAL_law_rows':21047,
       'scientific_neural_forward_rows':2*6593+320+64,
       'scientific_neural_forward_calls':2*((6593+63)//64)+1,
       'D_parameter_VJP_calls':(6593+63)//64,'C_parameter_reverse_calls':1,
       'full_D_endpoint_reverse_calls':1,'full_D_objective_calls':1,'replay_chunks_checked':(6593+63)//64}
expected = {k:v*50000 for k,v in per.items()}
assert expected == cfg['expected_completed_update_counts']
O.mkdir(parents=True, exist_ok=False)
# Preserve the changing observer bytes used in this particular report.
(O/'OBSERVED_PROGRESS.json').write_bytes((R/files[-1]).read_bytes())
report = f'''# R65 waiting analysis: work accounting and GAP dependency

This is static preparation during the single active R65 training run. It does
not change its method, source population, stochastic streams, 50000-update
endpoint or original A-F. No scientific arrays or weights were decoded.

## Full empirical work and observed timing

Each update includes two 6593-parent neural passes plus the original 384-row C
pass: 13570 neural rows and 209 forward calls. It uses 104 full-D parameter VJPs,
one C reverse pass, one combined gradient installation, one clip and one Adam
step. Across the fixed 50000 updates the committed accounting is 678500000
neural rows, 10450000 forward calls and 5200000 full-D VJPs. These are repeated
operations, not distinct training parents or extra independent measurements.
The full-D SOURCE and REAL empirical laws respectively contribute 329650000
and 1052350000 repeated rows. Each step still has only 6593 and 21047 unique
parents. Artificial checks, failed attempts and replayed work remain separate.

The two saved intervals at the same full-D parameter-replay phase average
{intervals[0]['mean_s_per_update_over_interval']:.3f} and
{intervals[1]['mean_s_per_update_over_interval']:.3f} seconds per update.
These are observed operational intervals, not a benchmark, a cause diagnosis,
or a reliable completion-date estimate. The variation is why neither a single
instantaneous GPU utilization value nor the sparse 100-step training log should
be treated as a stalled worker. No profiling workload or runtime change was made.

## Conditional empty-region consequence for the future GAP study

The published R3B eight-bin definition puts Q6 in
[{edges[5]:.15g},{edges[6]:.15g}) mV ns. That whole interval lies strictly inside
the original CAL q3 exclusion [{gap[0]:.15g},{gap[1]:.15g}) mV ns. Q6 and CAL q3
are different partition labels; their numerical endpoints must be compared.

The saved source chain uses the R3B width definition in R23 data_support, saves
it as TRAIN_PRIVATE.edges, carries it into the R33 affine constants and R42
unit-charge constants, then copies those constants into R43. R65 loads the
frozen R43 constant archive. The current full-population R63 receipt reports
399 SOURCE and 2624 REAL parents in raw region 5, the zero-based Q6 category.
Those are FULL counts from existing metadata, not newly filtered GAP counts.

If the GAP exclusion uses exactly the same signed saved-window charge and those
bound edges, Q6 must be empty after exclusion, independently of the individual
waveforms. This is a conditional consequence of interval containment and saved
source/metadata lineage. We did not decode the live constants or filter any
actual parents here. The registered preflight must still verify actual edge
bytes, source nominal-clean charge versus training-input charge, REAL charge,
parent identity and every remaining region count. It must not silently replace
the source's original GAP charge definition with a different measured quantity.

The unchanged R65 trainer requires all ten SOURCE and REAL regional pools to be
nonempty; ParentLayout rejects an empty regional list and the original C step
draws 32 occurrences per region. Refitting unit-charge moments from an empty Q6
REAL pool is also undefined. Therefore the future GAP preparation should first
test this concrete applicability dependency. It cannot merely remove a region,
reuse FULL moments, or alter regional mass/thresholds and claim the same recipe.
No GAP job is registered or executed, and no part of E is completed by this note.

The active FULL training continues unchanged. This issue affects the design and
interpretation of a conditional future scope study, not the current A-C verdict,
which is still unavailable. It does not establish that the original model goal
is impossible or authorize a change in the current scientific endpoint.
'''
(O/'REPORT.md').write_bytes(report.encode())
receipt = {'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),
 'status':'STATIC_PREPARATION_ONLY','per_update':per,'fixed_final_committed_counts':expected,
 'observed_same_phase_intervals':intervals,'completion_date_estimate_given':False,
 'fully_contained_published_Q_regions_1based':covered,'GAP_charge_interval_mVns':gap,
 'published_Q6_interval_mVns':edges[5:7],'FULL_Q6_SOURCE_REAL_counts':[full['source_counts'][6],full['REAL_counts'][6]],
 'actual_filtered_GAP_counts_computed':False,'conditional_same_charge_and_edges_required':True,
 'new_science_arrays_decoded':0,'scientific_imports':0,'new_scientific_jobs':0,
 'active_R65_modified':False,'original_A_F_modified':False,'CPU_affinity_mask':8,'RSS_cap_bytes':1024**3,
 'files':[{'path':name,'bytes':(R/name).stat().st_size,'sha256':sha(R/name)} for name in files],
 'report_SHA256':sha(O/'REPORT.md'),'observed_progress_snapshot_SHA256':sha(O/'OBSERVED_PROGRESS.json')}
(O/'SOURCE_BINDINGS_AND_RESULT.json').write_bytes((json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode())
assert json.loads((O/'SOURCE_BINDINGS_AND_RESULT.json').read_text('utf8')) == receipt
assert sha(O/'OBSERVED_PROGRESS.json') == receipt['files'][-1]['sha256']
print(json.dumps({'directory':str(O),'source_files':len(files),'phase_interval_seconds_per_update':
 [row['mean_s_per_update_over_interval'] for row in intervals],
 'conditional_empty_published_Q6':True,'scientific_arrays_decoded':0,'new_jobs':0,'report_SHA256':receipt['report_SHA256']},indent=2))
