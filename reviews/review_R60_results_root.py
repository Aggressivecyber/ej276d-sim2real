from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, time
START = time.monotonic()
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '4'
import ctypes
from ctypes import wintypes
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
handle = kernel.GetCurrentProcess()
process_mask = ctypes.c_size_t()
system_mask = ctypes.c_size_t()
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask))
affinity = [i for i in range(64) if process_mask.value & (1 << i)][:4]
assert kernel.SetProcessAffinityMask(handle, sum(1 << i for i in affinity))

class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in [
        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]

def memory_counter():
    memory = MemoryCounters()
    memory.cb = ctypes.sizeof(memory)
    get_memory = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    assert get_memory(handle, ctypes.byref(memory), memory.cb)
    return memory
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
R = Path(__file__).resolve().parents[1]
ID = 'R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001'
S = R / 'work' / ID
H = R / 'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text('utf8'))

def write(p, value):
    p.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def close(a, b, atol=1e-11, rtol=1e-10):
    assert np.allclose(a, b, atol=atol, rtol=rtol), (a, b)

B = read(S / 'INPUT_BINDING.json')
C = read(S / 'config.json')
result = read(S / 'RESULT.json')
full = read(S / 'FULL_TRAIN_SUMMARY.json')
records = read(S / 'GRADIENT_BATCH_RECORDS.json')
summary = read(S / 'GRADIENT_SUMMARY.json')
native = read(S / 'diagnose_train.py.NATIVE_RECEIPT.json')
checks = {}
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None and native['handshake_before_imports']
assert not read(S / 'LIVE_OBSERVATION.json')['actual_process']['exists']
assert all(x['native_exit'] == 0 for x in read(S / 'FETCH_RECEIPT.json')['events'])
assert result['frozen_weight_before_SHA'] == result['frozen_weight_after_SHA'] == B['frozen_weight_tensor_SHA']
assert all(sha(S / x['name']) == x['sha256'] for x in read(S / 'ROOT_IMPLEMENTATION_REVIEW.json')['files'])
assert all(sha(Path(x['path'])) == x['sha256'] for x in B['original_acceptance'])
assert all(sha(S / name) == sha(H / name) == x['sha256'] for name, x in B['frozen_R59_helpers'].items())
local_inputs = {}
for key, record in B['inputs'].items():
    if key == 'TRAIN_path':
        continue
    suffix = record['path'].split(C['historical_R59_root'] + '/', 1)[1]
    assert sha(H / suffix) == record['sha256']
    local_inputs[key] = {'local_path': str(H / suffix), 'sha256': record['sha256']}
checks['sealed_code_original_acceptance_and_seven_local_historical_inputs'] = True
checks['remote_all_eight_input_hashes_checked_by_successfully_executed_bound_entry'] = True
with np.load(S / 'FULL_TRAIN_DIAGNOSTICS_PRIVATE.npz', allow_pickle=False) as z:
    data = {name: z[name].copy() for name in z.files}
with np.load(S / 'GRADIENT_DIAGNOSTICS_PRIVATE.npz', allow_pickle=False) as z:
    grads = {name: z[name].copy() for name in z.files}
for arrays in [data, grads]:
    assert all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fiu')
assert len(data['source_ids']) == 6593 and len(data['REAL_raw_region']) == 21047
assert hashlib.sha256(np.ascontiguousarray(data['source_ids']).tobytes()).hexdigest() == B['source_parent_ID_SHA']
assert len(set(data['source_ids'])) == 6593
assert np.all(data['source_PSD_interval_low'] <= data['source_original_psd'])
assert np.all(data['source_PSD_interval_high'] >= data['source_original_psd'])
close(wasserstein_distance(data['source_output_psd'], data['REAL_reference_psd']), full['overall_SOURCE_PSD_W1'])
close(wasserstein_distance(data['REAL_output_psd'], data['REAL_reference_psd']), full['overall_same_treated_REAL_PSD_W1'])
close(spearmanr(data['source_original_psd'], data['source_output_psd']).statistic, full['source_PSD_rank_rho'])
close(spearmanr(data['source_original_q'], data['source_output_q']).statistic, full['source_Q_rank_rho'])
by_region = []
for item in full['per_original_region']:
    region = item['raw_region']
    sm = data['source_raw_region'] == region
    rm = data['REAL_raw_region'] == region
    assert sm.sum() == item['SOURCE_N'] and rm.sum() == item['REAL_N']
    w = wasserstein_distance(data['source_output_psd'][sm], data['REAL_reference_psd'][rm])
    close(w, item['SOURCE_PSD_W1'])
    close(wasserstein_distance(data['REAL_output_psd'][rm], data['REAL_reference_psd'][rm]), item['same_treated_REAL_PSD_W1'])
    by_region.append(w)
close(np.mean(by_region[1:8]), full['common_Q1_Q7_mean_SOURCE_PSD_W1'])
for key, prefix in [('SOURCE', 'source'), ('same_treated_REAL', 'REAL')]:
    saved = full['projection'][key]
    assert np.sum(data[prefix + '_constraint_projection_range_active']) == saved['range_active_events']
    close(data[prefix + '_constraint_projection_RMS_V'].mean(), saved['RMS_correction_V_mean'])
    assert np.sum(data[prefix + '_constraint_projection_rail_sample_count']) == saved['rail_sample_count']
    assert data[prefix + '_preADC_charge_relative_error_f32'].max() <= 1e-5
checks['saved_parent_population_all_regions_W1_ranks_and_projection'] = True
assert grads['loss_values'].shape == (32, 17) and grads['parameter_Gram'].shape == (32, 17, 17)
assert grads['raw_gradient_position_RMS'].shape == grads['raw_gradient_position_mean'].shape == (32, 17, 2, 480)
assert grads['raw_gradient_domain_norm'].shape == (32, 17, 2)
assert np.all(grads['raw_gradient_position_RMS'] >= 0)
names = grads['component_names'].tolist()
assert names == summary['component_names']
assert len(records) == 32
assert np.array_equal(data['source_raw_region'][grads['SOURCE_indices']], np.tile(np.repeat(np.arange(-1, 9), 32), (32, 1)))
assert np.array_equal(data['REAL_raw_region'][grads['REAL_reference_indices']], np.tile(np.repeat(np.arange(-1, 9), 32), (32, 1)))
assert grads['REAL_identity_indices'].shape == (32, 64) and grads['REAL_identity_indices'].max() < 21047
flag_counts = {name: 0 for name in result['diagnostic_markers']}
subcounts = {'roundoff_mass_at_least_half': 0, 'roundoff_norm_at_least_half': 0, 'roundoff_cos_below_minus_quarter': 0,
             'outside_norm_at_least_half': 0, 'outside_cos_below_minus_quarter': 0,
             'valley_norm_at_least_half_PSD': 0, 'valley_cos_below_minus_quarter': 0, 'undefined_valley_PSD_cosines': 0}
for i, record in enumerate(records):
    assert record['batch'] == i + 1
    close(grads['loss_values'][i], [record['losses'][n] for n in names])
    gram = grads['parameter_Gram'][i]
    close(gram, gram.T)
    norm = np.sqrt(np.maximum(np.diag(gram), 0))
    close(norm, [record['parameter_gradient_norms'][n] for n in names])
    assert all(v['pass'] for v in record['decomposition_checks'].values())
    assert all(v['gradient_relative_error'] <= C['gradient_recomposition_rtol'] for v in record['decomposition_checks'].values())
    m = record['marker_inputs']
    def cosine(indices_a, indices_b):
        denominator = np.sqrt(max(gram[np.ix_(indices_a, indices_a)].sum(), 0) * max(gram[np.ix_(indices_b, indices_b)].sum(), 0))
        return float(gram[np.ix_(indices_a, indices_b)].sum() / denominator) if denominator > 1e-30 else None
    expected = {'roundoff_response_cosine': cosine([11], [0, 1, 4]), 'outside_inside_cosine': cosine([14], [13]), 'valley_PSD_cosine': cosine([3], [1]),
                'roundoff_norm_fraction': norm[11] / (norm[11] + norm[12]), 'outside_norm_fraction': norm[14] / (norm[13] + norm[14]), 'valley_to_PSD_norm_ratio': norm[3] / norm[1]}
    for key, value in expected.items():
        if value is None:
            assert m[key] is None
        else:
            close(value, m[key])
    criteria = {'roundoff_mass_at_least_half': record['pair_masses']['roundoff_compatible_fraction_of_nonzero_mass'] >= .5,
                'roundoff_norm_at_least_half': m['roundoff_norm_fraction'] >= .5,
                'roundoff_cos_below_minus_quarter': m['roundoff_response_cosine'] is not None and m['roundoff_response_cosine'] < -.25,
                'outside_norm_at_least_half': m['outside_norm_fraction'] >= .5,
                'outside_cos_below_minus_quarter': m['outside_inside_cosine'] is not None and m['outside_inside_cosine'] < -.25,
                'valley_norm_at_least_half_PSD': m['valley_to_PSD_norm_ratio'] >= .5,
                'valley_cos_below_minus_quarter': m['valley_PSD_cosine'] is not None and m['valley_PSD_cosine'] < -.25,
                'undefined_valley_PSD_cosines': m['valley_PSD_cosine'] is None}
    flags = {'roundoff_order_conflict': all(criteria[k] for k in list(criteria)[:3]),
             'outside_full_window_conflict': criteria['outside_norm_at_least_half'] and criteria['outside_cos_below_minus_quarter'],
             'valley_PSD_conflict': criteria['valley_norm_at_least_half_PSD'] and criteria['valley_cos_below_minus_quarter']}
    assert flags == record['flags']
    for k in subcounts:
        subcounts[k] += int(criteria[k])
    for k in flag_counts:
        flag_counts[k] += int(flags[k])
for k, count in flag_counts.items():
    assert count == result['diagnostic_markers'][k]['batches']
    assert result['diagnostic_markers'][k]['activated'] == (count >= 24)
close(grads['loss_values'].mean(0), [summary['mean_loss'][n] for n in names])
close(grads['parameter_Gram'].mean(0), summary['parameter_Gram_mean'])
checks['saved_17_component_geometry_and_all_preset_marker_rules'] = True
trace = read(S / 'SAVED_R59_TRAINING_TRACE_SUMMARY.json')
with np.load(H / 'DIRECT_DISTRIBUTION_RESPONSE/LOSS_BY_UPDATE_PRIVATE.npz', allow_pickle=False) as z:
    history_values = z['values']; columns = z['columns'].tolist()
    assert history_values.shape == (50000, 20) and np.isfinite(history_values).all()
    historical_norm = history_values[:, columns.index('gradient_norm')]
    close(np.mean(historical_norm > 5), trace['gradient_clipped_fraction'])
    close(np.quantile(historical_norm, [0, .05, .5, .95, 1]), trace['gradient_norm_quantiles'])
    for i, block in enumerate(trace['blocks_of_1000']):
        close(history_values[i*1000:(i+1)*1000].mean(0), [block[name] for name in columns])
checks['frozen_R59_trace_recounted_without_new_training'] = True
pair = full['source_order']
assert pair['ordered_parent_pairs_total'] == 6593**2
assert pair['strict_nonzero_pairs'] + pair['exact_equal_pairs'] == 6593**2
assert sum(pair['groups'][k]['pairs'] for k in ['roundoff_compatible', 'resolved']) == pair['strict_nonzero_pairs']
close(sum(pair['groups'][k]['contribution_to_original_order_loss'] for k in ['roundoff_compatible', 'resolved']), pair['original_loss'])
assert result['counter']['neural_forward_rows_completed'] == 39928 and result['counter']['autograd_calls_completed'] == 544
assert all(result['counter'][k] == 0 for k in ['response_training_runs', 'response_updates', 'HGB_fits', 'density_fits', 'teacher_waveforms', 'DEV_read', 'EVAL_read', 'CAL_decode'])
memory = memory_counter()
assert memory.PeakWorkingSetSize < 4 * 1024**3
review = {'UTC': NOW, 'owner': '/root', 'pass': True, 'checks': checks, 'local_historical_inputs': local_inputs,
          'marker_subcondition_counts_out_of_32': subcounts, 'original_marker_counts': flag_counts,
          'CPU_affinity': affinity, 'sampled_RSS_bytes': memory.WorkingSetSize, 'peak_RSS_bytes': memory.PeakWorkingSetSize, 'elapsed_s': time.monotonic() - START,
          'new_model_forward_rows': 0, 'new_training_updates': 0, 'new_HGB_or_density_fits': 0,
          'independent_full_forward_replay': False, 'full_random_stream_reconstruction': False,
          'validation_scope': 'Saved artifact identity, dimensions, statistics and gradient geometry; not another full frozen-model inference or A-F acceptance.'}
write(S / 'FROZEN_ARTIFACT_REVIEW.json', review)
figures = {
    'full_TRAIN_PSD_regions.png': 'All ten original regions and domain sample counts visible; Q3/Q4 narrow SOURCE peaks and supported/outside discrepancies retained.',
    'gradient_cosine_matrix.png': 'All 17 components legible; raw gradient and zero-norm interpretation in title; no causal or optimizer-step claim.',
    'precision_order_and_flags.png': 'Storage-overlap mass, all 32 batches, undefined cosine gaps, raw norms and failed 24/32 preset markers visible.',
    'full_window_and_projection.png': 'All ten original regions retained, including outside mass, feature contributions and original versus generated overrange events.',
    'raw_gradient_boundary_profiles.png': 'Complete 480-point preprojection waveform-gradient profiles and sample100/150/400 lines visible; normalized and weighted magnitudes separated.',
    'saved_R59_training_trajectory.png': 'Historical 50000 updates and clipping fractions readable; zero new updates stated explicitly.',
    'complete_TRAIN_waveform.png': 'All 480 samples, raw SOURCE and both transformed domains, mean/SD and key fixed time boundaries visible.'}
assert set('figures/' + name for name in figures) == set(result['figures'])
write(S / 'FIGURE_QA.json', {'UTC': NOW, 'owner': '/root', 'all_figures_opened': True, 'layout_pass': True,
      'method': 'Actual view_image inspection of all seven PNGs by root after fetch.', 'figure_count': 7, 'figure_repairs': 0,
      'figures': [{'path': 'figures/' + n, 'SHA256': sha(S / 'figures' / n), 'opened': True, 'layout_pass': True, 'observation': v} for n, v in figures.items()],
      'scientific_acceptance_determined': False})
next_action = '封存R60，保留三个预设冲突标记均未触发的结论。下一轮预登记独立的单因素存储精度排序消融：相对R59只取消float32 PSD区间重叠父对的顺序罚，保留原非零父对分母、全部源父事件及其他六项损失；原网络、初始化、随机流、常数和唯一50000步终点保持。依据是直接核验的存储顺序不可分辨性，而非宣称17/32达到24/32。先检查新损失语义和复现绑定，再从零训练；同一冻结末点沿用原数值门与完整A–C，失败关闭，通过才继续D/E。原A–F、最终SOURCE人口和阈值不变。'
direction = {'UTC': NOW, 'owner': '/root', 'scientific_decision': 'close_completed_frozen_diagnostic_all_three_preset_markers_false',
             'model_goal_complete': False, 'application_goal_status': 'active', 'original_A_F_changed': False,
             'diagnostic_markers': result['diagnostic_markers'], 'marker_subcondition_counts_out_of_32': subcounts,
             'full_TRAIN_PSD': result['full_TRAIN_PSD'], 'source_order': pair, 'projection': full['projection'],
             'saved_R59_gradient_clipped_fraction': trace['gradient_clipped_fraction'],
             'roundoff_compatible_share_of_original_order_penalty': pair['groups']['roundoff_compatible']['contribution_to_original_order_loss'] / pair['original_loss'],
             'numerical_decompositions_pass': True, 'all_seven_figures_opened': True, 'new_response_training': 0,
             'next_protocol_registered': False, 'next_action': next_action,
             'interpretation_limits': [
                 'All three prespecified markers remain false; a separately preregistered ablation must not be presented as a triggered or validated mechanism.',
                 'Float32 interval overlap is conservative storage precision, not proof of identical true PSD or detector measurement uncertainty.',
                 'Local Euclidean gradient geometry does not determine the actual Adam trajectory or establish a cause of model failure.',
                 'TRAIN summaries neither replace original DEV gates nor establish independent-acquisition or downstream-utility validity.',
                 'Outside full-window gradients dominate their norm but are not consistently opposed to inside gradients; 0/32 is retained.',
                 'Diagnostic completion and a hypothesis for the next ablation do not complete the original model goal.']}
write(R / 'reviews/R60_ROOT_DIRECTION_001.json', direction)
print(json.dumps({'artifact_review_pass': True, 'figure_QA_pass': True, 'marker_counts': flag_counts, 'marker_subconditions': subcounts, 'next_action': next_action}, ensure_ascii=False, indent=2))
