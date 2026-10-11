"""Independent saved-artifact arithmetic and stream audit; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import math
import os
import threading
import time

START = time.monotonic()
R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
cache = S/'cache/root_saved_artifact_review'
cache.mkdir(parents=True, exist_ok=True)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
for key in ('TMP', 'TEMP', 'TMPDIR', 'MPLCONFIGDIR'):
    os.environ[key] = str(cache)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
cpus = [i for i in range(64) if allowed.value & (1 << i)]
cpu = cpus[min(3, len(cpus)-1)]
assert kernel.SetProcessAffinityMask(handle, 1 << cpu)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
          'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


getmem = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
getmem.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
resource = dict(actual_PID=os.getpid(), parent_PID=os.getppid(), CPU_affinity=[cpu],
                actual_peak_RSS_bytes=0, RSS_limit_bytes=2*1024**3, actual_process_guard_before_numpy=True)
done = threading.Event()


def write(name, value):
    (S/name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())


def probe():
    value = PMC()
    value.cb = ctypes.sizeof(value)
    assert getmem(handle, ctypes.byref(value), value.cb)
    resource['actual_peak_RSS_bytes'] = max(resource['actual_peak_RSS_bytes'], value.PeakWorkingSetSize)
    if value.WorkingSetSize > resource['RSS_limit_bytes']:
        write('ROOT_REVIEW_RESOURCE_FAILURE.json', resource)
        os._exit(91)


def guard():
    while not done.wait(.1):
        try:
            probe()
        except BaseException as exc:
            write('ROOT_REVIEW_RESOURCE_FAILURE.json', dict(resource, error=repr(exc)))
            os._exit(92)


probe()
watcher = threading.Thread(target=guard, daemon=True)
watcher.start()
import numpy as np

read = lambda name: json.loads((S/name).read_text('utf8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = {}
C, binding, result = read('config.json'), read('INPUT_BINDING.json'), read('RESULT.json')
native, fetch = read('diagnose_train.py.NATIVE_RECEIPT.json'), read('FETCH_RECEIPT.json')
assert native['actual_worker_native_exit'] == native['worker_self_receipt']['native_exit'] == 0
assert native['termination_reason'] is None and native['handshake_before_imports']
assert fetch['remote_completion']['actual_PID_gone'] and all(row['native_exit'] == 0 for row in fetch['events'])
assert read('DIAGNOSTIC_PROGRESS.json')['phase'] == 'registered_diagnostic_endpoint_complete'
assert not (S/'DIAGNOSTIC_FAILURE.json').exists()
assert native['actual_worker_RSS_peak_bytes'] <= C['RSS_limit_bytes']
assert native['GPU_process_memory_peak_bytes'] <= C['GPU_limit_bytes']
assert native['memory_probe_failures'] == 0 and native['GPU_query_failures'] == []
assert result['frozen_weight_before_SHA'] == result['frozen_weight_after_SHA'] == binding['frozen_weight_tensor_SHA']
assert result['all_input_hashes_unchanged_after'] and result['all_replayed_endpoints_bitwise']
for row in read('ROOT_IMPLEMENTATION_REVIEW.json')['files']:
    assert sha(S/row['name']) == row['sha256'], row['name']
for row in binding['original_acceptance']:
    assert sha(Path(row['path'])) == row['sha256'], row['path']
for name in ('CPU', 'CUDA'):
    fixture = read('DISTRIBUTION_ARTIFICIAL_'+name+'.json')
    assert fixture['compact']['passed'] and fixture['compact']['checks'] == 69
    assert len(fixture['tests']) == 69 and all(v['passed'] for v in fixture['tests'].values())
    assert fixture['compact']['caller_state_preserved']
    net = read('NETWORK_ARTIFICIAL_'+name+'.json')
    assert net['pass'] and all(r['bitwise_endpoint'] and r['canonical_codes'] and r['Box_free_mask'] for r in net['chunk_checks'])
counts = result['counter']
for key, expected in dict(full_SOURCE_rows_completed=6593, full_REAL_rows_completed=21047,
       neural_forward_rows_completed=245209, neural_forward_calls_completed=3865,
       parameter_VJP_calls_completed=33800, endpoint_objective_calls_completed=161,
       full_SOURCE_replay_passes_completed=33, paired_macro_batches_completed=32).items():
    assert counts[key] == expected, key
for key in ('response_training_runs', 'response_updates', 'HGB_fits', 'density_fits', 'teacher_waveforms', 'DEV_read', 'EVAL_read', 'CAL_decode'):
    assert counts[key] == 0, key
replay = read('ENDPOINT_REPLAY_RECEIPT.json')
assert len(replay['passes']) == 33
for k, row in enumerate(replay['passes']):
    assert row['VJPs_per_chunk'] == (5 if k == 0 else 10) and row['zero_derivative_VJP_skips'] == 0
    assert row['all_original_SOURCE_rows_replayed'] and len(row['chunks']) == 104
    for j, chunk in enumerate(row['chunks']):
        assert chunk['start'] == 64*j and chunk['stop'] == min(64*(j+1), 6593)
        assert chunk['endpoint_bitwise'] and chunk['codes_exact'] and chunk['free_mask_exact']
assert read('DATA_ACCESS_RECEIPT.json')['decoded_fields'] == ['source', 'target', 'source_ids', 'h', 'source_bin', 'target_bin']
checks['native_artificial_bindings_scope_and_all_3432_replay_chunks'] = True


def close(a, b, *, label='', atol=1e-10, rtol=1e-10):
    assert np.allclose(a, b, atol=atol, rtol=rtol), label


with np.load(S/'SOURCE_ENDPOINTS_PRIVATE.npz', allow_pickle=False) as z:
    source_bins = z['original_raw_bins']
    assert z['projected'].shape == (6593, 480) and z['projected'].dtype == np.float64
with np.load(S/'REAL_control_ENDPOINTS_PRIVATE.npz', allow_pickle=False) as z:
    real_bins = z['original_raw_bins']
    assert z['projected'].shape == (21047, 480) and z['projected'].dtype == np.float64
with np.load(S/'PARENT_IDENTITIES_PRIVATE.npz', allow_pickle=False) as z:
    assert hashlib.sha256(np.ascontiguousarray(z['SOURCE_ids']).tobytes()).hexdigest() == binding['source_parent_ID_SHA']
    assert np.array_equal(z['SOURCE_original_order'], np.arange(6593)) and np.array_equal(z['REAL_original_order'], np.arange(21047))
regions = list(range(-1, 9))
scount, rcount = np.array([(source_bins == r).sum() for r in regions]), np.array([(real_bins == r).sum() for r in regions])
geometry = read('ENDPOINT_GEOMETRY_SUMMARY.json')
assert np.array_equal(scount, geometry['population']['source_counts']) and np.array_equal(rcount, geometry['population']['REAL_counts'])
close(geometry['population']['source_mass'], scount/6593)
close(geometry['population']['REAL_mass'], rcount/21047)
streams = {key: np.random.Generator(np.random.PCG64(seed)) for key, seed in C['PCG64_seeds'].items()}
chains = {key: bytes(32) for key in streams}
assert {key: rng.bit_generator.state for key, rng in streams.items()} == read('INITIAL_RANDOM_STATES.json')


def consume(key, value):
    chains[key] = hashlib.sha256(chains[key]+np.ascontiguousarray(value).tobytes()).digest()


design = np.load(S/'FROZEN_RANDOM_DESIGN_PRIVATE.npz', allow_pickle=False)
for key, n, filename in [('full_SOURCE_epsilon', 6593, 'SOURCE_ENDPOINTS_PRIVATE.npz'), ('full_REAL_epsilon', 21047, 'REAL_control_ENDPOINTS_PRIVATE.npz')]:
    expected = streams[key].standard_normal((n, 480)).astype('<f4')
    with np.load(S/filename, allow_pickle=False) as z:
        assert np.array_equal(expected, z['epsilon']), key
    consume(key, expected)
    del expected
for j, dimension in enumerate((480, 480, 479, 478)):
    v = streams['full_feature_directions'].standard_normal((dimension, 64))
    v = (v/np.linalg.norm(v, axis=0, keepdims=True)).astype('<f4')
    assert np.array_equal(v, design['directions_'+str(j)])
    consume('full_feature_directions', v)
v = streams['temporal_feature_directions'].standard_normal((31, 64))
v = (v/np.linalg.norm(v, axis=0, keepdims=True)).astype('<f4')
assert np.array_equal(v, design['temporal_directions'])
consume('temporal_feature_directions', v)
saved_s, saved_r = design['macro_SOURCE_indices'], design['macro_REAL_indices']
assert saved_s.shape == saved_r.shape == (32, 10, 4, 32)
for macro in range(32):
    for j, region in enumerate(regions):
        for key, bins, saved in [('macro_SOURCE_rows', source_bins, saved_s), ('macro_REAL_rows', real_bins, saved_r)]:
            draw = streams[key].choice(np.flatnonzero(bins == region), size=128, replace=True).astype('<u2')
            assert np.array_equal(draw.reshape(4, 32), saved[macro, j])
            consume(key, draw)
bootstrap = streams['gradient_paired_bootstrap'].integers(0, 32, size=(1000, 32)).astype('<u2')
assert np.array_equal(bootstrap, design['paired_bootstrap_indices'])
consume('gradient_paired_bootstrap', bootstrap)
for scope in ['all']+['region'+str(r) for r in regions]:
    n = 6593 if scope == 'all' else int((source_bins == int(scope[6:])).sum())
    pool = np.arange(21047) if scope == 'all' else np.flatnonzero(real_bins == int(scope[6:]))
    for arm in ('A', 'B'):
        key = 'REAL_floor_'+arm+'_rows'
        draw = streams[key].choice(pool, size=(32, n), replace=True).astype('<u2')
        assert np.array_equal(draw, design['REAL_floor__'+scope+'__'+arm])
        consume(key, draw)
design.close()
assert {key: rng.bit_generator.state for key, rng in streams.items()} == read('FINAL_RANDOM_STATES.json')
assert {key: value.hex() for key, value in chains.items()} == read('RANDOM_STREAM_RECEIPT.json')['chains']
assert all(chains[key].hex() == value for key, value in C['expected_full_stream_hashes'].items())
checks['all_original_parents_separate_domain_masses_and_nine_random_streams_reconstructed'] = True
print(json.dumps(dict(phase='native_binding_random_stream_checks_passed')), flush=True)

geo = np.load(S/'ENDPOINT_GEOMETRY_PRIVATE.npz', allow_pickle=False)
term_names = ['full_window', 'coordinate_RMS', 'temporal_joint']
views = ['V480', 'U480', 'DU479', 'D2U478', 'temporal31']
full_terms, floors = [], []
for row in geometry['rows']:
    scope = row['scope']
    for label, prefix in [('SOURCE_vs_complete_REAL', 'SOURCE_REAL__'), ('same_treated_REAL_vs_complete_REAL', 'REAL_control_REAL__')]:
        computed = []
        for name, item in zip(views, row[label]['views']):
            w = geo[prefix+scope+'__'+name+'__coordinate_W1']
            sw = geo[prefix+scope+'__'+name+'__sliced_W1']
            assert np.isfinite(w).all() and np.isfinite(sw).all() and np.min(w) >= 0 and np.min(sw) >= 0
            values = [float(w.mean()), float(np.sqrt(np.mean(w*w))), float(sw.mean())]
            close(values, [item['coordinate_mean'], item['coordinate_RMS'], item['sliced_mean']], label=scope+name)
            computed.append(values)
        c = np.array(computed)
        terms = np.array([.5*(c[:4, 0]+c[:4, 2]).mean(), c[:4, 1].mean(), .5*(c[4, 0]+c[4, 2])])
        close(terms, [row[label]['terms'][key] for key in term_names])
        if label.startswith('SOURCE'):
            full_terms.append(terms)
    floor_values = []
    for name in views:
        w = geo['REAL_pair_floor__'+scope+'__'+name+'__coordinate_W1']
        sw = geo['REAL_pair_floor__'+scope+'__'+name+'__sliced_W1']
        assert w.shape[0] == sw.shape[0] == 32 and np.isfinite(w).all() and np.min(w) >= 0
        floor_values.append(np.stack([w.mean(1), np.sqrt(np.mean(w*w, axis=1)), sw.mean(1)], axis=1))
    c = np.stack(floor_values)
    ft = np.stack([.5*(c[:4, :, 0]+c[:4, :, 2]).mean(0), c[:4, :, 1].mean(0), .5*(c[4, :, 0]+c[4, :, 2])], axis=1)
    floors.append(ft)
    for j, name in enumerate(term_names):
        close(ft[:, j], row['matched_N_REAL_pair_floor']['terms'][name]['values'])
    if scope != 'all':
        b = row['charge_mean_bound']
        for sign, numerator in [('output_specific', max(0., b['Dout_q']-b['Erep_q'])),
                                ('source_charge_form', max(0., b['Dsrc_q']-b['actual_postADC_total_mean_absolute_charge_error_q']-b['Erep_q']))]:
            close(b[sign+'_LB_coordinate_mean'], numerator/b['denominator_mean'])
            close(b[sign+'_LB_coordinate_RMS'], numerator/b['denominator_RMS'])
        assert b['nominal_inequality_checks_pass'] and all(b['nominal_inequality_checks'].values())
full_terms, floors = np.stack(full_terms), np.stack(floors)
mass = scount/6593
close((full_terms[1:]*mass[:, None]).sum(0), [geometry['original_SOURCE_mass_weighted_regional_terms'][key] for key in term_names])
wf = (floors[1:]*mass[:, None, None]).sum(0)
for j, key in enumerate(term_names):
    close(wf[:, j], geometry['matched_N_REAL_pair_weighted_regional_terms'][key]['values'])
geo.close()
checks['all_geometry_coordinate_reductions_real_pairs_region_weights_and_nominal_bound_arithmetic'] = True

parameter = read('PARAMETER_GRADIENT_SUMMARY.json')
endpoint = read('ENDPOINT_GRADIENT_SUMMARY.json')
manifest = read('PARAMETER_MANIFEST.json')
assert manifest['count'] == 1369889 and len({p['name'] for p in manifest['parameters']}) == len(manifest['parameters'])
position = 0
for p in manifest['parameters']:
    assert p['start'] == position and p['stop']-p['start'] == math.prod(p['shape'])
    position = p['stop']
assert position == 1369889
names = list(C['gradient_components'])+['total_D']
assert parameter['reported_order'] == endpoint['reported_order'] == names
with np.load(S/'FULL_REFERENCE_PARAMETER_GRADIENTS_PRIVATE.npz', allow_pickle=False) as z:
    reference = z['components']
assert reference.shape == (5, position) and reference.dtype == np.float64 and np.isfinite(reference).all()
sum_a, sum_b = np.zeros_like(reference), np.zeros_like(reference)
norms, errors = np.zeros((32, 2, 6)), np.zeros((32, 2, 6))
component = lambda matrix, j: matrix[j] if j < 5 else matrix.sum(0)


def squared(v):
    return float(np.dot(v, v))


for j, name in enumerate(names):
    close(squared(component(reference, j)), parameter['reference'][name]['squared_norm'], label='reference '+name)
for macro in range(32):
    with np.load(S/'parameter_gradients_private'/('macro_'+str(macro).zfill(2)+'.npz'), allow_pickle=False) as z:
        a, b = z['A'], z['B']
    assert a.shape == b.shape == reference.shape and a.dtype == b.dtype == np.float64
    assert np.isfinite(a).all() and np.isfinite(b).all()
    sum_a += a
    sum_b += b
    record = parameter['macro_records'][macro]
    for j, name in enumerate(names):
        f, av, bv = component(reference, j), component(a, j), component(b, j)
        f2 = squared(f)
        for arm, value in enumerate((av, bv)):
            label = ('A', 'B')[arm]
            n2, dot, err = squared(value), float(np.dot(value, f)), squared(value-f)
            norms[macro, arm, j], errors[macro, arm, j] = n2, err
            target = record[label][name]
            close([n2, dot, err], [target['squared_norm'], target['dot_reference'], target['squared_l2_error']], label=str(macro)+label+name)
            if n2 == 0 or f2 == 0:
                assert target['cosine_reference'] is None
            else:
                close((dot/math.sqrt(n2))/math.sqrt(f2), target['cosine_reference'])
        pair = record['paired'][name]
        close([float(np.dot(av, bv)), squared(av-bv), errors[macro, 1, j]-errors[macro, 0, j]],
              [pair['dot_A_B'], pair['squared_l2_A_minus_B'], pair['B_minus_A_squared_l2_error']])
    del a, b, av, bv, f
    if macro % 8 == 7:
        print(json.dumps(dict(phase='saved_parameter_vectors_checked', macros=macro+1)), flush=True)
means = [sum_a/32., sum_b/32.]
with np.load(S/'PARAMETER_MEAN_GRADIENTS_PRIVATE.npz', allow_pickle=False) as z:
    close(means[0], z['mean_A_components'])
    close(means[1], z['mean_B_components'])
    changes = errors[:, 1]-errors[:, 0]
    bt = changes[bootstrap].mean(1)
    close(bt, z['bootstrap_error_changes'])
    ci = np.quantile(bt, [.025, .975], axis=0, method='linear')
for j, name in enumerate(names):
    f = component(reference, j)
    for arm, label in enumerate(('A', 'B')):
        m = component(means[arm], j)
        mse, discrepancy = float(errors[:, arm, j].mean()), squared(m-f)
        dispersion = float(norms[:, arm, j].mean())-squared(m)
        target = parameter['arms'][label][name]
        close([mse, discrepancy, dispersion], [target['mean_squared_l2_error'], target['mean_gradient']['squared_l2_error'],
                                              target['empirical_dispersion_mean_squared_l2']], label=label+name)
        close(mse, discrepancy+dispersion, label='independent second-moment identity '+label+name)
    close([changes[:, j].mean(), *ci[:, j]], [parameter['paired'][name]['mean_B_minus_A_squared_l2_error'],
          *parameter['paired'][name]['bootstrap_percentile_95_interval']])
del reference, sum_a, sum_b, means, f, m
checks['all_32_saved_float64_parameter_pairs_mean_vectors_norms_dots_errors_and_paired_bootstrap'] = True

# Macro endpoint arrays are intentionally not saved. Audit the saved full/mean
# vectors and every scalar record's algebra without claiming a second VJP run.
with np.load(S/'FULL_REFERENCE_ENDPOINT_GRADIENTS_PRIVATE.npz', allow_pickle=False) as z:
    reference = z['components'].reshape(5, -1)
assert reference.shape == (5, 6593*480) and reference.dtype == np.float64 and np.isfinite(reference).all()
with np.load(S/'ENDPOINT_MEAN_GRADIENTS_PRIVATE.npz', allow_pickle=False) as z:
    means = [z['mean_A_components'], z['mean_B_components']]
    saved_bootstrap = z['bootstrap_error_changes']
for j, name in enumerate(names):
    f = component(reference, j)
    f2 = squared(f)
    close(f2, endpoint['reference'][name]['squared_norm'])
    for arm, label in enumerate(('A', 'B')):
        m = component(means[arm], j)
        close(squared(m-f), endpoint['arms'][label][name]['mean_gradient']['squared_l2_error'])
        for row in endpoint['macro_records']:
            target = row[label][name]
            close(target['squared_l2_error'], target['squared_norm']+f2-2*target['dot_reference'])
    delta = np.array([row['paired'][name]['B_minus_A_squared_l2_error'] for row in endpoint['macro_records']])
    exact_delta = np.array([row['B'][name]['squared_l2_error']-row['A'][name]['squared_l2_error'] for row in endpoint['macro_records']])
    close(delta, exact_delta)
    bt = delta[bootstrap].mean(1)
    close(bt, saved_bootstrap[:, j])
    close(np.quantile(bt, [.025, .975], method='linear'), endpoint['paired'][name]['bootstrap_percentile_95_interval'])
checks['endpoint_full_and_mean_vectors_scalar_algebra_and_bootstrap_without_new_network_calls'] = True
assert parameter['all_finite_sample_decompositions_passed'] and endpoint['all_finite_sample_decompositions_passed']
assert result['primary_parameter_gradient_comparison'] == parameter['paired']['total_D']
assert result['primary_endpoint_gradient_comparison'] == endpoint['paired']['total_D']
assert len(result['figures']) == 4 and all((S/name).is_file() for name in result['figures'])
done.set()
watcher.join()
probe()
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), pass_review=True, checks=checks, resource=resource,
               elapsed_s=time.monotonic()-START, new_model_forwards=0, new_model_or_endpoint_VJPs=0,
               DEV_EVAL_CAL_reads=0, training_updates=0, root_visual_QA_still_required=True,
               arithmetic_tolerance=dict(atol=1e-10, rtol=1e-10),
               scope='Saved numerical artifacts, original population/random-stream reconstruction and independent float64 scalar checks; no claim of a second scientific network replay.')
write('FROZEN_ARTIFACT_REVIEW.json', receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
