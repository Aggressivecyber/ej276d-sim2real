"""Registered R64 frozen finite-reference diagnostic; no optimizer or fitting."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import sys
import time
import traceback

START = time.monotonic()
R = Path(__file__).resolve().parent
C = json.loads((R/'config.json').read_text('utf8'))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
sys.path.insert(0, str(R/'distribution_adapter'))
import numpy as np
import scipy
from scipy.stats import spearmanr
import torch
from network import ResponseNetwork
from noise_core import bins_for, continuous_logq, innovation, q_np
from source_flow_core import endpoint, canonical_adc_torch, canonical_REAL_torch, q_torch
from residual_math import clean, array_sha, draw_directions, boundaries, psd, exact_w1_columns, summary, REGIONS, BOUNDARIES
from distribution_gradient import FrozenDistributionObjective, ParentLayout, paired_layouts, D_NAMES
from endpoint_geometry import analyze_endpoint_geometry, SCOPES
from gradient_statistics import GradientAccumulator
from portable_fixture import run_check
from network_replay_artificial import check as network_check


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    path = R/name
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes((json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    tmp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


counter = dict(response_training_runs=0, response_updates=0, HGB_fits=0, density_fits=0,
               teacher_waveforms=0, DEV_read=0, EVAL_read=0, CAL_decode=0,
               full_SOURCE_rows_completed=0, full_REAL_rows_completed=0,
               neural_forward_rows_completed=0, neural_forward_calls_completed=0,
               parameter_VJP_calls_completed=0, endpoint_objective_calls_completed=0,
               full_SOURCE_replay_passes_completed=0, paired_macro_batches_completed=0)


def progress(phase, **more):
    value = dict(UTC=utc(), PID=os.getpid(), phase=phase, counter=counter,
                 elapsed_s=time.monotonic()-START, **more)
    write('DIAGNOSTIC_PROGRESS.json', value)
    print(json.dumps(clean(value)), flush=True)


def gpu_guard():
    if max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > C['GPU_limit_bytes']:
        raise MemoryError('registered GPU safety limit exceeded')


def describe_shared(source, real, source_ids, h, sb, rb, ys, yr, ref):
    """Pure summaries of cached endpoints, preserving all parents and regions."""
    curves, rows, projection = {}, [], {}
    for scope in SCOPES:
        si = np.arange(len(source)) if scope == 'all' else np.flatnonzero(sb == int(scope[6:]))
        ri = np.arange(len(real)) if scope == 'all' else np.flatnonzero(rb == int(scope[6:]))
        pv = {name: psd(v) for name, v in [('SOURCE_raw', source[si]), ('SOURCE_preADC', ys['projected'][si]),
                                        ('SOURCE_canonical', ys['canonical'][si]), ('REAL_raw', real[ri]),
                                        ('REAL_canonical', ref[ri]), ('REAL_control_canonical', yr['canonical'][ri])]}
        rr = dict(scope=scope, SOURCE_N=len(si), REAL_N=len(ri),
                  SOURCE_raw_vs_REAL_raw_PSD_W1=float(exact_w1_columns(pv['SOURCE_raw'], pv['REAL_raw'])[0]),
                  SOURCE_preADC_vs_REAL_canonical_PSD_W1=float(exact_w1_columns(pv['SOURCE_preADC'], pv['REAL_canonical'])[0]),
                  SOURCE_canonical_vs_REAL_canonical_PSD_W1=float(exact_w1_columns(pv['SOURCE_canonical'], pv['REAL_canonical'])[0]),
                  REAL_control_vs_REAL_canonical_PSD_W1=float(exact_w1_columns(pv['REAL_control_canonical'], pv['REAL_canonical'])[0]))
        rank = lambda a, b: None if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b).statistic)
        rr['source_rank'] = dict(original_Q_vs_output_Q=rank(q_np(source[si]), q_np(ys['canonical'][si])),
                                 original_PSD_vs_output_PSD=rank(pv['SOURCE_raw'], pv['SOURCE_canonical']))
        reference_boundary = boundaries(ref[ri])
        rr['fixed_boundaries'] = {name: dict(zip(BOUNDARIES, exact_w1_columns(boundaries(v), reference_boundary).tolist()))
                                  for name, v in [('SOURCE_raw', source[si]), ('SOURCE_canonical', ys['canonical'][si]),
                                                  ('REAL_control_canonical', yr['canonical'][ri])]}
        for name, value in [('SOURCE_raw', source[si]), ('SOURCE_canonical', ys['canonical'][si]),
                            ('REAL_canonical', ref[ri]), ('REAL_control_canonical', yr['canonical'][ri])]:
            curves[name+'__'+scope+'__mean'] = value.mean(0, dtype=np.float64)
            curves[name+'__'+scope+'__SD'] = value.std(0, ddof=1, dtype=np.float64)
        rows.append(rr)
    for name, raw, data in [('SOURCE', source, ys), ('REAL_control', real, yr)]:
        q = q_np(raw)
        changed = data['projected']-data['raw_endpoint']
        initial = raw.astype('f8')+data['innovation']
        delta = data['raw_endpoint']-initial
        out = data['projected']
        projection[name] = dict(rows=len(raw), raw_endpoint_outside_box_N=int((np.abs(data['raw_endpoint']) > .5).any(1).sum()),
                                active_rail_samples=int((~data['free_mask']).sum()),
                                preADC_relative_Q_error_f64=summary(np.abs(q_np(out)-q)/q),
                                preADC_relative_Q_error_f32=summary(np.abs(q_np(out.astype('f4'))-q)/q),
                                canonical_absolute_Q_error=summary(np.abs(q_np(data['canonical'])-q)),
                                neural_displacement_RMS=summary(np.sqrt(np.mean(delta*delta, axis=1))),
                                projection_RMS=summary(np.sqrt(np.mean(changed*changed, axis=1))),
                                mean_absolute_neural_boundary_change=np.abs(boundaries(delta)).mean(0),
                                mean_absolute_projection_boundary_change=np.abs(boundaries(changed)).mean(0))
    np.savez_compressed(R/'WAVEFORM_STATISTICS_PRIVATE.npz', **curves)
    write('SUPPORTING_SUMMARY.json', dict(rows=rows, projection=projection, boundary_order=BOUNDARIES,
                                        source_ids_SHA256=array_sha(source_ids), h_SHA256=array_sha(h),
                                        endpoint='direct projected TRAIN endpoint', neural_calls_added=0,
                                        interpretation='Pure complete TRAIN endpoint descriptions; no density/HGB fit or acceptance claim.'))


try:
    assert tuple(C['gradient_components']) == tuple(D_NAMES)
    assert C['paired_macro_batches'] == 32 and C['macro_per_region'] == 128
    assert C['microbatches_per_macro'] == 4 and C['microbatch_per_region'] == 32
    assert C['generation_chunk'] == 64 and C['neural_replay_full_SOURCE_passes'] == 33
    assert C['response_training_runs'] == C['response_updates'] == C['HGB_fits'] == C['density_fits'] == C['DEV_read'] == C['EVAL_read'] == C['CAL_decode'] == 0
    assert all(C[k] is None for k in ('science_budget', 'deadline_UTC', 'output_limit_bytes'))
    torch.set_num_threads(C['CPU_threads'])
    torch.set_num_interop_threads(C['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)
    binding = json.loads((R/'INPUT_BINDING.json').read_text('utf8'))
    for name, record in binding['frozen_R63_helpers'].items():
        assert file_sha(R/name) == record['sha256'], name
    review = json.loads((R/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
    for record in review['files']:
        assert file_sha(R/record['name']) == record['sha256'], record['name']
    assert torch.cuda.is_available()
    dev = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
    env = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, scipy=scipy.__version__,
               device=str(dev), GPU=torch.cuda.get_device_name(0), PID=os.getpid(), CPU_threads=torch.get_num_threads(),
               CPU_affinity=sorted(os.sched_getaffinity(0)), cwd=str(R), UTC=utc(),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               MHA_fastpath=torch.backends.mha.get_fastpath_enabled())
    assert all(env[k] == C['expected_R33_environment'][k] for k in ('python', 'numpy', 'torch', 'scipy', 'device', 'GPU'))
    write('ENVIRONMENT.json', env)
    progress('artificial_checks_before_scientific_files')
    for device, name in [('cpu', 'CPU'), (dev, 'CUDA')]:
        try:
            receipt = run_check(device=device, adapter_root=R/'distribution_adapter')
        except Exception as exc:
            if hasattr(exc, 'receipt'):
                write('DISTRIBUTION_ARTIFICIAL_'+name+'_FAILURE.json', exc.receipt)
            raise
        write('DISTRIBUTION_ARTIFICIAL_'+name+'.json', receipt)
        assert receipt['compact']['passed'] and receipt['compact']['checks'] == 69
        del receipt
        progress('integrated_artificial_'+name+'_passed')
    for device, chunk, name in [('cpu', 4, 'CPU'), (dev, 64, 'CUDA')]:
        receipt = network_check(device=device, chunk=chunk)
        write('NETWORK_ARTIFICIAL_'+name+'.json', receipt)
        assert receipt['pass']
        del receipt
        progress('actual_network_artificial_'+name+'_passed')
    gpu_guard()
    torch.cuda.empty_cache()

    def verify_inputs():
        for key, path in C['input_paths'].items():
            assert file_sha(path) == binding['inputs'][key]['sha256'], key

    verify_inputs()
    old = json.loads(Path(C['input_paths']['frozen_training_config']).read_text('utf8'))
    training = json.loads(Path(C['input_paths']['frozen_training_receipt']).read_text('utf8'))
    assert old['batch_per_region'] == 32 and old['projection_directions'] == 64
    assert training['updates'] == 50000 and training['final_state_SHA'] == binding['frozen_weight_tensor_SHA']
    with np.load(C['input_paths']['TRAIN_path'], allow_pickle=False) as z:
        source = z['source'].astype('f4')
        real = z['target'].astype('f4')
        source_ids, h = z['source_ids'].copy(), z['h'].copy()
        saved_s, saved_r = z['source_bin'].copy(), z['target_bin'].copy()
    assert source.shape == (6593, 480) and real.shape == (21047, 480)
    assert array_sha(source_ids) == binding['source_parent_ID_SHA']
    write('DATA_ACCESS_RECEIPT.json', dict(archive=C['input_paths']['TRAIN_path'],
          decoded_fields=['source', 'target', 'source_ids', 'h', 'source_bin', 'target_bin'],
          source_shape=source.shape, REAL_shape=real.shape, source_parent_ID_SHA=array_sha(source_ids),
          other_data_archives_decoded=[], DEV_read=0, EVAL_read=0, CAL_decode=0,
          scientific_arrays_opened_only_after_CPU_CUDA_artificial_pass=True))
    with np.load(C['input_paths']['constants_path'], allow_pickle=False) as z:
        mu, scale, edges = z['mu_unit_q'].copy(), z['scale_unit_q'].copy(), z['edges_logQ'].copy()
    with np.load(C['input_paths']['noise_calibration_path'], allow_pickle=False) as z:
        noise = {k: z[k].copy() for k in z.files}
    with np.load(C['input_paths']['distribution_statistics'], allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in z.files}
    with np.load(C['input_paths']['temporal_statistics'], allow_pickle=False) as z:
        for k in z.files:
            if k in stats:
                assert np.array_equal(stats[k], z[k]), 'overlapping constant mismatch: '+k
            else:
                stats[k] = z[k].copy()
    sb, sraw, sq = bins_for(source, edges)
    rb, rraw, rq = bins_for(real, edges)
    assert np.array_equal(sraw, saved_s) and np.array_equal(rraw, saved_r)
    assert np.array_equal(stats['source_counts'], [(sraw == b).sum() for b in REGIONS])
    assert np.array_equal(stats['REAL_counts'], [(rraw == b).sum() for b in REGIONS])
    state = torch.load(C['input_paths']['frozen_weight'], map_location='cpu', weights_only=True)
    initial_weight_hash = state_sha(state)
    assert initial_weight_hash == binding['frozen_weight_tensor_SHA']
    model = ResponseNetwork().to(dev)
    model.load_state_dict(state, strict=True)
    del state
    model.train()
    model.requires_grad_(True)
    parameters = tuple(model.parameters())
    manifest, offset = [], 0
    for name, p in model.named_parameters():
        manifest.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype), start=offset, stop=offset+p.numel()))
        offset += p.numel()
    assert offset == C['native_parameter_count'] == training['network_parameter_count']
    write('PARAMETER_MANIFEST.json', dict(parameters=manifest, count=offset, named_parameter_order=True))
    streams = {k: np.random.Generator(np.random.PCG64(seed)) for k, seed in C['PCG64_seeds'].items()}
    chains = {k: bytes(32) for k in streams}
    consumed = {k: dict(calls=0, elements=0, bytes=0) for k in streams}

    def consume(name, value):
        arr = np.ascontiguousarray(value)
        chains[name] = hashlib.sha256(chains[name]+arr.tobytes()).digest()
        consumed[name]['calls'] += 1
        consumed[name]['elements'] += arr.size
        consumed[name]['bytes'] += arr.nbytes

    write('INITIAL_RANDOM_STATES.json', {k: copy.deepcopy(v.bit_generator.state) for k, v in streams.items()})
    epsS = streams['full_SOURCE_epsilon'].standard_normal(source.shape).astype('<f4')
    epsR = streams['full_REAL_epsilon'].standard_normal(real.shape).astype('<f4')
    consume('full_SOURCE_epsilon', epsS)
    consume('full_REAL_epsilon', epsR)
    directions = draw_directions(streams['full_feature_directions'])
    for a in directions:
        consume('full_feature_directions', a)
    for key, sha in C['expected_full_stream_hashes'].items():
        assert chains[key].hex() == sha, key
    td = streams['temporal_feature_directions'].standard_normal((31, 64))
    td = (td/np.linalg.norm(td, axis=0, keepdims=True)).astype('<f4')
    consume('temporal_feature_directions', td)
    macro_s, macro_r = np.empty((32, 10, 4, 32), dtype='<u2'), np.empty((32, 10, 4, 32), dtype='<u2')
    for k in range(32):
        for j, region in enumerate(REGIONS):
            for name, bins, out in [('macro_SOURCE_rows', sraw, macro_s), ('macro_REAL_rows', rraw, macro_r)]:
                draw = streams[name].choice(np.flatnonzero(bins == region), size=128, replace=True).astype('<u2')
                consume(name, draw)
                out[k, j] = draw.reshape(4, 32)
    bootstrap = streams['gradient_paired_bootstrap'].integers(0, 32, size=(1000, 32)).astype('<u2')
    consume('gradient_paired_bootstrap', bootstrap)
    floor, design = {}, dict(macro_SOURCE_indices=macro_s, macro_REAL_indices=macro_r, paired_bootstrap_indices=bootstrap,
                             temporal_directions=td, **{'directions_'+str(j): a for j, a in enumerate(directions)})
    for scope in SCOPES:
        count = len(source) if scope == 'all' else int((sraw == int(scope[6:])).sum())
        pool = np.arange(len(real)) if scope == 'all' else np.flatnonzero(rraw == int(scope[6:]))
        floor[scope] = {}
        for arm in ('A', 'B'):
            name = 'REAL_floor_'+arm+'_rows'
            indices = streams[name].choice(pool, size=(32, count), replace=True).astype('<u2')
            consume(name, indices)
            floor[scope][arm] = indices
            design['REAL_floor__'+scope+'__'+arm] = indices
    np.savez_compressed(R/'FROZEN_RANDOM_DESIGN_PRIVATE.npz', **design)
    mt, st = (torch.as_tensor(v, dtype=torch.float64, device=dev) for v in (mu, scale))

    def prepare_inputs(x, eps):
        b, bins, q = bins_for(x, edges)
        hq = continuous_logq(q, b, edges)
        n = np.empty(x.shape, dtype='f8')
        for lo in range(0, len(x), 64):
            sl = slice(lo, min(lo+64, len(x)))
            n[sl], _ = innovation(x[sl], eps[sl], mu, edges, noise['p480'], noise['kappa2'], float(noise['tau2']), True)
        return dict(x=x, q=q, b=b, raw_bins=bins, condition=hq, innovation=n, epsilon=eps)

    sx, rx = prepare_inputs(source, epsS), prepare_inputs(real, epsR)

    def forward(data, sl):
        x, n, q, b, hq = [torch.as_tensor(data[k][sl], dtype=d, device=dev) for k, d in
                          [('x', torch.float64), ('innovation', torch.float64), ('q', torch.float64),
                           ('b', torch.int64), ('condition', torch.float64)]]
        out, raw = endpoint(model, x, n, q, b, hq, mt, st)
        counter['neural_forward_rows_completed'] += len(out)
        counter['neural_forward_calls_completed'] += 1
        return out, raw

    def collect(data, name):
        n = len(data['x'])
        saved = {key: np.empty((n, 480), dtype=dtype) for key, dtype in
                 [('projected', 'f8'), ('raw_endpoint', 'f8'), ('canonical', 'f8'), ('codes', '<u2'), ('free_mask', '?')]}
        saved['innovation'] = data['innovation']
        for lo in range(0, n, 64):
            sl = slice(lo, min(lo+64, n))
            with torch.no_grad():
                out, raw = forward(data, sl)
                can = canonical_adc_torch(out, False)
                saved['projected'][sl] = out.cpu().numpy()
                saved['raw_endpoint'][sl] = raw.cpu().numpy()
                saved['canonical'][sl] = can.cpu().numpy()
                saved['codes'][sl] = torch.round((can+.5)*16383).cpu().numpy().astype('<u2')
                saved['free_mask'][sl] = ((out > -.5) & (out < .5)).cpu().numpy()
            assert np.isfinite(saved['projected'][sl]).all()
            assert np.max(np.abs(q_np(saved['projected'][sl])-data['q'][sl])/data['q'][sl]) <= 1e-5
            counter['full_SOURCE_rows_completed' if name == 'SOURCE' else 'full_REAL_rows_completed'] = sl.stop
            if lo % (64*32) == 0:
                progress(name+'_full_forward', completed=sl.stop, total=n)
        gpu_guard()
        np.savez_compressed(R/(name+'_ENDPOINTS_PRIVATE.npz'), **saved, source_raw=data['x'],
                            epsilon=data['epsilon'], original_q=data['q'], original_raw_bins=data['raw_bins'],
                            clipped_network_bins=data['b'], continuous_condition=data['condition'])
        return saved

    ys, yr = collect(sx, 'SOURCE'), collect(rx, 'REAL_control')
    assert counter['neural_forward_rows_completed'] == C['full_parent_rows'] == 27640
    with torch.no_grad():
        ref_t = canonical_REAL_torch(torch.as_tensor(real, dtype=torch.float64, device=dev))
    ref = ref_t.cpu().numpy()
    np.savez_compressed(R/'PARENT_IDENTITIES_PRIVATE.npz', SOURCE_ids=source_ids, h=h,
                        SOURCE_original_order=np.arange(len(source)), REAL_original_order=np.arange(len(real)))
    write('FORWARD_RECEIPT.json', dict(endpoint=C['endpoint_definition'], SOURCE_rows=6593, REAL_rows=21047,
          complete_population_neural_calls=counter['neural_forward_calls_completed'], chunk=64,
          SOURCE_final_chunk=6593 % 64, REAL_final_chunk=21047 % 64, model_training_mode=model.training,
          MHA_fastpath=torch.backends.mha.get_fastpath_enabled(), reference='canonical raw REAL without network or innovation',
          SOURCE_projected_SHA256=array_sha(ys['projected']), REAL_control_projected_SHA256=array_sha(yr['projected'])))
    progress('full_shared_endpoint_geometry')
    geometry, private = analyze_endpoint_geometry(ys['projected'], ys['canonical'], ref, source, real, sraw, rraw, sq,
          stats, directions, td, floor, REAL_control_canonical=yr['canonical'], expected_population=(6593, 21047),
          expected_reference_replicates=32)
    assert geometry['charge_bound_aggregation']['all_nominal_inequality_checks_pass']
    write('ENDPOINT_GEOMETRY_SUMMARY.json', geometry)
    np.savez_compressed(R/'ENDPOINT_GEOMETRY_PRIVATE.npz', **private)
    del private
    describe_shared(source, real, source_ids, h, sraw, rraw, ys, yr, ref)
    objective = FrozenDistributionObjective(stats, C, directions, td, sraw, rraw, feature_chunk=64, device=dev)
    parent_y = torch.as_tensor(ys['projected'], dtype=torch.float64, device=dev)
    reference_raw = torch.as_tensor(real, dtype=torch.float64, device=dev)

    def dgrad(layout):
        result = objective.endpoint_gradients(parent_y, reference_raw, layout)
        counter['endpoint_objective_calls_completed'] += 1
        g = np.stack([result['gradients'][k].cpu().numpy() for k in D_NAMES])
        v = np.array([result['values'][k] for k in D_NAMES], dtype='f8')
        assert np.isfinite(g).all() and np.isfinite(v).all()
        del result
        gpu_guard()
        return v, g

    replay_receipts = []

    def parameter_vjp(gradients, label):
        # Every parent participates in the original 64-row schedule in every pass.
        assert gradients.ndim == 3 and gradients.shape[1:] == source.shape
        answer = np.zeros((len(gradients), offset), dtype='f8')
        checked = []
        for lo in range(0, len(source), 64):
            sl = slice(lo, min(lo+64, len(source)))
            out, raw = forward(sx, sl)
            same = torch.equal(out.detach(), torch.as_tensor(ys['projected'][sl], device=dev))
            can = canonical_adc_torch(out.detach(), False)
            code = torch.round((can+.5)*16383).to(torch.int32)
            same_codes = torch.equal(code, torch.as_tensor(ys['codes'][sl], dtype=torch.int32, device=dev))
            same_free = torch.equal((out.detach() > -.5) & (out.detach() < .5), torch.as_tensor(ys['free_mask'][sl], device=dev))
            assert same and same_codes and same_free, (label, lo, same, same_codes, same_free)
            for j, values in enumerate(gradients):
                gy = torch.as_tensor(values[sl], dtype=torch.float64, device=dev)
                vg = torch.autograd.grad(out, parameters, grad_outputs=gy, retain_graph=j+1 < len(gradients))
                flat = torch.cat([v.detach().reshape(-1) for v in vg]).cpu().numpy().astype('f8')
                assert np.isfinite(flat).all()
                answer[j] += flat
                counter['parameter_VJP_calls_completed'] += 1
                del vg, flat, gy
            checked.append(dict(start=lo, stop=sl.stop, endpoint_bitwise=True, codes_exact=True, free_mask_exact=True))
            del out, raw, can, code
            if lo % (64*16) == 0:
                progress('parameter_replay', pass_label=label, completed=sl.stop, total=len(source))
            gpu_guard()
        counter['full_SOURCE_replay_passes_completed'] += 1
        replay_receipts.append(dict(pass_label=label, VJPs_per_chunk=len(gradients), chunks=checked,
                                    zero_derivative_VJP_skips=0, all_original_SOURCE_rows_replayed=True))
        write('ENDPOINT_REPLAY_RECEIPT.json', dict(passes=replay_receipts, canonical_STE_applied_once_upstream=True,
              injected_at='projected output before ADC', Box_backward_applied_once=True,
              accumulation='named parameters; original chunks; float64 CPU; no additional division'))
        return answer

    progress('full_finite_reference_gradient')
    full_values, full_gy = dgrad(ParentLayout.full_reference(sraw, rraw))
    np.savez_compressed(R/'FULL_REFERENCE_ENDPOINT_GRADIENTS_PRIVATE.npz', components=full_gy)
    full_gp = parameter_vjp(full_gy, 'full_reference')
    np.savez_compressed(R/'FULL_REFERENCE_PARAMETER_GRADIENTS_PRIVATE.npz', components=full_gp)
    write('FULL_REFERENCE_LOSS.json', dict(components=dict(zip(D_NAMES, full_values.tolist())), total_D=float(full_values.sum()),
          numerical_device=str(dev), parameter_count=offset, SOURCE_N=6593, REAL_N=21047,
          coordinate='projected-before-ADC parent endpoint; original CUDA feature chain and selected STE gradients',
          reference_noise_law='one fixed innovation per SOURCE parent; canonical raw REAL',
          C_terms_excluded=C['excluded_gradient_components']))
    eg = GradientAccumulator(full_gy.reshape(5, -1), bootstrap, coordinate_label='endpoint')
    pg = GradientAccumulator(full_gp, bootstrap, coordinate_label='parameter')
    macro_losses = []
    vectors = R/'parameter_gradients_private'
    vectors.mkdir(exist_ok=False)
    for macro in range(32):
        progress('macro_endpoint_gradients', macro_index=macro)
        layouts, lb = paired_layouts(macro_s[macro], macro_r[macro], sraw, rraw)
        ga, va = np.zeros_like(full_gy), np.zeros(5, dtype='f8')
        micro_values = []
        for layout in layouts:
            v, g = dgrad(layout)
            va += v/4.
            ga += g/4.
            micro_values.append(v.tolist())
            del g
        vb, gb = dgrad(lb)
        er = eg.update(ga.reshape(5, -1), gb.reshape(5, -1))
        joined = np.concatenate((ga, gb), axis=0)
        parameters_ab = parameter_vjp(joined, 'macro_'+str(macro).zfill(2))
        del joined, ga, gb
        pa, pb = parameters_ab[:5], parameters_ab[5:]
        pr = pg.update(pa, pb)
        np.savez_compressed(vectors/('macro_'+str(macro).zfill(2)+'.npz'), A=pa, B=pb)
        record = dict(macro_index=macro, A_components=va, B_components=vb, A_total_D=float(va.sum()),
                      B_total_D=float(vb.sum()), four_micro_losses=micro_values, endpoint_record=er, parameter_record=pr)
        macro_losses.append(record)
        write('MACRO_LOSS_AND_GRADIENT_RECORDS.json', dict(component_order=D_NAMES, macros=macro_losses))
        counter['paired_macro_batches_completed'] = macro+1
        progress('macro_complete', macro_index=macro)
        del parameters_ab, pa, pb
    e_summary, p_summary = eg.finalize(), pg.finalize()
    for label, result in [('ENDPOINT', e_summary), ('PARAMETER', p_summary)]:
        write(label+'_GRADIENT_SUMMARY.json', result.summary)
        np.savez_compressed(R/(label+'_MEAN_GRADIENTS_PRIVATE.npz'), mean_A_components=result.mean_A[:5],
                            mean_B_components=result.mean_B[:5], bootstrap_error_changes=result.bootstrap_error_changes)
    assert counter['neural_forward_rows_completed'] == C['planned_total_network_rows'] == 245209
    assert counter['neural_forward_calls_completed'] == 433+33*104
    assert counter['parameter_VJP_calls_completed'] == 104*(5+32*10)
    assert counter['endpoint_objective_calls_completed'] == 1+32*5
    assert counter['full_SOURCE_replay_passes_completed'] == 33 and counter['paired_macro_batches_completed'] == 32
    assert all(p.grad is None and p.requires_grad for p in parameters)
    final_weight_hash = state_sha(model.state_dict())
    assert final_weight_hash == initial_weight_hash
    verify_inputs()
    write('FINAL_RANDOM_STATES.json', {k: copy.deepcopy(v.bit_generator.state) for k, v in streams.items()})
    write('RANDOM_STREAM_RECEIPT.json', dict(seeds=C['PCG64_seeds'], chains={k: v.hex() for k, v in chains.items()},
          consumed=consumed, full_three_chains_match_frozen_R60_R62=True,
          algorithm='SHA256(previous32bytes + contiguous consumed bytes); initial32 zero bytes; see REPRODUCTION for draw order',
          original_parent_order_preserved=True, original_two_domain_masses_preserved=True))
    result = dict(UTC=utc(), PID=os.getpid(), status='fixed_frozen_TRAIN_finite_reference_diagnostic_complete',
                  model_goal_complete=False, original_A_F_changed=False, counter=counter,
                  full_reference_loss=dict(zip(D_NAMES, full_values.tolist())),
                  primary_parameter_gradient_comparison=p_summary.summary['paired']['total_D'],
                  primary_endpoint_gradient_comparison=e_summary.summary['paired']['total_D'],
                  charge_bound=geometry['charge_bound_aggregation'],
                  frozen_weight_before_SHA=initial_weight_hash, frozen_weight_after_SHA=final_weight_hash,
                  all_input_hashes_unchanged_after=True, all_replayed_endpoints_bitwise=True,
                  elapsed_s=time.monotonic()-START, GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                  GPU_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                  artificial_network_rows_are_separate_from_scientific_counts=True,
                  scientific_scope='Conditional fixed finite TRAIN reference and five D terms only; no optimizer, new method or A-F determination.')
    write('RESULT.json', result)
    progress('rendering_registered_figures')
    from render_diagnostic import render
    figures = render(R)
    write('FIGURE_MANIFEST.json', dict(figures=figures, actual_root_visual_QA_pending=True))
    result.update(figures=figures, elapsed_s=time.monotonic()-START)
    write('RESULT.json', result)
    progress('registered_diagnostic_endpoint_complete')
except BaseException as exc:
    write('DIAGNOSTIC_FAILURE.json', dict(UTC=utc(), PID=os.getpid(), error=repr(exc), traceback=traceback.format_exc(),
          counter=counter, counter_completed_rows_is_lower_bound_after_exception=True, elapsed_s=time.monotonic()-START))
    raise
