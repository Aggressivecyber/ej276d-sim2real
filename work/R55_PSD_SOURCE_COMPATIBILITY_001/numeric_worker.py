from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text('utf8'))
START = time.monotonic()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CFG['CPU_threads']])
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[key] = str(CFG['CPU_threads'])
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits
from certificate_core import certify, dyadic, float_down

threadpool_limits(CFG['CPU_threads'])
COUNTER = {'LP_attempts': 0, 'certificate_calculations': 0, 'response_updates': 0,
           'HGB_fits': 0, 'density_fits': 0, 'ODE_generation': 0, 'EVAL_read': 0, 'CAL_decode': 0}


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(name, value):
    (ROOT / name).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


def progress(phase, **values):
    row = {'UTC': datetime.now(timezone.utc).isoformat(), 'PID': os.getpid(), 'phase': phase,
           'elapsed_s': time.monotonic() - START, 'counter': COUNTER, **values}
    write('PROGRESS.json', row)
    print(json.dumps(safe(row)), flush=True)


def load(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while part := f.read(1 << 20):
            h.update(part)
    return h.hexdigest()


def float_up(value):
    out = float(value)
    if dyadic(out) < value:
        out = math.nextafter(out, math.inf)
    assert dyadic(out) >= value
    return out


class Builder:
    def __init__(self, variables):
        self.variables = variables
        self.row = []
        self.col = []
        self.data = []
        self.rhs = []
        self.kinds = []
        self.kind_counts = Counter()
        self.max_rhs_relaxation = Fraction(0)
        self.sum_rhs_relaxation = Fraction(0)

    def add(self, terms, rhs, kind):
        exact = defaultdict(Fraction)
        for column, coefficient in terms:
            exact[int(column)] += coefficient
        rhs = Fraction(rhs)
        padding = Fraction(0)
        index = len(self.rhs)
        for column, coefficient in exact.items():
            if not coefficient:
                continue
            value = float(coefficient)
            delta = dyadic(value) - coefficient
            padding += max(delta, Fraction(0))  # every variable has upper bound one
            self.row.append(index)
            self.col.append(column)
            self.data.append(value)
        relaxed = float_up(rhs + padding)
        assert dyadic(relaxed) >= rhs + padding
        relaxation = dyadic(relaxed) - rhs
        self.max_rhs_relaxation = max(self.max_rhs_relaxation, relaxation)
        self.sum_rhs_relaxation += relaxation
        self.rhs.append(relaxed)
        self.kinds.append(kind)
        self.kind_counts[kind] += 1

    def equality(self, terms, rhs, kind):
        self.add(terms, rhs, kind + '_upper')
        self.add([(c, -v) for c, v in terms], -rhs, kind + '_lower')

    def materialize(self):
        a = csr_matrix((self.data, (self.row, self.col)), shape=(len(self.rhs), self.variables), dtype='f8')
        a.sort_indices()
        return a, np.asarray(self.rhs, dtype='f8')


def execute():
    progress('validate_frozen_inputs')
    binding = json.loads((ROOT / 'INPUT_BINDING.json').read_text('utf8'))
    for name, row in binding['inputs'].items():
        assert row['path'] == CFG['input_paths'][name]
        assert Path(row['path']).stat().st_size == row['bytes'] and hash_file(row['path']) == row['sha256'], name
    paths = CFG['input_paths']
    feature = load(paths['features'], ['SOURCE_INPUT__Qtotal_mVns', 'SOURCE_INPUT__valid',
                                    'REAL__Qtotal_mVns', 'REAL__PSD', 'REAL__valid'])
    source = load(paths['actual_inputs'], ['source_ids', 'source_waveforms_V'])
    meta = load(paths['source_metadata'], ['source_ids', 'composition'])
    output_ids = load(paths['response_ids'], ['source_ids'])['source_ids']
    assert np.array_equal(source['source_ids'], meta['source_ids']) and np.array_equal(source['source_ids'], output_ids)
    assert len(source['source_ids']) == CFG['SOURCE_N'] == 2000
    assert len(set(source['source_ids'].tolist())) == 2000
    assert len(feature['REAL__PSD']) == CFG['REAL_N'] == 2000
    assert feature['SOURCE_INPUT__valid'].all() and feature['REAL__valid'].all()
    assert np.isfinite(feature['REAL__PSD']).all()
    composition = meta['composition']
    assert dict(Counter(composition.tolist())) == {'mixed': 65, 'pure_EE': 417, 'pure_had': 1518}
    x = source['source_waveforms_V'].astype('f8')
    assert x.shape == (2000, 480) and np.isfinite(x).all() and np.max(np.abs(x)) <= 1.
    q_runtime = x[:, 100:400].sum(1) - 3. * x[:, :100].sum(1)
    q_source = feature['SOURCE_INPUT__Qtotal_mVns'] / 1000.
    assert (q_runtime > 0).all() and np.max(np.abs(q_runtime - q_source)) < 1e-7
    M = json.loads(Path(paths['metrics']).read_text('utf8'))
    valley = json.loads(Path(paths['valley_definition']).read_text('utf8'))
    width = json.loads(Path(paths['width_definition']).read_text('utf8'))
    edges = np.asarray(width['edges_logQ'], dtype='f8')
    assert np.array_equal(edges, M['PSD_main']['edges_logQ']) and len(edges) == 9

    def bins(q_mVns):
        logarithm = np.log(q_mVns)
        b = np.searchsorted(edges, logarithm, side='right') - 1
        b[logarithm == edges[-1]] = 7
        return b

    input_bins = bins(feature['SOURCE_INPUT__Qtotal_mVns'])
    real_bins = bins(feature['REAL__Qtotal_mVns'])
    labels = np.zeros(2000, dtype='i8')
    primary = M['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']['BASE']
    constraints = {}
    for b in [2, 3]:
        mask = (composition == 'pure_had') & (input_bins == b)
        assert int(mask.sum()) == CFG['expected_purehad_counts']['bin' + str(b + 1)]
        labels[mask] = b + 1
        original = primary['bin' + str(b + 1)]
        assert original['source_count'] == int(mask.sum()) and valley['bins'][b]['valid']
        lo, hi = valley['bins'][b]['low'], valley['bins'][b]['high']
        assert np.isfinite([lo, hi]).all() and lo < hi
        constraints[b + 1] = {'N': int(mask.sum()), 'low': lo, 'high': hi,
                              'BASE_fraction': original['valley_fraction']}
    radius = CFG['ADC_q_bound_LSB'] / 16383. + CFG['q_absolute_margin_Vns']
    qlo = q_runtime * (1. - CFG['q_relative_tolerance']) - radius
    qhi = q_runtime * (1. + CFG['q_relative_tolerance']) + radius
    loglo = np.full(2000, -np.inf)
    positive = qlo > 0
    loglo[positive] = np.log(1000. * qlo[positive]) - CFG['log_boundary_margin']
    loghi = np.log(1000. * qhi) + CFG['log_boundary_margin']
    ranges = {-1: (-np.inf, edges[0]), 8: (edges[-1], np.inf)}
    ranges.update({j: (edges[j], edges[j + 1]) for j in range(8)})
    event_possible = [tuple(j for j in range(-1, 9) if loghi[i] >= ranges[j][0] and loglo[i] <= ranges[j][1]) for i in range(2000)]
    assert all(p and int(input_bins[i]) in p for i, p in enumerate(event_possible))
    members = defaultdict(list)
    for i, possible in enumerate(event_possible):
        members[(int(labels[i]), possible)].append(i)
    groups = [{'label': key[0], 'possible_bins': key[1], 'indices': members[key], 'N': len(members[key])}
              for key in sorted(members)]
    assert sum(g['N'] for g in groups) == 2000
    masses = [Fraction(g['N'], 2000) for g in groups]
    allocations = [(g, j) for g, row in enumerate(groups) for j in row['possible_bins']]
    by_bin = {j: [i for i, (_, actual) in enumerate(allocations) if actual == j] for j in range(-1, 9)}
    by_group = {g: [i for i, (actual, _) in enumerate(allocations) if actual == g] for g in range(len(groups))}
    mass_bounds = {}
    for j in range(-1, 9):
        lower = Fraction(sum(g['N'] for g in groups if g['possible_bins'] == (j,)), 2000)
        upper = Fraction(sum(g['N'] for g in groups if j in g['possible_bins']), 2000)
        if j in range(7):
            lower = max(lower, Fraction(CFG['min_common_bin_N'], 2000))
        assert lower <= upper, ('support_bounds_conflict', j, lower, upper)
        mass_bounds[j] = (lower, upper)
    real_psd = feature['REAL__PSD']
    grid = np.unique(np.r_[real_psd, [v for row in constraints.values() for v in [row['low'], row['high']]]])
    assert np.isfinite(grid).all() and np.all(np.diff(grid) > 0)
    K = len(grid) - 1
    widths = [dyadic(grid[k + 1]) - dyadic(grid[k]) for k in range(K)]
    assert all(w > 0 for w in widths)
    real_cdf_counts = np.searchsorted(np.sort(real_psd), grid[:-1], side='right')
    conditional_counts = {}
    real_N = {}
    for j in range(7):
        values = np.sort(real_psd[real_bins == j])
        assert len(values) >= 40
        real_N[j] = len(values)
        conditional_counts[j] = np.searchsorted(values, grid[:-1], side='right')
    nalloc = len(allocations)
    cursor = 0
    mass_index = np.arange(cursor, cursor + nalloc)
    cursor += nalloc
    cdf_index = np.arange(cursor, cursor + nalloc * K).reshape(nalloc, K)
    cursor += nalloc * K
    F_index = np.arange(cursor, cursor + 7 * K).reshape(7, K)
    cursor += 7 * K
    E_index = np.arange(cursor, cursor + 7 * K).reshape(7, K)
    cursor += 7 * K
    T_index = np.arange(cursor, cursor + K)
    cursor += K
    objective = np.zeros(cursor, dtype='f8')
    objective[T_index] = [float_down(w) for w in widths]
    builder = Builder(cursor)
    progress('build_LP', groups=len(groups), allocations=nalloc, grid_intervals=K, variables=cursor)
    for g in range(len(groups)):
        builder.equality([(mass_index[a], 1) for a in by_group[g]], masses[g], 'group_mass')
    for j in range(-1, 9):
        lo, hi = mass_bounds[j]
        terms = [(mass_index[a], 1) for a in by_bin[j]]
        builder.add(terms, hi, 'bin_mass_upper')
        builder.add([(i, -c) for i, c in terms], -lo, 'bin_mass_lower')
    for a in range(nalloc):
        for k in range(K):
            builder.add([(cdf_index[a, k], 1), (mass_index[a], -1)], 0, 'joint_CDF_mass')
            if k:
                builder.add([(cdf_index[a, k - 1], 1), (cdf_index[a, k], -1)], 0, 'joint_CDF_monotone')
    for j in range(7):
        lo, hi = mass_bounds[j]
        st = [(mass_index[a], 1) for a in by_bin[j]]
        for k in range(K):
            f, e = F_index[j, k], E_index[j, k]
            ct = [(cdf_index[a, k], 1) for a in by_bin[j]]
            builder.add([(f, lo)] + [(i, -c) for i, c in ct], 0, 'McCormick_lower1')
            builder.add(ct + [(f, -hi)], 0, 'McCormick_upper1')
            builder.add([(f, hi)] + st + [(i, -c) for i, c in ct], hi, 'McCormick_lower2')
            builder.add(ct + [(f, -lo)] + [(i, -c) for i, c in st], -lo, 'McCormick_upper2')
            if k:
                builder.add([(F_index[j, k - 1], 1), (f, -1)], 0, 'conditional_CDF_monotone')
            target = Fraction(int(conditional_counts[j][k]), real_N[j])
            builder.add([(f, 1), (e, -1)], target, 'conditional_absolute_plus')
            builder.add([(f, -1), (e, -1)], -target, 'conditional_absolute_minus')
    margin = Fraction(str(CFG['diagnostic_PSD_margin']))
    conditional_limit = 7 * (Fraction(str(CFG['original_conditional_mean_W1_max'])) + margin)
    builder.add([(E_index[j, k], widths[k]) for j in range(7) for k in range(K)], conditional_limit, 'conditional_W1_budget')
    for k in range(K):
        ct = [(cdf_index[a, k], 1) for a in range(nalloc)]
        target = Fraction(int(real_cdf_counts[k]), 2000)
        builder.add(ct + [(T_index[k], -1)], target, 'overall_absolute_plus')
        builder.add([(i, -c) for i, c in ct] + [(T_index[k], -1)], -target, 'overall_absolute_minus')
    valley_lp_rows = {}
    for label, row in constraints.items():
        low = int(np.flatnonzero(grid == row['low'])[0])
        high = int(np.flatnonzero(grid == row['high'])[0]) - 1
        assert 0 <= low <= high < K
        cap = Fraction(row['N'], 2000) * (dyadic(row['BASE_fraction']) + Fraction(str(CFG['original_valley_delta_max'])) + margin)
        terms = []
        for a, (g, _) in enumerate(allocations):
            if groups[g]['label'] == label:
                terms.extend([(cdf_index[a, high], 1), (cdf_index[a, low], -1)])
        builder.add(terms, cap, 'source_valley_lower_integral')
        valley_lp_rows[label] = {'first_interval': low, 'last_interval': high, 'mass_cap': float(cap),
                                'endpoint_intervals_relaxed_not_exact_valley_mass': True}
    A, b = builder.materialize()
    construction = {
        'source_rows': 2000, 'REAL_rows': 2000, 'identity_order_bound': True,
        'source_composition': dict(Counter(composition.tolist())),
        'source_runtime_vs_feature_q_max_Vns': float(np.max(np.abs(q_runtime - q_source))),
        'source_abs_input_max_V': float(np.max(np.abs(x))), 'q_post_radius_constant_Vns': radius,
        'ADC_per_sample_allowance_LSB': 1., 'migration_uncertain_SOURCE_rows': sum(len(p) > 1 for p in event_possible),
        'raw_source_bin_counts': {str(j): int(np.sum(input_bins == j)) for j in range(-1, 9)},
        'allowed_bin_mass_bounds': {str(j): [float(v) for v in mass_bounds[j]] for j in range(-1, 9)},
        'groups': [{k: v for k, v in g.items() if k != 'indices'} for g in groups],
        'allocations': allocations, 'grid_points': len(grid), 'grid_intervals': K,
        'variables': cursor, 'rows': A.shape[0], 'nnz': A.nnz, 'row_kinds': dict(builder.kind_counts),
        'matrix_rounding': 'Every inequality RHS rounded outward for coefficient error on [0,1]; objective widths rounded downward.',
        'max_rhs_outward_relaxation': float_up(builder.max_rhs_relaxation),
        'sum_rhs_outward_relaxation': float_up(builder.sum_rhs_relaxation),
        'original_thresholds_unchanged': True, 'CDF_variables_are_interval_averages': True,
        'grid_exterior_distance_ignored': True, 'source_valley': constraints, 'valley_LP_rows': valley_lp_rows,
        'necessary_relaxation_not_constructed_waveforms': True,
    }
    write('LP_CONSTRUCTION.json', construction)
    np.savez_compressed(ROOT / 'INPUT_SUMMARY_PRIVATE.npz', source_ids=source['source_ids'], source_q_Vns=q_runtime,
                        source_original_bin=input_bins, source_constraint_label=labels, REAL_PSD=real_psd,
                        REAL_bin=real_bins, grid=grid, group_index=np.asarray([next(g for g, row in enumerate(groups) if i in row['indices']) for i in range(2000)]))
    del builder.row, builder.col, builder.data
    progress('solve_LP', variables=cursor, rows=A.shape[0], nnz=A.nnz)
    assert COUNTER['LP_attempts'] == 0
    COUNTER['LP_attempts'] += 1
    solve_start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solution = linprog(objective, A_ub=A, b_ub=b, bounds=(0., 1.), method=CFG['LP_method'], options=CFG['LP_options'])
    solve_seconds = time.monotonic() - solve_start
    warning_text = [str(w.message) for w in caught]
    status = {'status': int(solution.status), 'success': bool(solution.success), 'message': str(solution.message),
              'nit': int(solution.nit), 'method': CFG['LP_method'], 'options': CFG['LP_options'],
              'elapsed_s': solve_seconds, 'warnings': warning_text,
              'numerical_primal_objective_not_a_lower_bound': float(solution.fun) if solution.fun is not None else None}
    if solution.x is not None:
        violation = A @ solution.x - b
        status.update(max_inequality_violation=float(max(0., violation.max())),
                      min_x=float(solution.x.min()), max_x=float(solution.x.max()))
    write('LP_SOLVER_RECEIPT.json', status)
    dual = getattr(getattr(solution, 'ineqlin', None), 'marginals', None)
    certificate = None
    if dual is not None and len(dual) == len(b) and np.isfinite(dual).all():
        np.savez_compressed(ROOT / 'LP_PRIVATE.npz', A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
                            A_shape=np.asarray(A.shape), b=b, c=objective, lower=np.zeros(cursor), upper=np.ones(cursor),
                            primal=solution.x if solution.x is not None else np.zeros(0), dual_inequality=dual)
        progress('exact_certificate', dual_rows_total=int(np.count_nonzero(dual)))
        COUNTER['certificate_calculations'] += 1
        certificate = certify(ROOT / 'LP_PRIVATE.npz', progress)
        write('LOWER_BOUND_CERTIFICATE.json', certificate)
    else:
        np.savez_compressed(ROOT / 'LP_PRIVATE.npz', A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
                            A_shape=np.asarray(A.shape), b=b, c=objective, lower=np.zeros(cursor), upper=np.ones(cursor),
                            primal=solution.x if solution.x is not None else np.zeros(0), dual_inequality=np.zeros(0))
    decision = 'evidence_insufficient_no_valid_dual' if certificate is None else (
        'necessary_conditions_conflict_on_frozen_population' if certificate['exact_lower_bound_exceeds_diagnostic_threshold']
        else 'relaxed_diagnostic_does_not_rule_out_compatibility')
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'PID': os.getpid(), 'status': 'complete',
              'scientific_decision': decision, 'model_goal_complete': False, 'counter': COUNTER,
              'construction': construction, 'solver': status, 'certificate': certificate,
              'source_original_weights_modified': False, 'new_candidate_waveforms': 0,
              'original_A_F_changed': False, 'historically_exposed_DEV': True,
              'elapsed_s': time.monotonic() - START,
              'environment': {'Python': platform.python_version(), 'NumPy': np.__version__, 'SciPy': scipy.__version__,
                              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'threadpools': threadpool_info()}}
    write('RESULT.json', result)
    progress('complete', decision=decision)
    # Plot only fixed inputs, admissible masses and the actual LP/certificate outcome.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=145)
    js = np.arange(-1, 9)
    source_mass = np.array([np.mean(input_bins == j) for j in js])
    lower = np.array([float(mass_bounds[int(j)][0]) for j in js])
    upper = np.array([float(mass_bounds[int(j)][1]) for j in js])
    axes[0].bar(js - .17, source_mass, .32, label='Original SOURCE', color='#226ba2')
    axes[0].bar(js + .17, [np.mean(real_bins == j) for j in js], .32, label='REAL', color='#777777')
    axes[0].vlines(js - .17, lower, upper, color='#111111', linewidth=1.5, label='Allowed mass bounds')
    axes[0].set(xticks=js, xticklabels=['below'] + ['Q' + str(j + 1) for j in range(8)] + ['above'],
                ylabel='Fixed population mass', title='ADC and numerical migration retained')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(fontsize=7.5)
    sx = np.sort(real_psd)
    axes[1].step(sx, np.arange(1, 2001) / 2000., where='post', color='#222222', label='Frozen REAL PSD')
    for label, row in constraints.items():
        axes[1].axvspan(row['low'], row['high'], alpha=.17, label='Source Q' + str(label) + ' valley')
    axes[1].set(xlabel='Signed PSD', ylabel='Empirical CDF', title='One fixed grid; no scan')
    axes[1].legend(fontsize=7.5)
    axes[2].axis('off')
    lb = certificate['lower_bound']['float_rounded_down'] if certificate else None
    text = 'Original overall target: 0.010\nConflict test: L > 0.010001\n\n'
    text += 'Certified lower bound: ' + (format(lb, '.9g') if lb is not None else 'unavailable') + '\n'
    text += 'LP numerical objective: ' + str(status['numerical_primal_objective_not_a_lower_bound']) + '\n\n'
    text += ('Conflict certified for necessary conditions.' if certificate and certificate['exact_lower_bound_exceeds_diagnostic_threshold']
             else 'Compatibility is not ruled out.\nThis is NOT a waveform or A-F success.')
    axes[2].text(.02, .96, text, va='top', linespacing=1.65, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / 'compatibility_bounds.png', bbox_inches='tight')
    plt.close(fig)
    write('NUMERIC_RECEIPT.json', {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
                                  'status': 'complete', 'elapsed_s': time.monotonic() - START,
                                  'scientific_decision': decision, 'counter': COUNTER})
    print(json.dumps({'complete': decision, 'certified_lower_bound': lb, 'elapsed_s': time.monotonic() - START}), flush=True)


try:
    execute()
except BaseException as error:
    write('FAILURE.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'PID': os.getpid(),
                          'error': repr(error), 'traceback': traceback.format_exc(), 'counter': COUNTER,
                          'elapsed_s': time.monotonic() - START, 'scientific_success_claimed': False})
    raise
