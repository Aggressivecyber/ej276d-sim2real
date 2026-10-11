"""One fixed TRAIN scalar analysis; no network, fitting, or waveform decoding."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
from bisect import bisect_right
import hashlib
import json
import sys
import time

S = Path(__file__).resolve().parent
R = S.parents[1]
P = R / 'reviews/R68_TRAIN_CHARGE_COMPATIBILITY_001'
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
GUARD = apply(S)
START = time.monotonic()
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rational(x):
    return dict(numerator=x.numerator, denominator=x.denominator, decimal=float(x))


def certificate(q, real, radius):
    """Exact CDF-envelope bound for equal-parent laws on closed intervals."""
    ns, nr = len(q), len(real)
    assert ns > 0 and nr > 0 and len(radius) == ns and all(d >= 0 for d in radius)
    low = sorted(x - d for x, d in zip(q, radius))
    high = sorted(x + d for x, d in zip(q, radius))
    ref = sorted(real)
    thresholds = sorted(set(low + high + ref))
    best = 0
    witness = None
    curve = []
    for k, t in enumerate(thresholds):
        if k % 4096 == 0 and time.monotonic() - START > 50:
            raise TimeoutError('Fixed local analysis exceeded its 50-second worker reserve')
        lo = bisect_right(high, t)
        up = bisect_right(low, t)
        rr = bisect_right(ref, t)
        assert 0 <= lo <= up <= ns
        plus = lo * nr - rr * ns
        minus = rr * ns - up * nr
        for direction, score in [('SOURCE_CDF_above_REAL', plus), ('REAL_CDF_above_SOURCE', minus)]:
            if score > best:
                best = score
                witness = dict(threshold=rational(t), direction=direction,
                               SOURCE_lower_count=lo, SOURCE_upper_count=up, REAL_count=rr,
                               SOURCE_N=ns, REAL_N=nr, integer_gap_numerator=score,
                               integer_gap_denominator=ns * nr)
        curve.append((float(t), lo / ns, up / ns, rr / nr))
    lower = F(best, ns * nr)
    return dict(KS_lower_bound=rational(lower), best_threshold_BA_lower_bound=rational((1 + lower) / 2),
                exceeds_point60_threshold=best * 5 > ns * nr,
                unique_threshold_count=len(thresholds), witness=witness), np.asarray(curve, dtype='f8')


def main():
    registration = read(P / 'REGISTRATION.json')
    assert registration['status'] == 'REGISTERED_BEFORE_SCALAR_DECODE'
    for row in registration['files']:
        assert sha(R / row['path']) == row['sha256'], row['path']
    assert not (S / 'RESULT.json').exists()
    write(S / 'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), resource=GUARD,
                                 stage='R68_TRAIN_CHARGE_COMPATIBILITY_001'))
    # Check arithmetic and endpoint conventions before decoding scientific scalars.
    fixtures = [([F(0)], [F(1)], [F(0)], F(1)),
                ([F(1)], [F(0)], [F(0)], F(1)),
                ([F(0)], [F(1)], [F(1)], F(0)),
                ([F(0), F(2)], [F(0), F(2)], [F(0), F(0)], F(0)),
                ([F(0), F(0)], [F(0), F(1)], [F(0), F(0)], F(1, 2))]
    for fq, fr, fd, expected in fixtures:
        ans, _ = certificate(fq, fr, fd)
        item = ans['KS_lower_bound']
        assert F(item['numerator'], item['denominator']) == expected
    cfg = read(S / 'config.json')
    archive = R / cfg['input_archive']
    assert sha(archive) == cfg['input_SHA256']
    # NPZ access decodes only these two existing charge arrays, not waveforms or endpoints.
    with np.load(archive, allow_pickle=False) as saved:
        source = saved['SOURCE_original_q']
        target = saved['REAL_canonical_q']
    assert source.shape == (6593,) and target.shape == (21047,)
    assert source.dtype == target.dtype == np.dtype('float64')
    assert np.isfinite(source).all() and np.isfinite(target).all() and (source > 0).all()
    q = [F.from_float(float(x)) for x in source]
    real = [F.from_float(float(x)) for x in target]
    tolerance = F(1, 100000)
    adc_bound = F(600, 16383)
    numeric_slack = F(1, 10000000)
    radius = [tolerance * x + adc_bound + numeric_slack for x in q]
    raw, _ = certificate(q, real, [F(0)] * len(q))
    robust, curve = certificate(q, real, radius)
    # Independent O(N) exact verification of the selected witness, without binary searches.
    w = robust['witness']
    if w:
        t = F(w['threshold']['numerator'], w['threshold']['denominator'])
        lo = sum(x + d <= t for x, d in zip(q, radius))
        up = sum(x - d <= t for x, d in zip(q, radius))
        rr = sum(x <= t for x in real)
        assert (lo, up, rr) == (w['SOURCE_lower_count'], w['SOURCE_upper_count'], w['REAL_count'])
        numerator = lo * len(real) - rr * len(q) if w['direction'] == 'SOURCE_CDF_above_REAL' else rr * len(q) - up * len(real)
        assert numerator == w['integer_gap_numerator']
    assert robust['KS_lower_bound']['decimal'] <= raw['KS_lower_bound']['decimal']
    np.savez_compressed(S / 'EXTRACTED_TRAIN_CHARGE_PRIVATE.npz', SOURCE_original_q=source, REAL_canonical_q=target)
    np.savez_compressed(S / 'CDF_ENVELOPE_PRIVATE.npz', columns=curve)
    access = dict(archive=cfg['input_archive'], archive_SHA256=sha(archive),
                  decoded_fields=['SOURCE_original_q', 'REAL_canonical_q'], SOURCE_N=6593, REAL_N=21047,
                  weights='unchanged uniform original parent laws; all rows retained',
                  waveform_arrays_decoded=0, model_weights_decoded=0, DEV_read=0, EVAL_read=0, CAL_decode=0,
                  network_calls=0, optimizer_updates=0, classifier_fits=0, RNG_draws=0)
    write(S / 'DATA_ACCESS_RECEIPT.json', access)
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), status='COMPLETED_FINITE_TRAIN_CHARGE_CERTIFICATE',
                  raw_finite_charge_comparison=raw, robust_interval_result=robust,
                  interval_definition=dict(relative_tolerance=rational(tolerance), ADC_charge_radius=rational(adc_bound),
                                           float64_reduction_slack=rational(numeric_slack),
                                           units='V samples', per_coordinate_ADC_radius='1/16383 V'),
                  exact_fraction_interval_endpoints=True, exact_integer_CDF_comparisons=True,
                  independent_witness_count_verified=True, artificial_cases_passed=len(fixtures),
                  scope='All equal-parent TRAIN output laws supported in the declared per-parent charge intervals, including stochastic conditional laws.',
                  excludes_exact_joint_Q_PSD_equality=robust['KS_lower_bound']['numerator'] > 0,
                  fixed_HGB_or_original_DEV_AF_universal_failure_proven=False,
                  heldout_CI_or_population_inference=False, original_A_F_complete=False,
                  arbitrary_future_ADC_or_charge_map_covered=False, acceptance_or_population_changed=False,
                  R65_automatic_resume=False, data_access=access, analysis_elapsed_s=time.monotonic()-START,
                  resource=dict(GUARD, **memory()))
    write(S / 'RESULT.json', result)
    print(json.dumps(dict(status=result['status'], raw_KS=raw['KS_lower_bound']['decimal'],
                         robust_KS=robust['KS_lower_bound']['decimal'], robust_threshold_BA=robust['best_threshold_BA_lower_bound']['decimal'],
                         analysis_elapsed_s=result['analysis_elapsed_s'], resource=result['resource']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
