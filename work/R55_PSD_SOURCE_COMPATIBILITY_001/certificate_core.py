"""Exact dual-residual certificate for the saved float64 LP, 0 <= x <= 1."""
from fractions import Fraction
from functools import lru_cache
import math
import time


@lru_cache(maxsize=262144)
def dyadic(value):
    return Fraction.from_float(float(value))


def float_down(value):
    out = float(value)
    if dyadic(out) > value:
        out = math.nextafter(out, -math.inf)
    assert dyadic(out) <= value
    return out


def certify(path, progress=None):
    import numpy as np
    start = time.monotonic()
    with np.load(path, allow_pickle=False) as z:
        data = z['A_data'].copy()
        indices = z['A_indices'].copy()
        indptr = z['A_indptr'].copy()
        rhs = z['b'].copy()
        cost = z['c'].copy()
        raw_dual = z['dual_inequality'].copy()
        lower = z['lower'].copy()
        upper = z['upper'].copy()
    assert np.isfinite(data).all() and np.isfinite(rhs).all() and np.isfinite(cost).all()
    assert np.isfinite(raw_dual).all() and len(raw_dual) == len(rhs)
    assert np.array_equal(lower, np.zeros_like(cost)) and np.array_equal(upper, np.ones_like(cost))
    dual = np.minimum(raw_dual, 0.)
    residual = [dyadic(v) for v in cost]
    objective = Fraction(0)
    active = np.flatnonzero(dual)
    last = time.monotonic()
    for ordinal, row in enumerate(active):
        y = dyadic(dual[row])
        assert y <= 0
        objective += y * dyadic(rhs[row])
        for k in range(indptr[row], indptr[row + 1]):
            residual[int(indices[k])] -= y * dyadic(data[k])
        if progress is not None and time.monotonic() - last >= 3:
            progress('exact_certificate', dual_rows_done=ordinal + 1, dual_rows_total=len(active))
            last = time.monotonic()
    negative = [r for r in residual if r < 0]
    correction = sum(negative, Fraction(0))
    lower_bound = objective + correction
    original = Fraction(1, 100)
    threshold = original + Fraction(1, 1000000)
    answer = {
        'valid': True, 'calculation': 'exact fractions of the actual saved IEEE binary64 LP and clipped dual',
        'formula': 'yTb + sum(min(0, c - ATy)); y<=0; Ax<=b; every x in [0,1]',
        'rows': len(rhs), 'columns': len(cost), 'nnz': len(data),
        'positive_dual_entries_clipped_to_zero': int(np.sum(raw_dual > 0)),
        'nonzero_dual_rows': len(active), 'negative_dual_residual_columns': len(negative),
        'dual_objective': {'numerator': str(objective.numerator), 'denominator': str(objective.denominator)},
        'residual_correction': {'numerator': str(correction.numerator), 'denominator': str(correction.denominator)},
        'lower_bound': {'numerator': str(lower_bound.numerator), 'denominator': str(lower_bound.denominator),
                        'float_rounded_down': float_down(lower_bound)},
        'original_overall_threshold': 0.01, 'diagnostic_conflict_threshold': 0.010001,
        'exact_lower_bound_exceeds_diagnostic_threshold': lower_bound > threshold,
        'elapsed_s': time.monotonic() - start,
        'does_not_certify_original_A_F_feasibility': True,
    }
    return answer
