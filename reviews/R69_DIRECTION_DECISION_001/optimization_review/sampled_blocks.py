"""Pure helpers for uniform, without-replacement sampling of original VJP blocks.

Import performs no project I/O and creates/advances no random generator.
Only sample_blocks needs NumPy. The caller owns a NEW independent PCG64 stream;
this module cannot certify its independence from the scientific random streams.
The variance is for the original pre-clip program gradient at one fixed draw,
not for clipped/Adam updates or a changed training trajectory.
"""

import math
import numbers
import operator
import sys


# A scale-relative rounding allowance, with no absolute floor. Larger negative
# centered sums are inconsistent inputs, not variances to silently clamp.
ROUNDING_ULPS = 32


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(name + " must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(name + " must be an integer") from error
    if result < 1:
        raise ValueError(name + " must be positive")
    return int(result)


def _population_and_sample_size(M, k):
    M = _positive_integer(M, "M")
    k = _positive_integer(k, "k")
    if k > M:
        raise ValueError("k must not exceed M")
    return M, k


def _finite_nonnegative_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(name + " must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " must be representable as a finite float") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(name + " must be finite and nonnegative")
    return result


def original_schedule(n, chunk):
    """Return all original half-open row ranges, retaining a short final block."""
    n = _positive_integer(n, "n")
    chunk = _positive_integer(chunk, "chunk")
    return [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]


def sample_blocks(rng, M, k):
    """Return (sorted original block indices, float HT scale M/k).

    Selection is exactly rng.choice(M, k, replace=False). Sorting fixes the
    replay/addition order without changing inclusion probabilities. There is
    no block-length adjustment: the final one-row block has probability k/M.
    Only the supplied PCG64 stream is advanced, including when k == M.
    """
    import numpy as np

    M, k = _population_and_sample_size(M, k)
    if not isinstance(rng, np.random.Generator) or type(rng.bit_generator) is not np.random.PCG64:
        raise TypeError("rng must be a caller-owned numpy Generator with PCG64")
    indices = sorted(int(index) for index in rng.choice(M, k, replace=False))
    return indices, M / k


def conditional_variance_of_sum(M, k, sum_block_squared_norm, total_gradient_squared_norm):
    """Return the conditional HT total-gradient error variance trace as float.

    V = M*(M-k)/(k*(M-1)) * (sum ||G_b||^2 - ||sum G_b||^2/M).
    Norms must share the same parameter coordinates, with no dimension or row
    averaging. For k=M, V=0, including M=k=1. Inputs are still validated.

    A negative centered sum is clamped only if its magnitude is at most
    32*float-epsilon*max(sumNorm2, totalNorm2/M). There is no absolute floor.
    Materially inconsistent, nonfinite or overflowing inputs raise an error.
    """
    M, k = _population_and_sample_size(M, k)
    sum_norm = _finite_nonnegative_scalar(sum_block_squared_norm, "sum_block_squared_norm")
    total_norm = _finite_nonnegative_scalar(total_gradient_squared_norm, "total_gradient_squared_norm")
    try:
        mean_total_norm = total_norm / M
        centered = sum_norm - mean_total_norm
        allowance = ROUNDING_ULPS * sys.float_info.epsilon * max(sum_norm, mean_total_norm)
    except OverflowError as error:
        raise ValueError("variance inputs overflow float arithmetic") from error
    if centered < 0.0:
        if -centered <= allowance:
            centered = 0.0
        else:
            raise ValueError("sum block norms violate the total-gradient norm bound")
    if k == M:
        return 0.0
    try:
        variance = (M * (M - k) / (k * (M - 1))) * centered
    except OverflowError as error:
        raise ValueError("variance overflows float arithmetic") from error
    if not math.isfinite(variance):
        raise ValueError("variance is not finite")
    return max(0.0, variance)
