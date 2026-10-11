"""Independent float64 NumPy math reference for frozen R71 + R72 shape MLP.

No Torch imports, file I/O, state loading, or random draws.  Explicit epsilon
and already loaded array state dictionaries are the complete public inputs.
"""
from __future__ import annotations

import numpy as np


def q(x):
    return x[..., 100:400].sum(-1) - 3.0 * x[..., :100].sum(-1)


def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _finite(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f"nonfinite independent {name}")


def _state(state):
    result = {key: np.asarray(value, dtype="f8") for key, value in state.items()}
    for key, value in result.items():
        _finite(value, key)
    return result


def _bins(qs, edges):
    if not np.isfinite(qs).all() or not (qs > 0).all():
        raise ValueError("independent metadata charge must remain positive finite")
    logq = np.log(1000.0 * qs)
    raw = np.searchsorted(edges, logq, side="right") - 1
    raw[logq == edges[-1]] = 7
    return raw


def reference_numpy(q_state, shape_state, x, epsilon, mu, scale, edges, p480, tau2):
    """Return (pre32, canonical64, raw64, row_meta), independently of Torch.

    Shape float32 weights are promoted to float64 and the MLP is evaluated as
    float64 mathematics; the returned pre-ADC cast and legacy ADC remain the
    original float32 operations.  This is a reference, not an alternate model.
    """
    x, epsilon = np.asarray(x, dtype="f8"), np.asarray(epsilon, dtype="f8")
    mu, scale, edges, p = [np.asarray(v, dtype="f8") for v in (mu, scale, edges, p480)]
    if x.ndim != 2 or x.shape[1] != 480 or len(x) == 0 or epsilon.shape != x.shape:
        raise ValueError("independent input/epsilon shape")
    for name, value in [("input", x), ("epsilon", epsilon), ("mu", mu), ("scale", scale),
                        ("edges", edges), ("spectrum", p)]:
        _finite(value, name)
    if mu.shape != (8, 480) or scale.shape != (8, 480) or not (scale > 0).all():
        raise ValueError("independent conditional statistics")
    if edges.shape != (9,) or not (np.diff(edges) > 0).all():
        raise ValueError("independent edges")
    if p.shape != (241,) or not (p >= 0).all() or not np.isfinite(tau2) or tau2 < 0:
        raise ValueError("independent noise constants")
    Q, S = _state(q_state), _state(shape_state)
    if Q["feature_center"].shape != (480,) or Q["feature_scale"].shape != (480,) or not (Q["feature_scale"] > 0).all():
        raise ValueError("independent Q-head statistics")
    if S["feature_center"].shape != (482,) or S["feature_scale"].shape != (482,) or not (S["feature_scale"] >= 1e-6).all():
        raise ValueError("independent waveform statistics")
    for key, expected in {"hidden1.weight": (128, 482), "hidden1.bias": (128,),
                          "hidden2.weight": (128, 128), "hidden2.bias": (128,),
                          "output.weight": (480, 128), "output.bias": (480,)}.items():
        if S[key].shape != expected:
            raise ValueError("independent shape architecture must be the fixed 128 candidate")

    q0 = q(x)
    if not ((q0 > 0) & (q0 < 300)).all():
        raise ValueError("independent original Q support; no row deletion")
    unit0 = (x - x[:, :100].mean(1, keepdims=True)) / q0[:, None]
    features0 = (unit0 - Q["feature_center"]) / Q["feature_scale"]
    hidden = features0 @ Q["hidden.weight"].T + Q["hidden.bias"]
    hidden = hidden * sigmoid(hidden)
    ab = hidden @ Q["output.weight"].T + Q["output.bias"]
    slope = np.exp(ab[:, 1])
    _finite(slope, "Q logit slope")
    if not (slope > 0).all():
        raise ValueError("independent Q logit slope underflow")
    qt = 300.0 * sigmoid(ab[:, 0] + slope * (np.log(q0) - np.log(300.0 - q0)))
    if not np.isfinite(qt).all() or not ((qt > 0) & (qt < 300)).all():
        raise ValueError("independent Qstar failure")
    physical_slope = slope * qt * (300.0 - qt) / (q0 * (300.0 - q0))
    if not np.isfinite(physical_slope).all() or not (physical_slope > 0).all():
        raise ValueError("independent fixed-unit Q slope failure")

    centered_mu = mu - mu[:, :100].mean(1, keepdims=True)
    if not (q(centered_mu) > 0).all():
        raise ValueError("independent nonpositive template charge")
    unit_mu = centered_mu / q(centered_mu)[:, None]
    centers = (edges[:-1] + edges[1:]) * 0.5
    logqt = np.log(1000.0 * qt)
    insertion = np.searchsorted(centers, logqt, side="right")
    left, right = np.clip(insertion - 1, 0, 7), np.clip(insertion, 0, 7)
    alpha = np.zeros_like(logqt)
    interior = left != right
    alpha[interior] = (logqt[interior] - centers[left[interior]]) / (centers[right[interior]] - centers[left[interior]])
    m = (1.0 - alpha[:, None]) * unit_mu[left] + alpha[:, None] * unit_mu[right]
    m = m - m[:, :100].mean(1, keepdims=True)
    m = m / q(m)[:, None]
    local_scale = (1.0 - alpha[:, None]) * scale[left] + alpha[:, None] * scale[right]
    lifted = x + (qt - q0)[:, None] * m
    unit = (lifted - lifted[:, :100].mean(1, keepdims=True)) / qt[:, None]
    feature482 = np.concatenate((unit, np.log(q0)[:, None], np.log(qt)[:, None]), axis=1)
    feature = (feature482 - S["feature_center"]) / S["feature_scale"]
    h1 = feature @ S["hidden1.weight"].T + S["hidden1.bias"]
    h1 = h1 * sigmoid(h1)
    h2 = h1 @ S["hidden2.weight"].T + S["hidden2.bias"]
    h2 = h2 * sigmoid(h2)
    f = h2 @ S["output.weight"].T + S["output.bias"]
    d_raw = qt[:, None] * local_scale * f
    delta = d_raw - m * q(d_raw)[:, None]

    # Independent dense covariance, using the sealed R71 reference construction.
    a = np.zeros(480); a[:100] = -3; a[100:400] = 1
    k = np.fft.irfft(p, n=480)
    index = np.arange(480)
    K = k[(index[:, None] - index[None, :]) % 480]
    B = np.eye(480)[:100] - np.eye(480)[:100].mean(0, keepdims=True)
    base = np.trace(B @ K @ B.T) / 99.0
    bka, ama, bm = B @ (K @ a), a @ K @ a, m @ B.T
    kappa = base - 2.0 * (bm * bka).sum(1) / 99.0 + ama * (bm * bm).sum(1) / 99.0
    if not np.isfinite(kappa).all() or not (kappa > 0).all():
        raise ValueError("independent projected noise variance failure")
    variance = np.var(lifted[:, :100], axis=1, ddof=1)
    amp2 = np.maximum(float(tau2) - variance, 0.0)
    noise = np.fft.irfft(np.sqrt(p) * np.fft.rfft(epsilon, axis=1, norm="ortho"),
                         n=480, axis=1, norm="ortho")
    noise = (noise - q(noise)[:, None] * m) * np.sqrt(amp2 / kappa)[:, None]
    raw = lifted + noise + delta
    _finite(raw, "raw waveform")
    pre = raw.copy()
    active = ((raw < -0.5) | (raw > 0.5)).any(1)
    change = active | (np.abs(q(raw) - qt) > 1e-12)
    if change.any():
        v, qq, aa = raw[change], qt[change], a[:400]
        lo = np.where(aa > 0, (v[:, :400] - 0.5) / aa, (v[:, :400] + 0.5) / aa).min(1)
        hi = np.where(aa > 0, (v[:, :400] + 0.5) / aa, (v[:, :400] - 0.5) / aa).max(1)
        for _ in range(64):
            mid = (lo + hi) * 0.5
            charge = q(np.clip(v - mid[:, None] * a, -0.5, 0.5))
            lo = np.where(charge > qq, mid, lo)
            hi = np.where(charge > qq, hi, mid)
        pre[change] = np.clip(v - 0.5 * (lo + hi)[:, None] * a, -0.5, 0.5)
    error64 = np.abs(q(pre) - qt) / qt
    pre32 = pre.astype("f4")
    error32 = np.abs(q(pre32.astype("f8")) - qt) / qt
    if (error64 > 1e-5).any() or (error32 > 1e-5).any():
        raise FloatingPointError("independent original Qstar precision failure")
    legacy = (-np.float32(0.5) + np.rint((np.clip(pre32, -0.5, 0.5) + np.float32(0.5)) *
                                      np.float32(16383)) / np.float32(16383)).astype("f4")
    canonical = -0.5 + np.clip(np.rint((legacy.astype("f8") + 0.5) * 16383), 0, 16383) / 16383
    _finite(canonical, "canonical waveform")
    preq, adcq = q(pre32.astype("f8")), q(canonical)
    meta = {"Q0": q0, "Qstar": qt, "QpreADC": preq, "QADC": adcq,
            "original_bin_raw": _bins(q0, edges), "target_bin_raw": _bins(qt, edges),
            "preADC_bin_raw": _bins(preq, edges), "ADC_bin_raw": _bins(adcq, edges),
            "Q_relative_error_f64": error64, "Q_relative_error_f32": error32,
            "range_projection_active": active, "range_projection_fraction": active.mean(),
            "projection_RMS_V": np.sqrt(np.mean((pre - raw) ** 2, axis=1)),
            "residual_RMS_V": np.sqrt(np.mean(delta ** 2, axis=1)), "residual_Q": q(delta),
            "noise_kappa2": kappa, "noise_amplitude_squared": amp2,
            "target_to_input_Q_ratio": qt / q0, "ADC_to_input_Q_ratio": adcq / q0}
    return pre32, canonical, raw, meta
