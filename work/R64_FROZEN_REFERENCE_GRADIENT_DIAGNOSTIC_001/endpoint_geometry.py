"""R64 endpoint geometry: array-only, fixed empirical laws, no I/O or fitting.

The caller supplies the *direct training endpoint* before ADC and its actual
canonical_adc_torch(..., False) value.  This module never calls a network, solver,
random generator, optimizer, or data loader.  Values are descriptive float64
computations of finite empirical laws, not interval certificates.

The R62 pure features/exact_w1_columns algorithm is reused below, with bounded
column scratch space.  temporal_features follows R63, including its power floor;
R62 baseline_features is deliberately not substituted at zero/very low power.
"""
from __future__ import annotations

import numpy as np

REGIONS = tuple(range(-1, 9))
SCOPES = ("all",) + tuple("region" + str(b) for b in REGIONS)
FULLVIEWS = ("V480", "U480", "DU479", "D2U478")
VIEW_NAMES = FULLVIEWS + ("temporal31",)
DIMENSIONS = (480, 480, 479, 478, 31)
TERM_NAMES = ("full_window", "coordinate_RMS", "temporal_joint")
TEMPORAL_NAMES = (
    ("RMS_V",)
    + tuple("ACF_lag" + str(k) for k in range(1, 21))
    + tuple("rFFT_power_bin%d_%d_fraction" % (1 + 5*k, 5 + 5*k) for k in range(10))
)
CHARGE_A = np.r_[np.full(100, -3.), np.ones(300), np.zeros(80)]
NOMINAL_CHECK_ATOL = 2e-10


def _finite(a, name, ndim=None):
    x = np.asarray(a)
    if ndim is not None and x.ndim != ndim:
        raise ValueError(name + ": wrong rank")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError(name + ": nonfinite or nonnumeric; no row deletion")
    return x


def _wave(a, name):
    x = _finite(a, name, 2)
    if not len(x) or x.shape[1] != 480:
        raise ValueError(name + ": require nonempty N x 480")
    return np.asarray(x, dtype=np.float64)


def q_np(x):
    """Original code charge units (V samples); no factor of 1000."""
    v = np.asarray(x, dtype=np.float64)
    return v[..., 100:400].sum(-1) - 3.*v[..., :100].sum(-1)


def canonical_real(v):
    x = np.asarray(v, dtype=np.float64)
    return -.5 + np.clip(np.rint((x + .5)*16383), 0, 16383)/16383


def canonical_from_pre_adc(v):
    """R62/R63 two-stage generated ADC path, used only to check caller input."""
    f = np.asarray(v, dtype=np.float32)
    legacy = (-.5 + np.rint((np.clip(f, -.5, .5) + .5)*16383)/16383).astype(np.float32)
    return canonical_real(legacy)


def feature_vectors(v):
    """Pure R62 features / R63 R61_distribution_loss.feature_vectors."""
    x = np.asarray(v, dtype=np.float64)
    s = x - x[:, :100].mean(1, keepdims=True)
    u = s/np.maximum(s.max(1, keepdims=True), .0025)
    return [x, u, np.diff(u, axis=1), np.diff(u, n=2, axis=1)]


def temporal_features(v, power_floor=1e-20):
    """R63 temporal31 formula in float64, including Nyquist multiplicity one."""
    if power_floor != 1e-20:
        raise ValueError("R63 temporal_power_floor must remain 1e-20")
    b = np.asarray(v, dtype=np.float64)[..., :100]
    b = b - b.mean(-1, keepdims=True)
    power = (b*b).mean(-1)
    safe = np.maximum(power, power_floor)
    acf = np.stack([(b[..., k:]*b[..., :-k]).mean(-1)/safe for k in range(1, 21)], -1)
    spectrum = np.fft.rfft(b, axis=-1)[..., 1:]
    frequency_power = (spectrum.real**2 + spectrum.imag**2)/10000
    multiplicity = np.r_[np.full(49, 2.), 1.]
    bands = (frequency_power*multiplicity).reshape(*b.shape[:-1], 10, 5).sum(-1)/safe[..., None]
    return np.concatenate((np.sqrt(safe)[..., None], acf, bands), -1)


def empirical_w1_columns(a, b, column_chunk=32):
    """R62 quantile integral for equally weighted, possibly unequal-size laws.

    Duplicated rows retain their occurrence mass.  Sorting/subtraction/reduction
    use float64.  'Exact empirical' describes the formula, not numerical rigor.
    """
    x = _finite(a, "W1 left").astype(np.float64, copy=False)
    y = _finite(b, "W1 right").astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        raise ValueError("W1 dimensions or empty law")
    if column_chunk < 1:
        raise ValueError("column_chunk")
    ca = np.arange(1, len(x)+1, dtype=np.float64)/len(x)
    cb = np.arange(1, len(y)+1, dtype=np.float64)/len(y)
    edges = np.sort(np.r_[0., ca, cb], kind="stable")
    mid = .5*(edges[1:] + edges[:-1])
    mass = np.diff(edges)
    ia = np.minimum(np.searchsorted(ca, mid), len(x)-1)
    ib = np.minimum(np.searchsorted(cb, mid), len(y)-1)
    out = np.empty(x.shape[1], dtype=np.float64)
    for j in range(0, x.shape[1], column_chunk):
        k = slice(j, min(j + column_chunk, x.shape[1]))
        sa = np.sort(x[:, k], axis=0, kind="stable")
        sb = np.sort(y[:, k], axis=0, kind="stable")
        out[k] = (np.abs(sa[ia] - sb[ib])*mass[:, None]).sum(0)
    return out


def _summary(x):
    v = np.asarray(x, dtype=np.float64)
    return dict(N=int(v.size), mean=float(v.mean()),
                SD=float(v.std(ddof=1)) if v.size > 1 else 0.,
                min=float(v.min()), q10=float(np.quantile(v, .1)),
                q50=float(np.quantile(v, .5)), q90=float(np.quantile(v, .9)),
                max=float(v.max()))


def _ratio(n, d):
    return float(n/d) if d > 0 else None


def _leq(a, b):
    return bool(a <= b + NOMINAL_CHECK_ATOL*(1. + max(abs(a), abs(b))))


def charge_mean_bound(y, r, q_source, center, scale, represented_y, represented_r, coordinate_w1):
    """Nominal, output-specific V480 charge-mean lower bounds in code q units.

    Reconstruction residual x-(center+scale*zbar) includes the discrepancy of
    actual represented coordinates from the mathematical affine relation; it is
    not just an ideal half-ULP cast estimate.  Reconstruction and every bound
    operation remain ordinary float64, so this is not an interval certificate.
    """
    y = _wave(y, "bound Y")
    r = _wave(r, "bound R")
    qs = _finite(q_source, "bound original charge", 1).astype(np.float64)
    mu = _finite(center, "bound center", 1).astype(np.float64)
    s = _finite(scale, "bound scale", 1).astype(np.float64)
    zy = _finite(represented_y, "bound represented Y", 2).astype(np.float64)
    zr = _finite(represented_r, "bound represented R", 2).astype(np.float64)
    d = _finite(coordinate_w1, "bound coordinate W1", 1).astype(np.float64)
    if len(qs) != len(y) or mu.shape != (480,) or s.shape != (480,) or np.any(s <= 0):
        raise ValueError("bound population or scale")
    if zy.shape != y.shape or zr.shape != r.shape or d.shape != (480,) or np.any(d < 0):
        raise ValueError("bound represented dimensions")
    qy, qr = q_np(y), q_np(r)
    signed_output_gap = float(qy.mean() - qr.mean())
    signed_source_gap = float(qs.mean() - qr.mean())
    dout, dsrc = abs(signed_output_gap), abs(signed_source_gap)
    epsq = float(np.abs(qy - qs).mean())
    err_y = (np.abs(y - (mu + s*zy))*np.abs(CHARGE_A)).sum(1)
    err_r = (np.abs(r - (mu + s*zr))*np.abs(CHARGE_A)).sum(1)
    erep = float(err_y.mean() + err_r.mean())
    c = np.abs(CHARGE_A)*s
    den_mean = float(480*c.max())
    den_rms = float(np.sqrt(480)*np.linalg.norm(c))
    effective_output = max(0., dout - erep)
    effective_source = max(0., dsrc - epsq - erep)
    out_mean, out_rms = effective_output/den_mean, effective_output/den_rms
    src_mean, src_rms = effective_source/den_mean, effective_source/den_rms
    observed_mean = float(d.mean())
    observed_rms = float(np.sqrt(np.mean(d*d)))
    cd = float(np.dot(c, d))
    checks = {
        "output_charge_gap_le_c_dot_W1_plus_representation": _leq(dout, cd + erep),
        "source_charge_gap_le_output_gap_plus_actual_error": _leq(dsrc, dout + epsq),
        "c_dot_W1_le_mean_denominator_times_mean": _leq(cd, den_mean*observed_mean),
        "c_dot_W1_le_RMS_denominator_times_RMS": _leq(cd, den_rms*observed_rms),
        "mean_lower_bound_le_observed_mean": _leq(out_mean, observed_mean),
        "RMS_lower_bound_le_observed_RMS": _leq(out_rms, observed_rms),
        "source_form_le_output_form": _leq(effective_source, effective_output),
        "coordinate_mean_le_RMS": _leq(observed_mean, observed_rms),
    }
    observed_v = observed_mean/8. + observed_rms/4.
    return {
        "q_units": "original q = sum(v[100:400])-3*sum(v[0:100]); V samples; no x1000",
        "signed_output_minus_canonical_REAL_mean_q": signed_output_gap,
        "signed_original_SOURCE_minus_canonical_REAL_mean_q": signed_source_gap,
        "Dout_q": dout, "Dsrc_q": dsrc,
        "actual_postADC_total_mean_absolute_charge_error_q": epsq,
        "representation_reconstruction_error_Y_q": float(err_y.mean()),
        "representation_reconstruction_error_REAL_q": float(err_r.mean()),
        "Erep_q": erep, "c_dot_coordinate_W1_q": cd,
        "denominator_mean": den_mean, "denominator_RMS": den_rms,
        "output_specific_LB_coordinate_mean": out_mean,
        "output_specific_LB_coordinate_RMS": out_rms,
        "source_charge_form_LB_coordinate_mean": src_mean,
        "source_charge_form_LB_coordinate_RMS": src_rms,
        "observed_coordinate_mean": observed_mean,
        "observed_coordinate_RMS": observed_rms,
        "output_LB_fraction_of_V480_coordinate_mean": _ratio(out_mean, observed_mean),
        "output_LB_fraction_of_V480_coordinate_RMS": _ratio(out_rms, observed_rms),
        "output_LB_V480_two_term_contribution": out_mean/8. + out_rms/4.,
        "source_LB_V480_two_term_contribution": src_mean/8. + src_rms/4.,
        "observed_V480_two_coordinate_contributions": observed_v,
        "output_LB_fraction_of_V480_two_coordinate_contributions": _ratio(out_mean/8. + out_rms/4., observed_v),
        "nominal_inequality_checks": checks,
        "nominal_inequality_checks_pass": bool(all(checks.values())),
        "interval_certificate": False,
        "model_independent_irreducibility_or_A_F_claim": False,
    }


def _banks(v, bins, statistics):
    raw = feature_vectors(v) + [temporal_features(v)]
    represented = []
    for name, f in zip(VIEW_NAMES, raw):
        stem = "temporal" if name == "temporal31" else name
        center = np.asarray(statistics[stem + "_center"], dtype=np.float64)[bins + 1]
        scale = np.asarray(statistics[stem + "_scale"], dtype=np.float64)[bins + 1]
        z = ((f - center)/scale).astype(np.float32)
        if not np.isfinite(z).all():
            raise FloatingPointError("nonfinite represented feature; no dropping")
        represented.append(z)
    return raw, represented


def _metrics(banka, bankb, ia, ib, directions):
    views = []
    arrays = []
    for name, x, y, p in zip(VIEW_NAMES, banka, bankb, directions):
        xx, yy = x[ia], y[ib]
        w = empirical_w1_columns(xx, yy)
        # Both operands are float32: record the finite projection representation,
        # then integrate its W1 in float64.  No bitwise GPU-BLAS claim is made.
        sw = empirical_w1_columns(xx @ p, yy @ p)
        m, rms, sliced = float(w.mean()), float(np.sqrt(np.mean(w*w))), float(sw.mean())
        views.append(dict(view=name, coordinate_mean=m, coordinate_RMS=rms,
                          coordinate_max=float(w.max()), sliced_mean=sliced,
                          marginal_sliced_half_sum=.5*(m + sliced)))
        arrays.append((w, sw))
    terms = np.asarray([
        np.mean([v["marginal_sliced_half_sum"] for v in views[:4]]),
        np.mean([v["coordinate_RMS"] for v in views[:4]]),
        views[4]["marginal_sliced_half_sum"],
    ], dtype=np.float64)
    return {"views": views, "terms": dict(zip(TERM_NAMES, terms.tolist()))}, terms, arrays


def _save_metrics(private, prefix, arrays):
    for name, (w, sw) in zip(VIEW_NAMES, arrays):
        private[prefix + "__" + name + "__coordinate_W1"] = w
        private[prefix + "__" + name + "__projection_W1"] = sw


def _indices_for_scope(bins, scope):
    return np.arange(len(bins)) if scope == "all" else np.flatnonzero(bins == int(scope[6:]))


def _check_indices(a, nrep, nrows, real_bins, scope, name):
    ix = np.asarray(a)
    if ix.dtype.kind not in "iu" or ix.shape != (nrep, nrows):
        raise ValueError(name + ": integer global REAL indices shape must be (replicates, SOURCE scope N)")
    if np.any(ix < 0) or np.any(ix >= len(real_bins)):
        raise ValueError(name + ": REAL index out of range")
    if scope != "all" and np.any(real_bins[ix] != int(scope[6:])):
        raise ValueError(name + ": index left original raw-bin")
    return ix


def _charge_row(qs, qpre, qout, qr, qrc, si, ri):
    return {
        "SOURCE_original": _summary(qs[si]),
        "SOURCE_direct_preADC": _summary(qpre[si]),
        "SOURCE_canonical": _summary(qout[si]),
        "REAL_raw": _summary(qr[ri]),
        "REAL_canonical": _summary(qrc[ri]),
        "SOURCE_raw_to_REAL_raw_W1_q": float(empirical_w1_columns(qs[si], qr[ri])[0]),
        "SOURCE_raw_to_REAL_canonical_W1_q": float(empirical_w1_columns(qs[si], qrc[ri])[0]),
        "SOURCE_canonical_to_REAL_canonical_W1_q": float(empirical_w1_columns(qout[si], qrc[ri])[0]),
        "SOURCE_preADC_minus_original_q": _summary(qpre[si] - qs[si]),
        "SOURCE_canonical_minus_preADC_q": _summary(qout[si] - qpre[si]),
        "SOURCE_canonical_minus_original_q": _summary(qout[si] - qs[si]),
        "REAL_canonical_minus_raw_q": _summary(qrc[ri] - qr[ri]),
    }


def analyze_endpoint_geometry(
    source_out_preADC, source_canonical, canonical_REAL,
    source_raw, REAL_raw, source_bins, REAL_bins, source_charge,
    statistics, directions, temporal_directions, matched_real_indices, *,
    REAL_control_canonical=None, expected_population=(6593, 21047),
    expected_reference_replicates=32,
):
    """Return (JSON-ready summary, private NumPy arrays), without reading/writing.

    statistics uses unchanged R63 keys: source_counts, REAL_counts, each old
    VIEW_center/scale (10,dim), temporal_center/scale (10,31).  If supplied,
    temporal_REAL_counts is checked too.  No constants are estimated here.

    directions is four float32 (dim,64) matrices; temporal_directions is (31,64).
    matched_real_indices[scope]['A'/'B'] are caller-frozen integer global REAL
    row indices, shape (32, original SOURCE scope N); scope order is SCOPES.
    Samples are with replacement; repeats are retained.  The caller must bind
    original parent IDs/order, split, frozen input hashes, RNG seeds and streams.

    expected_population/reps may be reduced only for labeled artificial fixtures;
    a registered R64 scientific worker must use defaults, and preserve all rows.
    """
    pre = _wave(source_out_preADC, "SOURCE direct preADC endpoint")
    sy = _wave(source_canonical, "SOURCE canonical endpoint")
    rr = _wave(canonical_REAL, "canonical REAL reference")
    sx = _wave(source_raw, "original SOURCE")
    rx = _wave(REAL_raw, "original REAL")
    if (len(sx), len(rx)) != tuple(expected_population) or pre.shape != sx.shape or sy.shape != sx.shape or rr.shape != rx.shape:
        raise ValueError("full original population/shape binding; no population change")
    for name, x in (("SOURCE", sx), ("REAL", rx)):
        if not np.array_equal(x, x.astype(np.float32).astype(np.float64)):
            raise ValueError(name + " raw input must retain original float32 values")
    qs = _finite(source_charge, "original SOURCE charge", 1).astype(np.float64)
    if qs.shape != (len(sx),):
        raise ValueError("source charge length")
    raw_charge_error = float(np.max(np.abs(qs - q_np(sx))))
    if raw_charge_error > 1e-12*(1. + float(np.max(np.abs(qs)))):
        raise ValueError("source charge disagrees with original float32-to-float64 q_np")
    if np.any(pre < -.5) or np.any(pre > .5):
        raise ValueError("direct projected endpoint outside ADC box")
    if not np.allclose(sy, canonical_from_pre_adc(pre), rtol=0, atol=4e-15):
        raise ValueError("SOURCE canonical value disagrees with two-stage generated ADC")
    if not np.allclose(rr, canonical_real(rx), rtol=0, atol=4e-15):
        raise ValueError("REAL reference disagrees with canonical_REAL of original REAL")
    sb, rb = np.asarray(source_bins), np.asarray(REAL_bins)
    for name, bins, n in (("SOURCE", sb, len(sx)), ("REAL", rb, len(rx))):
        if bins.dtype.kind not in "iu" or bins.shape != (n,) or not np.isin(bins, REGIONS).all():
            raise ValueError(name + " original raw-bin binding; do not use clipped network bins")
    sc = np.asarray([(sb == b).sum() for b in REGIONS], dtype=np.int64)
    rc = np.asarray([(rb == b).sum() for b in REGIONS], dtype=np.int64)
    if np.any(sc == 0) or np.any(rc == 0):
        raise ValueError("all ten original regions required")
    if not np.array_equal(statistics["source_counts"], sc) or not np.array_equal(statistics["REAL_counts"], rc):
        raise ValueError("frozen statistics population mismatch")
    if "temporal_REAL_counts" in statistics and not np.array_equal(statistics["temporal_REAL_counts"], rc):
        raise ValueError("frozen temporal population mismatch")
    for name, dim in zip(VIEW_NAMES, DIMENSIONS):
        stem = "temporal" if name == "temporal31" else name
        for suffix in ("center", "scale"):
            v = _finite(statistics[stem + "_" + suffix], stem + " " + suffix, 2)
            if v.shape != (10, dim) or (suffix == "scale" and np.any(v <= 0)):
                raise ValueError(stem + ": frozen positive scale/center dimensions")
    if len(directions) != 4:
        raise ValueError("four frozen fullview directions required")
    pd = list(directions) + [temporal_directions]
    for p, dim in zip(pd, DIMENSIONS):
        p = _finite(p, "directions", 2)
        if p.shape != (dim, 64) or p.dtype != np.dtype("float32"):
            raise ValueError("fixed projection shape/dtype; directions are not normalized or redrawn here")
        if not np.allclose(np.linalg.norm(p.astype(np.float64), axis=0), 1., rtol=0, atol=3e-7):
            raise ValueError("frozen directions must be unit columns to float32 precision")
    if set(matched_real_indices) != set(SCOPES):
        raise ValueError("all and all ten original-region REAL floor designs required")
    nrep = expected_reference_replicates
    if not isinstance(nrep, int) or nrep < 1:
        raise ValueError("replicate count")
    mass_s, mass_r = sc/sc.sum(), rc/rc.sum()
    sraw, s = _banks(sy, sb, statistics)
    rraw, r = _banks(rr, rb, statistics)
    control = None
    if REAL_control_canonical is not None:
        ctl = _wave(REAL_control_canonical, "same-treated REAL canonical")
        if ctl.shape != rr.shape or not np.allclose(ctl, canonical_real(ctl), rtol=0, atol=4e-15):
            raise ValueError("same-treated REAL full population or canonical grid")
        _, control = _banks(ctl, rb, statistics)
    qpre, qout, qr, qrc = q_np(pre), q_np(sy), q_np(rx), q_np(rr)
    rows, bounds = [], []
    private = {
        "source_original_region_counts": sc.copy(), "REAL_original_region_counts": rc.copy(),
        "source_original_region_mass": mass_s.copy(), "REAL_original_region_mass": mass_r.copy(),
        "SOURCE_original_q": qs.copy(), "SOURCE_direct_preADC_q": qpre,
        "SOURCE_canonical_q": qout, "REAL_raw_q": qr, "REAL_canonical_q": qrc,
    }
    full_terms = np.zeros((11, 3), dtype=np.float64)
    control_terms = np.zeros((11, 3), dtype=np.float64) if control is not None else None
    floor_terms = np.zeros((11, nrep, 3), dtype=np.float64)
    for i, scope in enumerate(SCOPES):
        si, ri = _indices_for_scope(sb, scope), _indices_for_scope(rb, scope)
        pair, terms, arrays = _metrics(s, r, si, ri, pd)
        full_terms[i] = terms
        _save_metrics(private, "SOURCE_REAL__" + scope, arrays)
        physical = {}
        for name, x, y in zip(VIEW_NAMES, sraw, rraw):
            w = empirical_w1_columns(x[si], y[ri])
            private["SOURCE_REAL__" + scope + "__" + name + "__unstandardized_coordinate_W1"] = w
            physical[name] = dict(coordinate_mean=float(w.mean()), coordinate_RMS=float(np.sqrt(np.mean(w*w))), coordinate_max=float(w.max()))
        row = dict(scope=scope, raw_region=None if i == 0 else REGIONS[i-1],
                   SOURCE_N=int(len(si)), REAL_N=int(len(ri)),
                   SOURCE_parent_mass=1. if i == 0 else float(mass_s[i-1]),
                   REAL_parent_mass=1. if i == 0 else float(mass_r[i-1]),
                   SOURCE_vs_complete_REAL=pair,
                   unstandardized_coordinate_statistics=physical,
                   charge_statistics=_charge_row(qs, qpre, qout, qr, qrc, si, ri))
        if control is not None:
            cp, ct, ca = _metrics(control, r, ri, ri, pd)
            control_terms[i] = ct
            row["same_treated_REAL_vs_complete_REAL"] = cp
            _save_metrics(private, "REAL_control_REAL__" + scope, ca)
        if i:
            j = i-1
            bound = charge_mean_bound(sy[si], rr[ri], qs[si],
                                      statistics["V480_center"][j], statistics["V480_scale"][j],
                                      s[0][si], r[0][ri], arrays[0][0])
            bound["scope"] = scope
            bound["original_SOURCE_region_mass"] = float(mass_s[j])
            bound["output_LB_fraction_of_full_window_plus_coordinate_RMS"] = _ratio(bound["output_LB_V480_two_term_contribution"], float(terms[:2].sum()))
            row["charge_mean_bound"] = bound
            row["original_SOURCE_mass_weighted_terms"] = dict(zip(TERM_NAMES, (terms*mass_s[j]).tolist()))
            bounds.append(bound)
        supplied = matched_real_indices[scope]
        if set(supplied) != {"A", "B"}:
            raise ValueError(scope + ": require both A and B frozen draw arrays")
        ra = _check_indices(supplied["A"], nrep, len(si), rb, scope, scope + " A")
        rbidx = _check_indices(supplied["B"], nrep, len(si), rb, scope, scope + " B")
        floor_arrays = [(np.empty((nrep, dim)), np.empty((nrep, 64))) for dim in DIMENSIONS]
        floor_charge = np.empty((nrep, 2))
        for k in range(nrep):
            _, floor_terms[i, k], fa = _metrics(r, r, ra[k], rbidx[k], pd)
            for dst, src in zip(floor_arrays, fa):
                dst[0][k], dst[1][k] = src
            floor_charge[k] = [empirical_w1_columns(qr[ra[k]], qr[rbidx[k]])[0],
                               empirical_w1_columns(qrc[ra[k]], qrc[rbidx[k]])[0]]
        _save_metrics(private, "REAL_pair_floor__" + scope, floor_arrays)
        private["REAL_pair_floor__" + scope + "__A_indices"] = ra.copy()
        private["REAL_pair_floor__" + scope + "__B_indices"] = rbidx.copy()
        private["REAL_pair_floor__" + scope + "__charge_W1_raw_canonical_q"] = floor_charge
        row["matched_N_REAL_pair_floor"] = {
            "replicates": nrep, "N_each_draw": int(len(si)),
            "terms": {name: {**_summary(floor_terms[i, :, j]), "values": floor_terms[i, :, j].tolist(),
                              "SOURCE_vs_complete_REAL_minus_floor_values": (terms[j] - floor_terms[i, :, j]).tolist()}
                      for j, name in enumerate(TERM_NAMES)},
            "raw_charge_W1_q": {**_summary(floor_charge[:, 0]), "values": floor_charge[:, 0].tolist()},
            "canonical_charge_W1_q": {**_summary(floor_charge[:, 1]), "values": floor_charge[:, 1].tolist()},
            "A_unique_parent_counts": [int(len(np.unique(ix))) for ix in ra],
            "B_unique_parent_counts": [int(len(np.unique(ix))) for ix in rbidx],
            "duplicate_occurrences_retained": True,
        }
        rows.append(row)
    weighted = (full_terms[1:]*mass_s[:, None]).sum(0)
    weighted_floor = (floor_terms[1:]*mass_s[:, None, None]).sum(0)
    lb_output = float(sum(mass_s[i]*b["output_LB_V480_two_term_contribution"] for i, b in enumerate(bounds)))
    lb_source = float(sum(mass_s[i]*b["source_LB_V480_two_term_contribution"] for i, b in enumerate(bounds)))
    private.update(SOURCE_full_terms_by_scope=full_terms, REAL_pair_floor_terms_by_scope=floor_terms,
                   SOURCE_weighted_regional_terms=weighted, REAL_pair_floor_weighted_regional_terms=weighted_floor,
                   SOURCE_original_mass_region_contributions=full_terms[1:]*mass_s[:, None])
    if control_terms is not None:
        private["REAL_control_full_terms_by_scope"] = control_terms
    result = {
        "schema_version": 1,
        "status": "descriptive_finite_TRAIN_endpoint_geometry_no_new_acceptance",
        "endpoint": "direct source_flow_core.endpoint projected preADC; generated two-stage canonical ADC supplied by worker",
        "REFERENCE": "canonical_REAL of all original REAL TRAIN parents; same-treated REAL is separate control",
        "population": {"SOURCE_N": len(sx), "REAL_N": len(rx), "region_axis": list(REGIONS),
                       "source_counts": sc.tolist(), "REAL_counts": rc.tolist(),
                       "source_mass": mass_s.tolist(), "REAL_mass": mass_r.tolist(),
                       "source_charge_binding_max_abs_q_error": raw_charge_error},
        "rows": rows,
        "original_SOURCE_mass_weighted_regional_terms": dict(zip(TERM_NAMES, weighted.tolist())),
        "matched_N_REAL_pair_weighted_regional_terms": {n: {**_summary(weighted_floor[:, j]), "values": weighted_floor[:, j].tolist()}
                                                     for j, n in enumerate(TERM_NAMES)},
        "charge_bound_aggregation": {
            "output_specific_V480_lower_bound_full_window_plus_RMS": lb_output,
            "source_charge_form_V480_lower_bound_full_window_plus_RMS": lb_source,
            "observed_full_window_plus_RMS": float(weighted[:2].sum()),
            "output_lower_bound_fraction_of_observed_full_window_plus_RMS": _ratio(lb_output, float(weighted[:2].sum())),
            "source_lower_bound_fraction_of_observed_full_window_plus_RMS": _ratio(lb_source, float(weighted[:2].sum())),
            "coefficients": {"V480_coordinate_mean_in_full_window": .125, "V480_coordinate_RMS_in_coordinate_RMS": .25},
            "unit_outer_weights": {n: 1. for n in TERM_NAMES},
            "all_nominal_inequality_checks_pass": bool(all(b["nominal_inequality_checks_pass"] for b in bounds)),
        },
        "temporal_names": list(TEMPORAL_NAMES), "fixed_directions_per_view": 64,
        "numerical_definition": "float64 feature arithmetic; actual standardized representation float32; projected representation float32; empirical W1 integral and summaries float64",
        "nominal_check_scaled_atol": NOMINAL_CHECK_ATOL,
        "interpretation": [
            "All parents retain equal mass within their original full domain and raw-bin; no bin recomputation, clipping, filtering, fitting or reweighting.",
            "The all-scope represented law pools each parent's own frozen regional standardization; it is not the regional training-objective average. Unstandardized coordinate W1 is saved separately.",
            "Matched-N REAL floor compares two with-replacement empirical draws from the fixed finite REAL reference; all SOURCE comparisons still use complete REAL. These 32 values are not population confidence intervals.",
            "Endpoint noise is fixed per parent. Repeated parent resampling repeats its stored value, unlike independent noise at each training occurrence.",
            "Nominal charge bounds describe this endpoint's V480 coordinate contributions; no interval certificate, uniform lower bound, causality, optimization guarantee, or original A-F infeasibility follows.",
            "Fixed diagnostic projections do not replace the original redraw-at-each-update training directions. NumPy and device FFT/BLAS bitwise agreement is not claimed.",
        ],
        "scientific_work_added_by_this_module": {"neural_forwards": 0, "gradients": 0, "fits": 0, "optimizer_updates": 0, "random_draws": 0, "files_read": 0, "files_written": 0},
    }
    if control_terms is not None:
        result["same_treated_REAL_original_SOURCE_mass_weighted_regional_terms"] = dict(zip(TERM_NAMES, (control_terms[1:]*mass_s[:, None]).sum(0).tolist()))
    return result, private
