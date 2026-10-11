"""Run only small, artificial weighted-W1 and two-pass-VJP proof cases."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
from fractions import Fraction
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
from resource_guard import enter_guard, snapshot
INITIAL_RESOURCE = enter_guard(ROOT)  # Before importing numerical libraries.

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
assert torch.version.cuda is None, "Use the inspected local CPU-only torch runtime."


def _forbidden_load(*_args, **_kwargs):
    raise RuntimeError("Scientific arrays and model checkpoint loading are forbidden.")


np.load = _forbidden_load
torch.load = _forbidden_load

from weighted_empirical_w1 import (
    FrozenMass, quantile_plan, weighted_w1_numpy, weighted_w1_torch,
)

FIXTURE_PATH = ROOT / "artificial_fixture.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf8"))
TOL = FIXTURE["tolerances"]
TOY_COUNTERS = {
    "artificial_SOURCE_torch_calls": 0, "artificial_SOURCE_torch_rows": 0,
    "generic_REAL_torch_calls": 0, "generic_REAL_torch_rows": 0,
    "artificial_numpy_model_calls": 0,
}


def hash_file(path: Path) -> dict:
    content = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def immutable_inputs():
    result = []
    for record in FIXTURE["immutable_text_inputs"]:
        p = Path(record["path"]).resolve()
        binding = hash_file(p)
        if binding["sha256"] != record["sha256"]:
            raise AssertionError(f"Immutable text input changed: {p.name}")
        result.append({**binding, "expected_sha256": record["sha256"], "match": True})
    return result


INPUTS_BEFORE = immutable_inputs()


def close(actual, expected, tolerance, label):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: mismatched shape or nonfinite value")
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if error > tolerance:
        raise AssertionError(f"{label}: error {error:.17g} > {tolerance:.17g}")
    return error


def masses(values, label):
    return FrozenMass.from_normalized(values, label=label)


def torch_value_gradients(x, a, y, b):
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    value = weighted_w1_torch(tx, a, ty, b)
    gx, gy = torch.autograd.grad(value, (tx, ty))
    return float(value.detach()), gx.detach().numpy(), gy.detach().numpy()


def coordinate_CDF_oracle(x, a, y, b):
    """Independent coordinate-axis integral, not the helper's quantile plan.

    Every supplied binary64 mass AND coordinate becomes an exact Fraction.
    This computes sum_(consecutive distinct coordinates)
        abs(CDF_SOURCE - CDF_REAL) * coordinate_interval_length.
    It neither calls quantile_plan nor uses its sorted transport pairings.
    """
    xx, yy = [float(z) for z in x], [float(z) for z in y]
    ax, by = [Fraction(float(z)) for z in a], [Fraction(float(z)) for z in b]
    sx, sy = sum(ax), sum(by)
    ax, by = [z / sx for z in ax], [z / sy for z in by]
    points = sorted(set(xx + yy))
    answer = Fraction(0)
    for left, right in zip(points[:-1], points[1:]):
        fx = sum((w for v, w in zip(xx, ax) if v <= left), Fraction(0))
        fy = sum((w for v, w in zip(yy, by) if v <= left), Fraction(0))
        answer += abs(fx - fy) * (Fraction(right) - Fraction(left))
    return float(answer)


def analytic_unequal():
    f = FIXTURE["analytic_unequal"]
    ax, by = masses(f["source_mass"], "SOURCE"), masses(f["real_mass"], "REAL")
    r = weighted_w1_numpy(f["source"], ax, f["real"], by)
    tv, tx, ty = torch_value_gradients(f["source"], ax, f["real"], by)
    errors = {
        "numpy_value": close(r["value"], f["W1"], TOL["scalar_f64_absolute"], "analytic value"),
        "torch_value": close(tv, f["W1"], TOL["scalar_f64_absolute"], "torch analytic value"),
        "numpy_SOURCE_gradient": close(r["source_gradient"], f["source_gradient"],
                                      TOL["endpoint_gradient_f64_absolute"], "SOURCE gradient"),
        "numpy_REAL_gradient": close(r["target_gradient"], f["real_gradient"],
                                    TOL["endpoint_gradient_f64_absolute"], "REAL gradient"),
        "torch_SOURCE_gradient": close(tx, f["source_gradient"],
                                      TOL["endpoint_gradient_f64_absolute"], "torch SOURCE gradient"),
        "torch_REAL_gradient": close(ty, f["real_gradient"],
                                    TOL["endpoint_gradient_f64_absolute"], "torch REAL gradient"),
        "coordinate_CDF_value": close(
            coordinate_CDF_oracle(f["source"], f["source_mass"], f["real"], f["real_mass"]),
            f["W1"], TOL["scalar_f64_absolute"], "independent CDF value"),
    }
    step = TOL["finite_difference_step"]
    sx, ry = np.array(f["source"], dtype="f8"), np.array(f["real"], dtype="f8")
    gradients = []
    for domain, base in (("SOURCE", sx), ("REAL", ry)):
        result = []
        for i in range(len(base)):
            plus, minus = base.copy(), base.copy()
            plus[i] += step
            minus[i] -= step
            if domain == "SOURCE":
                v1 = coordinate_CDF_oracle(plus, f["source_mass"], ry, f["real_mass"])
                v0 = coordinate_CDF_oracle(minus, f["source_mass"], ry, f["real_mass"])
            else:
                v1 = coordinate_CDF_oracle(sx, f["source_mass"], plus, f["real_mass"])
                v0 = coordinate_CDF_oracle(sx, f["source_mass"], minus, f["real_mass"])
            result.append((v1 - v0) / (2 * step))
        gradients.append(result)
    errors["SOURCE_independent_finite_difference"] = close(
        gradients[0], tx, TOL["finite_difference_absolute"], "SOURCE finite difference"
    )
    errors["REAL_independent_finite_difference"] = close(
        gradients[1], ty, TOL["finite_difference_absolute"], "REAL finite difference"
    )
    close(tx.sum() + ty.sum(), 0.0, TOL["endpoint_gradient_f64_absolute"], "joint translation")
    return {"W1": tv, "errors": errors, "source_mass": ax.to_dict(),
            "real_mass": by.to_dict(), "quantile_plan": r["plan"].to_dict()}


def repeated_values_and_tie_ambiguity():
    f = FIXTURE["ties"]
    x, y = np.asarray(f["source"]), np.asarray(f["real"])
    a, b = masses(f["source_mass"], "SOURCE"), masses(f["real_mass"], "REAL")
    r = weighted_w1_numpy(x, a, y, b)
    permutation = np.asarray(f["source_permutation"], dtype="i8")
    ap = masses(np.asarray(f["source_mass"])[permutation], "SOURCE_permuted")
    rp = weighted_w1_numpy(x[permutation], ap, y, b)
    mapped = np.zeros(len(x), dtype="f8")
    mapped[permutation] = rp["source_gradient"]
    close(r["value"], 1.0, 0.0, "tied W1")
    close(rp["value"], r["value"], 0.0, "tie permutation value")
    close(r["source_gradient"], f["expected_original_gradient"], 0.0, "stable tie branch")
    close(mapped, f["expected_permuted_gradient_in_original_order"], 0.0, "alternate tie branch")
    j = np.asarray(f["distinct_parent_jacobians"])
    pg1, pg2 = float(r["source_gradient"] @ j), float(mapped @ j)
    if abs(pg1 - pg2) < TOL["negative_control_minimum_difference"]:
        raise AssertionError("Tie ambiguity example unexpectedly lost parameter-gradient difference.")
    for _ in range(3):
        tv, gx, gy = torch_value_gradients(x, a, y, b)
        close(tv, r["value"], 0.0, "repeat value")
        close(gx, r["source_gradient"], 0.0, "repeat SOURCE gradient")
        close(gy, r["target_gradient"], 0.0, "repeat REAL gradient")
    # Repeated values on BOTH sides, same law, different atom decomposition.
    ax = masses([0.25, 0.25, 0.5], "SOURCE")
    by = masses([0.5, 0.25, 0.25], "REAL")
    tv, gx, gy = torch_value_gradients([0, 0, 2], ax, [0, 2, 2], by)
    close(tv, 0.0, 0.0, "same repeated law")
    close(gx, [0, 0, 0], 0.0, "abs zero SOURCE convention")
    close(gy, [0, 0, 0], 0.0, "abs zero REAL convention")
    return {
        "same_value_after_permuting_tied_atoms": r["value"],
        "original_SOURCE_gradient": r["source_gradient"].tolist(),
        "permuted_gradient_mapped_to_original_parent_order": mapped.tolist(),
        "distinct_parent_jacobians": j.tolist(),
        "two_selected_parameter_gradients": [pg1, pg2],
        "stable_repeats_exact": True,
        "repeated_both_sides_zero_law_value_and_gradients": True,
        "interpretation": "A fixed stable branch is deterministic, not a unique derivative at ties.",
        "plan_original": r["plan"].to_dict(),
        "plan_permuted": rp["plan"].to_dict(),
    }


def zero_masses_and_sub_ULP_CDF_interval():
    a = masses([0, 0.25, 0.75, 0], "SOURCE")
    b = masses([0.5, 0.25, 0.25], "REAL")
    v, gx, gy = torch_value_gradients([-999, 0, 4, 999], a, [-1, 1, 5], b)
    close(v, 2.5, 0.0, "zero masses")
    close(gx, [0, 0.25, 0.25, 0], 0.0, "zero-mass endpoint gradients")
    close(gy, [-0.5, -0.25, 0.25], 0.0, "zero-mass REAL gradients")
    tiny = math.ldexp(1.0, -60)
    at = masses([1.0, tiny], "SOURCE_tiny_positive_atom")
    bt = masses([1.0], "REAL_singleton")
    r = weighted_w1_numpy([0, 1], at, [0], bt)
    oracle = coordinate_CDF_oracle([0, 1], [1, tiny], [0], [1])
    close(r["value"], oracle, 0.0, "tiny interval exact CDF oracle")
    if len(r["plan"].segment_numerators) != 2 or r["source_gradient"][1] <= 0:
        raise AssertionError("A positive interval smaller than an ulp at CDF=1 was dropped.")
    return {
        "zero_mass_gradients_exactly_zero": True,
        "tiny_mass_input": tiny,
        "tiny_W1": r["value"],
        "tiny_SOURCE_gradient": r["source_gradient"].tolist(),
        "positive_quantile_segments": len(r["plan"].segment_numerators),
        "tiny_mass_exact_normalization": at.to_dict(),
        "no_midpoint_or_float_CDF_collapse": True,
    }


def separate_global_mass_conventions():
    f = FIXTURE["global_mass_convention"]
    cs, cr = np.asarray(f["original_source_region_counts"]), np.asarray(f["original_real_region_counts"])
    ks, kr = np.asarray(f["source_draw_counts"]), np.asarray(f["real_draw_counts"])
    ms, mr = cs / cs.sum(), cr / cr.sum()
    coordinates = np.asarray(f["region_coordinate"], dtype="f8")
    x, y = np.repeat(coordinates, ks), np.repeat(coordinates, kr)
    a = masses(np.repeat(ms / ks, ks), "SOURCE_original_global_mass")
    b = masses(np.repeat(mr / kr, kr), "REAL_original_global_mass")
    r = weighted_w1_numpy(x, a, y, b)
    full_x, full_y = np.repeat(coordinates, cs), np.repeat(coordinates, cr)
    fa = masses(np.full(cs.sum(), 1.0 / cs.sum()), "SOURCE_full_parent_uniform")
    fb = masses(np.full(cr.sum(), 1.0 / cr.sum()), "REAL_full_parent_uniform")
    full = weighted_w1_numpy(full_x, fa, full_y, fb)
    wrong = weighted_w1_numpy(coordinates, masses(ms, "SOURCE"),
                              coordinates, masses(ms, "WRONG_REAL_using_SOURCE_mass"))
    close(r["value"], f["expected_correct_W1"], TOL["scalar_f64_absolute"], "correct global mixture")
    close(full["value"], r["value"], TOL["scalar_f64_absolute"], "full vs stratified global mixture")
    close(wrong["value"], f["expected_wrong_common_mass_W1"], 0.0, "wrong mass negative control")
    if abs(r["value"] - wrong["value"]) < TOL["negative_control_minimum_difference"]:
        raise AssertionError("Wrong shared global mass was not detected.")
    return {
        "SOURCE_original_region_mass": ms.tolist(), "REAL_original_region_mass": mr.tolist(),
        "SOURCE_draw_row_mass": a.as_float64().tolist(), "REAL_draw_row_mass": b.as_float64().tolist(),
        "source_draw_count": len(x), "real_draw_count": len(y),
        "full_original_parent_counts": [len(full_x), len(full_y)],
        "correct_W1": r["value"], "full_parent_W1": full["value"],
        "wrong_common_mass_W1": wrong["value"],
        "not_a_project_data_measurement": True,
    }


def equal_count_equal_mass_special_case():
    x0 = [-1.2, 0.3, 0.8, -0.4, 1.9, -0.6, 0.11]
    y0 = [0.05, 0.9, -1.4, 1.5, -0.8, 0.23, 0.67]
    a, b = masses(np.full(7, 1 / 7), "SOURCE"), masses(np.full(7, 1 / 7), "REAL")
    result = {}
    for dtype in (torch.float64, torch.float32):
        x = torch.tensor(x0, dtype=dtype, requires_grad=True)
        y = torch.tensor(y0, dtype=dtype, requires_grad=True)
        actual = weighted_w1_torch(x, a, y, b)
        expected = (torch.sort(x, stable=True).values - torch.sort(y, stable=True).values).abs().mean()
        ag = torch.autograd.grad(actual, (x, y), retain_graph=True)
        eg = torch.autograd.grad(expected, (x, y))
        tolerance = TOL["scalar_f64_absolute"] if dtype == torch.float64 else TOL["equal_weight_f32_absolute"]
        result[str(dtype)] = {
            "value_error": close(float(actual.detach()), float(expected.detach()), tolerance, "equal uniform value"),
            "SOURCE_gradient_error": close(ag[0].numpy(), eg[0].numpy(), tolerance, "equal uniform source gradient"),
            "REAL_gradient_error": close(ag[1].numpy(), eg[1].numpy(), tolerance, "equal uniform real gradient"),
            "tolerance_frozen_before_run": tolerance,
        }
    return {"numeric_equivalence_tested_not_bitwise_R63_claim": True, "results": result}


def independent_seeded_scalar_cases():
    f = FIXTURE["seeded_scalar_cases"]
    rng = np.random.Generator(np.random.PCG64(f["seed"]))
    details = []
    for k in range(f["count"]):
        n, m = [int(v) for v in rng.integers(1, f["maximum_atoms_per_domain"] + 1, size=2)]
        x, y = [rng.integers(-16, 17, size=z).astype("f8") * f["coordinate_grid_step"] for z in (n, m)]
        wa, wb = [rng.integers(0, 6, size=z) for z in (n, m)]
        if wa.sum() == 0:
            wa[0] = 1
        if wb.sum() == 0:
            wb[0] = 1
        wa, wb = wa / wa.sum(), wb / wb.sum()
        a, b = masses(wa, "SOURCE"), masses(wb, "REAL")
        r = weighted_w1_numpy(x, a, y, b)
        oracle = coordinate_CDF_oracle(x, wa, y, wb)
        tv, tx, ty = torch_value_gradients(x, a, y, b)
        details.append({
            "case": k, "SOURCE_atoms": n, "REAL_atoms": m,
            "value": tv,
            "independent_CDF_error": close(tv, oracle, TOL["scalar_f64_absolute"], "seeded CDF"),
            "numpy_value_error": close(tv, r["value"], TOL["scalar_f64_absolute"], "seeded numpy"),
            "SOURCE_gradient_error": close(tx, r["source_gradient"], TOL["endpoint_gradient_f64_absolute"], "seeded source gradient"),
            "REAL_gradient_error": close(ty, r["target_gradient"], TOL["endpoint_gradient_f64_absolute"], "seeded real gradient"),
        })
    return {"seed": f["seed"], "count": len(details), "cases": details,
            "independent_reference": "exact rational coordinate-space CDF integral"}


def input_contract_and_frozen_snapshot():
    rejected = []
    def reject(name, fn):
        try:
            fn()
        except (ValueError, TypeError, FloatingPointError):
            rejected.append(name)
        else:
            raise AssertionError(f"Invalid input was accepted: {name}")
    for name, values in (
        ("negative_mass", [-0.1, 1.1]), ("zero_total", [0, 0]),
        ("nonunit_total", [1, 1]), ("nan_mass", [float("nan"), 1]),
        ("infinite_mass", [float("inf"), 0]), ("empty_mass", []),
        ("two_dimensional_mass", [[0.5, 0.5]]), ("complex_mass", [1 + 1j]),
    ):
        reject(name, lambda values=values: masses(values, name))
    reject("trainable_mass", lambda: masses(torch.tensor([0.5, 0.5], requires_grad=True), "bad"))
    a = masses([0.5, 0.5], "SOURCE")
    b = masses([1], "REAL")
    reject("length_mismatch", lambda: weighted_w1_numpy([0], a, [0], b))
    reject("nan_endpoint", lambda: weighted_w1_numpy([0, float("nan")], a, [0], b))
    reject("multidimensional_endpoint", lambda: quantile_plan([[0, 1]], a, [0], b))
    reject("unfrozen_mass_type", lambda: quantile_plan([0, 1], [0.5, 0.5], [0], b))
    original = np.array([0.25, 0.75], dtype="f8")
    frozen = masses(original, "immutable_snapshot")
    original[:] = [0.8, 0.2]
    close(frozen.as_float64(), [0.25, 0.75], 0.0, "mass snapshot immutability")
    return {"rejected_invalid_cases": rejected, "mass_snapshot_unaffected_by_caller_mutation": True}


def toy_torch(theta, inputs, epsilon, *, domain):
    TOY_COUNTERS[domain + "_torch_calls"] += 1
    TOY_COUNTERS[domain + "_torch_rows"] += len(inputs)
    u, v = inputs[:, 0], inputs[:, 1]
    z = theta[1] * u + theta[2] * v + epsilon
    return theta[0] * torch.tanh(z) + theta[3] * u.square() + torch.sin(theta[4] * v + theta[5]) + 0.1 * theta[2] * u * v


def toy_numpy(theta, inputs, epsilon):
    TOY_COUNTERS["artificial_numpy_model_calls"] += 1
    u, v = np.asarray(inputs).T
    return theta[0] * np.tanh(theta[1] * u + theta[2] * v + epsilon) + theta[3] * u**2 + np.sin(theta[4] * v + theta[5]) + 0.1 * theta[2] * u * v


def toy_analytic_jacobian(theta, inputs, epsilon):
    u, v = np.asarray(inputs).T
    z = np.tanh(theta[1] * u + theta[2] * v + epsilon)
    c = np.cos(theta[4] * v + theta[5])
    return np.stack((z, theta[0] * (1 - z*z) * u,
                     theta[0] * (1 - z*z) * v + 0.1 * u * v,
                     u*u, c*v, c), axis=1)


def chunked_toy(theta, data, epsilon, chunk, domain):
    return torch.cat([
        toy_torch(theta, data[k:k+chunk], epsilon[k:k+chunk], domain=domain)
        for k in range(0, len(data), chunk)
    ])


def tiny_model_VJP(role):
    f = FIXTURE["tiny_model"]
    both = role == "two_sided_shared_theta"
    if role not in f["real_roles"]:
        raise ValueError(role)
    before_calls = dict(TOY_COUNTERS)
    theta_np = np.asarray(f["theta"], dtype="f8")
    sx, ry = np.asarray(f["source_inputs"], dtype="f8"), np.asarray(f["real_inputs"], dtype="f8")
    es = np.asarray(f["source_parent_epsilon"], dtype="f8")
    er = np.asarray(f["real_parent_epsilon_generic_case_only"], dtype="f8")
    ds, dr = np.asarray(f["source_draws"], dtype="i8"), np.asarray(f["real_draws"], dtype="i8")
    cs, cr = np.asarray(f["source_original_region_counts"]), np.asarray(f["real_original_region_counts"])
    ks, kr = np.asarray(f["source_draw_region_counts"]), np.asarray(f["real_draw_region_counts"])
    wa, wb = np.repeat((cs / cs.sum()) / ks, ks), np.repeat((cr / cr.sum()) / kr, kr)
    a, b = masses(wa, "SOURCE_own_original_mass"), masses(wb, "REAL_own_original_mass")
    ts, tr, tes, ter = [torch.tensor(z, dtype=torch.float64) for z in (sx, ry, es, er)]
    ids, idr = torch.tensor(ds), torch.tensor(dr)
    fixed_reference = torch.tensor(f["fixed_real_reference"], dtype=torch.float64)
    chunk = int(f["chunk_size"])

    # Retained full graph, assembled using the exact same chunk schedule.
    theta_retained = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    source_retained = chunked_toy(theta_retained, ts, tes, chunk, "artificial_SOURCE")
    real_retained = (
        chunked_toy(theta_retained, tr, ter, chunk, "generic_REAL") if both else fixed_reference
    )
    loss_retained = weighted_w1_torch(source_retained[ids], a, real_retained[idr], b)
    retained_gradient = torch.autograd.grad(loss_retained, theta_retained, retain_graph=True)[0].detach()
    retained_source_partial = torch.autograd.grad(loss_retained, source_retained, retain_graph=True)[0].detach()
    retained_real_partial = (
        torch.autograd.grad(loss_retained, real_retained, retain_graph=True)[0].detach()
        if both else None
    )

    # Pass 1: both unique parent endpoint arrays are cached without network graphs.
    theta_replay = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        source_cache = chunked_toy(theta_replay, ts, tes, chunk, "artificial_SOURCE")
        real_cache = (
            chunked_toy(theta_replay, tr, ter, chunk, "generic_REAL") if both else fixed_reference.clone()
        )
    if not torch.equal(source_cache, source_retained.detach()) or not torch.equal(real_cache, real_retained.detach()):
        raise AssertionError("Retained and no-grad cached endpoints differ.")
    source_leaf = source_cache.detach().requires_grad_(True)
    real_leaf = real_cache.detach().requires_grad_(both)
    endpoint_loss = weighted_w1_torch(source_leaf[ids], a, real_leaf[idr], b)
    leaves = (source_leaf, real_leaf) if both else (source_leaf,)
    partials = torch.autograd.grad(endpoint_loss, leaves)
    gs = partials[0].detach()
    gr = partials[1].detach() if both else None
    partial_error = close(gs.numpy(), retained_source_partial.numpy(),
                          TOL["endpoint_gradient_f64_absolute"], "cached SOURCE partial")
    if both:
        partial_error = max(partial_error, close(
            gr.numpy(), retained_real_partial.numpy(), TOL["endpoint_gradient_f64_absolute"],
            "cached REAL partial"
        ))

    # Independent occurrence partial -> parent scatter-add check.
    scalar = weighted_w1_numpy(source_cache.numpy()[ds], a, real_cache.numpy()[dr], b)
    gs_expected = np.zeros(len(sx), dtype="f8")
    gr_expected = np.zeros(len(ry), dtype="f8")
    np.add.at(gs_expected, ds, scalar["source_gradient"])
    np.add.at(gr_expected, dr, scalar["target_gradient"])
    close(gs.numpy(), gs_expected, TOL["endpoint_gradient_f64_absolute"], "duplicate source scatter")
    if both:
        close(gr.numpy(), gr_expected, TOL["endpoint_gradient_f64_absolute"], "duplicate real scatter")

    # Pass 2: replay and immediately free each chunk graph. Sum BOTH sides if needed.
    g_source = torch.zeros_like(theta_replay)
    g_real = torch.zeros_like(theta_replay)
    for data, epsilon, cache, partial, accumulator, domain in (
        (ts, tes, source_cache, gs, g_source, "artificial_SOURCE"),
        (tr, ter, real_cache, gr, g_real, "generic_REAL"),
    ):
        if partial is None:
            continue  # Fixed REAL reference has no network Jacobian.
        for k in range(0, len(data), chunk):
            out = toy_torch(theta_replay, data[k:k+chunk], epsilon[k:k+chunk], domain=domain)
            if not torch.equal(out.detach(), cache[k:k+chunk]):
                raise AssertionError("Replay endpoint differs from frozen cache.")
            contribution = torch.autograd.grad(out, theta_replay, grad_outputs=partial[k:k+chunk])[0]
            accumulator += contribution.detach()
    two_pass = g_source + g_real
    loss_error = close(float(endpoint_loss.detach()), float(loss_retained.detach()),
                       TOL["scalar_f64_absolute"], "cached loss")
    gradient_error = close(two_pass.numpy(), retained_gradient.numpy(),
                           TOL["parameter_gradient_f64_absolute"], "two-pass vs retained graph")
    js = toy_analytic_jacobian(theta_np, sx, es)
    analytic = js.T @ gs_expected
    if both:
        analytic += toy_analytic_jacobian(theta_np, ry, er).T @ gr_expected
    analytic_error = close(two_pass.numpy(), analytic,
                           TOL["parameter_gradient_f64_absolute"], "independent analytic Jacobian")

    # Independent coordinate-space CDF finite differences of the tiny smooth model.
    def independent_scalar(theta):
        outputs = toy_numpy(theta, sx, es)[ds]
        refs = toy_numpy(theta, ry, er)[dr] if both else np.asarray(f["fixed_real_reference"])[dr]
        return coordinate_CDF_oracle(outputs, wa, refs, wb)
    step, fd = TOL["finite_difference_step"], []
    for j in range(len(theta_np)):
        high, low = theta_np.copy(), theta_np.copy()
        high[j] += step
        low[j] -= step
        fd.append((independent_scalar(high) - independent_scalar(low)) / (2 * step))
    fd_error = close(two_pass.numpy(), fd, TOL["finite_difference_absolute"], "independent model finite difference")
    close(theta_replay.detach().numpy(), theta_np, 0.0, "no parameter updates")

    # Dropping the transformed REAL branch must fail in the generic two-sided case.
    omitted_real_error = float(torch.max(torch.abs(g_source - retained_gradient)))
    if both and omitted_real_error < TOL["negative_control_minimum_difference"]:
        raise AssertionError("Generic two-sided guard failed to detect missing REAL VJP.")
    if not both:
        close(g_real.numpy(), np.zeros(len(theta_np)), 0.0, "fixed reference parameter contribution")
        if TOY_COUNTERS["generic_REAL_torch_calls"] != before_calls["generic_REAL_torch_calls"]:
            raise AssertionError("Fixed-reference case invoked a REAL network.")
    wrong_chunk_average = two_pass / (math.ceil(len(sx)/chunk) + (math.ceil(len(ry)/chunk) if both else 0))
    wrong_average_error = float(torch.max(torch.abs(wrong_chunk_average - retained_gradient)))
    if wrong_average_error < TOL["negative_control_minimum_difference"]:
        raise AssertionError("Unexpectedly insensitive to erroneous extra VJP chunk normalization.")
    return {
        "role": role, "meaning": f["real_roles"][role],
        "SOURCE_parents": len(sx), "REAL_reference_parents": len(ry),
        "SOURCE_draw_slots": len(ds), "REAL_draw_slots": len(dr),
        "SOURCE_duplicate_draws": len(ds) - len(np.unique(ds)),
        "REAL_duplicate_draws": len(dr) - len(np.unique(dr)),
        "chunk_size": chunk, "frozen_parent_noise": True,
        "SOURCE_global_row_mass": a.as_float64().tolist(),
        "REAL_global_row_mass": b.as_float64().tolist(),
        "loss": float(loss_retained.detach()),
        "retained_graph_gradient": retained_gradient.numpy().tolist(),
        "two_pass_gradient": two_pass.numpy().tolist(),
        "source_parameter_contribution": g_source.numpy().tolist(),
        "real_parameter_contribution": g_real.numpy().tolist(),
        "independent_analytic_Jacobian_gradient": analytic.tolist(),
        "independent_CDF_parameter_finite_difference": fd,
        "errors": {"loss": loss_error, "endpoint_partial": partial_error,
                   "parameter_VJP": gradient_error, "analytic_Jacobian": analytic_error,
                   "independent_parameter_finite_difference": fd_error},
        "negative_controls": {
            "missing_REAL_VJP_applicable": both,
            "missing_REAL_VJP_error": omitted_real_error,
            "wrong_extra_chunk_average_error": wrong_average_error,
        },
        "retained_no_grad_and_replayed_endpoints_bitwise_equal": True,
        "parameters_unchanged": True,
        "fixed_REAL_reference_network_calls": 0 if not both else "not_applicable_generic_transformed_toy",
        "counts_this_case": {k: TOY_COUNTERS[k] - before_calls[k] for k in TOY_COUNTERS},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if ROOT not in output.parents:
        raise RuntimeError("All outputs must remain under c1_preparation.")
    output.mkdir(parents=True, exist_ok=True)
    checks = [
        ("analytic_unequal_counts_and_both_endpoint_gradients", analytic_unequal),
        ("repeated_values_and_tie_gradient_ambiguity", repeated_values_and_tie_ambiguity),
        ("zero_masses_and_sub_ULP_CDF_segment", zero_masses_and_sub_ULP_CDF_interval),
        ("separate_SOURCE_REAL_global_mass_conventions", separate_global_mass_conventions),
        ("equal_count_equal_mass_special_case_f64_f32", equal_count_equal_mass_special_case),
        ("independent_seeded_coordinate_CDF_oracle", independent_seeded_scalar_cases),
        ("frozen_mass_and_input_validation_contract", input_contract_and_frozen_snapshot),
        ("SOURCE_parameterized_REAL_fixed_reference_two_pass_VJP",
         lambda: tiny_model_VJP("source_only_fixed_reference")),
        ("generic_both_endpoint_laws_share_theta_two_pass_VJP",
         lambda: tiny_model_VJP("two_sided_shared_theta")),
    ]
    results = []
    for name, check in checks:
        start = time.perf_counter()
        try:
            details = check()
            result = {"name": name, "passed": True, "details": details}
        except Exception as exc:
            result = {"name": name, "passed": False, "error": repr(exc),
                      "traceback": traceback.format_exc()}
        result["elapsed_seconds"] = time.perf_counter() - start
        results.append(result)
        snapshot()
    inputs_after = immutable_inputs()
    passed = all(row["passed"] for row in results)
    result_path = output / "ARTIFICIAL_TEST_RESULTS.json"
    write_json(result_path, {
        "status": "passed" if passed else "failed",
        "artificial_only": True, "test_groups": len(results),
        "passed_groups": sum(row["passed"] for row in results),
        "tolerances_frozen_in_fixture": TOL, "results": results,
        "artificial_model_counters": TOY_COUNTERS,
        "scientific_forward_gradient_fit_or_training_counts": 0,
        "scope_and_limitations": FIXTURE["limitations"],
    })
    source_names = [
        "weighted_empirical_w1.py", "artificial_vjp_proof.py", "resource_guard.py",
        "run_artificial.ps1", "artificial_fixture.json",
    ]
    source_bindings = [hash_file(ROOT / name) for name in source_names]
    resource = {
        **snapshot(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cuda_version": torch.version.cuda,
        "device": "cpu",
        "numpy_load_disabled": True, "torch_load_disabled": True,
    }
    resource_path = output / "RESOURCE_EVIDENCE.json"
    write_json(resource_path, resource)
    receipt = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "artificial_only_code_preparation": True,
        "not_adopted_scientific_protocol": True,
        "not_exact_R63_objective": True,
        "runtime": {
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "numpy_file": np.__file__,
            "torch_version": torch.__version__, "torch_file": torch.__file__,
            "platform": platform.platform(),
        },
        "input_bindings_before": INPUTS_BEFORE,
        "input_bindings_after": inputs_after,
        "immutable_reports_byte_identical": INPUTS_BEFORE == inputs_after,
        "source_and_fixture_bindings": source_bindings,
        "output_bindings": [hash_file(result_path), hash_file(resource_path)],
        "resource_evidence": resource,
        "test_groups": len(results), "passed_groups": sum(row["passed"] for row in results),
        "artificial_model_counters": TOY_COUNTERS,
        "scientific_arrays_read": 0, "scientific_model_checkpoints_loaded": 0,
        "scientific_forwards_or_gradients_or_fits": 0, "network_training_updates": 0,
        "R63_source_or_protocol_changes": 0, "root_metadata_changes": 0,
        "git_operations": 0, "new_agents": 0,
        "limitations": FIXTURE["limitations"],
        "two_REAL_roles": FIXTURE["tiny_model"]["real_roles"],
        "tie_policy": "Stable input-order branch; abs(0)=0; no unique tie-gradient claim.",
        "exactness": "Exact integer CDF partition for normalized binary64 masses; binary64 endpoint distances/reductions.",
        "full_reference_interpretation": "A parent-fixed-noise diagnostic would remain a conditional finite empirical objective. R63 uses independent per-occurrence noise.",
        "receipt_self_hash_intentionally_not_embedded": True,
    }
    proof_path = output / "PROOF_ATTEMPT_RECEIPT.json"
    write_json(proof_path, receipt)
    # A final independent Windows peak query is emitted for the launcher.
    end_resource = snapshot()
    print(json.dumps({
        "status": receipt["status"], "proof_receipt": str(proof_path),
        "passed_groups": receipt["passed_groups"], "test_groups": receipt["test_groups"],
        "peak_RSS_bytes": end_resource["peak_RSS_bytes_Windows_PeakWorkingSetSize"],
        "CPU_affinity_count": end_resource["cpu_affinity_count"],
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "scientific_arrays_or_models_loaded": 0,
    }), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
