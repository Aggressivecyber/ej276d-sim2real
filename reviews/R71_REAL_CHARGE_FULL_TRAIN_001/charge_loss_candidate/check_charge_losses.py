"""Artificial-only native checks. Never opens actual scientific data."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
GUARD = ROOT / "reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py"
if os.name == "nt":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 4)
spec = importlib.util.spec_from_file_location("r71_artificial_guard", GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
LIMITS = guard.apply(OUT)

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from charge_losses import FullChargeObjective

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
START = time.perf_counter()
DTYPE = torch.float64
KEYS = ("charge_distribution", "REAL_identity", "source_charge_order")


def vector(values, device, grad=False):
    return torch.tensor(values, dtype=DTYPE, device=device,
                        requires_grad=grad)


def dense_w1(a, b):
    # Independent dense TOY oracle, not an implementation for full populations.
    a = torch.sort(a, stable=True).values
    b = torch.sort(b, stable=True).values
    n, m = len(a), len(b)
    lo_a = torch.arange(n, device=a.device, dtype=DTYPE) / n
    hi_a = torch.arange(1, n + 1, device=a.device, dtype=DTYPE) / n
    lo_b = torch.arange(m, device=b.device, dtype=DTYPE) / m
    hi_b = torch.arange(1, m + 1, device=b.device, dtype=DTYPE) / m
    overlap = (torch.minimum(hi_a[:, None], hi_b[None, :])
               - torch.maximum(lo_a[:, None], lo_b[None, :])).clamp_min(0)
    return (overlap * (a[:, None] - b[None, :]).abs()).sum()


def dense_order(q0, prediction, sd):
    qsorted, ix = torch.sort(q0, stable=True)
    ordered = prediction.log()[ix]
    target = torch.sort(ordered, stable=True).values
    result = prediction.sum() * 0
    i = 0
    while i < len(q0):
        j = i + 1
        while j < len(q0) and bool(qsorted[j] == qsorted[i]):
            j += 1
        result = result + (j - i) * (
            ordered[i:j].mean() - target[i:j].mean()).abs()
        i = j
    return result / len(q0) / sd


def dense_terms(q0, rc, sp, rp, ri):
    sq, sl = rc.std(unbiased=False), rc.log().std(unbiased=False)
    return {
        "charge_distribution": .5 * (
            dense_w1(sp, rc) / sq + dense_w1(sp.log(), rc.log()) / sl),
        "REAL_identity": ((rp.log() - ri.log()).square().mean() / sl.square()),
        "source_charge_order": dense_order(q0, sp, sl),
    }


CASES = [
    ("unequal_sizes_smooth", [1., 2., 4.], [1.1, 1.6, 2.4, 3.8, 5.2],
     [3.1, 1.3, 4.5], [1.2, 1.4, 2.6, 3.5, 5.1],
     [1., 1.8, 2.1, 3.4, 5.]),
    ("repeated_predictions_and_reference", [2., 1., 2., 4.],
     [1.5, 1.5, 4.5], [3., 3., 1.5, 3.], [1.5, 1.7, 4.7],
     [1.4, 1.5, 4.5]),
    ("original_ties_with_crossing", [1., 1., 2., 2.], [1., 2., 4.],
     [1., 4., 2., 3.], [1., 2.2, 3.9], [1., 2., 4.]),
    ("single_source", [2.], [1., 3., 6.], [4.], [1., 3., 6.],
     [1., 3., 6.]),
    ("identical_law_different_counts", [1., 1., 3., 3.], [1., 3.],
     [1., 1., 3., 3.], [1., 3.], [1., 3.]),
]
checks = []
max_value_error = max_gradient_error = max_scipy_error = 0.


def compare_device(device):
    global max_value_error, max_gradient_error, max_scipy_error
    for name, q0, rc, s, r, ri in CASES:
        q0, rc, ri = (vector(v, device) for v in (q0, rc, ri))
        sp, rp = vector(s, device, True), vector(r, device, True)
        obj = FullChargeObjective(q0, rc)
        total, terms = obj(sp, rp, ri)
        oracle = dense_terms(q0, rc, sp, rp, ri)
        assert tuple(terms) == KEYS
        for key in KEYS:
            assert terms[key].dtype == DTYPE and terms[key].device == sp.device
            torch.testing.assert_close(terms[key], oracle[key],
                                       rtol=2e-12, atol=2e-12)
            a = torch.autograd.grad(terms[key], (sp, rp),
                                    retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(oracle[key], (sp, rp),
                                    retain_graph=True, allow_unused=True)
            for index, value in enumerate((sp, rp)):
                ga = torch.zeros_like(value) if a[index] is None else a[index]
                gb = torch.zeros_like(value) if b[index] is None else b[index]
                torch.testing.assert_close(ga, gb, rtol=2e-11, atol=2e-12)
                max_gradient_error = max(
                    max_gradient_error, float((ga - gb).abs().max()))
            max_value_error = max(
                max_value_error, float((terms[key] - oracle[key]).abs()))
        gs = torch.autograd.grad(total, (sp, rp))
        gd = torch.autograd.grad(sum(oracle.values()), (sp, rp))
        for a, b in zip(gs, gd):
            torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)

        scipy_q = wasserstein_distance(s, rc.cpu().numpy())
        scipy_log = wasserstein_distance(np.log(s), np.log(rc.cpu().numpy()))
        scipy_d = .5 * (scipy_q / float(obj.std_Q)
                        + scipy_log / float(obj.std_logQ))
        assert abs(float(terms["charge_distribution"]) - scipy_d) < 2e-12
        max_scipy_error = max(max_scipy_error,
                             abs(float(terms["charge_distribution"]) - scipy_d))
        assert obj.std_Q.dtype == obj.std_logQ.dtype == DTYPE
        assert obj.std_Q.device == obj.std_logQ.device == sp.device
        if name == "identical_law_different_counts":
            assert float(total) == 0. and all(bool((a == 0).all()) for a in gs)
        checks.append({"device": str(device), "case": name, "pass": True})

    # No within-tie artificial order, including the all-tied original population.
    for q0, a, b in [
        ([1., 1., 2., 2.], [2., 1.5, 3.5, 3.], [1.5, 2., 3., 3.5]),
        ([2., 2., 2., 2.], [9., 1., 7., 2.], [1., 9., 2., 7.]),
    ]:
        rc = vector([1., 2., 4.], device)
        obj = FullChargeObjective(vector(q0, device), rc)
        sa, sb = vector(a, device, True), vector(b, device, True)
        ta, pa = obj(sa, rc, rc)
        tb, pb = obj(sb, rc, rc)
        assert float(pa["source_charge_order"]) == 0.
        assert float(pb["source_charge_order"]) == 0.
        assert bool((torch.autograd.grad(pa["source_charge_order"], sa)[0] == 0).all())
        torch.testing.assert_close(ta, tb, rtol=2e-12, atol=2e-12)
    checks.append({"device": str(device), "case": "within_original_tie_permutation",
                   "pass": True})


compare_device("cpu")

# Smooth-case independent central finite difference, with no sort/sign crossings.
name, q0, rc, s, r, ri = CASES[0]
obj = FullChargeObjective(vector(q0, "cpu"), vector(rc, "cpu"))
sp, rp = vector(s, "cpu", True), vector(r, "cpu", True)
total, _ = obj(sp, rp, vector(ri, "cpu"))
grad = torch.autograd.grad(total, (sp, rp))
max_fd_error = 0.
for domain, values in enumerate((s, r)):
    for i in range(len(values)):
        plus, minus = list(values), list(values)
        h = 1e-6
        plus[i] += h
        minus[i] -= h
        sources = (plus, minus) if domain == 0 else (s, s)
        reals = (r, r) if domain == 0 else (plus, minus)
        a = float(obj(vector(sources[0], "cpu"), vector(reals[0], "cpu"),
                      vector(ri, "cpu"))[0])
        b = float(obj(vector(sources[1], "cpu"), vector(reals[1], "cpu"),
                      vector(ri, "cpu"))[0])
        error = abs((a - b) / (2 * h) - float(grad[domain][i]))
        max_fd_error = max(max_fd_error, error)
        assert error < 2e-8
checks.append({"device": "cpu", "case": "central_finite_difference", "pass": True})

def must_reject(label, fn):
    try:
        fn()
    except (ValueError, TypeError, FloatingPointError):
        checks.append({"case": label, "rejected": True})
    else:
        raise AssertionError(label + " should fail")


base = vector([1., 2., 4.], "cpu")
obj = FullChargeObjective(base, base)
must_reject("zero_reference_std", lambda: FullChargeObjective(base, base * 0 + 2))
must_reject("empty_source", lambda: FullChargeObjective(base[:0], base))
must_reject("wrong_dtype", lambda: FullChargeObjective(base.float(), base))
must_reject("zero_source_input", lambda: FullChargeObjective(vector([0., 1.], "cpu"), base))
must_reject("NaN_source_prediction", lambda: obj(vector([float("nan"), 2., 3.], "cpu"), base, base))
must_reject("nonpositive_REAL_prediction", lambda: obj(base, vector([0., 2., 3.], "cpu"), base))
must_reject("nonfinite_REAL_input", lambda: obj(base, base, vector([float("inf"), 2., 3.], "cpu")))
must_reject("parent_count_changed", lambda: obj(base[:2], base, base))
tiny = vector([1., 1. + 1e-9, 1. + 2e-9], "cpu")
small = FullChargeObjective(base, tiny)
assert float(small.std_Q) < 1e-8 and float(small.std_logQ) < 1e-8
assert bool(small.std_Q == tiny.std(unbiased=False))
checks.append({"case": "positive_small_std_not_floored", "pass": True})

# Artificial full dimensions only; this is not a model-training time forecast.
start_full = time.perf_counter()
q0 = 1. + torch.div(torch.arange(6593), 3, rounding_mode="floor").double() / 3000
rc = torch.linspace(1., 7., 21047, dtype=DTYPE)
sp = (q0 + .05 * torch.sin(torch.arange(6593, dtype=DTYPE))).requires_grad_()
rp = (rc * 1.001).requires_grad_()
large = FullChargeObjective(q0, rc)
loss, terms = large(sp, rp, rc)
grad = torch.autograd.grad(loss, (sp, rp))
assert all(bool(torch.isfinite(v).all()) for v in grad)
full_seconds = time.perf_counter() - start_full
assert large._quantile_mass.numel() <= 6593 + 21047 - 1
assert all(t.dtype == DTYPE for t in terms.values())
checks.append({"case": "artificial_6593_21047_full_forward_backward", "pass": True})

if torch.cuda.is_available():
    compare_device("cuda")
    cuda_status = "passed_artificial_value_gradient_and_tie_checks"
else:
    cuda_status = "unknown_no_local_CUDA_available"

receipt = {
    "status": "PASS_ARTIFICIAL_ONLY",
    "scientific_array_reads": 0, "actual_model_training_or_inference": 0,
    "CPU": "passed", "CUDA": cuda_status,
    "torch_version": torch.__version__, "scipy_version": __import__("scipy").__version__,
    "checks": checks,
    "max_dense_value_abs_error": max_value_error,
    "max_dense_gradient_abs_error": max_gradient_error,
    "max_scipy_distribution_abs_error": max_scipy_error,
    "max_smooth_central_difference_abs_error": max_fd_error,
    "artificial_full_dimension_forward_backward_seconds": full_seconds,
    "full_dimension_timing_scope": "synthetic scalar losses only, not full-model training",
    "elapsed_seconds": time.perf_counter() - START,
    "resource_limits": LIMITS, "resource_memory": guard.memory(),
    "source_SHA256": {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                      for name in ("charge_losses.py", "check_charge_losses.py")},
}
(OUT / "ARTIFICIAL_CHECK_RECEIPT.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=False))
