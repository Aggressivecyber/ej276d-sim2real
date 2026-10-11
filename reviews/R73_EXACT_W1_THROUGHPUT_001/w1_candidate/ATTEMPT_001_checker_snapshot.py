"""Artificial float64 checks; importing this module does not read files/draw RNG."""
from __future__ import annotations

import ast
from collections import OrderedDict
from fractions import Fraction
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np
import torch

from exact_uniform_w1 import UniformW1Cache


def original_class(source_path):
    """Compile only the exact sealed class and its validator, not the big loss."""
    path = Path(source_path)
    raw = path.read_bytes()
    parsed = ast.parse(raw, filename=str(path))
    selected = [node for node in parsed.body if getattr(node, "name", None)
                in ("_tensor", "UniformW1Cache")]
    if len(selected) != 2:
        raise AssertionError("sealed original validator/class not uniquely found")
    env = {"torch": torch, "math": math, "OrderedDict": OrderedDict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return env["UniformW1Cache"], hashlib.sha256(raw).hexdigest()


def _fraction_cells(a, b):
    """Independent rational value + stable per-occurrence abs-sign derivatives."""
    n, m = len(a), len(b)
    oa, ob = sorted(range(n), key=lambda i: a[i]), sorted(range(m), key=lambda j: b[j])
    edges = sorted({Fraction(i, n) for i in range(n + 1)} |
                   {Fraction(j, m) for j in range(m + 1)})
    value = Fraction(0)
    ga, gb = [Fraction(0) for _ in a], [Fraction(0) for _ in b]
    for lo, hi in zip(edges[:-1], edges[1:]):
        i, j = oa[int(lo * n)], ob[int(lo * m)]
        d, mass = Fraction(a[i]) - Fraction(b[j]), hi - lo
        s = (d > 0) - (d < 0)
        value += abs(d) * mass
        ga[i] += s * mass
        gb[j] -= s * mass
    return value, ga, gb


def _fraction_cdf(a, b):
    n, m = len(a), len(b)
    result = [Fraction(0) for _ in a]
    for i, parent in enumerate(sorted(range(n), key=lambda j: a[j])):
        lo, hi = Fraction(i, n), Fraction(i + 1, n)
        left, right = Fraction(sum(v < a[parent] for v in b), m), Fraction(sum(v <= a[parent] for v in b), m)
        result[parent] = max(min(left, hi) - lo, 0) - max(hi - max(right, lo), 0)
    return result


def run_checks(device="cpu", original_source=None) -> dict:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if original_source is None:
        original_source = Path(__file__).resolve().parents[2] / "work/R72_FULL_WAVEFORM_FULL_TRAIN_001/full_waveform_loss.py"
    Original, source_sha = original_class(original_source)
    eps = torch.finfo(torch.float64).eps
    summaries = {"forward_bitwise_cases": 0, "gradient_sides_compared": 0,
                 "max_gradient_absolute_difference": 0.0, "max_gradient_error_to_roundoff_bound": 0.0,
                 "rational_formula_cases": 0, "comparison_cases": 0}
    checks = []

    def require(name, condition):
        if not bool(condition):
            raise AssertionError(name)

    def bytes_equal(a, b):
        return torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64))

    def compare(a_array, b_array, upstream=None, *, b_sorted=False,
                need_a=True, need_b=True, noncontiguous=False, rational=False):
        a_array, b_array = np.asarray(a_array, dtype="f8"), np.asarray(b_array, dtype="f8")
        if b_sorted:
            b_array = np.sort(b_array, axis=0, kind="stable")
        dimensions = 1 if a_array.ndim == 1 else a_array.shape[1]
        if upstream is None:
            upstream = np.array([(-1.0 if i % 2 else 1.0) * (i + .125) for i in range(dimensions)], dtype="f8")
        upstream = torch.as_tensor(upstream, dtype=torch.float64, device=device)

        def tensor(array, needs):
            value = torch.as_tensor(array.copy(), dtype=torch.float64, device=device)
            if noncontiguous and value.ndim == 2:
                value = value.T.contiguous().T
            return value.requires_grad_(needs)

        a0, b0, a1, b1 = tensor(a_array, need_a), tensor(b_array, need_b), tensor(a_array, need_a), tensor(b_array, need_b)
        old = Original(device).columns(a0, b0, b_sorted=b_sorted)
        new = UniformW1Cache(device).columns(a1, b1, b_sorted=b_sorted)
        require("forward cells must match original by bytes", bytes_equal(old, new))
        summaries["forward_bitwise_cases"] += 1
        if need_a or need_b:
            parents0 = [v for v in (a0, b0) if v.requires_grad]
            parents1 = [v for v in (a1, b1) if v.requires_grad]
            g0 = torch.autograd.grad(old, parents0, grad_outputs=upstream)
            g1 = torch.autograd.grad(new, parents1, grad_outputs=upstream)
            counts = [(len(a_array), len(b_array))] if need_a and not need_b else [(len(b_array), len(a_array))] if need_b and not need_a else [(len(a_array), len(b_array)), (len(b_array), len(a_array))]
            for before, after, (n, m) in zip(g0, g1, counts):
                # Original rounded cell masses, per-cell multiplication and a
                # reduction can differ from one exact rational division. This
                # bound scales by the maximum overlapping cells, not waveform
                # magnitudes, and is not a scientific acceptance threshold.
                limit = 16 * eps * (math.ceil(m / n) + 3) * upstream.abs() / n
                diff = (before - after).abs()
                if diff.ndim == 1:
                    diff = diff[:, None]
                require("original first derivative within roundoff bound", (diff <= limit[None, :]).all())
                require("zero upstream gives exactly zero derivative", (after[..., upstream == 0] == 0).all() if after.ndim == 2 else (after == 0).all() if bool((upstream == 0).all()) else True)
                summaries["gradient_sides_compared"] += 1
                summaries["max_gradient_absolute_difference"] = max(summaries["max_gradient_absolute_difference"], float(diff.max()))
                ratio = diff / limit.clamp_min(torch.finfo(torch.float64).tiny)[None, :]
                summaries["max_gradient_error_to_roundoff_bound"] = max(summaries["max_gradient_error_to_roundoff_bound"], float(ratio.max()))
            if rational:
                av, bv = a_array.tolist(), b_array.tolist()
                value, fa, fb = _fraction_cells(av, bv)
                require("Fraction interval/CDF a derivative identity", fa == _fraction_cdf(av, bv))
                require("Fraction interval/CDF b derivative identity", fb == _fraction_cdf(bv, av))
                summaries["rational_formula_cases"] += 1
                expected = [fa, fb]
                for actual, fractions in zip(g1, expected):
                    expected_tensor = torch.tensor([float(v) for v in fractions], device=device, dtype=torch.float64) * upstream[0]
                    require("analytical derivative equals exact Fraction division", bytes_equal(actual, expected_tensor))
                # Forward retains original arithmetic, so compare rational
                # value at a tight arithmetic bound rather than byte identity.
                require("finite cells equal rational W1", abs(float(new[0]) - float(value)) <= 8 * eps * max(1.0, abs(float(value))))
        else:
            require("no-grad output stays detached", not old.requires_grad and not new.requires_grad)
        summaries["comparison_cases"] += 1

    alphabet = (-2, 0, 1)
    for n in range(1, 4):
        for m in range(1, 4):
            for a in product(alphabet, repeat=n):
                for b in product(alphabet, repeat=m):
                    compare(a, b, [1.0], b_sorted=False, rational=True)
    checks.append("1521 exhaustive Fraction cases: unequal counts, stable duplicates, zero abs ties, both gradients")

    rng = np.random.Generator(np.random.PCG64(730001))
    configurations = ((1, 7, 2), (7, 1, 3), (7, 9, 4), (13, 8, 5),
                      (257, 251, 7), (101, 101, 3), (211, 4, 8), (6593, 21047, 2))
    for n, m, dimensions in configurations:
        a, b = rng.normal(size=(n, dimensions)), rng.normal(size=(m, dimensions))
        a[::3] = np.rint(a[::3])
        b[::4] = np.rint(b[::4])
        upstream = np.resize(np.array([-.75, 0, 2, 1e4, -1e-8, math.pi], dtype="f8"), dimensions)
        for sorted_b in (False, True):
            for need_a, need_b in ((True, True), (True, False), (False, True), (False, False)):
                compare(a, b, upstream, b_sorted=sorted_b, need_a=need_a, need_b=need_b, noncontiguous=True)
    checks.append("multiple columns, noncontiguous rows, arbitrary upstream, all requires_grad combinations, b_sorted both")
    checks.append("artificial 6593/21047 counts only; no scientific endpoints")

    base_a = np.array([[0., 1., -1.], [0., 2., 2.], [1., 2., 2.], [-2., 0., 2.]])
    base_b = np.array([[0., 2., 3.], [1., 2., -1.], [1., -1., 2.]])
    for shift, scale in ((0., 1.), (2.0**40, 2.0**10), (-2.0**40, 2.0**20), (0., 2.0**-40), (0., 2.0**32)):
        for sorted_b in (False, True):
            compare(shift + scale * base_a, shift + scale * base_b, [1., -2., .125], b_sorted=sorted_b)
    for sorted_b in (False, True):
        compare(base_a, base_a, [-2., 0., 1.], b_sorted=sorted_b)
        compare(np.array([-0., 0., -0., 0.]), np.array([0., -0., 0.]), [-1.], b_sorted=sorted_b)
    checks.append("large translations, large/small positive scales, identical laws, signed zero equality")

    # b_sorted is allowed to be a non-leaf produced by the caller's stable sort.
    for Cache in (Original, UniformW1Cache):
        a = torch.as_tensor(base_a.copy(), device=device, dtype=torch.float64).requires_grad_()
        b = torch.as_tensor(base_b.copy(), device=device, dtype=torch.float64).requires_grad_()
        out = Cache(device).columns(a, torch.sort(b, dim=0, stable=True).values, b_sorted=True)
        gradient = torch.autograd.grad(out, (a, b), grad_outputs=torch.tensor([-.125, 3., 0.], device=device, dtype=torch.float64))
        if Cache is Original:
            previous = gradient
        else:
            require("caller sorted nonleaf b chain", all(torch.allclose(x, y, atol=8*eps, rtol=8*eps) for x, y in zip(previous, gradient)))
    shared = torch.tensor([0., 0., -1., 2.], device=device, dtype=torch.float64).requires_grad_()
    out = UniformW1Cache(device).columns(shared, shared)
    require("same leaf on both sides zero", (torch.autograd.grad(out.sum(), shared)[0] == 0).all())
    checks.append("external sorted-b graph and shared-a-b leaf")

    old_cache, new_cache = Original(device, max_entries=2, max_cells=20), UniformW1Cache(device, max_entries=2, max_cells=20)
    for n, m in ((2, 3), (4, 5), (2, 3), (8, 9), (100, 101), (4, 5), (4, 5)):
        p0, p1 = old_cache._plan(n, m), new_cache._plan(n, m)
        require("original exact plan preserved", all(torch.equal(x, y) if x.dtype == torch.int64 else bytes_equal(x, y) for x, y in zip(p0, p1)))
        require("LRU receipt preserved", old_cache.receipt() == new_cache.receipt())
    checks.append("exact ia/ib/mass plans and LRU caps/hits/misses preserved")

    cases = [
        (torch.empty((0, 1), device=device, dtype=torch.float64), torch.ones((1, 1), device=device, dtype=torch.float64)),
        (torch.ones((2, 2), device=device, dtype=torch.float64), torch.ones((1, 3), device=device, dtype=torch.float64)),
        (torch.tensor([float('nan')], device=device, dtype=torch.float64), torch.ones(1, device=device, dtype=torch.float64)),
        (torch.ones(1, device=device, dtype=torch.float32), torch.ones(1, device=device, dtype=torch.float64)),
    ]
    for a, b in cases:
        errors = []
        for Cache in (Original, UniformW1Cache):
            try:
                Cache(device).columns(a, b)
            except Exception as error:
                errors.append(type(error).__name__)
        require("original validation failures retained", len(errors) == 2 and errors[0] == errors[1])
    checks.append("validation failure types match original")

    return {"pass_all": True, "evidence_level": "artificial_" + device.type + "_float64_first_order_only",
            "device": str(device), "original_source": str(original_source), "original_source_SHA256": source_sha,
            "checks": checks, "measurements": summaries,
            "forward_bitwise_expected": True, "gradient_bitwise_equivalence_claimed": False,
            "higher_order_derivatives_validated": False, "scientific_arrays_or_weights_read": False,
            "scientific_training_or_probe_run": False, "real_CUDA_speedup_claimed": False}
