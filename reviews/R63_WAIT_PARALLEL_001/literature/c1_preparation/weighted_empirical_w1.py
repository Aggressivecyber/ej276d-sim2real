"""Exact finite 1-D weighted empirical W1 with a documented first-order branch.

This is standalone artificial-data code preparation, NOT the R63 objective.
Both laws require their own explicit frozen, nonnegative, unit-total masses.
No common SOURCE/REAL region mass is inferred.

The quantile partition is computed with Python integer arithmetic from the
exact binary64 mass ratios; endpoint distances and the final sum are binary64.
Thus "exact" means no quantile-grid, truncation, entropic, or pairing
approximation. It does not mean arbitrary-precision endpoint arithmetic.
Stable input order chooses the within-law tie branch; abs(0) has derivative 0.
This branch is deterministic, not a unique derivative at nondifferentiability.
CPU only. First-order endpoint/parameter differentiation only; mass derivatives
and higher derivatives are deliberately outside the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

MASS_SUM_ATOL = 8.0 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class FrozenMass:
    """An immutable snapshot of one explicitly normalized finite law.

    A binary64 unit-sum vector can differ from exactly one by rounding.
    After the unit-sum check, its exact integer ratios are normalized by their
    exact total. This final normalization is explicit in to_dict().
    Use separately constructed SOURCE and REAL masses for global mixtures.
    """
    label: str
    ticks: tuple[int, ...]
    denominator: int
    input_float64: tuple[float, ...]
    input_sum_float64: float

    def __post_init__(self):
        if not self.label or not self.ticks or any(type(t) is not int or t < 0 for t in self.ticks):
            raise ValueError("FrozenMass requires a label and nonnegative integer ticks.")
        if type(self.denominator) is not int or self.denominator <= 0 or sum(self.ticks) != self.denominator:
            raise ValueError("FrozenMass integer masses must sum to the positive denominator.")
        if len(self.input_float64) != len(self.ticks) or not math.isfinite(self.input_sum_float64):
            raise ValueError("FrozenMass input metadata are inconsistent.")
        if abs(self.input_sum_float64 - 1.0) > MASS_SUM_ATOL:
            raise ValueError("FrozenMass input was not explicitly normalized.")

    @classmethod
    def from_normalized(
        cls, values: Sequence[float], *, label: str
    ) -> "FrozenMass":
        if getattr(values, "requires_grad", False):
            raise ValueError("Masses must be frozen; requires_grad=True is rejected.")
        if not isinstance(label, str) or not label:
            raise ValueError("A nonempty mass label is required.")
        if np.iscomplexobj(values):
            raise ValueError("Complex masses are not supported.")
        a = np.array(values, dtype=np.float64, copy=True)
        if a.ndim != 1 or not a.size:
            raise ValueError("Masses must be a nonempty one-dimensional vector.")
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Masses must be finite and nonnegative.")
        try:
            total = math.fsum(float(z) for z in a)
        except OverflowError as exc:
            raise ValueError("Mass total is not finite.") from exc
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Mass total must be finite and positive.")
        if abs(total - 1.0) > MASS_SUM_ATOL:
            raise ValueError(
                f"Masses must be explicitly normalized: sum={total!r}; "
                f"absolute tolerance={MASS_SUM_ATOL!r}."
            )
        ratios = [float(z).as_integer_ratio() for z in a]
        # Binary64 denominators are powers of two, including the denominator 1.
        common = max(q for _, q in ratios)
        ticks = tuple(p * (common // q) for p, q in ratios)
        denominator = sum(ticks)
        if denominator <= 0:
            raise ValueError("Mass total is zero after exact conversion.")
        divisor = math.gcd(*ticks)
        if divisor > 1:
            ticks = tuple(t // divisor for t in ticks)
            denominator //= divisor
        return cls(label, ticks, denominator, tuple(map(float, a)), total)

    def __len__(self) -> int:
        return len(self.ticks)

    def as_float64(self) -> np.ndarray:
        result = np.array(
            [t / self.denominator for t in self.ticks], dtype=np.float64
        )
        if any(t > 0 and z == 0.0 for t, z in zip(self.ticks, result)):
            raise FloatingPointError("A positive normalized mass underflows binary64.")
        return result

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "count": len(self),
            "input_float64": list(self.input_float64),
            "input_sum_float64": self.input_sum_float64,
            "unit_sum_check_absolute_tolerance": MASS_SUM_ATOL,
            "normalized_float64": self.as_float64().tolist(),
            "exact_normalization": "integer ticks / sum(integer ticks)",
            "integer_ticks_decimal": [str(t) for t in self.ticks],
            "integer_denominator_decimal": str(self.denominator),
            "zero_mass_atoms": sum(t == 0 for t in self.ticks),
            "frozen": True,
        }


@dataclass(frozen=True)
class QuantilePlan:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    segment_numerators: tuple[int, ...]
    denominator: int
    source_stable_sort: tuple[int, ...]
    target_stable_sort: tuple[int, ...]
    source_tie_groups: tuple[tuple[int, ...], ...]
    target_tie_groups: tuple[tuple[int, ...], ...]

    def segment_float64(self) -> np.ndarray:
        mass = np.array(
            [n / self.denominator for n in self.segment_numerators],
            dtype=np.float64,
        )
        if any(n > 0 and z == 0.0 for n, z in zip(self.segment_numerators, mass)):
            raise FloatingPointError("A positive quantile segment underflows binary64.")
        return mass

    def to_dict(self) -> dict:
        return {
            "segments": len(self.segment_numerators),
            "source_indices": list(self.source_indices),
            "target_indices": list(self.target_indices),
            "segment_mass_float64": self.segment_float64().tolist(),
            "exact_segment_numerators_decimal": [
                str(n) for n in self.segment_numerators
            ],
            "exact_denominator_decimal": str(self.denominator),
            "source_stable_sort": list(self.source_stable_sort),
            "target_stable_sort": list(self.target_stable_sort),
            "source_equal_value_groups": [list(g) for g in self.source_tie_groups],
            "target_equal_value_groups": [list(g) for g in self.target_tie_groups],
            "positive_segments_sum_exactly_to_one":
                sum(self.segment_numerators) == self.denominator,
            "zero_length_segments_not_materialized": True,
            "tie_policy": "stable input order; abs derivative is zero at equal endpoints",
            "unique_derivative_claimed": False,
        }


def _coordinates(values, name: str) -> np.ndarray:
    if getattr(values, "requires_grad", False):
        raise ValueError(f"{name}: use weighted_w1_torch for differentiable tensors.")
    if np.iscomplexobj(values):
        raise ValueError(f"{name}: complex coordinates are not supported.")
    a = np.array(values, dtype=np.float64, copy=True)
    if a.ndim != 1 or not a.size:
        raise ValueError(f"{name}: expected a nonempty one-dimensional vector.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: coordinates must be finite.")
    return a


def _tie_groups(values: np.ndarray, order: np.ndarray) -> tuple[tuple[int, ...], ...]:
    groups = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        if end - start > 1:
            groups.append(tuple(int(i) for i in order[start:end]))
        start = end
    return tuple(groups)


def quantile_plan(
    x, source_mass: FrozenMass, y, target_mass: FrozenMass
) -> QuantilePlan:
    """Build the full weighted quantile integral, including all positive mass.

    Stable-sort equal coordinates without merging distinct atoms. Mass-zero
    atoms remain in sort diagnostics and have exactly zero endpoint gradient.
    Quantile CDF breakpoints are exact integers on the common product
    denominator. Taking the next CDF edge avoids float midpoint rounding.
    """
    if not isinstance(source_mass, FrozenMass) or not isinstance(target_mass, FrozenMass):
        raise TypeError("Pass two explicit FrozenMass objects.")
    xx, yy = _coordinates(x, "SOURCE"), _coordinates(y, "REAL")
    if len(xx) != len(source_mass) or len(yy) != len(target_mass):
        raise ValueError("Coordinate and mass lengths must match within each law.")
    order_x = np.argsort(xx, kind="stable")
    order_y = np.argsort(yy, kind="stable")
    ids_x = [int(i) for i in order_x if source_mass.ticks[int(i)] > 0]
    ids_y = [int(i) for i in order_y if target_mass.ticks[int(i)] > 0]
    denom = source_mass.denominator * target_mass.denominator
    i = j = 0
    edge_x = source_mass.ticks[ids_x[0]] * target_mass.denominator
    edge_y = target_mass.ticks[ids_y[0]] * source_mass.denominator
    edge = 0
    sources, targets, numerators = [], [], []
    while i < len(ids_x) and j < len(ids_y):
        next_edge = min(edge_x, edge_y)
        length = next_edge - edge
        if length <= 0:
            raise ArithmeticError("A positive-mass CDF segment did not advance.")
        sources.append(ids_x[i])
        targets.append(ids_y[j])
        numerators.append(length)
        edge = next_edge
        advance_x = edge_x == next_edge
        advance_y = edge_y == next_edge
        if advance_x:
            i += 1
            if i < len(ids_x):
                edge_x += source_mass.ticks[ids_x[i]] * target_mass.denominator
        if advance_y:
            j += 1
            if j < len(ids_y):
                edge_y += target_mass.ticks[ids_y[j]] * source_mass.denominator
    if edge != denom or i != len(ids_x) or j != len(ids_y):
        raise ArithmeticError("Quantile plan does not cover both unit laws.")
    result = QuantilePlan(
        tuple(sources), tuple(targets), tuple(numerators), denom,
        tuple(map(int, order_x)), tuple(map(int, order_y)),
        _tie_groups(xx, order_x), _tie_groups(yy, order_y),
    )
    # Also fail explicitly if converting a positive segment cannot be represented.
    result.segment_float64()
    return result


def weighted_w1_numpy(
    x, source_mass: FrozenMass, y, target_mass: FrozenMass
) -> dict:
    """Return binary64 W1, selected branch gradients for both endpoints, and plan."""
    xx, yy = _coordinates(x, "SOURCE"), _coordinates(y, "REAL")
    plan = quantile_plan(xx, source_mass, yy, target_mass)
    ix = np.asarray(plan.source_indices, dtype=np.int64)
    iy = np.asarray(plan.target_indices, dtype=np.int64)
    mass = plan.segment_float64()
    delta = xx[ix] - yy[iy]
    if not np.isfinite(delta).all():
        raise FloatingPointError("Endpoint differences overflow binary64.")
    value = math.fsum(float(w) * abs(float(d)) for w, d in zip(mass, delta))
    if not math.isfinite(value):
        raise FloatingPointError("W1 overflows binary64.")
    signed_mass = mass * np.sign(delta)  # np.sign(0) = 0.
    gx = np.zeros_like(xx)
    gy = np.zeros_like(yy)
    np.add.at(gx, ix, signed_mass)
    np.add.at(gy, iy, -signed_mass)
    return {"value": value, "source_gradient": gx, "target_gradient": gy, "plan": plan}


def weighted_w1_torch(
    x, source_mass: FrozenMass, y, target_mass: FrozenMass
):
    """A first-order CPU torch scalar with both endpoint graphs retained.

    Sorting/quantile segments are chosen from detached coordinates and frozen
    masses. The differentiable expression still uses BOTH x and y, so shared
    parameters receive J_x^T g_x + J_y^T g_y. It intentionally has no derivative
    through the chosen mass partition. No model or data is loaded.
    """
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch tensors.")
    for name, v in (("SOURCE", x), ("REAL", y)):
        if v.device.type != "cpu" or v.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name}: CPU float32/float64 endpoints required.")
        if v.ndim != 1 or not v.numel() or not bool(torch.isfinite(v).all()):
            raise ValueError(f"{name}: finite nonempty 1-D endpoints required.")
    plan = quantile_plan(
        x.detach().to(torch.float64).numpy(), source_mass,
        y.detach().to(torch.float64).numpy(), target_mass,
    )
    ix = torch.tensor(plan.source_indices, dtype=torch.int64, device="cpu")
    iy = torch.tensor(plan.target_indices, dtype=torch.int64, device="cpu")
    mass = torch.tensor(plan.segment_float64(), dtype=torch.float64, device="cpu")
    delta = x.to(torch.float64)[ix] - y.to(torch.float64)[iy]
    value = (mass * delta.abs()).sum()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("W1 overflows binary64.")
    return value
