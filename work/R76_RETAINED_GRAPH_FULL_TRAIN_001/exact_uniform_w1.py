"""Exact original finite-cell W1 forward; analytical first-order backward.

Drop-in UniformW1Cache for float64 endpoints. No import-time I/O, RNG,
model/data loading, new library, approximation grid, or custom compilation.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import torch


def _tensor(value, name, device):
    if not isinstance(value, torch.Tensor) or value.device != device:
        raise TypeError(name + " must be a tensor on the common device")
    if value.dtype != torch.float64:
        raise TypeError(name + " must be float64")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(name + " is nonfinite; no row deletion")
    return value


def _sorted_gradient(values, other):
    """One derivative per sorted occurrence, including separate stable ties.

    A_i=[i/n,(i+1)/n]. Positive length is A_i intersect [0,F_other(x-));
    negative length is A_i intersect [F_other(x),1]. Equality contributes zero.
    Use exact int64 interval lengths in the original LCM denominator when safe.
    """
    n, m = len(values), len(other)
    boundary = other.transpose(0, 1).contiguous()
    query = values.transpose(0, 1).contiguous()
    left = torch.searchsorted(boundary, query, right=False)
    right = torch.searchsorted(boundary, query, right=True)
    denominator = math.lcm(n, m)
    if denominator <= torch.iinfo(torch.int64).max:
        step, other_step = denominator // n, denominator // m
        lo = torch.arange(n, device=values.device, dtype=torch.int64) * step
        hi = lo + step
        positive = (torch.minimum(left * other_step, hi) - lo).clamp_min(0)
        negative = (hi - torch.maximum(right * other_step, lo)).clamp_min(0)
        answer = (positive - negative).to(torch.float64) / denominator
    else:
        # Preserve the formula for hypothetical counts exceeding int64 LCM.
        # The registered project counts use the integer path exclusively.
        lo = torch.arange(n, device=values.device, dtype=torch.float64) / n
        hi = (torch.arange(n, device=values.device, dtype=torch.float64) + 1) / n
        positive = (torch.minimum(left.to(torch.float64) / m, hi) - lo).clamp_min(0)
        negative = (hi - torch.maximum(right.to(torch.float64) / m, lo)).clamp_min(0)
        answer = positive - negative
    return answer.transpose(0, 1)


class _ExactCellsAnalytical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, ia, ib, mass, b_sorted):
        sa, order_a = torch.sort(a, dim=0, stable=True)
        if b_sorted:
            sb, order_b = b, None
        else:
            sb, order_b = torch.sort(b, dim=0, stable=True)
        # Exact original expression, including cell order and float64 reduction.
        result = ((sa[ia] - sb[ib]).abs() * mass[:, None]).sum(0)
        ctx.b_sorted = b_sorted
        ctx.a_dtype, ctx.b_dtype = a.dtype, b.dtype
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(sa, sb,
                order_a if ctx.needs_input_grad[0] else None,
                order_b if ctx.needs_input_grad[1] and not b_sorted else None)
        return result

    @staticmethod
    def backward(ctx, upstream):
        sa, sb, order_a, order_b = ctx.saved_tensors
        ga = gb = None
        if ctx.needs_input_grad[0]:
            sorted_ga = _sorted_gradient(sa, sb) * upstream[None, :]
            ga = torch.empty_like(sa)
            # Each column's sort indices are a permutation: no repeated writes,
            # no scatter_add, and no floating atomic summation.
            ga.scatter_(0, order_a, sorted_ga.to(ctx.a_dtype))
        if ctx.needs_input_grad[1]:
            sorted_gb = _sorted_gradient(sb, sa) * upstream[None, :]
            if ctx.b_sorted:
                gb = sorted_gb.to(ctx.b_dtype)
            else:
                gb = torch.empty_like(sb)
                gb.scatter_(0, order_b, sorted_gb.to(ctx.b_dtype))
        return ga, gb, None, None, None, None


class UniformW1Cache:
    """Original bounded device-local LRU and exact finite uniform cells.

    b_sorted=True has the same caller contract as the original: b is already
    sorted in each column, and its gradient is returned in that supplied order.
    This implementation is validated for first-order float64 training.
    """
    def __init__(self, device, max_entries=32, max_cells=262144):
        if max_entries < 1 or max_cells < 1:
            raise ValueError("cache caps must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_entries, self.max_cells = int(max_entries), int(max_cells)
        self._cache = OrderedDict()
        self.cached_cells = self.hits = self.misses = 0

    def _plan(self, n, m):
        if n <= 0 or m <= 0:
            raise ValueError("W1 requires both nonempty finite laws")
        key = (int(n), int(m))
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        denominator = math.lcm(n, m)
        step_a, step_b = denominator // n, denominator // m
        edges = sorted(set(range(0, denominator + 1, step_a)).union(
                       range(0, denominator + 1, step_b)))
        left, right = edges[:-1], edges[1:]
        plan = (torch.tensor([x // step_a for x in left], device=self.device,
                             dtype=torch.int64),
                torch.tensor([x // step_b for x in left], device=self.device,
                             dtype=torch.int64),
                torch.tensor([(b - a) / denominator for a, b in zip(left, right)],
                             device=self.device, dtype=torch.float64))
        cells = len(left)
        if cells <= self.max_cells:
            while self._cache and (len(self._cache) >= self.max_entries or
                                    self.cached_cells + cells > self.max_cells):
                _, previous = self._cache.popitem(last=False)
                self.cached_cells -= previous[2].numel()
            self._cache[key] = plan
            self.cached_cells += cells
        return plan

    def columns(self, a, b, *, b_sorted=False, check=True):
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("W1 feature dimensions differ")
        if check:
            _tensor(a, "W1 candidate", self.device)
            _tensor(b, "W1 reference", self.device)
        ia, ib, mass = self._plan(len(a), len(b))
        return _ExactCellsAnalytical.apply(a, b, ia, ib, mass, bool(b_sorted))

    def receipt(self):
        return {"entries": len(self._cache), "cells": self.cached_cells,
                "max_entries": self.max_entries, "max_cells": self.max_cells,
                "hits": self.hits, "misses": self.misses}
