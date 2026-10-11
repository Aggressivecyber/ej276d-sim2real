"""Full finite-population charge losses; float64 values, uniform parent mass.

No binning, sampling, weighting changes, or fitted scientific constants.
At nonsmooth sort/absolute-value ties, ordinary stable-sort PyTorch autograd
provides a subgradient. Original-input charge ties impose no internal order.
"""
from __future__ import annotations

import math
import torch


def _positive_vector(value: torch.Tensor, name: str, *, device=None, size=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(name + " must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(name + " must be float64; no silent cast")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(name + " must be a nonempty one-dimensional vector")
    if device is not None and value.device != device:
        raise ValueError(name + " is on the wrong device; no silent transfer")
    if size is not None and value.numel() != size:
        raise ValueError(name + " parent count changed")
    if not bool(torch.isfinite(value).all() and (value > 0).all()):
        raise ValueError(name + " must be finite and strictly positive")
    return value


def _uniform_quantile_plan(n: int, m: int, device):
    """Merge exact integer uniform-CDF endpoints; cache quantile cells.

    Floating-point masses are conversions of exact rational cell lengths.
    The plan depends only on population sizes, never on prediction values.
    """
    denominator = math.lcm(n, m)
    step_s, step_r = denominator // n, denominator // m
    edges = sorted(set(range(0, denominator + 1, step_s)).union(
        range(0, denominator + 1, step_r)))
    left, right = edges[:-1], edges[1:]
    source_index = [a // step_s for a in left]
    real_index = [a // step_r for a in left]
    mass = [(b - a) / denominator for a, b in zip(left, right)]
    return (
        torch.tensor(source_index, dtype=torch.int64, device=device),
        torch.tensor(real_index, dtype=torch.int64, device=device),
        torch.tensor(mass, dtype=torch.float64, device=device),
    )


class FullChargeObjective:
    """Three equally weighted terms on the original full parent populations.

    Init references are detached/cloned constants. Call inputs must keep their
    counts, float64 dtype and device. q_real_input is the frozen original REAL
    input charge for the paired identity term, not q_real_prediction.

    Returns (total, terms), where terms keys are:
      charge_distribution, REAL_identity, source_charge_order.
    Exposes scalar float64 std_Q/std_logQ and source_count/real_count.
    """

    def __init__(self, q_source_input, q_real_canonical):
        _positive_vector(q_source_input, "q_source_input")
        self.device = q_source_input.device
        _positive_vector(q_real_canonical, "q_real_canonical",
                         device=self.device)
        self.source_count = q_source_input.numel()
        self.real_count = q_real_canonical.numel()
        source = q_source_input.detach().clone()
        reference = q_real_canonical.detach().clone()
        log_reference = reference.log()
        self.std_Q = reference.std(unbiased=False)
        self.std_logQ = log_reference.std(unbiased=False)
        for name, value in (("std_Q", self.std_Q),
                            ("std_logQ", self.std_logQ)):
            if not bool(torch.isfinite(value) and (value > 0)):
                raise ValueError(name + " must be finite and positive; no floor")

        self._reference_sorted_Q = torch.sort(reference, stable=True).values
        self._reference_sorted_logQ = torch.sort(log_reference,
                                                stable=True).values
        (self._source_quantile_index, self._real_quantile_index,
         self._quantile_mass) = _uniform_quantile_plan(
             self.source_count, self.real_count, self.device)

        original, self._source_order = torch.sort(source, stable=True)
        new_group = torch.ones(self.source_count, dtype=torch.bool,
                               device=self.device)
        new_group[1:] = original[1:] != original[:-1]
        starts = torch.nonzero(new_group, as_tuple=False).flatten()
        ends = torch.cat((starts[1:], starts.new_tensor([self.source_count])))
        self._tie_starts, self._tie_ends = starts, ends
        self._tie_counts = (ends - starts).to(torch.float64)
        self._tie_group = new_group.to(torch.int64).cumsum(0) - 1

    def _w1(self, prediction, reference_sorted):
        sorted_prediction = torch.sort(prediction, stable=True).values
        difference = (
            sorted_prediction[self._source_quantile_index]
            - reference_sorted[self._real_quantile_index]
        ).abs()
        return (difference * self._quantile_mass).sum()

    def _group_means(self, values):
        prefix = torch.cat((values.new_zeros(1), values.cumsum(0)))
        return ((prefix[self._tie_ends] - prefix[self._tie_starts])
                / self._tie_counts)

    def _source_charge_order(self, log_prediction):
        ordered = log_prediction[self._source_order]
        sorted_values, value_order = torch.sort(ordered, stable=True)

        # Sort inside each ORIGINAL-charge tie group before summation. This
        # leaves each group mean unchanged, but makes its numerical result
        # independent of artificial order inside a tied input group.
        grouped_order = torch.argsort(self._tie_group[value_order],
                                      stable=True)
        tie_invariant_ordered = sorted_values[grouped_order]
        a = self._group_means(tie_invariant_ordered)
        b = self._group_means(sorted_values)
        return ((self._tie_counts * (a - b).abs()).sum()
                / self.source_count / self.std_logQ)

    def __call__(self, q_source_prediction, q_real_prediction, q_real_input):
        _positive_vector(q_source_prediction, "q_source_prediction",
                         device=self.device, size=self.source_count)
        _positive_vector(q_real_prediction, "q_real_prediction",
                         device=self.device, size=self.real_count)
        _positive_vector(q_real_input, "q_real_input",
                         device=self.device, size=self.real_count)

        log_source = q_source_prediction.log()
        q_w1 = self._w1(q_source_prediction, self._reference_sorted_Q)
        log_w1 = self._w1(log_source, self._reference_sorted_logQ)
        distribution = .5 * (q_w1 / self.std_Q + log_w1 / self.std_logQ)
        identity = ((q_real_prediction.log()
                     - q_real_input.detach().log()).square().mean()
                    / self.std_logQ.square())
        order = self._source_charge_order(log_source)
        terms = {
            "charge_distribution": distribution,
            "REAL_identity": identity,
            "source_charge_order": order,
        }
        total = distribution + identity + order
        if not bool(torch.isfinite(torch.stack(tuple(terms.values()))).all()
                    and torch.isfinite(total)):
            raise FloatingPointError("nonfinite charge objective; no masking")
        return total, terms
