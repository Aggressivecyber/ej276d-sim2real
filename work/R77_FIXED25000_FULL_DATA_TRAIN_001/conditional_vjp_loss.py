"""R74 laws with unique conditional feature writes and one domain-view VJP.

Only iter_gradient_groups' first pair changes its first-order implementation.
This module contains no I/O, random draws, fitting, or scientific state loading.
The existing R74 combined_loss/full_waveform_loss/exact_uniform_w1 are required.
"""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable

from combined_loss import FullWaveformLoss as R74FullWaveformLoss
from full_waveform_loss import TERM_NAMES, VIEWS, feature_vectors


def _domain_pair(v, bins, counts, directions, owner, projected_reference, need_gradient):
    """Keep original values; build each local graph only until its exact VJP.

    Current bins partition every row. Each full feature-gradient row is written
    once, avoiding one dense gather-backward allocation/add per occupied bin.
    The four global view gradients are sent to the canonical waveform together.
    """
    with torch.enable_grad():
        canonical = v.detach().requires_grad_(need_gradient)
        arrays = feature_vectors(canonical)
        full, coordinate = v.new_zeros(()), v.new_zeros(())
        view_gradients = []
        for name, features, direction in zip(VIEWS, arrays, directions):
            feature_full, feature_coordinate = v.new_zeros(()), v.new_zeros(())
            if need_gradient:
                feature_gradient = torch.empty_like(features)
            for slot, n in enumerate(counts):
                if not n:
                    continue
                use = bins == slot - 1
                scale = owner.stats[name + "_scale"][slot]
                with torch.no_grad():
                    # Exact old value expression; no gradient graph from this
                    # conditional gather/normalization back to global features.
                    a = ((features[use] - owner.stats[name + "_center"][slot]) / scale)
                a = a.detach().requires_grad_(need_gradient)
                marginal = owner.w1.columns(a, owner._reference_sorted[name][slot],
                                            b_sorted=True, check=False)
                key = (name, slot)
                if key not in projected_reference:
                    projected_reference[key] = torch.sort(
                        owner._reference_features[name][slot] @ direction,
                        dim=0, stable=True).values
                sliced = owner.w1.columns(a @ direction, projected_reference[key],
                                          b_sorted=True, check=False).mean()
                regional_full = .5 * (marginal.mean() + sliced)
                regional_coordinate = torch.linalg.vector_norm(marginal) / math.sqrt(features.shape[1])
                # Keep the same scalar accumulation order as R74, independently
                # of the backward coefficient and its rounding order.
                feature_full = feature_full + (n / len(v)) * regional_full.detach()
                feature_coordinate = feature_coordinate + (n / len(v)) * regional_coordinate.detach()
                if need_gradient:
                    coefficient = v.new_tensor(((n / len(v)) / len(VIEWS)) * .5)
                    local_gradient, = torch.autograd.grad(
                        regional_full + regional_coordinate, a, grad_outputs=coefficient)
                    with torch.no_grad():
                        feature_gradient[use] = local_gradient / scale
                    del local_gradient, coefficient
                del a, marginal, sliced, regional_full, regional_coordinate
            full = full + feature_full
            coordinate = coordinate + feature_coordinate
            if need_gradient:
                view_gradients.append(feature_gradient)
        if need_gradient:
            canonical_gradient, = torch.autograd.grad(arrays, canonical, grad_outputs=view_gradients)
            canonical_gradient = canonical_gradient.detach()
        else:
            canonical_gradient = None
        return full / len(VIEWS), coordinate / len(VIEWS), canonical_gradient


class _ConditionalPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, real, owner, bins, counts, directions):
        projected_reference = {}
        full, coordinate, gradients = [], [], []
        for i, v in enumerate((source, real)):
            f, c, g = _domain_pair(v, bins[i], counts[i], directions, owner,
                                    projected_reference, ctx.needs_input_grad[i])
            full.append(f)
            coordinate.append(c)
            gradients.append(g)
        # Original R74 domain averaging and final first-group scalar expression.
        full_report, coordinate_report = .5 * sum(full), .5 * sum(coordinate)
        combined = full_report + coordinate_report
        ctx.save_for_backward(*gradients)
        ctx.mark_non_differentiable(full_report, coordinate_report)
        return combined, full_report, coordinate_report

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream, unused_full, unused_coordinate):
        source_gradient, real_gradient = ctx.saved_tensors
        return (None if source_gradient is None else source_gradient * upstream,
                None if real_gradient is None else real_gradient * upstream,
                None, None, None, None)


class FullWaveformLoss(R74FullWaveformLoss):
    """Same constructor/state and eight gradient-group interface as R74.

    First group's reports are detached; combined scalar has the exact local
    first-order gradient. Other seven terms, validation, and the ordinary
    iter_terms/__call__ interface are inherited without changes. Higher-order
    differentiation of the optimized group is intentionally not supported.
    """
    def iter_gradient_groups(self, source_canonical, real_processed_canonical,
                             source_preADC, REAL_preADC, source_raw, REAL_raw,
                             directions4, temporal_directions):
        context = self._context(source_canonical, real_processed_canonical,
                                source_preADC, REAL_preADC, source_raw, REAL_raw,
                                directions4, temporal_directions)
        values, bins, counts, directions, _, diagnostics = context
        total, full, coordinate = _ConditionalPair.apply(values[0], values[1], self, bins, counts, directions)
        pair = {"full_window": full, "coordinate_RMS": coordinate}
        if not all(bool(torch.isfinite(v)) for v in pair.values()):
            raise FloatingPointError("nonfinite shared full/coordinate law")
        yield tuple(pair), total, pair, diagnostics
        for name in TERM_NAMES[2:]:
            value = self._compute_term(name, context)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("nonfinite " + name)
            yield (name,), value, {name: value}, diagnostics
