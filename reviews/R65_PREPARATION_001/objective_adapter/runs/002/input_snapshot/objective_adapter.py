"""R65 thin objective calls; no data I/O, model construction, RNG, or tests.

The caller supplies already constructed, hash-bound original objective instances.
D returns detached values and a detached first-order endpoint derivative.
C keeps the original four selected pieces and both out/raw graph paths alive.
"""
from __future__ import annotations

import torch

D_NAMES = ("full_window", "PSD", "noise", "coordinate_RMS", "temporal_joint")
D_PIECE_ORDER = ("full_window", "coordinate_RMS", "PSD", "noise", "temporal_joint")
C_NAMES = ("source_order", "source_valley", "REAL_identity", "box_correction")


def joint_total_D(objective, source_endpoint, reference, layout):
    """Call all five D terms once on one projected-before-ADC endpoint leaf.

    Returns dict(total, pieces, GY, diagnostics, piece_order), all value/gradient
    tensors detached. GY already includes canonical STE and original masses,
    weights and reductions; inject it into the original projected out once.
    It must not be injected into raw or divided/averaged again.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("joint_total_D requires ordinary autograd, not inference_mode")
    with torch.enable_grad():
        leaf = source_endpoint.detach().clone().requires_grad_(True)
        result = objective(leaf, reference, layout, terms=D_NAMES)
        if tuple(result.pieces) != D_PIECE_ORDER:
            raise RuntimeError("Bound joint D piece/reduction order changed")
        if result.total.ndim != 0:
            raise ValueError("Bound joint D total must be scalar")
        gradient, = torch.autograd.grad(result.total, leaf, create_graph=False, retain_graph=False)
    return {
        "total": result.total.detach(),
        "pieces": {name: result.pieces[name].detach() for name in D_PIECE_ORDER},
        "GY": gradient.detach(),
        "diagnostics": result.diagnostics,
        "piece_order": D_PIECE_ORDER,
    }


def original_C_microbatch(
    objective, output, raw, reference, identity_input, identity_bins,
    source_psd, source_h, directions, source_interval_low,
    source_interval_high, temporal_directions,
):
    """Call original nine-term loss once; directly add its four original C terms.

    No C formula is reimplemented. Five microbatch D pieces are evaluated but
    returned only as detached discarded values. The caller differentiates total
    through the original shared output/raw network graph before the joint update.
    """
    nine_total, all_pieces, diagnostics = objective(
        output, raw, reference, identity_input, identity_bins, source_psd,
        source_h, directions, source_interval_low, source_interval_high,
        temporal_directions,
    )
    if set(all_pieces) != set(D_NAMES + C_NAMES):
        raise RuntimeError("Bound original nine-term piece set changed")
    if any(objective.cfg["loss_weights"][name] != 1.0 for name in C_NAMES):
        raise ValueError("Original R65 C weights must all remain one")
    pieces = {name: all_pieces[name] for name in C_NAMES}
    total = sum(objective.cfg["loss_weights"][name] * pieces[name] for name in C_NAMES)
    return {
        "total": total,
        "pieces": pieces,
        "discarded_D_values": {name: all_pieces[name].detach() for name in D_NAMES},
        "original_nine_total_value": nine_total.detach(),
        "diagnostics": diagnostics,
    }

