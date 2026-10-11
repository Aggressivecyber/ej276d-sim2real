"""Common charge-coordinate response; no data I/O or import-time random draws.

Physical coordinates, buffers, and the small MLP use float64 by default.
All rows have the same public path.
No domain, parent identity, event weight, loss, or training schedule enters it.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn

SAMPLES = 480
REGIONS = 8
CHARGE_LIMIT = 300.0


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains nonfinite values; no row deletion")


def _waveform(value: Tensor, name: str = "waveform") -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if value.ndim != 2 or value.shape[1] != SAMPLES or value.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (N, {SAMPLES})")
    value = value.to(dtype=torch.float64)
    _finite(value, name)
    return value


def window_charge(value: Tensor) -> Tensor:
    """Original Q functional on the last 480-sample axis, in float64."""
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError("charge input must be a floating-point Tensor")
    if value.ndim < 1 or value.shape[-1] != SAMPLES:
        raise ValueError("charge input must end in 480 samples")
    value = value.to(dtype=torch.float64)
    return value[..., 100:400].sum(-1) - 3.0 * value[..., :100].sum(-1)


def _positive_charge(value: Tensor, name: str, limit: float | None = None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional floating-point Tensor")
    value = value.to(dtype=torch.float64)
    _finite(value, name)
    if bool((value <= 0).any()) or (limit is not None and bool((value >= limit).any())):
        interval = f"(0, {limit})" if limit is not None else "(0, infinity)"
        raise ValueError(f"all {name} must lie in {interval}; no row deletion or charge clamp")
    return value


def bounded_charge_map(q_input: Tensor, a: Tensor, b: Tensor, *,
                       check_input: bool = True) -> Dict[str, Tensor]:
    """Q*=300 sigmoid(a+exp(b) logit(Q0/300)), including fixed-u slope.

    The logit is evaluated as log(Q0)-log(300-Q0).  Finite-precision
    saturation to a rail fails instead of changing the registered map.
    """
    if check_input:
        q_input = _positive_charge(q_input, "q_input", CHARGE_LIMIT)
    elif q_input.dtype != torch.float64 or q_input.ndim != 1:
        raise TypeError("prevalidated q_input must be a one-dimensional float64 Tensor")
    a, b = a.to(dtype=torch.float64), b.to(dtype=torch.float64)
    if a.shape != q_input.shape or b.shape != q_input.shape:
        raise ValueError("a, b, and q_input must have the same one-dimensional shape")
    if a.device != q_input.device or b.device != q_input.device:
        raise ValueError("charge map tensors must be on the same device")
    _finite(a, "a")
    _finite(b, "b")
    slope = torch.exp(b)
    _finite(slope, "logit-coordinate slope")
    if bool((slope <= 0).any()):
        raise FloatingPointError("logit-coordinate slope underflow; no replacement")
    logit = torch.log(q_input) - torch.log(CHARGE_LIMIT - q_input)
    coordinate = a + slope * logit
    _finite(coordinate, "target charge coordinate")
    q_target = CHARGE_LIMIT * torch.sigmoid(coordinate)
    q_target = _positive_charge(q_target, "q_target", CHARGE_LIMIT)
    physical_slope = slope * q_target * (CHARGE_LIMIT - q_target) / (
        q_input * (CHARGE_LIMIT - q_input)
    )
    _finite(physical_slope, "fixed-unit-waveform physical Q slope")
    if bool((physical_slope <= 0).any()):
        raise FloatingPointError("fixed-unit-waveform Q slope must be positive")
    return {
        "q_input": q_input, "q_target": q_target, "a": a, "b": b,
        "logit_input": logit, "logit_slope": slope,
        "fixed_unit_q_slope": physical_slope,
    }


class PublicConditionalChargeHead(nn.Module):
    """Frozen full-waveform standardization -> 480/16/2 SiLU charge head.

    Last layer is zero initialized; this is a mathematical identity at
    initialization, subject to measured floating-point roundoff.  Constructor
    initialization uses the caller's Torch RNG; importing this module does not.
    Feature statistics must come from the applicable authorized TRAIN fit.
    Keep physical/statistics buffers in float64 when moving this module.
    """
    def __init__(self, feature_center: Tensor, feature_scale: Tensor,
                 hidden: int = 16, charge_limit: float = CHARGE_LIMIT) -> None:
        super().__init__()
        if hidden != 16 or not math.isfinite(charge_limit) or charge_limit != CHARGE_LIMIT:
            raise ValueError("this candidate is fixed to hidden=16 and charge_limit=300")
        center = torch.as_tensor(feature_center, dtype=torch.float64).detach().clone()
        scale = torch.as_tensor(feature_scale, dtype=torch.float64).detach().clone()
        if center.shape != (SAMPLES,) or scale.shape != (SAMPLES,):
            raise ValueError("feature statistics must each have shape (480,)")
        if center.device != scale.device:
            raise ValueError("feature statistics must be on the same device")
        _finite(center, "feature_center")
        _finite(scale, "feature_scale")
        if bool((scale <= 0).any()):
            raise ValueError("feature_scale must be strictly positive; no scale floor")
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)
        self.hidden = nn.Linear(SAMPLES, hidden, device=center.device, dtype=torch.float64)
        self.output = nn.Linear(hidden, 2, device=center.device, dtype=torch.float64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def standardize_input(self, waveform: Tensor) -> Dict[str, Tensor]:
        """Check fixed parent input once; return public reusable coordinates.

        No detach is applied.  A caller caching these tensors must bind them to
        this same head's frozen statistics and the unchanged input waveform.
        """
        x = _waveform(waveform)
        if x.device != self.feature_center.device:
            raise ValueError("waveform and head must be on the same device")
        q_input = _positive_charge(window_charge(x), "q_input", CHARGE_LIMIT)
        baseline = x[:, :100].mean(-1, keepdim=True)
        unit = (x - baseline) / q_input[:, None]
        _finite(unit, "unit_waveform")
        features = (unit - self.feature_center) / self.feature_scale
        _finite(features, "standardized head input")
        return {"features": features, "q_input": q_input,
                "unit_waveform": unit, "baseline": baseline[:, 0]}

    def forward_features(self, features: Tensor, q_input: Tensor, *,
                         check_inputs: bool = True) -> Dict[str, Tensor]:
        """MLP/map for checked standardized inputs, including benchmark use.

        check_inputs=False is solely for unchanged coordinates previously
        returned by standardize_input.  Output rails/nonfinites/slopes always
        retain failure checks.  It does not bypass SOURCE/REAL processing.
        """
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating-point Tensor")
        if features.ndim != 2 or features.shape[1] != SAMPLES or features.shape[0] == 0:
            raise ValueError("features must have nonempty shape (N, 480)")
        if q_input.shape != (features.shape[0],):
            raise ValueError("one input charge per standardized waveform is required")
        if features.device != self.hidden.weight.device or q_input.device != features.device:
            raise ValueError("features, charge, and head must share a device")
        if check_inputs:
            _finite(features, "standardized head input")
        features = features.to(dtype=self.hidden.weight.dtype)
        hidden = torch.nn.functional.silu(self.hidden(features))
        outputs = self.output(hidden).to(dtype=torch.float64)
        return bounded_charge_map(q_input, outputs[:, 0], outputs[:, 1], check_input=check_inputs)

    def forward(self, waveform: Tensor) -> Dict[str, Tensor]:
        prepared = self.standardize_input(waveform)
        result = self.forward_features(prepared["features"], prepared["q_input"], check_inputs=False)
        result.update({"unit_waveform": prepared["unit_waveform"], "baseline": prepared["baseline"]})
        return result


def _edges(log_edges: Tensor, device: torch.device) -> Tensor:
    edges = torch.as_tensor(log_edges, dtype=torch.float64, device=device)
    if edges.shape != (REGIONS + 1,):
        raise ValueError("log_edges must have shape (9,)")
    _finite(edges, "log_edges")
    if bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("log_edges must be strictly increasing")
    return edges


def target_charge_condition(q_target: Tensor, log_edges: Tensor) -> Dict[str, Tensor]:
    """Original bin convention and differentiable within-bin logQ condition.

    Bin clipping chooses a conditional-statistics row only; raw out-of-support
    bins remain explicit.  Q and the continuous condition are never clipped.
    The final edge belongs to bin 7, matching noise_core.bins_for.
    """
    q_target = _positive_charge(q_target, "condition charge")
    edges = _edges(log_edges, q_target.device)
    logq = torch.log(1000.0 * q_target)
    _finite(logq, "logQ condition")
    raw = torch.searchsorted(edges, logq.contiguous(), right=True) - 1
    raw = torch.where(logq == edges[-1], torch.full_like(raw, REGIONS - 1), raw)
    index = raw.clamp(0, REGIONS - 1)
    lo, hi = edges[index], edges[index + 1]
    continuous = (logq - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    return {"bin_clipped": index, "bin_raw": raw,
            "logq": logq, "continuous_condition": continuous}


def normalized_template_bank(mu: Tensor) -> Tensor:
    """Baseline-center frozen TRAIN templates and normalize each Q to one."""
    mu = _waveform(mu, "template bank")
    if mu.shape[0] != REGIONS:
        raise ValueError("template bank must contain exactly eight rows")
    centered = mu - mu[:, :100].mean(-1, keepdim=True)
    q_mu = window_charge(centered)
    _finite(q_mu, "template charge")
    if bool((q_mu <= 0).any()):
        raise ValueError("each template must have strictly positive charge")
    answer = centered / q_mu[:, None]
    _finite(answer, "unit-charge template bank")
    return answer


def interpolate_charge_statistics(q_target: Tensor, mu: Tensor, scale: Tensor,
                                  log_edges: Tensor) -> Dict[str, Tensor]:
    """Continuous linear interpolation at original eight REAL logQ centers.

    Outside the first/last center, templates and scales extend constantly.
    The returned template is baseline centered with Q=1.  Do not interpolate
    old noise kappa2 constants with this function: they require separate proof.
    """
    q_target = _positive_charge(q_target, "interpolation charge")
    edges = _edges(log_edges, q_target.device)
    mu = normalized_template_bank(mu)
    scale = _waveform(scale, "scale bank")
    if scale.shape != (REGIONS, SAMPLES) or mu.device != q_target.device or scale.device != q_target.device:
        raise ValueError("eight-row statistics and charge must share a device")
    if bool((scale <= 0).any()):
        raise ValueError("scale bank must be strictly positive")
    centers = 0.5 * (edges[:-1] + edges[1:])
    logq = torch.log(1000.0 * q_target)
    _finite(logq, "interpolation logQ")
    insertion = torch.searchsorted(centers, logq.contiguous(), right=True)
    left = (insertion - 1).clamp(0, REGIONS - 1)
    right = insertion.clamp(0, REGIONS - 1)
    interior = left != right
    denominator = torch.where(interior, centers[right] - centers[left], torch.ones_like(logq))
    alpha = torch.where(interior, (logq - centers[left]) / denominator, torch.zeros_like(logq))
    weight = alpha[:, None]
    template = (1.0 - weight) * mu[left] + weight * mu[right]
    # This coordinate normalization retains its full autograd graph.  It only
    # removes floating-point error in an algebraically unit-charge convex sum.
    template = template - template[:, :100].mean(-1, keepdim=True)
    template = template / window_charge(template)[:, None]
    local_scale = (1.0 - weight) * scale[left] + weight * scale[right]
    _finite(template, "interpolated template")
    _finite(local_scale, "interpolated scale")
    return {"unit_template": template, "scale": local_scale,
            "left_center": left, "right_center": right, "alpha": alpha,
            "logq": logq, "centers_logq": centers}


def _row_template(template: Tensor, x: Tensor) -> Tensor:
    template = _waveform(template, "row template")
    if template.shape != x.shape or template.device != x.device:
        raise ValueError("one unit-charge template per waveform is required")
    # Check implementation coordinates, not a new scientific acceptance rule.
    if bool((torch.abs(window_charge(template) - 1.0) > 1e-10).any()):
        raise ValueError("row template is not normalized to unit charge")
    return template


def charge_coordinate_lift(waveform: Tensor, q_target: Tensor, unit_template: Tensor) -> Tensor:
    """x*=x+(Q*-Q0)m(Q*); no ADC clipping and no tensor detachment."""
    x = _waveform(waveform)
    q_target = _positive_charge(q_target, "lifting charge", CHARGE_LIMIT)
    if q_target.shape != (x.shape[0],) or q_target.device != x.device:
        raise ValueError("one target charge per waveform on the same device is required")
    template = _row_template(unit_template, x)
    result = x + (q_target - window_charge(x))[:, None] * template
    _finite(result, "lifted waveform")
    return result


def zero_charge_project(value: Tensor, unit_template: Tensor) -> Tensor:
    """Original zero-Q projection using the current continuous Q* template."""
    value = _waveform(value, "zero-charge projection input")
    template = _row_template(unit_template, value)
    return value - window_charge(value)[:, None] * template


def common_charge_coordinates(head: PublicConditionalChargeHead, waveform: Tensor,
                              mu: Tensor, scale: Tensor,
                              log_edges: Tensor) -> Dict[str, Tensor]:
    """Shared SOURCE/REAL coordinate path; caller adds bound noise/Box/ADC.

    Retain parent IDs/weights outside this numeric API.  Record pre-ADC/ADC
    bins after those stages using target_charge_condition, rather than reusing
    input bins or pretending charge has stayed at the original parent value.
    """
    result = head(waveform)
    original = target_charge_condition(result["q_input"], log_edges)
    target = target_charge_condition(result["q_target"], log_edges)
    stats = interpolate_charge_statistics(result["q_target"], mu, scale, log_edges)
    lifted = charge_coordinate_lift(waveform, result["q_target"], stats["unit_template"])
    result.update({"lifted_waveform": lifted, "unit_template": stats["unit_template"],
                   "scale": stats["scale"], "interpolation_alpha": stats["alpha"],
                   "left_center": stats["left_center"], "right_center": stats["right_center"],
                   "input_bin_raw": original["bin_raw"], "input_bin_clipped": original["bin_clipped"],
                   "target_bin_raw": target["bin_raw"], "target_bin_clipped": target["bin_clipped"],
                   "target_logq": target["logq"], "continuous_condition": target["continuous_condition"]})
    return result
