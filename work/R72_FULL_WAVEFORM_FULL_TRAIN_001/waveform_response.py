"""One shared frozen-Q/full-waveform response, with explicit epsilon input.

Dependencies are the sealed R71 public_charge_head, common_response, and
physical_operators modules.  No file I/O, RNG draw, or model creation at import.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from public_charge_head import common_charge_coordinates, window_charge, target_charge_condition
from common_response import projected_noise_variance
from physical_operators import bounded_charge, canonical_adc_torch

ROW_KEYS = (
    "features482", "lifted", "mu_condition", "scale_condition", "residual_scale_V",
    "Q0", "Qstar", "original_bin_raw", "original_bin_clipped", "target_bin_raw",
    "target_bin_clipped", "continuous_condition", "fixed_unit_Q_slope",
    "lifted_baseline_variance", "noise_amplitude_squared", "noise_amplitude", "noise_kappa2",
)
SHARED_KEYS = ("edges_logQ", "noise_sqrt_p480")


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}; no event deletion")


class ShapeResidualNetwork(nn.Module):
    """Frozen 482-d standardization + float32 482/128/128/480 SiLU MLP."""
    def __init__(self, feature_center482: Tensor, feature_scale482: Tensor,
                 hidden: int = 128) -> None:
        super().__init__()
        if hidden != 128:
            raise ValueError("only the fixed hidden=128 candidate is implemented")
        center = torch.as_tensor(feature_center482, dtype=torch.float64).detach().clone()
        scale = torch.as_tensor(feature_scale482, dtype=torch.float64).detach().clone()
        if center.shape != (482,) or scale.shape != (482,) or center.device != scale.device:
            raise ValueError("frozen feature statistics must have matching shape (482,) and device")
        _finite(center, "feature center")
        _finite(scale, "feature scale")
        if bool((scale < 1e-6).any()):
            raise ValueError("root must pass TRAIN population std already floored at 1e-6")
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)
        self.hidden1 = nn.Linear(482, hidden, device=center.device, dtype=torch.float32)
        self.hidden2 = nn.Linear(hidden, hidden, device=center.device, dtype=torch.float32)
        self.output = nn.Linear(hidden, 480, device=center.device, dtype=torch.float32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def standardize_input(self, features482: Tensor) -> Tensor:
        if not isinstance(features482, Tensor) or features482.ndim != 2 or features482.shape[1] != 482:
            raise ValueError("network input must have shape (N,482)")
        if features482.shape[0] == 0 or features482.device != self.feature_center.device:
            raise ValueError("network input must be nonempty on the network device")
        if self.feature_center.dtype != torch.float64 or self.feature_scale.dtype != torch.float64:
            raise TypeError("physical feature-standardization buffers must remain float64")
        features = (features482.to(torch.float64) - self.feature_center) / self.feature_scale
        features = features.to(torch.float32)
        _finite(features, "standardized waveform input")
        return features

    def forward_standardized(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 482 or features.dtype != torch.float32:
            raise TypeError("standardized input must be (N,482) float32")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("the fixed waveform MLP parameters must remain float32")
        h = torch.nn.functional.silu(self.hidden1(features))
        h = torch.nn.functional.silu(self.hidden2(h))
        answer = self.output(h)
        _finite(answer, "shape residual prediction")
        return answer

    def forward(self, features482: Tensor) -> Tensor:
        return self.forward_standardized(self.standardize_input(features482))


@torch.no_grad()
def prepare_static(head: nn.Module, x: Tensor, mu: Tensor, scale: Tensor,
                   edges: Tensor, p480: Tensor, tau2: float) -> dict[str, Tensor]:
    """Reusable public raw-input prelude; contains no parent identity lookup.

    The head must be root-loaded/frozen.  Perturbed/new raw input must call this
    again.  Cache only while raw input, frozen head, and TRAIN statistics agree.
    features482 is NOT standardized, permitting a REAL-only TRAIN statistics fit.
    """
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError("Q head must be frozen before preparing waveform coordinates")
    if any(p.dtype != torch.float64 for p in head.parameters()):
        raise TypeError("frozen Q head must remain float64")
    coordinates = common_charge_coordinates(head, x, mu, scale, edges)
    lifted = coordinates["lifted_waveform"]
    q0, qs = coordinates["q_input"], coordinates["q_target"]
    unit = (lifted - lifted[:, :100].mean(1, keepdim=True)) / qs[:, None]
    # These are log Q in the original window-Q numeric unit, without a domain flag.
    features = torch.cat((unit, q0.log()[:, None], qs.log()[:, None]), dim=1)
    _finite(features, "frozen 482-d waveform input")
    p = torch.as_tensor(p480, dtype=torch.float64, device=lifted.device)
    if p.shape != (241,) or not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("fixed R71 spectrum must be a finite nonnegative 241-vector")
    if not math.isfinite(float(tau2)) or tau2 < 0:
        raise ValueError("fixed R71 target baseline variance must be finite and nonnegative")
    variance = lifted[:, :100].var(1, unbiased=True)
    amp2 = (float(tau2) - variance).clamp_min(0.0)
    kappa = projected_noise_variance(coordinates["unit_template"], p)
    active = amp2 > 0
    ratio = torch.where(active, amp2 / kappa, torch.ones_like(amp2))
    amplitude = torch.where(active, torch.sqrt(ratio), torch.zeros_like(amp2))
    return {
        "features482": features.detach(), "lifted": lifted.detach(),
        "mu_condition": coordinates["unit_template"].detach(),
        "scale_condition": coordinates["scale"].detach(),
        "residual_scale_V": (qs[:, None] * coordinates["scale"]).detach(),
        "Q0": q0.detach(), "Qstar": qs.detach(),
        "original_bin_raw": coordinates["input_bin_raw"].detach(),
        "original_bin_clipped": coordinates["input_bin_clipped"].detach(),
        "target_bin_raw": coordinates["target_bin_raw"].detach(),
        "target_bin_clipped": coordinates["target_bin_clipped"].detach(),
        "continuous_condition": coordinates["continuous_condition"].detach(),
        "fixed_unit_Q_slope": coordinates["fixed_unit_q_slope"].detach(),
        "lifted_baseline_variance": variance.detach(), "noise_amplitude_squared": amp2.detach(),
        "noise_amplitude": amplitude.detach(), "noise_kappa2": kappa.detach(),
        "edges_logQ": torch.as_tensor(edges, dtype=torch.float64, device=lifted.device).detach(),
        "noise_sqrt_p480": torch.sqrt(p).detach(),
    }


def slice_prepared(prepared: dict[str, Tensor], start: int, stop: int) -> dict[str, Tensor]:
    """Slice named row tensors, retaining shared edges/spectrum unmodified."""
    n = prepared["Qstar"].shape[0]
    if not 0 <= start < stop <= n:
        raise ValueError("prepared slice must be nonempty within the original rows")
    return {**{key: prepared[key][start:stop] for key in ROW_KEYS},
            **{key: prepared[key] for key in SHARED_KEYS}}


def forward_prepared(shape: ShapeResidualNetwork, prepared: dict[str, Tensor], epsilon: Tensor,
                     straight_through: bool = True):
    """Return (pre32, canonical, raw64, meta), with explicit full-wave noise.

    All meta values are row tensors.  range_projection_fraction is a scalar
    batch summary; use range_projection_active for population aggregation.
    """
    lifted, qs, m = prepared["lifted"], prepared["Qstar"], prepared["mu_condition"]
    eps = torch.as_tensor(epsilon, dtype=torch.float64, device=lifted.device)
    if eps.shape != lifted.shape:
        raise ValueError("epsilon must match every row and all 480 samples")
    _finite(eps, "explicit epsilon")
    f = shape(prepared["features482"]).to(torch.float64)
    d_raw = prepared["residual_scale_V"] * f
    delta = d_raw - m * window_charge(d_raw)[:, None]
    n = torch.fft.irfft(prepared["noise_sqrt_p480"] * torch.fft.rfft(eps, norm="ortho"),
                        n=480, norm="ortho")
    n = (n - m * window_charge(n)[:, None]) * prepared["noise_amplitude"][:, None]
    raw = lifted + n + delta
    pre64 = bounded_charge(raw, qs)
    pre32 = pre64.to(torch.float32)
    error64 = (window_charge(pre64) - qs).abs() / qs
    error32 = (window_charge(pre32) - qs).abs() / qs
    _finite(pre64, "pre-ADC waveform")
    if bool((error64 > 1e-5).any()) or bool((error32 > 1e-5).any()):
        raise FloatingPointError("registered R71 relative Qstar precision failed; no row deletion")
    if bool(((pre32 < -0.5) | (pre32 > 0.5)).any()):
        raise FloatingPointError("pre-ADC waveform exceeds original ADC range")
    canonical = canonical_adc_torch(pre32, straight_through=straight_through)
    _finite(canonical, "canonical waveform")
    preq, adcq = window_charge(pre32), window_charge(canonical)
    before = target_charge_condition(preq, prepared["edges_logQ"])
    after = target_charge_condition(adcq, prepared["edges_logQ"])
    active = ((raw < -0.5) | (raw > 0.5)).any(1)
    meta = {
        "Q0": prepared["Q0"], "Qstar": qs, "QpreADC": preq, "QADC": adcq,
        "original_bin_raw": prepared["original_bin_raw"], "target_bin_raw": prepared["target_bin_raw"],
        "preADC_bin_raw": before["bin_raw"], "ADC_bin_raw": after["bin_raw"],
        "original_bin_clipped": prepared["original_bin_clipped"],
        "target_bin_clipped": prepared["target_bin_clipped"],
        "preADC_bin_clipped": before["bin_clipped"], "ADC_bin_clipped": after["bin_clipped"],
        "original_inside_support": (prepared["original_bin_raw"] >= 0) & (prepared["original_bin_raw"] < 8),
        "target_inside_support": (prepared["target_bin_raw"] >= 0) & (prepared["target_bin_raw"] < 8),
        "preADC_inside_support": (before["bin_raw"] >= 0) & (before["bin_raw"] < 8),
        "ADC_inside_support": (after["bin_raw"] >= 0) & (after["bin_raw"] < 8),
        "noise_condition_Qstar": qs, "continuous_condition": prepared["continuous_condition"],
        "noise_kappa2": prepared["noise_kappa2"], "noise_amplitude_squared": prepared["noise_amplitude_squared"],
        "lifted_baseline_variance": prepared["lifted_baseline_variance"],
        "fixed_unit_Q_slope": prepared["fixed_unit_Q_slope"],
        "Q_relative_error_f64": error64, "Q_relative_error_f32": error32,
        "range_projection_active": active, "range_projection_fraction": active.to(torch.float64).mean(),
        "projection_RMS_V": (pre64 - raw).square().mean(1).sqrt(),
        "residual_RMS_V": delta.square().mean(1).sqrt(), "residual_Q": window_charge(delta),
        "target_to_input_Q_ratio": qs / prepared["Q0"], "ADC_to_input_Q_ratio": adcq / prepared["Q0"],
    }
    return pre32, canonical, raw, meta


def forward_raw(shape: ShapeResidualNetwork, head: nn.Module, x: Tensor, epsilon: Tensor,
                mu: Tensor, scale: Tensor, edges: Tensor, p480: Tensor, tau2: float,
                straight_through: bool = True):
    """Public path for new/perturbed raw input; never looks up parent IDs."""
    prepared = prepare_static(head, x, mu, scale, edges, p480, tau2)
    return forward_prepared(shape, prepared, epsilon, straight_through=straight_through)
