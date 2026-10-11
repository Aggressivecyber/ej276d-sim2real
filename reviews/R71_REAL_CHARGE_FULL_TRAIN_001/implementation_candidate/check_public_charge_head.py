"""Artificial CPU/autograd implementation checks, not scientific evidence.

Import performs no file I/O, random draw, model construction, or self-run.
run_checks() restores the caller's CPU Torch RNG.  Optional run_box_checks()
accepts already loaded frozen Box/ADC functions without loading a network.
"""
from __future__ import annotations

import math

import torch

from public_charge_head import (
    PublicConditionalChargeHead, bounded_charge_map, charge_coordinate_lift,
    common_charge_coordinates, interpolate_charge_statistics,
    normalized_template_bank, target_charge_condition, window_charge,
    zero_charge_project,
)


def _fixtures():
    t = torch.arange(480, dtype=torch.float64)
    charge_axis = torch.zeros(480, dtype=torch.float64)
    charge_axis[:100] = -3.0
    charge_axis[100:400] = 1.0
    charges = torch.tensor([5.0, 35.0, 110.0, 245.0], dtype=torch.float64)
    wiggle = 0.003 * torch.sin(t / 11.0)
    wiggle = wiggle - window_charge(wiggle) * charge_axis / charge_axis.square().sum()
    x = 0.004 + charges[:, None] * charge_axis / charge_axis.square().sum() + wiggle
    bank = []
    for region in range(8):
        pulse = torch.exp(-0.5 * ((t - (150 + 13 * region)) / (12 + 2 * region)).square())
        pulse = pulse + 0.03 * (region + 1) * torch.sin(t / 19.0)
        bank.append(pulse)
    mu = torch.stack(bank)
    scale = (0.2 + torch.arange(8, dtype=torch.float64)[:, None] * 0.025).expand(8, 480).clone()
    edges = torch.log(1000.0 * torch.pow(2.0, torch.arange(9, dtype=torch.float64)))
    return x, mu, scale, edges


def run_checks() -> dict:
    checks = []
    measurements = {}

    def verify(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def close(name, actual, expected, atol=1e-10, rtol=1e-10):
        actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
        verify(name, torch.allclose(actual, expected, atol=atol, rtol=rtol))

    def rejected(name, call):
        try:
            call()
        except (ValueError, TypeError, FloatingPointError):
            checks.append(name)
        else:
            raise AssertionError(name)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(710001)
        x, mu, scale, edges = _fixtures()
        head = PublicConditionalChargeHead(torch.zeros(480), torch.ones(480)).double()
        verify("all_parameters_and_buffers_float64", all(
            p.dtype == torch.float64 for p in list(head.parameters()) + list(head.buffers())
        ))
        verify("480_16_2_architecture", head.hidden.in_features == 480 and
               head.hidden.out_features == head.output.in_features == 16 and head.output.out_features == 2)
        verify("last_layer_zero", torch.count_nonzero(head.output.weight) == 0 and
               torch.count_nonzero(head.output.bias) == 0)
        initial = head(x)
        q0 = window_charge(x)
        identity_rel = ((initial["q_target"] - q0).abs() / q0).max().item()
        measurements["identity_initialization_max_relative_roundoff"] = identity_rel
        verify("identity_roundoff_below_implementation_1e_minus_5", identity_rel <= 1e-5)
        close("unit_waveform_charge_one", window_charge(initial["unit_waveform"]), torch.ones_like(q0))
        close("unit_waveform_baseline_zero", initial["unit_waveform"][:, :100].mean(-1), torch.zeros_like(q0))
        close("identity_fixed_unit_slope_one", initial["fixed_unit_q_slope"], torch.ones_like(q0))
        cached = head.standardize_input(x)
        quick = head.forward_features(cached["features"], cached["q_input"], check_inputs=False)
        close("cached_feature_path_equals_public_path", quick["q_target"], initial["q_target"], 0, 0)
        close("same_public_call_source_real", head(x)["q_target"], head(x.clone())["q_target"], 0, 0)
        permutation = torch.tensor([3, 0, 2, 1])
        close("no_cross_parent_state", head(x[permutation])["q_target"], initial["q_target"][permutation], 0, 0)

        q = torch.tensor([5.0, 70.0, 180.0, 275.0], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([0.13, -0.09, 0.07, -0.11], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([-0.06, 0.1, -0.03, 0.04], dtype=torch.float64, requires_grad=True)
        mapped = bounded_charge_map(q, a, b)
        actual_q, actual_a, actual_b = torch.autograd.grad(mapped["q_target"].sum(), (q, a, b))
        gate = mapped["q_target"] * (1.0 - mapped["q_target"] / 300.0)
        close("fixed_unit_q_slope_autograd", actual_q, mapped["fixed_unit_q_slope"])
        close("charge_a_derivative", actual_a, gate)
        close("charge_b_derivative", actual_b, gate * torch.exp(b) * mapped["logit_input"])
        verify("bounded_positive_map_and_slope", (mapped["q_target"] > 0).all() and
               (mapped["q_target"] < 300).all() and (actual_q > 0).all())

        with torch.no_grad():
            head.output.weight.copy_(torch.linspace(-0.035, 0.045, 32, dtype=torch.float64).reshape(2, 16))
            head.output.bias.copy_(torch.tensor([0.08, -0.02], dtype=torch.float64))
        differentiable_x = x.clone().requires_grad_(True)
        differentiable_output = head(differentiable_x)["q_target"]
        gx = torch.autograd.grad(differentiable_output.sum(), differentiable_x)[0]
        verify("input_autograd_finite_and_full_tail_connected", torch.isfinite(gx).all() and
               (gx[:, [0, 99, 100, 399, 400, 479]].abs() > 0).all())
        direction = torch.sin(torch.arange(x.numel(), dtype=torch.float64)).reshape_as(x) * 0.1
        eps = 1e-5
        finite_difference = (head(x + eps * direction)["q_target"].sum() -
                             head(x - eps * direction)["q_target"].sum()) / (2 * eps)
        close("full_waveform_head_directional_derivative", (gx * direction).sum(), finite_difference,
              atol=1e-7, rtol=1e-7)
        measurements["head_directional_derivative_absolute_error"] = (
            (gx * direction).sum() - finite_difference).abs().item()
        rng_before = torch.get_rng_state().clone()
        head(x)
        verify("head_forward_draws_no_rng", torch.equal(rng_before, torch.get_rng_state()))

        unit_bank = normalized_template_bank(mu)
        close("frozen_templates_unit_charge", window_charge(unit_bank), torch.ones(8, dtype=torch.float64))
        close("frozen_templates_baseline_zero", unit_bank[:, :100].mean(-1), torch.zeros(8, dtype=torch.float64))
        centers = 0.5 * (edges[:-1] + edges[1:])
        center_q = torch.exp(centers) / 1000.0
        at_centers = interpolate_charge_statistics(center_q, mu, scale, edges)
        close("interpolation_matches_centers", at_centers["unit_template"], unit_bank)
        midpoint_q = torch.exp(0.5 * (centers[2] + centers[3])).reshape(1) / 1000.0
        midpoint = interpolate_charge_statistics(midpoint_q, mu, scale, edges)
        close("interpolation_midpoint_template", midpoint["unit_template"], 0.5 * (unit_bank[2:3] + unit_bank[3:4]))
        close("interpolation_midpoint_scale", midpoint["scale"], 0.5 * (scale[2:3] + scale[3:4]))
        extension = interpolate_charge_statistics(torch.tensor([0.5, 280.0], dtype=torch.float64), mu, scale, edges)
        close("constant_extension_at_both_ends", extension["unit_template"], unit_bank[[0, 7]])
        near_q = center_q[3].repeat(2) * torch.exp(torch.tensor([-1e-8, 1e-8], dtype=torch.float64))
        near = interpolate_charge_statistics(near_q, mu, scale, edges)
        verify("template_value_continuous_at_knot", (near["unit_template"][0] - near["unit_template"][1]).abs().max() < 1e-8)
        q_grad = midpoint_q.clone().requires_grad_(True)
        stats_grad = interpolate_charge_statistics(q_grad, mu, scale, edges)
        template_scalar = stats_grad["unit_template"][0, 217]
        actual_template_gradient = torch.autograd.grad(template_scalar, q_grad)[0]
        expected_template_gradient = (unit_bank[3, 217] - unit_bank[2, 217]) / (
            q_grad * (centers[3] - centers[2]))
        close("continuous_template_charge_derivative", actual_template_gradient, expected_template_gradient)
        verify("template_gradient_nonzero_between_centers", actual_template_gradient.abs().item() > 0)

        target = torch.tensor([4.0, 48.0, 105.0, 220.0], dtype=torch.float64, requires_grad=True)
        stats = interpolate_charge_statistics(target, mu, scale, edges)
        lifted = charge_coordinate_lift(x, target, stats["unit_template"])
        close("lifting_charge_equals_independent_target", window_charge(lifted), target)
        lift_g = torch.autograd.grad(window_charge(lifted).sum(), target)[0]
        close("lifting_charge_target_derivative_one", lift_g, torch.ones_like(target))
        measurements["lifting_max_absolute_charge_error"] = (window_charge(lifted) - target).abs().max().item()
        projected = zero_charge_project(x, stats["unit_template"])
        close("zero_charge_projection", window_charge(projected), torch.zeros_like(target))
        x32 = x.to(torch.float32)
        lifted32 = charge_coordinate_lift(x32, target.detach(), stats["unit_template"].detach())
        close("float32_input_lifts_in_float64", window_charge(lifted32), target.detach())
        verify("float32_input_physical_output_float64", lifted32.dtype == torch.float64)
        condition_q = torch.tensor([3.0, 12.0, 70.0], dtype=torch.float64, requires_grad=True)
        condition = target_charge_condition(condition_q, edges)
        cg = torch.autograd.grad(condition["continuous_condition"].sum(), condition_q)[0]
        width = 0.5 * (edges[condition["bin_clipped"] + 1] - edges[condition["bin_clipped"]])
        close("continuous_logQ_condition_gradient", cg, 1.0 / (condition_q * width))
        bins = target_charge_condition(torch.tensor([0.5, 256.0, 280.0], dtype=torch.float64), edges)
        verify("raw_outside_bins_and_final_edge_preserved", torch.equal(bins["bin_raw"], torch.tensor([-1, 7, 8])) and
               torch.equal(bins["bin_clipped"], torch.tensor([0, 7, 7])))
        combined = common_charge_coordinates(head, x, mu, scale, edges)
        close("common_coordinate_end_to_end_charge", window_charge(combined["lifted_waveform"]), combined["q_target"])
        verify("parent_count_order_preserved", combined["lifted_waveform"].shape == x.shape and
               torch.equal(combined["q_input"], window_charge(x)))

        rejected("reject_nonpositive_original_Q", lambda: head(torch.zeros_like(x)))
        axis = torch.zeros_like(x)
        axis[:, 100:400] = 1.0
        rejected("reject_original_Q_equal_300", lambda: head(axis))
        bad_x = x.clone()
        bad_x[0, 479] = math.nan
        rejected("reject_nonfinite_full_waveform", lambda: head(bad_x))
        rejected("reject_scale_zero_without_floor", lambda: PublicConditionalChargeHead(torch.zeros(480), torch.zeros(480)))
        rejected("reject_wrong_hidden_candidate", lambda: PublicConditionalChargeHead(torch.zeros(480), torch.ones(480), hidden=8))
        rejected("reject_zero_charge_template", lambda: normalized_template_bank(torch.zeros(8, 480)))
        rejected("reject_nonmonotone_edges", lambda: target_charge_condition(q0, edges.flip(0)))
        rejected("reject_Q_saturation_without_clamp", lambda: bounded_charge_map(
            q0, torch.full_like(q0, 1000.0), torch.zeros_like(q0)))
        rejected("reject_slope_underflow_without_floor", lambda: bounded_charge_map(
            q0, torch.zeros_like(q0), torch.full_like(q0, -1000.0)))
        rejected("reject_nonfinite_prediction_parameters", lambda: bounded_charge_map(
            q0, torch.full_like(q0, math.nan), torch.zeros_like(q0)))
    return {"status": "PASS", "evidence_level": "artificial_CPU_PyTorch_autograd_only",
            "scientific_data_read": False, "scientific_inference_or_training": False,
            "CUDA_executed": False, "check_count": len(checks), "checks": checks,
            "measurements": measurements, "torch_version": torch.__version__}


def run_box_checks(box_projection, canonical_adc) -> dict:
    """Artificial compatibility with the caller's exact frozen Box/ADC code."""
    x, mu, scale, edges = _fixtures()
    target = torch.tensor([4.0, 48.0, 105.0, 220.0], dtype=torch.float64, requires_grad=True)
    stats = interpolate_charge_statistics(target, mu, scale, edges)
    lifted = charge_coordinate_lift(x, target, stats["unit_template"])
    addition = 0.55 * torch.sin(torch.arange(480, dtype=torch.float64) / 7.0).expand_as(x)
    raw = lifted + zero_charge_project(addition, stats["unit_template"])
    bounded = box_projection.apply(raw, target)
    if not bool(torch.isfinite(bounded).all() and (bounded.abs() <= 0.5).all()):
        raise AssertionError("frozen Box rails/finiteness")
    charge_error = (window_charge(bounded) - target).abs().max().item()
    if charge_error > 1e-9:
        raise AssertionError("frozen Box charge target")
    grad = torch.autograd.grad(window_charge(bounded).sum(), target, retain_graph=True)[0]
    if not torch.allclose(grad, torch.ones_like(target), atol=1e-10, rtol=1e-10):
        raise AssertionError("frozen Box charge gradient")
    adc = canonical_adc(bounded)
    if adc.dtype != torch.float64 or not bool(torch.isfinite(adc).all() and (adc.abs() <= 0.5).all()):
        raise AssertionError("frozen ADC output type/rails/finiteness")
    adc_charge_error = (window_charge(adc) - target).abs().max().item()
    # Sum of absolute charge coefficients=600.  This is a descriptive
    # quantization bound, not a proposed scientific pass/fail threshold.
    descriptive_bound = 600.0 / 16383.0 + 1e-7
    if adc_charge_error > descriptive_bound:
        raise AssertionError("frozen ADC descriptive quantization bound")
    ste = canonical_adc(bounded, straight_through=True)
    ste_grad = torch.autograd.grad(window_charge(ste).sum(), target)[0]
    if not torch.allclose(ste_grad, torch.ones_like(target), atol=1e-10, rtol=1e-10):
        raise AssertionError("frozen ADC declared STE gradient")
    return {"status": "PASS", "evidence_level": "artificial_CPU_exact_frozen_Box_ADC_only",
            "scientific_data_read": False, "CUDA_executed": False, "check_count": 5,
            "checks": ["Box_rails_finite", "Box_independent_target_charge", "Box_target_autograd",
                       "canonical_ADC_type_rails_quantization", "canonical_ADC_declared_STE"],
            "Box_max_absolute_Q_error": charge_error,
            "ADC_max_absolute_Q_error": adc_charge_error,
            "ADC_descriptive_Q_error_bound": descriptive_bound}
