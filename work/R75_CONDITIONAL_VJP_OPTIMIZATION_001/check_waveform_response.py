"""Artificial checks callable by root on CPU/CUDA before scientific data I/O."""
from __future__ import annotations

import numpy as np
import torch

from public_charge_head import PublicConditionalChargeHead, window_charge
from common_response import common_response as r71_response
from numpy_reference import reference as r71_numpy_reference
from waveform_response import ShapeResidualNetwork, prepare_static, slice_prepared, forward_prepared, forward_raw
from waveform_numpy_reference import reference_numpy


def _fixtures(device):
    t = np.arange(480, dtype="f8")
    a = np.zeros(480); a[:100] = -3; a[100:400] = 1
    q0 = np.array([0.5, 1.0, 3.0, 9.0, 18.0, 40.0, 80.0, 180.0])
    wiggle = 0.002 * np.sin(t / 11)
    wiggle -= (wiggle[100:400].sum() - 3 * wiggle[:100].sum()) * a / (a @ a)
    x = 0.001 + q0[:, None] * a / (a @ a) + wiggle
    mu = np.stack([np.exp(-0.5 * ((t - (150 + 13 * b)) / (12 + 2 * b)) ** 2)
                   + 0.002 * (b + 1) * np.sin(t / 19) for b in range(8)])
    scale = np.repeat((0.001 + np.arange(8) * 0.00025)[:, None], 480, axis=1)
    edges = np.log(1000 * np.power(2.0, np.arange(9)))
    p = 0.1 + np.exp(-np.arange(241) / 18)
    eps = np.random.Generator(np.random.PCG64(720001)).standard_normal(x.shape).astype("f4")
    values = [torch.as_tensor(v, device=device) for v in (x, mu, scale, edges, p)]
    return *values, torch.as_tensor(eps, device=device), 3e-5


def make_fixture(device="cpu") -> dict:
    """Eight artificial raw parents; no model construction or file I/O.

    This module fixture includes one outside-support row.  It is not a fully
    populated regional-objective fixture; root must construct that separately.
    """
    x, mu, scale, edges, p, eps, tau = _fixtures(torch.device(device))
    return {"x": x, "epsilon": eps, "mu": mu, "scale": scale,
            "edges": edges, "p480": p, "tau2": tau}


def _state_numpy(model):
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def _bits_equal(left, right):
    a, b = left.detach().cpu().contiguous().numpy(), right.detach().cpu().contiguous().numpy()
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def run_artificial(device="cpu") -> dict:
    """No file I/O, science arrays, or persistent RNG change; root selects device."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    checks, measurements = [], {}

    def require(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def close(name, a, b, atol=1e-10, rtol=1e-10):
        require(name, torch.allclose(a, b, atol=atol, rtol=rtol))

    def reject(name, call):
        try:
            call()
        except (ValueError, TypeError, FloatingPointError):
            checks.append(name)
        else:
            raise AssertionError(name)

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(720001)
        x, mu, scale, edges, p, eps, tau2 = _fixtures(device)
        head = PublicConditionalChargeHead(torch.zeros(480, device=device), torch.ones(480, device=device))
        with torch.no_grad():
            head.output.weight.copy_(torch.linspace(-0.035, 0.045, 32, device=device, dtype=torch.float64).reshape(2, 16))
            head.output.bias.copy_(torch.tensor([0.08, -0.02], device=device, dtype=torch.float64))
        head.requires_grad_(False)
        q_state = _state_numpy(head)
        prepared = prepare_static(head, x, mu, scale, edges, p, tau2)
        # Artificial common REAL-only-like population fit; no scientific fitting.
        center = prepared["features482"].mean(0)
        standard_deviation = prepared["features482"].std(0, unbiased=False).clamp_min(1e-6)
        shape = ShapeResidualNetwork(center, standard_deviation)
        require("fixed_482_128_128_480_architecture", shape.hidden1.in_features == 482 and
                shape.hidden1.out_features == shape.hidden2.in_features == shape.hidden2.out_features == 128 and
                shape.output.in_features == 128 and shape.output.out_features == 480)
        require("parameters_float32_buffers_float64", all(v.dtype == torch.float32 for v in shape.parameters()) and
                all(v.dtype == torch.float64 for v in shape.buffers()))
        require("last_layer_zero_initialization", torch.count_nonzero(shape.output.weight) == 0 and
                torch.count_nonzero(shape.output.bias) == 0)
        require("frozen_prepared_tensors_and_no_Q_head_grads", all(not v.requires_grad for v in prepared.values()) and
                all(not v.requires_grad for v in head.parameters()))
        close("residual_voltage_units_include_Qstar", prepared["residual_scale_V"],
              prepared["Qstar"][:, None] * prepared["scale_condition"], atol=0, rtol=0)
        close("input_unit_charge_one", window_charge(prepared["features482"][:, :480]),
              torch.ones_like(prepared["Qstar"]))
        close("features_logQ0_logQstar", prepared["features482"][:, -2:],
              torch.stack((prepared["Q0"].log(), prepared["Qstar"].log()), dim=1), atol=0, rtol=0)
        old = r71_response(head, x, eps, mu, scale, edges, p, tau2)
        zero = forward_prepared(shape, prepared, eps, straight_through=False)
        require("zero_residual_preADC_R71_bitwise", _bits_equal(zero[0], old[0]))
        require("zero_residual_canonical_R71_bitwise", _bits_equal(zero[1], old[1]))
        close("zero_residual_Q", zero[3]["residual_Q"], torch.zeros_like(prepared["Qstar"]), atol=0, rtol=0)
        old_np = r71_numpy_reference(q_state, x.cpu().numpy(), eps.cpu().numpy(), mu.cpu().numpy(),
                                      scale.cpu().numpy(), edges.cpu().numpy(), p.cpu().numpy(), tau2)
        zero_np = reference_numpy(q_state, _state_numpy(shape), x.cpu().numpy(), eps.cpu().numpy(),
                                  mu.cpu().numpy(), scale.cpu().numpy(), edges.cpu().numpy(), p.cpu().numpy(), tau2)
        require("independent_zero_residual_matches_R71_NumPy", np.array_equal(zero_np[0], old_np[0]) and
                np.array_equal(zero_np[1], old_np[1]))
        uncached = forward_raw(shape, head, x, eps, mu, scale, edges, p, tau2, straight_through=False)
        require("cached_uncached_bitwise", all(_bits_equal(zero[i], uncached[i]) for i in range(3)))
        same = forward_raw(shape, head, x.clone(), eps.clone(), mu, scale, edges, p, tau2, straight_through=False)
        require("same_input_SOURCE_REAL_identical_public_path", all(_bits_equal(uncached[i], same[i]) for i in range(3)))
        sl = slice_prepared(prepared, 2, 6)
        require("slice_preserves_shared_edges_and_spectrum", sl["edges_logQ"] is prepared["edges_logQ"] and
                sl["noise_sqrt_p480"] is prepared["noise_sqrt_p480"] and sl["Qstar"].shape == (4,))
        rng_before = torch.get_rng_state().clone()
        forward_prepared(shape, prepared, eps, straight_through=False)
        require("forward_draws_no_CPU_randomness", torch.equal(rng_before, torch.get_rng_state()))

        # Nonzero artificial weights exercise actual full-waveform residual/AD.
        with torch.no_grad():
            shape.output.weight.copy_(torch.linspace(-0.0007, 0.0009, 480 * 128, device=device).reshape(480, 128))
            shape.output.bias.copy_(0.005 * torch.sin(torch.arange(480, device=device) / 11.0))
        learned = forward_prepared(shape, prepared, eps, straight_through=False)
        close("nonzero_residual_zero_Q", learned[3]["residual_Q"], torch.zeros_like(prepared["Qstar"]), atol=1e-12)
        close("raw_waveform_Qstar", window_charge(learned[2]), prepared["Qstar"], atol=1e-11)
        require("Box_Qstar_precision_and_ADC_rails", bool((learned[3]["Q_relative_error_f64"] <= 1e-5).all()) and
                bool((learned[3]["Q_relative_error_f32"] <= 1e-5).all()) and bool((learned[0].abs() <= 0.5).all()))
        learned_uncached = forward_raw(shape, head, x, eps, mu, scale, edges, p, tau2, straight_through=False)
        require("nonzero_cached_uncached_bitwise", all(_bits_equal(learned[i], learned_uncached[i]) for i in range(3)))
        permutation = torch.tensor([7, 0, 4, 2, 6, 1, 5, 3], device=device)
        perm = forward_raw(shape, head, x[permutation], eps[permutation], mu, scale, edges, p, tau2, straight_through=False)
        measurements["row_permutation_byte_equal"] = {
            name: _bits_equal(perm[i], learned[i][permutation])
            for i, name in enumerate(["pre32", "canonical", "raw64"])
        }
        # Root revised this engineering fixture assertion after the preserved
        # first native failure identified float32 GEMM row-position rounding.
        # No scientific numeric/replay threshold is altered by this assertion.
        permutation_tolerance = 2.0 * torch.finfo(torch.float32).eps
        measurements["row_permutation_fixture_atol_rtol"] = permutation_tolerance
        require("parent_row_permutation_numerical_geometry", all(
            torch.allclose(perm[i].double(), learned[i][permutation].double(),
                           atol=permutation_tolerance, rtol=permutation_tolerance)
            for i in [0, 2]))
        perm_codes = torch.round((perm[1] + 0.5) * 16383).to(torch.int64)
        original_codes = torch.round((learned[1][permutation] + 0.5) * 16383).to(torch.int64)
        require("parent_row_permutation_fixture_actual_ADC_codes", torch.equal(perm_codes, original_codes))
        no_ste = learned[1]
        ste = forward_prepared(shape, prepared, eps, straight_through=True)
        codes_no_ste = torch.round((no_ste + 0.5) * 16383).to(torch.int64)
        codes_ste = torch.round((ste[1] + 0.5) * 16383).to(torch.int64)
        require("STE_same_actual_ADC_integer_codes", torch.equal(codes_no_ste, codes_ste))
        probe = torch.cos(torch.arange(480, dtype=torch.float64, device=device) / 9)[None, :]
        loss = (ste[1] * probe).sum()
        parameters = tuple(shape.parameters())
        grads = torch.autograd.grad(loss, parameters, retain_graph=True)
        require("shape_Box_ADC_STE_gradients_finite_nonzero", all(torch.isfinite(v).all() for v in grads) and
                grads[-1].abs().sum() > 0)
        q_grads = torch.autograd.grad(window_charge(ste[0]).sum(), parameters)
        measurements["charge_parameter_gradient_max_abs"] = max(v.abs().max().item() for v in q_grads)
        require("Qstar_independent_of_shape_parameters", measurements["charge_parameter_gradient_max_abs"] < 1e-5)
        require("frozen_Q_head_has_no_gradient", all(v.grad is None for v in head.parameters()))
        np_result = reference_numpy(q_state, _state_numpy(shape), x.cpu().numpy(), eps.cpu().numpy(), mu.cpu().numpy(),
                                    scale.cpu().numpy(), edges.cpu().numpy(), p.cpu().numpy(), tau2)
        difference = learned[0].detach().cpu().numpy().astype("f8") - np_result[0].astype("f8")
        rms_lsb = np.sqrt(np.mean(difference ** 2, axis=1)) * 16383
        measurements["independent_NumPy_nonzero_max_RMS_LSB"] = float(rms_lsb.max())
        require("independent_NumPy_nonzero_preADC_reference", float(rms_lsb.max()) <= 0.25)
        require("independent_NumPy_Qstar_reference", np.allclose(prepared["Qstar"].cpu().numpy(), np_result[3]["Qstar"], atol=1e-12, rtol=1e-12))
        learned_codes = torch.round((learned[1].detach().cpu() + 0.5) * 16383).to(torch.int64).numpy()
        numpy_codes = np.rint((np_result[1] + 0.5) * 16383).astype("i8")
        require("independent_NumPy_actual_ADC_codes", np.array_equal(learned_codes, numpy_codes))
        with torch.no_grad():
            shape.output.weight.zero_()
            shape.output.bias.copy_(1.7 * torch.sin(torch.arange(480, device=device) / 11.0))
        projected = forward_prepared(shape, prepared, eps, straight_through=False)
        require("nontrivial_Box_projection_exercised", projected[3]["range_projection_active"].any() and
                (projected[3]["projection_RMS_V"] > 0).any())
        require("projected_waveform_charge_and_rails", (projected[3]["Q_relative_error_f64"] <= 1e-5).all() and
                (projected[0].abs() <= 0.5).all())
        outside = (prepared["original_bin_raw"] < 0) | (prepared["original_bin_raw"] > 7)
        require("outside_support_original_rows_preserved", outside.any() and projected[0].shape[0] == x.shape[0] and
                torch.equal(projected[3]["original_bin_raw"], prepared["original_bin_raw"]))
        require("four_stage_migration_metadata_present", all(k in projected[3] for k in
                ["original_bin_raw", "target_bin_raw", "preADC_bin_raw", "ADC_bin_raw"]))
        reject("reject_wrong_architecture_32", lambda: ShapeResidualNetwork(center, standard_deviation, hidden=32))
        reject("reject_unfloored_zero_std", lambda: ShapeResidualNetwork(center, torch.zeros_like(standard_deviation)))
        reject("reject_epsilon_shape", lambda: forward_prepared(shape, prepared, eps[:, :100]))
        bad_eps = eps.clone(); bad_eps[0, 479] = float("nan")
        reject("reject_nonfinite_explicit_epsilon", lambda: forward_prepared(shape, prepared, bad_eps))
        reject("reject_empty_cache_slice", lambda: slice_prepared(prepared, 2, 2))
        head.requires_grad_(True)
        reject("reject_trainable_Q_head", lambda: prepare_static(head, x, mu, scale, edges, p, tau2))
        head.requires_grad_(False)
        measurements["max_Qstar_relative_error_f64"] = projected[3]["Q_relative_error_f64"].max().item()
        measurements["max_Qstar_relative_error_f32"] = projected[3]["Q_relative_error_f32"].max().item()
    return {"pass_all": True, "evidence_level": f"artificial_{device.type}_implementation_only",
            "device": str(device), "check_count": len(checks), "checks": checks, "measurements": measurements,
            "scientific_arrays_or_weights_read": False, "scientific_training_or_probe_run": False,
            "fixed_architecture": "482-128-128-480 SiLU", "Q_head_frozen": True}
