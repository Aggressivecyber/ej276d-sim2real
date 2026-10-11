"""Portable synthetic proofs for the R65 thin objective calls.

run_check(device='cpu', *, objective_adapter_root=...) returns a JSON receipt.
The caller owns resources, torch numerical settings and output writes. This
module opens only bound Python/JSON text. No scientific arrays or checkpoints.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

TOLERANCE_SHA256 = "0875588c423a23561e1d45dc326906ef93b136197475b5e113a458edd91ec6d6"
SPEC_SHA256 = "4f01f27e00c895de6ad6c9662304c0dd8f8e7cea236489f67bc049bd93afb683"
R64_CPU_ARRAYS_SHA256 = "8358df412f174e62fad3d848e38e50a6d39f906b6099b9c36696c5c56d2c3f1e"


class ArtificialCheckFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt["error_type"] + ": " + receipt["error"])


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _verify(root):
    manifest = json.loads((root / "PREEXECUTION_BINDING.json").read_text("utf-8"))
    for item in manifest["files"]:
        path = root / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("R65 frozen helper mismatch: " + item["file"])
    for name, expected in (("FROZEN_TOLERANCES.json", TOLERANCE_SHA256),
                           ("FIXTURE_SPEC.json", SPEC_SHA256)):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("R65 frozen tolerance/spec mismatch: " + name)
    return manifest


def run_check(device="cpu", *, objective_adapter_root):
    started = time.monotonic()
    root = Path(objective_adapter_root).resolve(strict=True)
    state = {"status": "RUNNING", "PID": os.getpid(), "device_requested": str(device),
             "tests": {}, "stages": [], "scientific_arrays_opened": 0,
             "scientific_weights_loaded": 0, "DEV_EVAL_CAL_reads": 0,
             "scientific_forward_or_gradient_runs": 0, "optimizer_updates": 0,
             "real_ResponseNetwork_executed": False}
    try:
        manifest = _verify(root)
        state["binding"] = manifest
        state["tolerance_sha256_before_run"] = TOLERANCE_SHA256
        state["fixture_spec_sha256_before_run"] = SPEC_SHA256
        import numpy as np
        import torch
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Select CPU or CUDA")
        old = _module("_r65_bound_portable", root / "bound_r64/portable_fixture.py")
        bound_root, _, _ = old._verified_inputs(root / "bound_r64")
        api = _module("_r65_objective_api", root / "objective_adapter.py")
        factory = _module("_r65_fixture_factory", root / "fixture_factory.py")
        tolerance = json.loads((root / "FROZEN_TOLERANCES.json").read_text("utf-8"))
        before = old._state_fingerprint(torch, np, device)
        state["caller_state_before"] = before
        with old._adapter_modules(bound_root) as (dg, R63, FrozenMass):
            _run(state, torch, np, device, api, factory, dg, R63, FrozenMass,
                 bound_root, tolerance, started)
        after = old._state_fingerprint(torch, np, device)
        state["caller_state_after"] = after
        if before != after:
            raise RuntimeError("Artificial helper changed caller numerical or random state")
        _verify(root)
        state["inputs_unchanged_after"] = True
        state["status"] = "ARTIFICIAL_OBJECTIVE_ADAPTER_PASSED_NOT_SCIENTIFIC_REGISTRATION"
        state["check_count"] = len(state["tests"])
        state["elapsed_seconds"] = time.monotonic() - started
        state["torch"] = torch.__version__
        state["numpy"] = np.__version__
        state["device"] = str(device)
        return state
    except Exception as exc:
        state.update(status="ARTIFICIAL_OBJECTIVE_ADAPTER_FAILED",
                     error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc(),
                     elapsed_seconds=time.monotonic() - started)
        raise ArtificialCheckFailure(state) from exc


def _run(state, torch, np, device, api, factory, dg, R63, FrozenMass,
         bound_root, tolerance, started):
    tests = state["tests"]

    def stage(name):
        state["stages"].append({"stage": name, "elapsed_seconds": time.monotonic() - started})

    def check(name, condition, **details):
        passed = bool(condition)
        tests[name] = dict(passed=passed, **details)
        if not passed:
            raise AssertionError((name, details))

    def compare(name, actual, expected, rule="exact"):
        a = torch.as_tensor(actual, device=device, dtype=torch.float64).detach()
        b = torch.as_tensor(expected, device=device, dtype=torch.float64).detach()
        if a.shape != b.shape:
            raise AssertionError((name, tuple(a.shape), tuple(b.shape)))
        delta = (a - b).abs()
        tol = tolerance[rule]
        check(name, torch.isfinite(a).all() and torch.isfinite(b).all() and
              (delta <= tol["atol"] + tol["rtol"] * b.abs()).all(),
              max_abs_error=float(delta.max()), reference_absmax=float(b.abs().max()),
              atol=tol["atol"], rtol=tol["rtol"], rule=rule)

    def norm(x):
        return float(torch.linalg.vector_norm(x.detach().double()))

    def hash_arrays(arrays):
        h = hashlib.sha256()
        for a in arrays:
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            a = np.asarray(a)
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    f = factory.build_fixture(dg, R63, device, bound_root)
    source, real, Y = f["source"], f["real"], f["cached_y"]
    source_regions, real_regions = f["source_regions"], f["real_regions"]
    cfg, stats, directions, tp = f["cfg"], f["stats"], f["directions"], f["tp"]
    objective, full, equal = f["objective"], f["full"], f["alayouts"][0]
    state["fixture_arrays_sha256"] = f["fixture_hash"].hexdigest()
    if device.type == "cpu":
        check("unchanged_R64_CPU_fixture_arrays",
              state["fixture_arrays_sha256"] == R64_CPU_ARRAYS_SHA256)
    canonical = R63.core.canonical_adc_torch(Y)
    check("original_unequal_population_and_duplicate_tie_fixture",
          len(source) == 38 and len(real) == 40 and
          not np.array_equal(f["source_counts"], f["real_counts"]) and
          len(torch.unique(canonical, dim=0)) < len(source) and
          max(np.bincount(equal.source_global, minlength=len(source))) > 1 and
          bool((Y.abs() == .5).any()) and bool((Y.abs() < .5).any()),
          SOURCE_parents=len(source), REAL_parents=len(real),
          distinct_ADC_waveforms=len(torch.unique(canonical, dim=0)),
          active_Box_coordinates=int((Y.abs() == .5).sum()))
    stage("original_artificial_fixture_constructed")

    results = {}
    for label, layout in (("full_unequal", full), ("equal32_duplicates", equal)):
        class CountCall:
            def __init__(self, delegate):
                self.delegate, self.calls = delegate, 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                return self.delegate(*args, **kwargs)
        counted = CountCall(objective)
        joint = api.joint_total_D(counted, Y, real, layout)
        separate = objective.endpoint_gradients(Y, real, layout)
        results[label] = joint
        check(label + "_one_joint_objective_call_and_order",
              counted.calls == 1 and joint["piece_order"] == api.D_PIECE_ORDER and
              not joint["total"].requires_grad and not joint["GY"].requires_grad,
              call_count=counted.calls, actual_piece_order=list(joint["pieces"]))
        for name in api.D_NAMES:
            compare(label + "_joint_vs_separate_value_" + name,
                    joint["pieces"][name], separate["values"][name], "unequal_oracle_value")
        compare(label + "_joint_vs_separate_total_value", joint["total"],
                sum(separate["values"][name] for name in api.D_PIECE_ORDER), "unequal_oracle_value")
        compare(label + "_joint_GY_vs_sum_five", joint["GY"],
                sum(separate["gradients"].values()), "equal_R63_f32_vs_f64_endpoint_gradient")
        leaf = Y.detach().clone().requires_grad_(True)
        direct = objective(leaf, real, layout, terms=api.D_NAMES)
        direct_gradient, = torch.autograd.grad(direct.total, leaf)
        compare(label + "_joint_vs_direct_total", joint["total"], direct.total)
        compare(label + "_joint_vs_direct_GY", joint["GY"], direct_gradient)
        check(label + "_nonzero_total_GY", norm(joint["GY"]) > tolerance["minimum_nonzero_gradient_norm"],
              gradient_norm=norm(joint["GY"]))
        del direct, direct_gradient, leaf, separate
    stage("joint_total_vs_five_and_direct_passed")

    # A second declared stream creates genuinely fresh full-window and temporal
    # directions; constructors must bind these tensors and must not alias inputs.
    next_rng = np.random.Generator(np.random.PCG64(640032129))
    next_directions = []
    for dim in (480, 480, 479, 478):
        p2 = next_rng.standard_normal((dim, 64))
        p2 /= np.linalg.norm(p2, axis=0, keepdims=True)
        next_directions.append(p2.astype(np.float32))
    next_temporal = next_rng.standard_normal((31, 64))
    next_temporal /= np.linalg.norm(next_temporal, axis=0, keepdims=True)
    next_temporal = next_temporal.astype(np.float32)
    next_objective = dg.FrozenDistributionObjective(
        stats, cfg, next_directions, next_temporal, source_regions, real_regions, device=device)
    for j, p2 in enumerate(next_directions):
        compare("fresh_step_direction_bound_" + str(j), next_objective.directions[j],
                torch.as_tensor(p2, device=device))
    compare("fresh_step_temporal_direction_bound", next_objective.temporal_directions,
            torch.as_tensor(next_temporal, device=device))
    second = api.joint_total_D(next_objective, Y, real, full)
    fw_gap = float((second["pieces"]["full_window"] -
                    results["full_unequal"]["pieces"]["full_window"]).abs())
    temporal_gap = float((second["pieces"]["temporal_joint"] -
                          results["full_unequal"]["pieces"]["temporal_joint"]).abs())
    gradient_gap = float((second["GY"] - results["full_unequal"]["GY"]).abs().max())
    check("fresh_directions_change_both_directional_terms_and_joint_GY",
          fw_gap > 0 and temporal_gap > 0 and gradient_gap > 0,
          full_window_value_gap=fw_gap, temporal_value_gap=temporal_gap,
          max_abs_GY_gap=gradient_gap)
    state["direction_sets"] = {
        "first_PCG64_seed": 640032128, "second_PCG64_seed": 640032129,
        "first_arrays_sha256": hash_arrays([*directions, tp]),
        "second_arrays_sha256": hash_arrays([*next_directions, next_temporal]),
    }
    for p2 in next_directions:
        p2.fill(0.)
    next_temporal.fill(0.)
    unchanged = api.joint_total_D(next_objective, Y, real, full)
    compare("fresh_step_constructor_directions_are_cloned_total",
            unchanged["total"], second["total"])
    compare("fresh_step_constructor_directions_are_cloned_GY", unchanged["GY"], second["GY"])
    stage("two_fresh_direction_sets_and_constructor_binding_passed")


    layout = equal
    sy, ry = list(layout.source_global), list(layout.real_global)
    identity_ids = np.arange(64) % len(real)
    identity = R63.core.canonical_REAL_torch(real[identity_ids])
    identity_bins = torch.from_numpy(real_regions[identity_ids]).to(device)
    source_psd = R63.old_loss.psd(source)[sy]
    source_h = torch.ones(320, dtype=torch.int64, device=device)
    leaf = Y.detach().clone().requires_grad_(True)
    output = torch.cat((leaf[sy], identity))
    raw = torch.cat((f["cached_raw"][sy], identity))
    original = R63.loss.DistributionLoss(stats, cfg, device)
    _, pieces, _ = original(output, raw, real[ry], real[identity_ids], identity_bins,
        source_psd, source_h, directions, source_psd-1e-8, source_psd+1e-8, tp)
    old_D = sum(cfg["loss_weights"][name] * pieces[name] for name in api.D_PIECE_ORDER)
    old_GY, = torch.autograd.grad(old_D, leaf)
    compare("joint_equal32_vs_original_R63_total", results["equal32_duplicates"]["total"],
            old_D, "equal_R63_f32_vs_f64_value")
    compare("joint_equal32_duplicate_scatter_vs_original_R63_GY",
            results["equal32_duplicates"]["GY"], old_GY,
            "equal_R63_f32_vs_f64_endpoint_gradient")
    del output, raw, old_D, pieces, leaf, old_GY

    common = R63.core.canonical_REAL_torch(real[:1])
    zero_y, zero_real = common.repeat(len(source), 1), common.repeat(len(real), 1)
    zero_full = api.joint_total_D(objective, zero_y, zero_real, full)
    compare("full_unequal_zero_coordinate_RMS_value", zero_full["pieces"]["coordinate_RMS"], 0.)
    zero_rms = objective.endpoint_gradients(zero_y, zero_real, full, terms=("coordinate_RMS",))
    compare("full_unequal_zero_coordinate_RMS_selected_gradient",
            zero_rms["gradients"]["coordinate_RMS"], torch.zeros_like(zero_y))
    state["full_unequal_identical_waveform_numerical_values"] = {
        k: float(v) for k, v in zero_full["pieces"].items()}
    zero = api.joint_total_D(objective, zero_y, zero_real, equal)
    for name in api.D_NAMES:
        compare("all_tied_zero_value_" + name, zero["pieces"][name], 0.)
    compare("all_tied_zero_total_GY_selected_branch", zero["GY"], torch.zeros_like(zero_y))

    class CastProbe(dg.FrozenDistributionObjective):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cast_records = []
        def _w1(self, a, b, *mass):
            self.cast_records.append((a.shape[1], str(a.dtype), str(b.dtype)))
            return super()._w1(a, b, *mass)
    probe = CastProbe(stats, cfg, directions, tp, source_regions, real_regions, device=device)
    probe_result = api.joint_total_D(probe, Y, real, full)
    cast_counts = Counter(a_dtype for _, a_dtype, _ in probe.cast_records)
    check("preserved_float32_feature_casts_and_float64_PSD_noise",
          cast_counts == {"torch.float32": 100, "torch.float64": 9} and
          all(a == b == ("torch.float64" if dimension in (1, 2) else "torch.float32")
              for dimension, a, b in probe.cast_records),
          calls_by_dtype=dict(cast_counts))
    compare("cast_probe_joint_GY_unchanged", probe_result["GY"], results["full_unequal"]["GY"])

    class WrongMass(dg.FrozenDistributionObjective):
        def _mass(self, layout, domain):
            if domain == "SOURCE":
                return super()._mass(layout, domain)
            weights = [self.source_mass[k] / len(layout.real[k])
                       for k in range(10) for _ in layout.real[k]]
            return FrozenMass.from_normalized(weights, label="DELIBERATELY_WRONG_REAL_MASS")
    wrong = WrongMass(stats, cfg, directions, tp, source_regions, real_regions, device=device)
    bad = api.joint_total_D(wrong, Y, real, full)
    value_gap = float((bad["total"] - results["full_unequal"]["total"]).abs())
    gradient_gap = float((bad["GY"] - results["full_unequal"]["GY"]).abs().max())
    check("joint_total_own_population_mass_negative_control",
          value_gap > tolerance["negative_wrong_mass_min_value_difference"] and
          gradient_gap > tolerance["negative_wrong_mass_min_gradient_difference"],
          value_gap=value_gap, max_abs_gradient_gap=gradient_gap)
    try:
        dg.FrozenDistributionObjective(dict(stats, REAL_counts=f["source_counts"]),
            cfg, directions, tp, source_regions, real_regions, device=device)
    except ValueError:
        check("wrong_original_REAL_population_counts_rejected", True)
    else:
        check("wrong_original_REAL_population_counts_rejected", False)
    stage("ties_zero_RMS_casts_and_own_masses_passed")

    graph_y, graph_raw = f["endpoint_all"]()
    direct = objective(graph_y, real, full, terms=api.D_NAMES)
    direct_GY, direct_parameter = torch.autograd.grad(
        direct.total, (graph_y, f["net"].coefficients))
    compare("joint_GY_vs_original_endpoint_same_graph", results["full_unequal"]["GY"], direct_GY)
    accumulator = torch.zeros_like(f["net"].coefficients, dtype=torch.float64)
    replay = []
    for a, b in f["schedule"]:
        y, raw = f["endpoint_chunk"](a, b)
        bitwise = (torch.equal(y.detach(), Y[a:b]) and
                   torch.equal(raw.detach(), f["cached_raw"][a:b]) and
                   torch.equal(R63.core.canonical_adc_torch(y.detach()),
                               R63.core.canonical_adc_torch(Y[a:b])) and
                   torch.equal((y.detach() > -.5) & (y.detach() < .5),
                               (Y[a:b] > -.5) & (Y[a:b] < .5)))
        check("joint_D_replay_chunk_%d_%d" % (a, b), bitwise)
        g, = torch.autograd.grad(y, f["net"].coefficients,
                                grad_outputs=results["full_unequal"]["GY"][a:b])
        accumulator += g.detach().double()
        replay.append({"start": a, "stop": b, "Y_raw_ADC_free_bitwise_equal": bitwise})
    compare("joint_D_parameter_VJP_float64_chunk_accumulation", accumulator,
            direct_parameter, "same_schedule_two_pass_parameter_gradient")
    check("joint_D_nonzero_toy_parameter_gradient",
          norm(accumulator) > tolerance["minimum_nonzero_gradient_norm"],
          gradient_norm=norm(accumulator))
    state["toy_joint_D_parameter_gradient"] = accumulator.cpu().tolist()
    state["same_schedule_replay"] = replay
    del graph_y, graph_raw, direct, direct_GY, direct_parameter, accumulator
    stage("joint_D_toy_endpoint_and_parameter_replay_passed")

    # Extra artificial C inputs explicitly frozen in FIXTURE_SPEC before execution.
    cstats = dict(stats, source_h_one_fraction=np.array([.625, .875]),
                  source_raw_valley_fraction=np.array([0., .05]),
                  valley_bounds=np.array([[.2, .7], [.2, .7]]))
    old_C = R63.loss.DistributionLoss(cstats, cfg, device)
    normalized_psd = (source_psd-source_psd.min())/(source_psd.max()-source_psd.min())
    target_psd = .45-.20*normalized_psd
    for region_index in (3, 4):
        ix = slice(region_index*32, (region_index+1)*32)
        target_psd[ix] = .2005+.0005*(torch.arange(32, device=device) % 4)
    out_source = torch.zeros((320, 480), dtype=torch.float64, device=device)
    out_source[:, 100:150] = 6*(1-target_psd[:, None])/50
    out_source[:, 150:400] = 6*target_psd[:, None]/250
    id_out = identity.clone()
    id_out[:, 420] += .005
    out_base = torch.cat((out_source, id_out))
    out_base[:, 450], out_base[:, 451] = .5, -.5
    raw_base = out_base.clone()
    raw_base[:, 450], raw_base[:, 451] = .75, -.73
    ch = ((torch.arange(320, device=device) % 2) == 0).to(torch.int64)
    low, high = source_psd-1e-8, source_psd+1e-8
    low, high = low.clone(), high.clone()
    low[::3], high[::3] = 0., 1.
    cargs = (real[ry], real[identity_ids], identity_bins, source_psd, ch,
             directions, low, high, tp)
    state["C_fixture_arrays_sha256"] = hash_arrays(
        [out_base, raw_base, source_psd, ch, low, high, identity_bins,
         cstats["source_h_one_fraction"], cstats["source_raw_valley_fraction"],
         cstats["valley_bounds"]])
    check("C_fixture_valid_range_positive_charge_and_outer_identity_regions",
          (out_base.abs() <= .5).all() and
          (R63.core.q_torch(R63.core.canonical_adc_torch(out_base)) > 0).all() and
          (identity_bins == -1).any() and (identity_bins == 8).any() and
          (raw_base.abs() > .5).any(),
          source_rows=320, identity_rows=64,
          active_output_rail_coordinates=int((out_base.abs() == .5).sum()))
    class CountC:
        def __init__(self, delegate):
            self.delegate, self.cfg, self.calls = delegate, delegate.cfg, 0
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.delegate(*args, **kwargs)
    counted_C = CountC(old_C)
    out, raw = out_base.clone().requires_grad_(), raw_base.clone().requires_grad_()
    c = api.original_C_microbatch(counted_C, out, raw, *cargs)
    ref_out, ref_raw = out_base.clone().requires_grad_(), raw_base.clone().requires_grad_()
    _, original_pieces, original_diag = old_C(ref_out, ref_raw, *cargs)
    direct_C = sum(cfg["loss_weights"][name] * original_pieces[name] for name in api.C_NAMES)
    check("C_calls_original_nine_terms_once_selects_four_without_subtraction",
          counted_C.calls == 1 and tuple(c["pieces"]) == api.C_NAMES and
          set(c["discarded_D_values"]) == set(api.D_NAMES) and
          all(not x.requires_grad for x in c["discarded_D_values"].values()) and
          c["total"].requires_grad, call_count=counted_C.calls)
    for name in api.C_NAMES:
        compare("C_original_value_" + name, c["pieces"][name], original_pieces[name])
    compare("C_original_four_term_sum", c["total"], direct_C)
    partials = torch.autograd.grad(c["total"], (out, raw), retain_graph=True)
    direct_partials = torch.autograd.grad(direct_C, (ref_out, ref_raw), retain_graph=True)
    for label, got, wanted in zip(("out", "raw"), partials, direct_partials):
        compare("C_original_independent_" + label + "_partial", got, wanted)
        check("C_nonzero_independent_" + label + "_partial",
              norm(got) > tolerance["minimum_nonzero_gradient_norm"], gradient_norm=norm(got))
    component_partial_norms = {}
    for name in api.C_NAMES:
        gout, graw = torch.autograd.grad(c["pieces"][name], (out, raw),
                                         retain_graph=True, allow_unused=True)
        component_partial_norms[name] = {"out": norm(gout), "raw": 0. if graw is None else norm(graw)}
        check("C_nonzero_value_and_output_derivative_" + name,
              float(c["pieces"][name].detach()) > 0 and
              norm(gout) > tolerance["minimum_nonzero_gradient_norm"],
              value=float(c["pieces"][name].detach()), **component_partial_norms[name])
        if name == "box_correction":
            compare("C_box_raw_and_out_partials_are_opposite", gout, -graw)
        else:
            check("C_raw_disconnected_" + name, graw is None)
    state["C_partial_norms"] = component_partial_norms

    # Independent assertions below are test oracles, never replacement C formulas.
    yp = R63.old_loss.psd(R63.core.canonical_adc_torch(out_base[:320], True))
    p = source_psd.detach().cpu().numpy()
    y = yp.detach().cpu().numpy()
    w = old_C.sw.detach().cpu().numpy()
    lo, hi = low.detach().cpu().numpy(), high.detach().cpu().numpy()
    ds = p[:, None]-p[None, :]
    pair = (w[:, None]*w[None, :])*(ds != 0)
    resolved = (lo[:, None] > hi[None, :]) | (hi[:, None] < lo[None, :])
    penalty = np.maximum(-np.sign(ds)*(y[:, None]-y[None, :]), 0.)
    denominator = float(pair.sum())
    resolved_denominator = float((pair*resolved).sum())
    numerator = float((pair*resolved*penalty).sum())
    expected_order = numerator/max(denominator, 1e-30)/cfg["PSD_loss_scale"]
    wrong_order = numerator/max(resolved_denominator, 1e-30)/cfg["PSD_loss_scale"]
    compare("C_order_original_all_nonzero_pair_denominator",
            c["pieces"]["source_order"], expected_order, "unequal_oracle_value")
    compare("C_order_denominator_diagnostic", c["diagnostics"]["original_nonzero_pair_mass"],
            denominator, "unequal_oracle_value")
    check("C_order_resolved_only_denominator_negative_control",
          0 < resolved_denominator < denominator and abs(expected_order-wrong_order) >
          tolerance["negative_wrong_mass_min_value_difference"],
          original_pair_mass=denominator, resolved_pair_mass=resolved_denominator,
          wrong_value_gap=abs(expected_order-wrong_order))

    expected_valleys, wrong_probabilities, per_row_nonlinear = [], [], []
    for j, k in enumerate((3, 4)):
        ix = slice(k*32, (k+1)*32)
        vlo, vhi = old_C.stats["valley_bounds"][j]
        smooth = torch.sigmoid((yp[ix]-vlo)/cfg["valley_smoothing"])*torch.sigmoid(
            (vhi-yp[ix])/cfg["valley_smoothing"])
        masked = smooth*(ch[ix] == 1)
        probability = masked.mean()/old_C.stats["source_h_one_fraction"][j]
        wrong_probability = masked.sum()/(ch[ix] == 1).sum()
        key = ("Q3", "Q4")[j] + "_h_one_soft_valley"
        compare("C_valley_original_32_slot_population_denominator_" + str(j),
                c["diagnostics"][key], probability, "unequal_oracle_value")
        wrong_probabilities.append(float((probability-wrong_probability).abs()))
        excess = torch.relu(probability-old_C.stats["source_raw_valley_fraction"][j]-
                            cfg["valley_delta"])/cfg["valley_delta"]
        expected_valleys.append(excess.square())
        per_row_excess = torch.relu(masked/old_C.stats["source_h_one_fraction"][j]-
            old_C.stats["source_raw_valley_fraction"][j]-cfg["valley_delta"])/cfg["valley_delta"]
        per_row_nonlinear.append(per_row_excess.square().mean())
    compare("C_valley_original_batch_nonlinear_reduction", c["pieces"]["source_valley"],
            torch.stack(expected_valleys).mean(), "unequal_oracle_value")
    check("C_valley_sampled_h_count_denominator_negative_control",
          min(wrong_probabilities) > tolerance["negative_wrong_mass_min_value_difference"],
          probability_gaps=wrong_probabilities)
    nonlinear_gap = float((torch.stack(expected_valleys).mean()-
                           torch.stack(per_row_nonlinear).mean()).abs())
    check("C_valley_mean_then_square_negative_control",
          nonlinear_gap > tolerance["negative_wrong_mass_min_value_difference"],
          wrong_reduction_value_gap=nonlinear_gap)

    ident_y = R63.core.canonical_adc_torch(out_base[320:], True)
    ident_real = R63.core.canonical_REAL_torch(real[identity_ids])
    scale = old_C.stats["V480_scale"][identity_bins+1]
    expected_identity = ((ident_y-ident_real)/scale).square().mean()
    compare("C_identity_original_outer_region_scale", c["pieces"]["REAL_identity"],
            expected_identity)
    wrong_scale = old_C.stats["V480_scale"][identity_bins.clamp(0, 7)+1]
    wrong_identity = ((ident_y-ident_real)/wrong_scale).square().mean()
    identity_gap = float((expected_identity-wrong_identity).abs())
    check("C_identity_clipped_network_bins_negative_control",
          identity_gap > tolerance["negative_wrong_mass_min_value_difference"],
          wrong_identity_value_gap=identity_gap)
    correction = (out_base-raw_base).square().mean(1)
    expected_box = .5*((correction[:320]*old_C.sw).sum()+correction[320:].mean())/(.5**2)
    compare("C_box_original_source_weighted_identity_mean", c["pieces"]["box_correction"], expected_box)
    precision = R63.precision.artificial_precision_check(device=device)
    check("original_precision_order_overlap_touch_equal_and_finite_difference",
          precision["unchanged_original_nonzero_pair_denominator"] and
          precision["all_overlap_zero"] and precision["touching_endpoints_overlap"] and
          precision["equal_PSD_and_duplicate_parent_handling"],
          original_receipt=precision)
    state["C_four_values"] = {k: float(v.detach()) for k, v in c["pieces"].items()}
    state["D_full_five_values"] = {k: float(v) for k, v in results["full_unequal"]["pieces"].items()}
    state["C_total"] = float(c["total"].detach())
    stage("C_original_values_independent_partials_and_semantics_passed")
    state["limitations"] = [
        "Synthetic leaves and a four-parameter toy use the original endpoint code; no actual ResponseNetwork is executed here.",
        "Combined D+C actual-network VJP, clipping and Adam equivalence remain the separately assigned integrated proof.",
        "CUDA is callable but this receipt only establishes the selected device actually used.",
        "Shared float32 feature graphs may change first-order accumulation rounding; joint-versus-five uses the pre-frozen R64 mixed-precision gradient category.",
        "Stable-tie STE and zero-RMS selected derivatives are algorithmic first-order branches, not hard-ADC finite-difference gradients.",
    ]

