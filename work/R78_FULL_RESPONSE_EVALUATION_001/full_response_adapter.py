"""Full frozen Q+shape adapter. No import-time Torch, I/O, RNG, or execution."""
from __future__ import annotations

import importlib
from pathlib import Path
import platform
import sys

import numpy as np

from evaluation_io import (AdmissionError, adc_representations, array_sha, binding,
                           check_payload, feature_Q_PSD, finite_number, owned_runtime,
                           preflight_training, q_window, raw_bins, raw_bins_mVns, require,
                           verify_file, wave_rows)


def state_digest(state):
    """Exact train_full.state_hash spelling, including statistics buffers."""
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_exact(a, b):
    import torch
    return (set(a) == set(b) and all(
        isinstance(a[k], torch.Tensor) and isinstance(b[k], torch.Tensor) and
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and
        a[k].detach().cpu().contiguous().numpy().tobytes() ==
        b[k].detach().cpu().contiguous().numpy().tobytes()
        for k in a))


def load_runtime_bundle(c):
    directory = Path(c["runtime_source_directory"]).resolve()
    names = ["public_charge_head", "physical_operators", "common_response", "waveform_response"]
    for name in names:
        expected = verify_file(c["runtime_sources"][name + ".py"], name)
        require(expected.resolve().parent == directory, "runtime package inconsistent")
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == expected.resolve(),
                    "runtime module already imported from an unbound historical directory")
    sys.path.insert(0, str(directory))
    try:
        modules = {name: importlib.import_module(name) for name in names}
    finally:
        sys.path.pop(0)
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == Path(c["runtime_sources"][name + ".py"]["path"]).resolve(),
                "runtime source origin mismatch")
    return modules


class FullResponseAdapter:
    """Only use in separately registered generation/replay, under owned worker.

    raw/white/z/style have the same original row order. Identity and domain are
    not model arguments. white is explicitly rounded once to float32, exactly
    as the current R74 postcheck; no new noise is drawn.
    """
    def __init__(self, config, entry_name):
        self.config = config
        # This admission deliberately precedes Torch import and every PT/NPZ.
        self.training_admission = preflight_training(config)
        self.resource_handshake = owned_runtime(config, entry_name, Path(__file__).parent)
        import torch
        self.torch = torch
        expected = config["expected_environment"]
        require(sys.version == expected["python"] and np.__version__ == expected["numpy"] and
                torch.__version__ == expected["torch"], "bound environment mismatch")
        require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == expected["GPU"],
                "bound GPU unavailable; no alternative device/model")
        torch.set_num_threads(config["CPU_threads"])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_per_process_memory_fraction(
            config["GPU_limit_bytes"] / torch.cuda.get_device_properties(0).total_memory, 0)
        bundle = load_runtime_bundle(config)
        self.response = bundle["waveform_response"]
        self.head_module = bundle["public_charge_head"]
        model = config["model"]
        paths = {key: verify_file(row, "model." + key) for key, row in model.items()}
        frozen = torch.load(paths["frozen_Q_weights"], map_location="cpu", weights_only=True)
        original = torch.load(paths["original_Q_weights"], map_location="cpu", weights_only=True)
        require(state_exact(frozen, original),
                "FROZEN_Q_HEAD and original Q state differ; container SHA equality is not required")
        shape_state = torch.load(paths["shape_weights"], map_location="cpu", weights_only=True)
        require(state_digest(shape_state) == self.training_admission["final_state_SHA256"],
                "completed shape parameter/buffer state mismatch")
        with np.load(paths["shape_statistics"], allow_pickle=False) as packet:
            names = ["feature_center", "feature_scale", "mu", "scale", "edges", "p480", "tau2"]
            require(set(names) <= set(packet.files), "full frozen shape/physical statistics absent")
            stats = {key: packet[key].copy() for key in names}
        for key in ["feature_center", "feature_scale"]:
            saved = shape_state[key].detach().cpu().numpy()
            require(saved.dtype == stats[key].dtype and saved.shape == stats[key].shape and
                    saved.tobytes() == stats[key].tobytes(),
                    "shape statistics file does not match completed state buffers")
        self.device = torch.device(config["inference_device"])
        # Constructors consume only temporary CPU initialization RNG; loaded
        # weights determine inference. Preserve caller RNG and never draw noise.
        with torch.random.fork_rng(devices=[]):
            head = self.head_module.PublicConditionalChargeHead(
                frozen["feature_center"], frozen["feature_scale"])
            shape = self.response.ShapeResidualNetwork(
                shape_state["feature_center"], shape_state["feature_scale"], hidden=128)
        head.load_state_dict(frozen, strict=True)
        shape.load_state_dict(shape_state, strict=True)
        self.head = head.to(self.device).double().eval().requires_grad_(False)
        self.shape = shape.to(self.device).eval().requires_grad_(False)
        require(all(p.dtype == torch.float32 for p in self.shape.parameters()) and
                self.shape.feature_center.dtype == torch.float64 and
                all(p.dtype == torch.float64 for p in self.head.parameters()),
                "registered model/physical precision changed")
        self.edges = np.asarray(stats["edges"], dtype=np.float64)
        self.stats = [torch.as_tensor(stats[k], device=self.device, dtype=torch.float64)
                      for k in ["mu", "scale", "edges", "p480"]]
        self.tau2 = finite_number(float(stats["tau2"]), "tau2")
        require(self.tau2 >= 0, "negative frozen noise target")
        raw_bins(np.array([1.]), self.edges)  # Validate the frozen edges only.
        self.counts = {"Q_head_forward_calls": 0, "Q_head_forward_rows": 0,
                       "shape_forward_calls": 0, "shape_forward_rows": 0,
                       "Box_capture_calls": 0, "Box_capture_rows": 0,
                       "Box_bisection_iterations_per_batch_sum": 0, "Box_projected_row_iterations": 0,
                       "diagnostic_noise_FFT_calls": 0, "new_noise_draws": 0,
                       "training_updates": 0, "GPU_allocated_peak_bytes": 0,
                       "GPU_reserved_peak_bytes": 0}
        self.bindings = {"model_files": model, "runtime_sources": config["runtime_sources"],
                         "Q_state_equal_after_container_load": True,
                         "Q_state_SHA256": state_digest(frozen),
                         "shape_state_SHA256": state_digest(shape_state),
                         "shape_buffers_equal_bound_statistics": True,
                         "straight_through": False, "chunk": 256,
                         "same_SOURCE_REAL_public_path": True,
                         "unused_random_inputs": ["z", "style"],
                         "caller_constructor_RNG_preserved": True}
        del frozen, original, shape_state, stats

    def _gpu_check(self):
        torch = self.torch
        allocated = torch.cuda.max_memory_allocated(0)
        reserved = torch.cuda.max_memory_reserved(0)
        self.counts["GPU_allocated_peak_bytes"] = max(self.counts["GPU_allocated_peak_bytes"], allocated)
        self.counts["GPU_reserved_peak_bytes"] = max(self.counts["GPU_reserved_peak_bytes"], reserved)
        require(allocated <= self.config["GPU_limit_bytes"] and
                reserved <= self.config["GPU_limit_bytes"], "registered GPU allocated/reserved limit")

    def _batch(self, x_np, epsilon_np):
        torch = self.torch
        x = torch.as_tensor(x_np, device=self.device, dtype=torch.float64)
        epsilon32 = torch.as_tensor(epsilon_np, device=self.device, dtype=torch.float32)
        prepared = self.response.prepare_static(self.head, x, *self.stats, self.tau2)
        # Save head prediction before the waveform and independently of output.
        target_before = prepared["Qstar"].detach().cpu().numpy().copy()
        captured = []
        original_box = self.response.bounded_charge

        def capture_box(v, q):
            pre64 = original_box(v, q)  # One actual original Box invocation.
            captured.append((v, q, pre64))
            return pre64

        self.response.bounded_charge = capture_box
        try:
            pre32, canonical, raw, meta = self.response.forward_prepared(
                self.shape, prepared, epsilon32,
                straight_through=self.config["straight_through"])
        finally:
            self.response.bounded_charge = original_box
        require(len(captured) == 1 and captured[0][2].shape == x.shape,
                "one full pre64 capture required for every actual partial batch")
        raw_seen, target_seen, pre64 = captured[0]
        require(raw_seen is raw and torch.equal(target_seen, prepared["Qstar"]),
                "captured Box does not belong to this complete forward")
        a = self.head_module.window_charge
        change = ((raw < -.5) | (raw > .5)).any(1) | ((a(raw)-prepared["Qstar"]).abs() > 1e-12)
        self.counts["Q_head_forward_calls"] += 1
        self.counts["shape_forward_calls"] += 1
        self.counts["Q_head_forward_rows"] += len(x_np)
        self.counts["shape_forward_rows"] += len(x_np)
        self.counts["Box_capture_calls"] += 1
        self.counts["Box_capture_rows"] += len(x_np)
        self.counts["Box_bisection_iterations_per_batch_sum"] += 64 if bool(change.any()) else 0
        self.counts["Box_projected_row_iterations"] += 64 * int(change.sum())
        # Diagnostic reconstruction of the exact frozen innovation expression.
        # This is two extra FFTs, counted; it does not call either neural network
        # or change the primary response output. Old R65 also recomputed noise.
        eps64 = epsilon32.to(torch.float64)
        noise = torch.fft.irfft(prepared["noise_sqrt_p480"] *
                               torch.fft.rfft(eps64, norm="ortho"), n=480, norm="ortho")
        noise = (noise - prepared["mu_condition"] * a(noise)[:, None]) * \
                prepared["noise_amplitude"][:, None]
        self.counts["diagnostic_noise_FFT_calls"] += 2
        correction = pre64 - raw
        row_meta = {}
        for key, value in meta.items():
            if value.ndim == 0:
                require(key == "range_projection_fraction", "unmapped scalar metadata: " + key)
                continue  # Overall fraction is derived from all per-row active flags.
            require(value.shape[0] == len(x_np), "partial batch lost row metadata: " + key)
            row_meta[key] = value.detach().cpu().numpy().copy()
        additional = {
            "predicted_target_Q_Vns": target_before,
            "input_variance100_V2": x[:, :100].var(1, unbiased=True),
            "innovation_variance100_actual_V2": noise[:, :100].var(1, unbiased=True),
            "initial_variance100_actual_V2": (prepared["lifted"]+noise)[:, :100].var(1, unbiased=True),
            "noise_completion_triggered": prepared["noise_amplitude_squared"] > 0,
            "pre_constraint_out_of_range_sample_count": ((raw < -.5) | (raw > .5)).sum(1),
            "constraint_projection_maxabs_V": correction.abs().amax(1),
            "constraint_projection_rail_sample_count": (pre64.abs() == .5).sum(1),
            "pre_constraint_charge_relative_error": (a(raw)-prepared["Qstar"]).abs()/prepared["Qstar"],
            "Box_projection_changed": change,
        }
        for key, value in additional.items():
            row_meta[key] = value if isinstance(value, np.ndarray) else value.detach().cpu().numpy().copy()
        result = {
            "pre_ADC_V": pre32.detach().cpu().numpy().copy(),
            "waveforms_V": canonical.detach().cpu().numpy().copy(),
            "raw_preconstraint_V": raw.detach().cpu().numpy().copy(),
            "pre_constraint_projected_f64_V": pre64.detach().cpu().numpy().copy(),
            "lifted_input_V": prepared["lifted"].detach().cpu().numpy().copy(),
            "projected_colored_innovation_V": noise.detach().cpu().numpy().copy(),
            **row_meta,
        }
        self._gpu_check()
        return result

    def infer(self, raw, actual_white, z, style):
        n = len(raw)
        require(n > 0, "empty response population")
        wave_rows(raw, n, "raw public input")
        wave_rows(actual_white, n, "actual frozen white")
        require(isinstance(z, np.ndarray) and isinstance(style, np.ndarray) and
                z.shape[0] == n and style.shape[0] == n, "original unused random rows missing")
        consumed = np.ascontiguousarray(actual_white, dtype=np.float32)
        require(np.isfinite(consumed).all(), "float32 epsilon conversion nonfinite")
        parts = []
        with self.torch.no_grad():
            for start in range(0, n, 256):
                stop = min(start + 256, n)
                parts.append(self._batch(raw[start:stop], consumed[start:stop]))
        keys = set(parts[0])
        require(all(set(part) == keys for part in parts), "partial-batch schema drift")
        result = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
        codes, strict = adc_representations(result["waveforms_V"])
        eval_qmV, eval_psd = feature_Q_PSD(strict)
        actual_pre_error = np.abs(q_window(result["pre_ADC_V"])-
                                  result["predicted_target_Q_Vns"])/result["predicted_target_Q_Vns"]
        require(np.isfinite(actual_pre_error).all() and (actual_pre_error <= 1e-5).all(),
                "actual saved pre32 Qstar implementation error failed")
        result.update(ADC_integer_codes=codes, evaluation_canonical_V=strict,
                      evaluation_canonical_Q_Vns=eval_qmV/1000.,
                      evaluation_canonical_Q_mVns=eval_qmV, evaluation_canonical_PSD=eval_psd,
                      evaluation_output_bin_raw=raw_bins_mVns(eval_qmV, self.edges),
                      saved_preADC_relative_Qstar_error_f32=actual_pre_error,
                      response_applied_all_rows=np.ones(n, dtype=np.bool_))
        check_payload(result, n)
        return result, {"rows": n, "batches": (n+255)//256,
                        "last_batch_rows": (n-1)%256+1,
                        "actual_white_original_SHA256": array_sha(actual_white),
                        "actual_white_consumed_float32_SHA256": array_sha(consumed),
                        "z_SHA256": array_sha(z), "style_SHA256": array_sha(style),
                        "z_style_unused": True,
                        "range_projection_fraction": float(result["range_projection_active"].mean()),
                        "Box_capture_calls": (n+255)//256, "Box_capture_rows": n,
                        "strict_integer_codes_from_actual_pipeline_output": True,
                        "canonical_voltage_and_integer_code_equality_separate": True,
                        "physical_noise_split": "unknown"}
