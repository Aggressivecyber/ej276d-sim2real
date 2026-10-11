"""Guarded artificial audit of R71 noise/reference code, no scientific data."""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    work = root / "work/R71_REAL_CHARGE_FULL_TRAIN_001"
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r71_noise_audit_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(output)
    started = time.perf_counter()
    sys.path.insert(0, str(work))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import common_response as response
    import numpy_reference as independent
    import public_charge_head as charge
    from check_public_charge_head import _fixtures

    checks, notes, measurements = [], [], {}
    loaded = [Path(response.__file__), Path(independent.__file__),
              Path(charge.__file__), work / "physical_operators.py", guard_path]
    source_rows = [{"path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in loaded]

    def require(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    exit_code, error = 1, None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(710023)
            x, mu, scale, edges = _fixtures()
            m = charge.normalized_template_bank(mu)[[0, 2, 5, 7]]
            a = np.zeros(480); a[:100] = -3; a[100:400] = 1
            B = np.eye(480)[:100]
            B = B - B.mean(0, keepdims=True)
            index = np.arange(480)
            rng = np.random.Generator(np.random.PCG64(710023))
            spectra = [np.ones(241), np.linspace(0.2, 2.3, 241),
                       0.1 + np.exp(-np.arange(241) / 18.0),
                       0.2 + rng.uniform(0, 1, 241)]
            worst = 0.0
            fft_worst = 0.0
            for si, p in enumerate(spectra):
                k = np.fft.irfft(p, n=480)
                K = k[(index[:, None] - index[None, :]) % 480]
                # Deterministic 480 independent basis innovations: no sampling
                # approximation, explicitly checks the two ortho FFT factors.
                A = np.fft.irfft(np.sqrt(p) * np.fft.rfft(np.eye(480), axis=1, norm="ortho"),
                                 n=480, axis=1, norm="ortho")
                fft_error = np.max(np.abs(A.T @ A - K))
                fft_worst = max(fft_worst, float(fft_error))
                require(f"ortho_FFT_covariance_spectrum_{si}", fft_error < 1e-12)
                dense = []
                for row in m.numpy():
                    P = np.eye(480) - row[:, None] * a[None, :]
                    BP = B @ P
                    dense.append(np.trace(BP @ K @ BP.T) / 99.0)
                dense = np.array(dense)
                actual = response.projected_noise_variance(m, p).numpy()
                difference = np.max(np.abs(actual - dense))
                worst = max(worst, float(difference))
                require(f"analytic_matches_dense_projected_ddof1_spectrum_{si}",
                        np.allclose(actual, dense, atol=1e-12, rtol=1e-12))
            measurements["analytic_dense_max_absolute_variance_error"] = worst
            measurements["ortho_FFT_covariance_max_absolute_error"] = fft_worst

            p = spectra[2]
            q = torch.tensor([12.0, 48.0], dtype=torch.float64, requires_grad=True)
            def variance_at(qvalue):
                row = charge.interpolate_charge_statistics(qvalue, mu, scale, edges)["unit_template"]
                return response.projected_noise_variance(row, p)
            actual_gradient = torch.autograd.grad(variance_at(q).sum(), q)[0]
            h = 1e-5
            fd = (variance_at(q.detach() + h) - variance_at(q.detach() - h)) / (2 * h)
            require("continuous_Qstar_kappa_gradient", torch.allclose(actual_gradient, fd, atol=1e-8, rtol=1e-6))
            measurements["kappa_Q_derivative_max_absolute_fd_error"] = (actual_gradient - fd).abs().max().item()

            amp = torch.tensor([0.0, 0.25, 2.0], dtype=torch.float64, requires_grad=True)
            kappa = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
            active = amp > 0
            safe = torch.where(active, torch.sqrt(torch.where(active, amp / kappa, torch.ones_like(amp))),
                               torch.zeros_like(amp))
            require("proposed_safe_sqrt_forward_equals_original", torch.equal(safe, torch.sqrt(amp / kappa)))
            ga, gk = torch.autograd.grad(safe.sum(), (amp, kappa))
            require("proposed_safe_sqrt_finite_backward_inactive_zero", torch.isfinite(ga).all() and
                    torch.isfinite(gk).all() and ga[0] == 0 and gk[0] == 0)
            unsafe_amp = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
            unsafe_grad = torch.autograd.grad((torch.sqrt(unsafe_amp) * 0.0).sum(), unsafe_amp)[0]
            require("original_zero_sqrt_backward_hazard_reproduced", torch.isnan(unsafe_grad).all())

            head = charge.PublicConditionalChargeHead(torch.zeros(480), torch.ones(480)).double()
            with torch.no_grad():
                head.output.weight.copy_(torch.linspace(-0.035, 0.045, 32, dtype=torch.float64).reshape(2, 16))
                head.output.bias.copy_(torch.tensor([0.08, -0.02], dtype=torch.float64))
            eps = rng.standard_normal(x.shape)
            state = {key: value.detach().numpy().copy() for key, value in head.state_dict().items()}
            for tau in [0.0, 0.002]:
                pre, adc, meta = response.common_response(head, x, eps, mu, scale, edges, p, tau)
                npre, nadc, nqt = independent.reference(state, x.numpy(), eps, mu.numpy(), scale.numpy(),
                                                      edges.numpy(), p, tau)
                require(f"independent_Qstar_tau_{tau}", np.allclose(meta["Qstar"].detach().numpy(), nqt, atol=1e-12, rtol=1e-12))
                require(f"independent_preADC_tau_{tau}", np.array_equal(pre.detach().numpy(), npre))
                require(f"independent_ADC_tau_{tau}", np.array_equal(adc.detach().numpy(), nadc))
            zero_pre, _, _ = response.common_response(head, x, eps, mu, scale, edges, p, 0.0)
            gradients = torch.autograd.grad(charge.window_charge(zero_pre).sum(), tuple(head.parameters()))
            measurements["loaded_common_response_zero_amp_backward_all_finite"] = all(
                bool(torch.isfinite(value).all()) for value in gradients)
            if not measurements["loaded_common_response_zero_amp_backward_all_finite"]:
                notes.append("Loaded common_response still has zero-amplitude sqrt backward NaNs; no_grad generate is unaffected.")

            underflow_state = {key: value.copy() for key, value in state.items()}
            underflow_state["output.weight"][:] = 0
            underflow_state["output.bias"][:] = [0.1, -1000.0]
            try:
                independent.reference(underflow_state, x.numpy(), eps, mu.numpy(), scale.numpy(), edges.numpy(), p, 0.0)
            except (ValueError, FloatingPointError):
                measurements["NumPy_rejects_logit_slope_underflow"] = True
            else:
                measurements["NumPy_rejects_logit_slope_underflow"] = False
                notes.append("NumPy reference accepts exp(b)==0 unlike the public head failure contract.")
            bad_eps = eps.copy(); bad_eps[0, 0] = np.nan
            try:
                independent.reference(state, x.numpy(), bad_eps, mu.numpy(), scale.numpy(), edges.numpy(), p, 0.0)
            except (ValueError, FloatingPointError):
                measurements["NumPy_rejects_nonfinite_epsilon"] = True
            else:
                measurements["NumPy_rejects_nonfinite_epsilon"] = False
                notes.append("NumPy reference returns nonfinite pre/ADC for nonfinite epsilon without raising.")
        exit_code = 0
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    resources.update(guard.memory())
    resources.update({"exit_code": exit_code, "torch_threads": torch.get_num_threads(),
                      "torch_interop_threads": torch.get_num_interop_threads(),
                      "elapsed_wall_seconds": time.perf_counter() - started,
                      "evidence_level": "artificial_native_CPU_only", "CUDA_executed": False,
                      "science_arrays_weights_read": False, "full_waveform_network_run": False})
    result = {"status": "PASS" if exit_code == 0 else "FAIL", "checks": checks,
              "check_count": len(checks), "measurements": measurements, "notes": notes,
              "error": error, "sources": source_rows, "resources": resources,
              "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (output / "COMMON_RESPONSE_ARTIFICIAL_REVIEW.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": len(checks),
                      "measurements": measurements, "notes": notes, "error": error,
                      "peak_RSS_bytes": resources["peak_RSS_bytes"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
