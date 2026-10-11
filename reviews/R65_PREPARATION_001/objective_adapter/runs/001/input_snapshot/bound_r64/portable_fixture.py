"""Portable form of the frozen 69-check R64 artificial fixture.

Entry: run_check(device='cpu', adapter_root=Path(.../'distribution_adapter')).
The caller owns CPU/GPU/RSS guards, torch backend/thread configuration, -B,
logging and result writes.  This module reads only the explicit hash-bound
adapter Python/config text files, never scientific arrays or checkpoints.

The fixture's toy coefficients, waveforms, counts, PCG64 draws, checks and
tolerances are unchanged.  Every tensor feature/standardization/projection and
both comparison arms use the selected device.  Only detached empirical-law
columns needed by the generic W1 / independent CDF oracle move to CPU.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ORIGINAL_FIXTURE_SOURCE_SHA256 = "5273878b94ef509e466a0ed5250b4f6c3b7b87ac4e87cee6c10b48fa06126c6e"
ORIGINAL_CPU_RECEIPT_SHA256 = "9fc29894450691905490b7f6a659e5c9d781e952a0fb18584d06dc71bd8bb3a1"
ORIGINAL_CPU_FIXTURE_ARRAYS_SHA256 = "8358df412f174e62fad3d848e38e50a6d39f906b6099b9c36696c5c56d2c3f1e"
FROZEN_TOLERANCES_SHA256 = "24c3c78dad18846fd56180fccfff19a1e0f835908f649ab2eecae50fe43c266c"
EXPECTED_ADAPTER_HASHES = {'original_bindings.py': '1d2e66275f1a52e5079775928046cd19791980049deda285daee3dacaf60907f', 'weighted_empirical_w1.py': 'c30c6e3d766879ffe0e0396238f843d4c22473a2083f6496ccfe4ba9525474b3', 'distribution_gradient.py': 'ed627502a736aa3f8fcab63ad2c98d18c5750b5c3d8f4f331ef072880343c965', 'SOURCE_BINDINGS.json': '971cfb7ad6c092740c98cb7be508e4077a02507e626bcc074f80bf6abe1b733d', 'FROZEN_TOLERANCES.json': '24c3c78dad18846fd56180fccfff19a1e0f835908f649ab2eecae50fe43c266c', 'r63_original/network.py': 'f7cce14e3f91c1682265d8559c2fc85cc5d604941f1637fc89372a3e3e255cd7', 'r63_original/noise_core.py': '5242214124d9f493c9cd25ff958ac26933a5115d76153caa70970ae721c9dc10', 'r63_original/source_flow_core.py': '3a9228b0d9fa30018b6f7f747e990e6339ed2e375a3981cdf25f79b6669b407d', 'r63_original/precision_order.py': 'f92fea23d677405475679b53b984c1b3c75f3a21da84079c899931b629369eb2', 'r63_original/R61_distribution_loss.py': '2dc13fda9817e1e0a71520b42ad6eb611c78151885cbae730e293556101ed27a', 'r63_original/distribution_loss.py': '9aa98106beae30993860c77fb23c97b9d60790150578b7a34d8ad8715efb2e26', 'r63_original/config.json': '9e5fcfa7330e400b2f159e30ad0500fe2ab7fd05c41c9ecb57e34a5cd67134c5'}
EXPECTED_TEST_NAMES = ('R63_equal32_value_full_window', 'R63_equal32_gradient_full_window', 'R63_equal32_value_PSD', 'R63_equal32_gradient_PSD', 'R63_equal32_value_noise', 'R63_equal32_gradient_noise', 'R63_equal32_value_coordinate_RMS', 'R63_equal32_gradient_coordinate_RMS', 'R63_equal32_value_temporal_joint', 'R63_equal32_gradient_temporal_joint', 'uniform_fastpath_vs_generic_full_value_full_window', 'uniform_fastpath_vs_generic_full_gradient_full_window', 'uniform_fastpath_vs_generic_full_value_PSD', 'uniform_fastpath_vs_generic_full_gradient_PSD', 'uniform_fastpath_vs_generic_full_value_noise', 'uniform_fastpath_vs_generic_full_gradient_noise', 'uniform_fastpath_vs_generic_full_value_coordinate_RMS', 'uniform_fastpath_vs_generic_full_gradient_coordinate_RMS', 'uniform_fastpath_vs_generic_full_value_temporal_joint', 'uniform_fastpath_vs_generic_full_gradient_temporal_joint', 'uniform_fastpath_vs_generic_equal32_value_full_window', 'uniform_fastpath_vs_generic_equal32_gradient_full_window', 'uniform_fastpath_vs_generic_equal32_value_PSD', 'uniform_fastpath_vs_generic_equal32_gradient_PSD', 'uniform_fastpath_vs_generic_equal32_value_noise', 'uniform_fastpath_vs_generic_equal32_gradient_noise', 'uniform_fastpath_vs_generic_equal32_value_coordinate_RMS', 'uniform_fastpath_vs_generic_equal32_gradient_coordinate_RMS', 'uniform_fastpath_vs_generic_equal32_value_temporal_joint', 'uniform_fastpath_vs_generic_equal32_gradient_temporal_joint', 'full_unequal_oracle_value_full_window', 'full_unequal_oracle_gradient_full_window', 'full_unequal_oracle_value_PSD', 'full_unequal_oracle_gradient_PSD', 'full_unequal_oracle_value_noise', 'full_unequal_oracle_gradient_noise', 'full_unequal_oracle_value_coordinate_RMS', 'full_unequal_oracle_gradient_coordinate_RMS', 'full_unequal_oracle_value_temporal_joint', 'full_unequal_oracle_gradient_temporal_joint', 'zero_RMS_exact_zero_value_and_selected_gradient', 'duplicates_original_slot_scatter_vs_preparation', 'duplicate_exact_global_mass_scatter', 'uniform_fastpath_vs_generic_equal128_value_full_window', 'uniform_fastpath_vs_generic_equal128_gradient_full_window', 'uniform_fastpath_vs_generic_equal128_value_PSD', 'uniform_fastpath_vs_generic_equal128_gradient_PSD', 'uniform_fastpath_vs_generic_equal128_value_noise', 'uniform_fastpath_vs_generic_equal128_gradient_noise', 'uniform_fastpath_vs_generic_equal128_value_coordinate_RMS', 'uniform_fastpath_vs_generic_equal128_gradient_coordinate_RMS', 'uniform_fastpath_vs_generic_equal128_value_temporal_joint', 'uniform_fastpath_vs_generic_equal128_gradient_temporal_joint', 'wrong_microbatch_major_layout_rejected', 'wrong_population_mass_binding_rejected', 'wrong_global_mass_detectably_changes_values_and_gradients', 'empty_duplicate_and_C_gradient_requests_rejected', 'same_schedule_firstpass_vs_singlegraph_Y', 'two_pass_parent_endpoint_gradient_full_window', 'two_pass_parameter_gradient_full_window', 'two_pass_parent_endpoint_gradient_PSD', 'two_pass_parameter_gradient_PSD', 'two_pass_parent_endpoint_gradient_noise', 'two_pass_parameter_gradient_noise', 'two_pass_parent_endpoint_gradient_coordinate_RMS', 'two_pass_parameter_gradient_coordinate_RMS', 'two_pass_parent_endpoint_gradient_temporal_joint', 'two_pass_parameter_gradient_temporal_joint', 'two_pass_total_parameter_gradient')


class ArtificialCheckFailure(RuntimeError):
    """Failure with a JSON-compatible partial receipt for the owned worker."""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt["error_type"] + ": " + receipt["error"])


def _verified_inputs(adapter_root):
    root = Path(adapter_root).resolve(strict=True)
    manifest = []
    for relative, expected in EXPECTED_ADAPTER_HASHES.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Frozen artificial adapter binding mismatch: " + relative)
        manifest.append({"file": relative, "sha256": actual})
    tol_bytes = (root / "FROZEN_TOLERANCES.json").read_bytes()
    if hashlib.sha256(tol_bytes).hexdigest() != FROZEN_TOLERANCES_SHA256:
        raise RuntimeError("Frozen tolerance binding mismatch")
    return root, json.loads(tol_bytes.decode("utf-8-sig")), manifest


@contextmanager
def _adapter_modules(root):
    """Bind this explicit adapter temporarily without modifying sys.path.

    original_bindings itself loads only the checked definition modules.  Restore
    all its temporary module keys as well, so CPU then CUDA calls do not reuse a
    stale objective/module cache or replace the worker's own adapter imports.
    This scoped loader is intended for the single owned supervisor thread.
    """
    names = ("original_bindings", "weighted_empirical_w1", "distribution_gradient")
    original_names = tuple("_r64_preparation_original_" + n for n in
                           ("network", "noise_core", "source_flow_core", "precision_order",
                            "R61_distribution_loss", "distribution_loss"))
    keys = names + original_names
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in keys}
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
            if spec is None or spec.loader is None:
                raise ImportError(name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield (loaded["distribution_gradient"], loaded["original_bindings"].R63,
               loaded["weighted_empirical_w1"].FrozenMass)
    finally:
        for name, previous in saved.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _state_fingerprint(torch, np, device):
    """Read state only.  No seed, thread or backend setter is called."""
    legacy_np = np.random.get_state()
    answer = {
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_grad_enabled": torch.is_grad_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
        "torch_CPU_rng_sha256": hashlib.sha256(torch.get_rng_state().detach().cpu().numpy().tobytes()).hexdigest(),
        "numpy_legacy_rng": [legacy_np[0], hashlib.sha256(legacy_np[1].tobytes()).hexdigest(),
                             int(legacy_np[2]), int(legacy_np[3]), float(legacy_np[4])],
    }
    if device.type == "cuda":
        answer["torch_selected_CUDA_rng_sha256"] = hashlib.sha256(torch.cuda.get_rng_state(device).detach().cpu().numpy().tobytes()).hexdigest()
        answer["CUDA_current_device"] = torch.cuda.current_device()
    return answer


def run_check(device="cpu", *, adapter_root=None):
    """Run the same 69 artificial checks and return a detailed JSON dictionary.

    No writes, network access, process/resource controls, torch thread/backend
    setters or torch RNG operations occur here.  NumPy uses its original local
    PCG64(640032128), which does not consume NumPy's global random state.

    Default adapter_root is the adjacent distribution_adapter folder.  Explicit
    paths are preferred in the root worker.  Python must already run with -B.
    Failure raises ArtificialCheckFailure with a .receipt dict for the caller;
    input-binding failures raise before fixture construction.
    """
    if not sys.dont_write_bytecode:
        raise RuntimeError("Owned supervisor must launch Python with -B")
    if adapter_root is None:
        adapter_root = Path(__file__).resolve().parent / "distribution_adapter"
    root, tol, manifest = _verified_inputs(adapter_root)
    import numpy as np
    import torch
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("Artificial fixture supports explicit CPU/CUDA only")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA artificial check is unavailable")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Fixture expects the original torch.float32 default; caller must configure it before invocation")
    before = _state_fingerprint(torch, np, dev)
    state = {"tests": {}, "stages": []}
    started = time.monotonic()
    try:
        with _adapter_modules(root) as (dg, R63, FrozenMass):
            with torch.enable_grad():
                result = _run_fixture(dev, root, dg, R63, FrozenMass, tol,
                                      FROZEN_TOLERANCES_SHA256, manifest, state)
        after = _state_fingerprint(torch, np, dev)
        if before != after:
            raise RuntimeError("Artificial fixture changed caller thread/backend/RNG state")
        if tuple(result["tests"]) != EXPECTED_TEST_NAMES or len(result["tests"]) != 69:
            raise RuntimeError("Frozen 69-check identity/count mismatch")
        if not all(item["passed"] for item in result["tests"].values()):
            raise RuntimeError("Artificial check table contains a failure")
        result.update(
            portable_entry="portable_fixture.run_check",
            original_fixture_source_sha256=ORIGINAL_FIXTURE_SOURCE_SHA256,
            original_CPU_receipt_sha256=ORIGINAL_CPU_RECEIPT_SHA256,
            historical_CPU_fixture_arrays_sha256=ORIGINAL_CPU_FIXTURE_ARRAYS_SHA256,
            selected_device=str(dev),
            caller_state_before=before, caller_state_after=after,
            caller_state_preserved=True,
            torch_random_draws=0,
            fixture_hash_semantics=("Original ordered shape/dtype/raw-byte stream; tensors use detach().cpu().numpy(). "
                                    "It identifies this device's materialized fixture. Native CUDA sin/reductions may "
                                    "change bytes relative to CPU; CPU/CUDA bitwise equivalence is not a gate."),
            all_comparison_arms_same_selected_device=True,
            selected_device_CUDA_checked=dev.type == "cuda",
            resource_controls_and_result_writes_owned_by_supervisor=True,
            frozen_tolerance_resource_fields_for_provenance={key: tol[key] for key in
                                                            ("RSS_limit_bytes", "CPU_affinity_count", "torch_threads")},
            stages=state["stages"],
        )
        result["compact"] = {
            "status": result["status"], "passed": True, "checks": 69,
            "device": str(dev), "fixture_arrays_sha256": result["fixture_arrays_sha256"],
            "tolerance_sha256": result["tolerance_sha256_before_run"],
            "caller_state_preserved": True,
            "elapsed_seconds": result["elapsed_seconds"],
        }
        json.dumps(result, allow_nan=False)
        return result
    except Exception as exc:
        after = _state_fingerprint(torch, np, dev)
        partial = {
            "status": "ARTIFICIAL_FAILED_NOT_A_SCIENTIFIC_RESULT",
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "device": str(dev),
            "elapsed_seconds": time.monotonic() - started,
            "tests": state["tests"], "completed_test_entries": len(state["tests"]),
            "stages": state["stages"],
            "tolerance_sha256_before_run": FROZEN_TOLERANCES_SHA256,
            "original_fixture_source_sha256": ORIGINAL_FIXTURE_SOURCE_SHA256,
            "caller_state_before": before, "caller_state_after": after,
            "caller_state_preserved": before == after,
            "scientific_arrays_opened": 0, "scientific_weights_loaded": 0,
            "DEV_EVAL_CAL_reads": 0, "scientific_forward_or_gradient_runs": 0,
        }
        raise ArtificialCheckFailure(partial) from exc


# BEGIN GENERATED FROZEN FIXTURE BODY
def _run_fixture(device, ROOT, dg, R63, FrozenMass, TOL, TOL_SHA, manifest, state):
    START = time.monotonic()

    def stage(name):
        state['stages'].append({'stage': name, 'elapsed_s': round(time.monotonic() - START, 3)})
    import numpy as np
    import torch
    TESTS = state['tests']

    def compare(name, actual, expected, rule):
        a = torch.as_tensor(actual, dtype=torch.float64, device=device).detach()
        b = torch.as_tensor(expected, dtype=torch.float64, device=device).detach()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        atol, rtol = (TOL[rule]['atol'], TOL[rule]['rtol'])
        delta = (a - b).abs()
        passed = bool((delta <= atol + rtol * b.abs()).all())
        result = {'max_abs_error': float(delta.max()), 'reference_absmax': float(b.abs().max()), 'atol': atol, 'rtol': rtol, 'passed': passed}
        TESTS[name] = result
        assert passed, (name, result)
        return result
    source_counts = np.array([3, 4, 5, 2, 6, 3, 4, 5, 2, 4], dtype=np.int64)
    real_counts = np.array([2, 5, 3, 6, 4, 2, 5, 3, 6, 4], dtype=np.int64)
    source_regions = np.repeat(dg.CATEGORIES, source_counts)
    real_regions = np.repeat(dg.CATEGORIES, real_counts)
    t = np.arange(480, dtype=np.float64)

    def waveform(region, parent, real=False):
        if region == -1:
            code = np.full(480, 8192, dtype=np.float64)
            code[100:150], code[150:400] = (10100, 8750)
            return -0.5 + code / 16383
        j = parent if parent < 2 else parent % 2
        base = 0.0008 * np.sin(t * (0.11 + 0.013 * region) + j) + 0.00035 * np.cos(t * 0.19 + region)
        pulse = np.zeros(480)
        amp = 0.11 + 0.012 * region + 0.004 * j + (0.005 if real else 0.0)
        tau = 27.0 + 2.5 * region + 1.5 * j + (4.0 if real else 0.0)
        pulse[100:400] = amp * np.exp(-np.arange(300) / tau)
        return (base + pulse).astype(np.float32).astype(np.float64)
    source_np = np.stack([waveform(int(r), j) for r, n in zip(dg.CATEGORIES, source_counts) for j in range(n)])
    real_np = np.stack([waveform(int(r), j, True) for r, n in zip(dg.CATEGORIES, real_counts) for j in range(n)])
    source = torch.from_numpy(source_np).to(device)
    real = torch.from_numpy(real_np).to(device)
    q = R63.core.q_torch(source)
    bins = torch.from_numpy(np.clip(source_regions, 0, 7)).to(device)
    condition = torch.from_numpy(source_regions.astype(np.float64)).to(device)
    mu = torch.zeros((8, 480), dtype=torch.float64, device=device)
    mu[:, 100:400] = 1 / 300
    scale = torch.full((8, 480), 0.012, dtype=torch.float64, device=device)
    noise = 0.0004 * torch.sin(torch.arange(len(source), dtype=torch.float64, device=device)[:, None] + torch.arange(480, device=device)[None, :] * 0.131)
    noise = R63.core.project(noise, mu[bins])
    noise[condition < 0] = 0

    class ToyNetwork(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.tensor([0.08, 0.035, 0.4, 0.018], dtype=torch.float32, device=device))
            basis = torch.sin(torch.arange(480, dtype=torch.float32, device=device) * 0.041)
            basis[450], basis[451] = (28.0, -24.0)
            self.register_buffer('basis', basis)

        def forward(self, noisy, b, guide, h):
            y, g, hh = (noisy.float(), guide.float(), h.float()[:, None])
            c = self.coefficients
            return (hh >= 0).float() * (c[0] * y + c[1] * g + c[2] * self.basis[None, :] + c[3] * (b.float()[:, None] + hh))
    net = ToyNetwork().train()
    schedule = tuple(((k, min(k + 7, len(source))) for k in range(0, len(source), 7)))

    def endpoint_chunk(a, b):
        return R63.core.endpoint(net, source[a:b], noise[a:b], q[a:b], bins[a:b], condition[a:b], mu, scale)

    def endpoint_all():
        pairs = [endpoint_chunk(a, b) for a, b in schedule]
        return (torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))
    cfg = json.loads((ROOT / 'r63_original/config.json').read_text('utf8'))
    stats = {'source_counts': source_counts, 'REAL_counts': real_counts, 'noise_scale': np.array([0.0012, 0.2]), 'valley_bounds': np.array([[0.2, 0.7], [0.2, 0.7]]), 'source_h_one_fraction': np.ones(2), 'source_raw_valley_fraction': np.zeros(2)}
    for name, dim, value in zip(dg.NAMES, (480, 480, 479, 478), (0.15, 0.5, 0.25, 0.25)):
        stats[name + '_center'] = np.zeros((10, dim))
        stats[name + '_scale'] = value * (1 + 0.08 * np.arange(10)[:, None]) * (1 + 0.05 * np.sin(np.arange(dim)[None, :] * 0.03))
    stats['temporal_center'] = np.zeros((10, 31))
    stats['temporal_scale'] = np.tile(np.r_[0.005, np.full(30, 0.35)], (10, 1)) * (1 + 0.04 * np.arange(10)[:, None])
    rng = np.random.Generator(np.random.PCG64(640032128))
    directions = []
    for dim in (480, 480, 479, 478):
        p = rng.standard_normal((dim, 64))
        p /= np.linalg.norm(p, axis=0, keepdims=True)
        directions.append(p.astype(np.float32))
    tp = rng.standard_normal((31, 64))
    tp /= np.linalg.norm(tp, axis=0, keepdims=True)
    tp = tp.astype(np.float32)
    objective = dg.FrozenDistributionObjective(stats, cfg, directions, tp, source_regions, real_regions, device=device)
    full = dg.ParentLayout.full_reference(source_regions, real_regions)
    sslots = np.stack([rng.choice(np.flatnonzero(source_regions == r), size=(4, 32), replace=True) for r in dg.CATEGORIES])
    rslots = np.stack([rng.choice(np.flatnonzero(real_regions == r), size=(4, 32), replace=True) for r in dg.CATEGORIES])
    alayouts, blayout = dg.paired_layouts(sslots, rslots, source_regions, real_regions)
    fixture_hash = hashlib.sha256()
    for a in (source_np, real_np, source_counts, real_counts, noise.detach().cpu().numpy(), mu.detach().cpu().numpy(), scale.detach().cpu().numpy(), q.detach().cpu().numpy(), bins.detach().cpu().numpy(), condition.detach().cpu().numpy(), net.coefficients.detach().detach().cpu().numpy(), net.basis.detach().cpu().numpy(), *(np.asarray(stats[k]) for k in sorted(stats)), *directions, tp, sslots, rslots):
        fixture_hash.update(str(a.shape).encode())
        fixture_hash.update(str(a.dtype).encode())
        fixture_hash.update(a.tobytes())
    with torch.no_grad():
        cached_y, cached_raw = endpoint_all()
    assert bool((cached_y.abs() == 0.5).any()), 'Fixture must exercise active Box rails'
    assert bool((cached_y.abs() < 0.5).any()), 'Fixture must retain free coordinates'
    assert bool((q > 0).all())
    stage('fixture_constructed_only_from_artificial_constants')
    original = R63.loss.DistributionLoss(stats, cfg, device)
    layout = alayouts[0]
    sy, ry = (list(layout.source_global), list(layout.real_global))
    identity_ids = np.arange(64) % len(real)
    ident = R63.core.canonical_REAL_torch(real[identity_ids])
    ident_bins = torch.from_numpy(real_regions[identity_ids]).to(device)
    source_psd = R63.old_loss.psd(source)[sy]
    source_h = torch.ones(320, dtype=torch.int64, device=device)
    equal_leaf = cached_y.clone().requires_grad_(True)
    output = torch.cat([equal_leaf[sy], ident])
    raw = torch.cat([cached_raw[sy], ident])
    _, old, old_diag = original(output, raw, real[ry], real[identity_ids], ident_bins, source_psd, source_h, directions, source_psd - 1e-08, source_psd + 1e-08, tp)
    old_values = {name: float(value.detach()) for name, value in old.items()}
    old_gradients = {name: torch.autograd.grad(old[name], equal_leaf, retain_graph=True)[0].detach() for name in dg.D_NAMES}
    prepared = objective.endpoint_gradients(cached_y, real, layout)
    for name in dg.D_NAMES:
        compare('R63_equal32_value_' + name, prepared['values'][name], old_values[name], 'equal_R63_f32_vs_f64_value')
        compare('R63_equal32_gradient_' + name, prepared['gradients'][name], old_gradients[name], 'equal_R63_f32_vs_f64_endpoint_gradient')
        assert float(torch.linalg.vector_norm(prepared['gradients'][name])) > TOL['minimum_nonzero_gradient_norm']
    del output, old, old_gradients, equal_leaf
    stage('original_R63_equal32_values_and_algorithm_gradients_passed')

    def dense_oracle_columns(a, b, ma, mb):
        aa, bb = (a.detach().double().detach().cpu().numpy(), b.detach().double().detach().cpu().numpy())
        ww, gg, hh = ([], [], [])
        for j in range(aa.shape[1]):
            ix, iy = (np.argsort(aa[:, j], kind='stable'), np.argsort(bb[:, j], kind='stable'))
            ax = [ma.ticks[int(i)] * mb.denominator for i in ix]
            by = [mb.ticks[int(i)] * ma.denominator for i in iy]
            cx = [0]
            cy = [0]
            for v in ax:
                cx.append(cx[-1] + v)
            for v in by:
                cy.append(cy[-1] + v)
            denominator = ma.denominator * mb.denominator
            overlap = np.array([[max(0, min(cx[i + 1], cy[k + 1]) - max(cx[i], cy[k])) / denominator for k in range(len(iy))] for i in range(len(ix))])
            delta = aa[ix, j, None] - bb[iy, j][None, :]
            ww.append(math.fsum((float(z) for z in (overlap * np.abs(delta)).reshape(-1))))
            gx = np.zeros(len(ix))
            gy = np.zeros(len(iy))
            gx[ix] = (overlap * np.sign(delta)).sum(1)
            gy[iy] = -(overlap * np.sign(delta)).sum(0)
            gg.append(gx)
            hh.append(gy)
        ga = torch.from_numpy(np.stack(gg, axis=1)).to(device)
        gb = torch.from_numpy(np.stack(hh, axis=1)).to(device)
        return torch.tensor(ww, dtype=torch.float64, device=device) + ((a - a.detach()).double() * ga).sum(0) + ((b - b.detach()).double() * gb).sum(0)

    def oracle_full(y):
        v = R63.core.canonical_adc_torch(y, True)
        ref = R63.core.canonical_REAL_torch(real)
        sm = source_counts / source_counts.sum()
        full_terms = []
        rms_terms = []
        uniform = lambda n: FrozenMass.from_normalized(np.full(n, 1 / n), label='independent_uniform')
        for name, fy, fr, p in zip(dg.NAMES, R63.old_loss.feature_vectors(v), R63.old_loss.feature_vectors(ref), directions):
            f = []
            c = []
            for k, r in enumerate(dg.CATEGORIES):
                si = np.flatnonzero(source_regions == r)
                ri = np.flatnonzero(real_regions == r)
                a = ((fy[si] - torch.tensor(stats[name + '_center'][k], device=device)) / torch.tensor(stats[name + '_scale'][k], device=device)).float()
                b = ((fr[ri] - torch.tensor(stats[name + '_center'][k], device=device)) / torch.tensor(stats[name + '_scale'][k], device=device)).float()
                ma, mb = (uniform(len(si)), uniform(len(ri)))
                w = dense_oracle_columns(a, b, ma, mb)
                sp = dense_oracle_columns(a @ torch.from_numpy(p).to(device), b @ torch.from_numpy(p).to(device), ma, mb)
                f.append(0.5 * (w.mean() + sp.mean()))
                c.append(torch.linalg.vector_norm(w) / math.sqrt(len(w)))
            full_terms.append(sum((sm[k] * f[k] for k in range(10))))
            rms_terms.append(sum((sm[k] * c[k] for k in range(10))))
        yp, rp = (R63.old_loss.psd(v), R63.old_loss.psd(ref))
        overall = dense_oracle_columns(yp[:, None], rp[:, None], uniform(len(yp)), uniform(len(rp)))[0]
        cond = []
        for r in range(7):
            si = np.flatnonzero(source_regions == r)
            ri = np.flatnonzero(real_regions == r)
            cond.append(dense_oracle_columns(yp[si, None], rp[ri, None], uniform(len(si)), uniform(len(ri)))[0])
        nn = dense_oracle_columns(R63.old_loss.noise_features(v), R63.old_loss.noise_features(ref), uniform(len(v)), uniform(len(ref)))
        tf, tr = (R63.loss.temporal_features(v), R63.loss.temporal_features(ref))
        temporal = []
        for k, r in enumerate(dg.CATEGORIES):
            si = np.flatnonzero(source_regions == r)
            ri = np.flatnonzero(real_regions == r)
            a = ((tf[si] - torch.tensor(stats['temporal_center'][k], device=device)) / torch.tensor(stats['temporal_scale'][k], device=device)).float()
            b = ((tr[ri] - torch.tensor(stats['temporal_center'][k], device=device)) / torch.tensor(stats['temporal_scale'][k], device=device)).float()
            ma, mb = (uniform(len(si)), uniform(len(ri)))
            m = dense_oracle_columns(a, b, ma, mb)
            p = dense_oracle_columns(a @ torch.from_numpy(tp).to(device), b @ torch.from_numpy(tp).to(device), ma, mb)
            temporal.append(0.5 * (m.mean() + p.mean()))
        return dict(full_window=torch.stack(full_terms).mean(), coordinate_RMS=torch.stack(rms_terms).mean(), PSD=0.5 * (overall + torch.stack(cond).mean()) / 0.01, noise=(nn / torch.tensor(stats['noise_scale'], device=device)).mean(), temporal_joint=sum((sm[k] * temporal[k] for k in range(10))))
    full_result = objective.endpoint_gradients(cached_y, real, full)
    generic_objective = dg.FrozenDistributionObjective(stats, cfg, directions, tp, source_regions, real_regions, uniform_fastpath=False, device=device)
    generic_full = generic_objective.endpoint_gradients(cached_y, real, full)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_full_value_' + name, full_result['values'][name], generic_full['values'][name], 'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_full_gradient_' + name, full_result['gradients'][name], generic_full['gradients'][name], 'unequal_oracle_endpoint_gradient')
    generic_equal = generic_objective.endpoint_gradients(cached_y, real, layout)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_equal32_value_' + name, prepared['values'][name], generic_equal['values'][name], 'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_equal32_gradient_' + name, prepared['gradients'][name], generic_equal['gradients'][name], 'unequal_oracle_endpoint_gradient')
    stage('uniform_integer_plan_fastpath_vs_generic_full_and_equal32_passed')
    oracle_leaf = cached_y.clone().requires_grad_(True)
    oracle = oracle_full(oracle_leaf)
    for name in dg.D_NAMES:
        g = torch.autograd.grad(oracle[name], oracle_leaf, retain_graph=True)[0].detach()
        compare('full_unequal_oracle_value_' + name, full_result['values'][name], oracle[name], 'unequal_oracle_value')
        compare('full_unequal_oracle_gradient_' + name, full_result['gradients'][name], g, 'unequal_oracle_endpoint_gradient')
    del oracle, oracle_leaf
    assert bool((full_result['diagnostics']['coordinate_RMS_by_region'][:, 0] == 0).all())
    assert bool((full_result['gradients']['coordinate_RMS'][source_regions == -1] == 0).all())
    TESTS['zero_RMS_exact_zero_value_and_selected_gradient'] = {'passed': True, 'zero_region': -1}
    stage('full_unequal_counts_dense_CDF_oracle_and_zero_RMS_passed')
    occurrence = cached_y[sy].clone().requires_grad_(True)
    old_result = original(torch.cat([occurrence, ident]), raw, real[ry], real[identity_ids], ident_bins, source_psd, source_h, directions, source_psd - 1e-08, source_psd + 1e-08, tp)[1]
    occurrence_total = sum((old_result[name] for name in dg.D_NAMES))
    gslot = torch.autograd.grad(occurrence_total, occurrence)[0]
    scatter = torch.zeros_like(cached_y).index_add_(0, torch.tensor(sy, device=device), gslot)
    compare('duplicates_original_slot_scatter_vs_preparation', prepared['total_gradient'], scatter, 'equal_R63_f32_vs_f64_endpoint_gradient')
    expanded_regions = source_regions[sy]
    expanded_real_regions = real_regions[ry]
    expanded_stats = dict(stats, source_counts=np.full(10, 32), REAL_counts=np.full(10, 32))
    duplicate_leaf = cached_y.clone().requires_grad_(True)
    detached_occurrence = duplicate_leaf[sy].detach().requires_grad_(True)
    exact_parent_result = objective(duplicate_leaf, real, layout)
    exact_g = torch.autograd.grad(exact_parent_result.total, duplicate_leaf)[0]
    original_column_w1 = dg.column_w1
    ma = objective._mass(layout, 'SOURCE')
    mb = objective._mass(layout, 'REAL')
    psg = R63.old_loss.psd(R63.core.canonical_adc_torch(detached_occurrence, True))
    pr = R63.old_loss.psd(R63.core.canonical_REAL_torch(real[ry]))
    scalar = dg.column_w1(psg[:, None], pr[:, None], ma, mb)[0]
    gocc = torch.autograd.grad(scalar, detached_occurrence)[0]
    pl = cached_y.clone().requires_grad_(True)
    val = dg.column_w1(R63.old_loss.psd(R63.core.canonical_adc_torch(pl, True))[sy, None], pr[:, None], ma, mb)[0]
    gparent = torch.autograd.grad(val, pl)[0]
    compare('duplicate_exact_global_mass_scatter', torch.zeros_like(cached_y).index_add_(0, torch.tensor(sy, device=device), gocc), gparent, 'duplicate_scatter_gradient')
    assert max(np.bincount(np.asarray(sy), minlength=len(source))) > 1
    stage('duplicate_parent_gradient_accumulation_passed')
    a_results = [prepared] + [objective.endpoint_gradients(cached_y, real, l) for l in alayouts[1:]]
    b_result = objective.endpoint_gradients(cached_y, real, blayout)
    generic_b = generic_objective.endpoint_gradients(cached_y, real, blayout)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_equal128_value_' + name, b_result['values'][name], generic_b['values'][name], 'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_equal128_gradient_' + name, b_result['gradients'][name], generic_b['gradients'][name], 'unequal_oracle_endpoint_gradient')
    a_values = {name: sum((x['values'][name] for x in a_results)) / 4 for name in dg.D_NAMES}
    a_gradients = {name: sum((x['gradients'][name] for x in a_results)) / 4 for name in dg.D_NAMES}
    for k in range(10):
        assert blayout.source[k] == tuple((int(i) for i in sslots[k].reshape(-1)))
        assert blayout.real[k] == tuple((int(i) for i in rslots[k].reshape(-1)))
    try:
        wrong_s = sslots.transpose(1, 0, 2).reshape(10, 128)
        wrong_r = rslots.transpose(1, 0, 2).reshape(10, 128)
        dg.ParentLayout.region_major(wrong_s, wrong_r, source_regions, real_regions)
        raise AssertionError('Incorrect microbatch-major layout was accepted')
    except ValueError as exc:
        TESTS['wrong_microbatch_major_layout_rejected'] = {'passed': True, 'error': str(exc)}
    try:
        badstats = dict(stats, REAL_counts=source_counts)
        dg.FrozenDistributionObjective(badstats, cfg, directions, tp, source_regions, real_regions, device=device)
        raise AssertionError('Wrong population masses were accepted')
    except ValueError as exc:
        TESTS['wrong_population_mass_binding_rejected'] = {'passed': True, 'error': str(exc)}

    class WrongMass(dg.FrozenDistributionObjective):

        def _mass(self, layout, domain):
            if domain == 'SOURCE':
                return super()._mass(layout, domain)
            weights = [self.source_mass[k] / len(layout.real[k]) for k in range(10) for _ in layout.real[k]]
            return FrozenMass.from_normalized(weights, label='DELIBERATE_WRONG_REAL_USES_SOURCE')
    wrong = WrongMass(stats, cfg, directions, tp, source_regions, real_regions, device=device).endpoint_gradients(cached_y, real, layout, terms=('PSD', 'noise'))
    wrong_gap = max((abs(wrong['values'][k] - prepared['values'][k]) for k in ('PSD', 'noise')))
    wrong_grad_gap = max((float((wrong['gradients'][k] - prepared['gradients'][k]).abs().max()) for k in ('PSD', 'noise')))
    assert wrong_gap > TOL['negative_wrong_mass_min_value_difference']
    assert wrong_grad_gap > TOL['negative_wrong_mass_min_gradient_difference']
    TESTS['wrong_global_mass_detectably_changes_values_and_gradients'] = {'passed': True, 'max_value_gap': wrong_gap, 'max_gradient_gap': wrong_grad_gap}
    for wrong_terms in ((), ('PSD', 'PSD'), ('source_order',)):
        try:
            objective.endpoint_gradients(cached_y, real, layout, terms=wrong_terms)
            raise AssertionError('Invalid or C term selection accepted')
        except ValueError:
            pass
    TESTS['empty_duplicate_and_C_gradient_requests_rejected'] = {'passed': True}
    stage('A_four32_B128_region_major_and_negative_controls_passed')
    graph_y, graph_raw = endpoint_all()
    graph = objective(graph_y, real, full)
    graph_parameter = {}
    graph_endpoint = {}
    for name in dg.D_NAMES:
        gg = torch.autograd.grad(graph.pieces[name], (graph_y, net.coefficients), retain_graph=True)
        graph_endpoint[name], graph_parameter[name] = (gg[0].detach(), gg[1].detach())
    compare('same_schedule_firstpass_vs_singlegraph_Y', cached_y, graph_y, 'same_schedule_two_pass_endpoint')
    del graph, graph_y, graph_raw
    replay_parameter = {name: torch.zeros_like(net.coefficients) for name in dg.D_NAMES}
    replay_checks = []
    for a, b in schedule:
        y, raw2 = endpoint_chunk(a, b)
        assert torch.equal(y.detach(), cached_y[a:b]), 'Projected endpoint replay drift'
        assert torch.equal(raw2.detach(), cached_raw[a:b]), 'Raw endpoint replay drift'
        assert torch.equal(R63.core.canonical_adc_torch(y.detach()), R63.core.canonical_adc_torch(cached_y[a:b])), 'ADC replay drift'
        assert torch.equal((y.detach() > -0.5) & (y.detach() < 0.5), (cached_y[a:b] > -0.5) & (cached_y[a:b] < 0.5)), 'Box active set drift'
        for j, name in enumerate(dg.D_NAMES):
            replay_parameter[name] += torch.autograd.grad(y, net.coefficients, grad_outputs=full_result['gradients'][name][a:b], retain_graph=j < len(dg.D_NAMES) - 1)[0].detach()
        replay_checks.append({'start': a, 'stop': b, 'Y_raw_codes_free_bitwise_equal': True})
    for name in dg.D_NAMES:
        compare('two_pass_parent_endpoint_gradient_' + name, full_result['gradients'][name], graph_endpoint[name], 'same_schedule_two_pass_endpoint')
        compare('two_pass_parameter_gradient_' + name, replay_parameter[name], graph_parameter[name], 'same_schedule_two_pass_parameter_gradient')
        assert float(torch.linalg.vector_norm(replay_parameter[name])) > TOL['minimum_nonzero_gradient_norm']
    compare('two_pass_total_parameter_gradient', sum(replay_parameter.values()), sum(graph_parameter.values()), 'same_schedule_two_pass_parameter_gradient')
    stage('original_endpoint_Box_STE_two_pass_VJP_passed')
    canonical = R63.core.canonical_adc_torch(cached_y)
    duplicate_waveforms = int(len(canonical) - len(torch.unique(canonical, dim=0)))
    assert duplicate_waveforms > 0
    receipt = {'status': 'ARTIFICIAL_PREPARATION_PASSED_NOT_SCIENTIFIC_REGISTRATION_OR_PASS', 'PID': os.getpid(), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(), 'elapsed_seconds': time.monotonic() - START, 'tolerance_sha256_before_run': TOL_SHA, 'fixture_arrays_sha256': fixture_hash.hexdigest(), 'fixture': {'waveform_dimension': 480, 'regions': list(dg.CATEGORIES), 'source_counts': source_counts.tolist(), 'REAL_counts': real_counts.tolist(), 'all_original_artificial_parents_in_F': True, 'independent_population_masses': True, 'same_parent_repeated_slots': True, 'SOURCE_unique_parents': len(source), 'REAL_unique_parents': len(real), 'ADC_duplicate_parent_waveforms': duplicate_waveforms, 'active_Box_rail_coordinates': int((cached_y.abs() == 0.5).sum()), 'exact_zero_RMS_region': -1, 'toy_network_nonzero_coefficients': net.coefficients.detach().tolist()}, 'tests': TESTS, 'original_nine_piece_values_32': old_values, 'D_F_values': full_result['values'], 'D_A_four32_values': a_values, 'D_B128_values': b_result['values'], 'D_F_parent_gradient_norms': {k: float(torch.linalg.vector_norm(v)) for k, v in full_result['gradients'].items()}, 'D_F_parameter_gradients': {k: v.tolist() for k, v in replay_parameter.items()}, 'D_A_B_endpoint_gradient_distance': {k: float(torch.linalg.vector_norm(a_gradients[k] - b_result['gradients'][k])) for k in dg.D_NAMES}, 'same_schedule_replay_checks': replay_checks, 'source_bindings': json.loads((ROOT / 'SOURCE_BINDINGS.json').read_text('utf-8-sig')), 'implementation_files': manifest, 'scientific_arrays_opened': 0, 'scientific_weights_loaded': 0, 'DEV_EVAL_CAL_reads': 0, 'scientific_forward_or_gradient_runs': 0, 'optimizer_updates': 0, 'real_R63_ResponseNetwork_executed': False, 'original_endpoint_Box_STE_feature_code_executed_with_artificial_data': True, 'C_terms_in_primary_gradient_comparison': False, 'limitations': ['Toy network only: real R63 attention/kernel replay not tested here.', 'Fixed per-parent noise and directions define a conditional finite empirical law, not original independent-occurrence noise/fresh-direction expectation.', 'Algorithm gradient uses canonical STE, stable original-row tie branch, abs(0)=0 and exact zero-RMS branch; no unique nonsmooth derivative or hard-ADC finite-difference equivalence is claimed.', 'Selected-device artificial fixture only; first-order gradients only; no scientific arrays, weights or training are used.']}
    return receipt
