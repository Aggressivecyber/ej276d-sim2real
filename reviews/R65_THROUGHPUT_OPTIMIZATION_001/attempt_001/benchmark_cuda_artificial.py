"""Prepared for root: CUDA artificial pack AND actual 64-row forward+VJP.

No saved model/data/state, training imports, optimizer steps, SSH, or SIGSTOP.
Root owns isolation, CPU/RAM/GPU guards and training-process resume in finally.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--pack-repetitions", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    args = parser.parse_args()
    assert 1 <= args.threads <= 12 and 3 <= args.samples <= 21 and 1 <= args.pack_repetitions <= 100
    assert 0 < args.max_seconds <= 150.0
    sys.dont_write_bytecode = True
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.parent / "tmp"
    cache.mkdir(exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
        os.environ[name] = str(cache)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from artificial_common import original_helper, assert_bits, digest_tensor, synthetic_native
    import gradient_pack_candidate as candidate
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script cannot claim a CUDA result")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    def bound():
        if time.perf_counter() > deadline:
            raise TimeoutError("Artificial CUDA time bound reached; root must resume training in finally")
    old = original_helper(args.source_root)
    special = [0.0, -0.0, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126, -(2.0 ** -126),
               1.0, -1.0, 3.4028234663852886e38]
    checks = []
    for dtype in (torch.float32, torch.float64):
        gradient = torch.tensor(special, dtype=dtype, device="cuda")
        if dtype == torch.float64:
            gradient = torch.cat((gradient, torch.tensor([2.0 ** -1074, -(2.0 ** -1074), 2.0 ** -1022], dtype=dtype, device="cuda")))
        values = (gradient[:4].reshape(2, 2).t(), gradient[4:])
        parameters = tuple(torch.nn.Parameter(torch.zeros_like(g)) for g in values)
        reference = old._flatten_gradients_cpu_f64(values, parameters)
        observed = candidate.flatten_gradients_cpu_f64(values, parameters)
        assert_bits(torch, reference, observed, "CUDA native transfer/cast " + str(dtype))
        checks.append({"check": "original to(cpu,f64) vs native D2H then CPU cast; signed zero/subnormal/noncontiguous", "dtype": str(dtype), "byte_equal": True,
                       "sha256": digest_tensor(observed)})
    for values in (
        (torch.tensor([0.0, -0.0, 2.0 ** -149], dtype=torch.float32, device="cuda"), torch.tensor([2.0 ** -1074], dtype=torch.float64, device="cuda")),
        (torch.tensor([0.0, -0.0, 2.0 ** -149], dtype=torch.float32, device="cuda"), torch.tensor([1.0], dtype=torch.float32, device="cpu")),
    ):
        parameters = tuple(torch.nn.Parameter(torch.zeros_like(g)) for g in values)
        assert_bits(torch, old._flatten_gradients_cpu_f64(values, parameters), candidate.flatten_gradients_cpu_f64(values, parameters), "CUDA mixed fallback")
    checks.append({"check": "mixed dtype and mixed device fallback", "byte_equal": True})
    model, flow, inputs, gy = synthetic_native(torch, args.source_root, "cuda", 64)
    parameters = tuple(model.parameters())
    versions = tuple(p._version for p in parameters)
    rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        cached, cached_raw = flow.endpoint(model, *inputs)
    def forward_vjp(fn):
        bound()
        output, raw = flow.endpoint(model, *inputs)
        return fn(output, parameters, gy)
    output, raw = flow.endpoint(model, *inputs)
    assert_bits(torch, output, cached, "CUDA original64 endpoint replay")
    assert_bits(torch, raw, cached_raw, "CUDA raw endpoint replay")
    values = torch.autograd.grad(output, parameters, grad_outputs=gy, create_graph=False, retain_graph=False, allow_unused=False)
    reference = old._flatten_gradients_cpu_f64(values, parameters)
    observed = candidate.flatten_gradients_cpu_f64(values, parameters)
    assert_bits(torch, reference, observed, "CUDA native VJP pack")
    assert all(bool(torch.count_nonzero(g)) for g in values), "degenerate synthetic VJP tensor"
    del output, raw
    before_full = forward_vjp(old.parameter_vjp_cpu_f64)
    after_full = forward_vjp(candidate.parameter_vjp_cpu_f64)
    assert_bits(torch, before_full, after_full, "CUDA independently replayed VJP")
    checks.append({"check": "actual artificial ResponseNetwork 64-row endpoint+replay+VJP", "byte_equal": True,
                   "all_gradient_tensors_nonzero": True, "parameter_tensors": len(parameters), "parameter_count": reference.numel(),
                   "sha256": digest_tensor(reference), "projection_active_rows": int(((cached_raw < -.5) | (cached_raw > .5)).any(1).sum().cpu())})
    def timed(fn, repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            bound()
            result = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / repetitions, result
    def series(old_fn, new_fn, repetitions):
        # Warm both arms; alternate order to reduce one-sided thermal/drift bias.
        for fn in (old_fn, new_fn):
            fn()
        arms = {"original_seconds": [], "candidate_seconds": []}
        for index in range(args.samples):
            order = (("original_seconds", old_fn), ("candidate_seconds", new_fn))
            if index % 2:
                order = tuple(reversed(order))
            results = {}
            for label, fn in order:
                elapsed, result = timed(fn, repetitions)
                arms[label].append(elapsed)
                results[label] = result
            assert_bits(torch, results["original_seconds"], results["candidate_seconds"], "timed byte equality")
        a, b = statistics.median(arms["original_seconds"]), statistics.median(arms["candidate_seconds"])
        return dict(arms, original_median_seconds=a, candidate_median_seconds=b,
                    median_ratio_original_over_candidate=a / b, candidate_fractional_time_reduction=1 - b / a,
                    repetitions_per_sample=repetitions)
    packing = series(lambda: old._flatten_gradients_cpu_f64(values, parameters),
                     lambda: candidate.flatten_gradients_cpu_f64(values, parameters), args.pack_repetitions)
    full = series(lambda: forward_vjp(old.parameter_vjp_cpu_f64),
                  lambda: forward_vjp(candidate.parameter_vjp_cpu_f64), 1)
    assert tuple(p._version for p in parameters) == versions
    assert_bits(torch, rng_cpu, torch.get_rng_state(), "CPU random stream unchanged")
    assert_bits(torch, rng_cuda, torch.cuda.get_rng_state(), "CUDA random stream unchanged")
    receipt = {"status": "passed", "evidence": "Artificial CUDA only; no scientific arrays or saved weights read",
               "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(), "threads": args.threads,
               "cudnn_deterministic": True, "TF32": False, "checks": checks,
               "packing_microbenchmark": packing, "actual64_forward_plus_VJP_benchmark": full,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
               "elapsed_seconds": time.perf_counter() - started,
               "limitations": "No optimizer steps, no joint full_D objective, no full scientific update timing, no 50000-step ETA inference"}
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
