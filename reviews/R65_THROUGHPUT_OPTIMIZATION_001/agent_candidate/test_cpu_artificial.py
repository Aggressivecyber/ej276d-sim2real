"""CPU-only synthetic checks; no scientific data/state I/O or training import."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from artificial_common import (HERE, configure_local_cpu, memory_receipt, original_helper,
                               assert_bits, digest_tensor, synthetic_native)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--native-batch", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()
    configure_local_cpu()
    import torch
    import gradient_pack_candidate as candidate
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    old = original_helper(args.source_root)
    start = time.perf_counter()
    checks = []
    special = [0.0, -0.0, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126, -(2.0 ** -126),
               2.0 ** -125, -(2.0 ** -125), 1.0, -1.0, 3.4028234663852886e38]
    for dtype in (torch.float32, torch.float64):
        g = torch.tensor(special, dtype=dtype)
        if dtype == torch.float64:
            g = torch.cat((g, torch.tensor([2.0 ** -1074, -(2.0 ** -1074), 2.0 ** -1022], dtype=dtype)))
        gradients = (g[:4].reshape(2, 2).t(), g[4:])  # noncontiguous and signed zeros
        params = tuple(torch.nn.Parameter(torch.zeros_like(v)) for v in gradients)
        reference = old._flatten_gradients_cpu_f64(gradients, params)
        observed = candidate.flatten_gradients_cpu_f64(gradients, params)
        assert_bits(torch, reference, observed, "special " + str(dtype))
        checks.append({"check": "native dtype + signed zero + subnormal + noncontiguous", "dtype": str(dtype), "byte_equal": True,
                       "sha256": digest_tensor(observed)})
    gradients = (torch.tensor([0.0, -0.0, 2.0 ** -149], dtype=torch.float32), torch.tensor([1.0, 2.0 ** -1074], dtype=torch.float64))
    params = tuple(torch.nn.Parameter(torch.zeros_like(v)) for v in gradients)
    assert_bits(torch, old._flatten_gradients_cpu_f64(gradients, params), candidate.flatten_gradients_cpu_f64(gradients, params), "mixed dtype fallback")
    checks.append({"check": "mixed dtype fallback", "byte_equal": True})
    p = torch.nn.Parameter(torch.ones(2))
    bad_cases = [((None,), (p,), RuntimeError), ((torch.ones(3),), (p,), RuntimeError),
                 ((torch.ones(2, dtype=torch.float64),), (p,), RuntimeError),
                 ((), (), ValueError), ((torch.ones(2),), (), ValueError)]
    for value in (float("nan"), float("inf"), -float("inf")):
        bad_cases.append(((torch.tensor([value, 1.0]),), (p,), FloatingPointError))
    for index, (gradients, params, expected) in enumerate(bad_cases):
        for fn in (old._flatten_gradients_cpu_f64, candidate.flatten_gradients_cpu_f64):
            try:
                fn(gradients, params)
            except expected:
                pass
            else:
                raise AssertionError("rejection case " + str(index))
    checks.append({"check": "metadata and NaN/Inf rejection", "cases": len(bad_cases), "passed": True})
    # Three toy updates retain original chunk boundaries, accumulation order,
    # one unchanged final installation, clip=5, and Adam. No RNG inside helper.
    torch.manual_seed(650105)
    model0 = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.SiLU(), torch.nn.Linear(7, 3))
    models = [copy.deepcopy(model0), copy.deepcopy(model0)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) for m in models]
    x, gy = torch.randn(129, 4), torch.randn(129, 3, dtype=torch.float64) * 20
    vjp_fns = (old.parameter_vjp_cpu_f64, candidate.parameter_vjp_cpu_f64)
    c_fns = (old.scalar_parameter_gradient_cpu_f64, candidate.scalar_parameter_gradient_cpu_f64)
    for step in range(3):
        vectors, metas = [], []
        for model, optimizer, vjp, c_fn in zip(models, optimizers, vjp_fns, c_fns):
            parameters = tuple(model.parameters())
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
            rng_before = torch.get_rng_state().clone()
            for lo, hi in ((0, 64), (64, 128), (128, 129)):
                total += vjp(model(x[lo:hi]).to(torch.float64), parameters, gy[lo:hi])
            c_gradient = c_fn(model(x[:11]).to(torch.float64).square().mean(), parameters)
            combined = total + c_gradient
            assert_bits(torch, rng_before, torch.get_rng_state(), "helper consumes no RNG")
            vectors.append(combined)
            metas.append(old.install_cpu_f64_clip_step(parameters, combined, optimizer, clip=5.0, capture=True))
        assert_bits(torch, vectors[0], vectors[1], "CPU f64 D then C step " + str(step))
        for key in ("before_clip", "after_clip"):
            assert_bits(torch, metas[0][key], metas[1][key], key)
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            assert_bits(torch, a, b, "Adam parameters")
        for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert_bits(torch, a[key], b[key], "Adam state " + key)
        assert metas[0]["did_clip"] and metas[1]["did_clip"]
    checks.append({"check": "3 updates: original64 chunk addition, D+C, one f64->f32/clip5/Adam, optimizer state, RNG", "byte_equal": True})
    if args.native_batch:
        model, flow, inputs, native_gy = synthetic_native(torch, args.source_root, "cpu", args.native_batch)
        parameters = tuple(model.parameters())
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            cache, _ = flow.endpoint(model, *inputs)
        output, _ = flow.endpoint(model, *inputs)
        assert_bits(torch, output, cache, "native endpoint replay")
        values = torch.autograd.grad(output, parameters, grad_outputs=native_gy, create_graph=False, retain_graph=False, allow_unused=False)
        reference = old._flatten_gradients_cpu_f64(values, parameters)
        observed = candidate.flatten_gradients_cpu_f64(values, parameters)
        assert_bits(torch, reference, observed, "native parameter VJP pack")
        assert_bits(torch, before, torch.get_rng_state(), "native replay consumes no RNG")
        assert all(bool(torch.count_nonzero(g)) for g in values), "degenerate synthetic VJP tensor"
        checks.append({"check": "actual ResponseNetwork/endpoint artificial CPU forward+replay+VJP", "batch": args.native_batch,
                       "parameter_tensors": len(parameters), "parameter_count": observed.numel(), "all_gradient_tensors_nonzero": True,
                       "byte_equal": True, "sha256": digest_tensor(observed)})
    receipt = {"status": "passed", "evidence": "CPU artificial only; CUDA not executed", "torch_version": torch.__version__,
               "source_root": str(Path(args.source_root).resolve()), "checks": checks,
               "elapsed_seconds": time.perf_counter() - start, "resources": memory_receipt()}
    (HERE / "CPU_ARTIFICIAL_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
