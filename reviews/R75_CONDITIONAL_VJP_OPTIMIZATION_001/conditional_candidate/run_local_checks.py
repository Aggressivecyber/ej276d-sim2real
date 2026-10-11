"""Existing native local guard; immutable pure-artificial CPU attempt receipts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be alphanumeric")
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    work = root / "work/R74_EXACT_LOSS_FULL_TRAIN_001"
    spec = importlib.util.spec_from_file_location("r75_existing_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(out)
    sys.path.append(str(work))
    prefix = "ATTEMPT_" + args.attempt + "_"
    for name in (prefix+"ARTIFICIAL_CHECK.json", prefix+"NATIVE_RESOURCE_RECEIPT.json"):
        if (out / name).exists():
            raise FileExistsError("never overwrite an earlier attempt: " + name)
    started = time.perf_counter()
    result, error, exit_code = None, None, 1
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        before = torch.get_rng_state().clone()
        from check_conditional_vjp import run_checks
        result = run_checks("cpu", seed=750001)
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError("artificial check must restore caller CPU RNG")
        result["caller_CPU_RNG_restored"] = True
        exit_code = 0
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    resources.update(guard.memory())
    source_names = ("combined_loss.py", "full_waveform_loss.py", "exact_uniform_w1.py", "check_combined_loss.py", "artificial_checks.py")
    resources.update({"native_exit_requested": exit_code, "elapsed_wall_seconds": time.perf_counter()-started,
        "python_executable": sys.executable, "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "CPU_threads": 1, "interop_threads": 1, "CUDA_executed": False,
        "scientific_arrays_weights_read": False, "scientific_training_probe_run": False,
        "guard_SHA256": hashlib.sha256(guard_path.read_bytes()).hexdigest(),
        "R74_source_SHA256": {name: hashlib.sha256((work/name).read_bytes()).hexdigest() for name in source_names},
        "attempt_source_SHA256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest()
            for name in ("conditional_vjp_loss.py", "check_conditional_vjp.py", "run_local_checks.py")}})
    payload = {"status": "PASS" if exit_code == 0 else "FAIL", "result": result, "error": error}
    for name, value in ((prefix+"ARTIFICIAL_CHECK.json", payload), (prefix+"NATIVE_RESOURCE_RECEIPT.json", resources)):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "error": error, "peak_RSS_bytes": resources["peak_RSS_bytes"],
        "case_labels": list(result["cases"]) if result else None}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
