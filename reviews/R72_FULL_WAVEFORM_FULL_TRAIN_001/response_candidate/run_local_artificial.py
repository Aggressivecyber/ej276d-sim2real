"""Apply existing one-CPU/1-GiB guard, then run the artificial CPU checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    gp = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r72_existing_local_guard", gp)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(out)
    sys.path.append(str(root / "work/R71_REAL_CHARGE_FULL_TRAIN_001"))
    start = time.perf_counter()
    exit_code, error, result = 1, None, None
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from check_waveform_response import run_artificial
        before = torch.get_rng_state().clone()
        result = run_artificial("cpu")
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError("artificial checks must restore caller CPU RNG")
        result["caller_CPU_RNG_restored"] = True
        exit_code = 0
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    resources.update(guard.memory())
    resources.update({"native_exit_code": exit_code, "elapsed_wall_seconds": time.perf_counter() - start,
                      "python_executable": sys.executable, "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
                      "sys_dont_write_bytecode": sys.dont_write_bytecode, "CPU_threads": 1, "interop_threads": 1,
                      "CUDA_executed": False, "scientific_arrays_weights_read": False,
                      "scientific_training_or_probe_run": False, "guard_sha256": hashlib.sha256(gp.read_bytes()).hexdigest()})
    payload = {"status": "PASS" if exit_code == 0 else "FAIL", "result": result, "error": error}
    (out / "ARTIFICIAL_CHECK.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "NATIVE_RESOURCE_RECEIPT.json").write_text(json.dumps(resources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "check_count": result.get("check_count") if result else None,
                      "error": error, "peak_RSS_bytes": resources["peak_RSS_bytes"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
