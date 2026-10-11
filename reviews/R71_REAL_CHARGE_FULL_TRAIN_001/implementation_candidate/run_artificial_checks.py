"""Native local launcher: apply existing resource guard before importing Torch."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r71_local_resource_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resource = guard.apply(output)
    resource["PYTHONDONTWRITEBYTECODE"] = os.environ.get("PYTHONDONTWRITEBYTECODE")
    resource["sys_dont_write_bytecode"] = sys.dont_write_bytecode
    started = time.perf_counter()
    exit_code = 1
    result = None
    error = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        import check_public_charge_head
        ambient_before = torch.get_rng_state().clone()
        result = check_public_charge_head.run_checks()
        if not torch.equal(ambient_before, torch.get_rng_state()):
            raise AssertionError("run_checks must restore ambient CPU RNG")
        result["caller_CPU_RNG_restored"] = True
        source_path = root / "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/source_flow_core.py"
        source_bytes = source_path.read_bytes()
        original = ast.parse(source_bytes.decode("utf-8"))
        names = {"q_torch", "charge_vector", "BoxChargeProjection", "canonical_adc_torch"}
        nodes = [node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if len(nodes) != len(names):
            raise AssertionError("exact frozen Box/ADC definitions not found")
        namespace = {"torch": torch}
        code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                       str(source_path), "exec")
        exec(code, namespace)
        result["frozen_Box_ADC"] = check_public_charge_head.run_box_checks(
            namespace["BoxChargeProjection"], namespace["canonical_adc_torch"])
        result["frozen_Box_ADC"]["source_path"] = str(source_path)
        result["frozen_Box_ADC"]["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        resource["torch_num_threads"] = torch.get_num_threads()
        resource["torch_num_interop_threads"] = torch.get_num_interop_threads()
        exit_code = 0
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    resource.update(guard.memory())
    resource.update({"exit_code": exit_code, "elapsed_wall_seconds": time.perf_counter() - started,
                     "script_path": str(Path(__file__).resolve()), "python_executable": sys.executable,
                     "science_arrays_weights_read": False, "full_waveform_network_run": False,
                     "CUDA_executed": False, "guard_sha256": hashlib.sha256(guard_path.read_bytes()).hexdigest()})
    payload = {"status": "PASS" if exit_code == 0 else "FAIL", "result": result, "error": error}
    (output / "ARTIFICIAL_CHECK.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "NATIVE_RESOURCE_RECEIPT.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "exit_code": exit_code,
                      "check_count": result.get("check_count") if result else None,
                      "peak_RSS_bytes": resource["peak_RSS_bytes"], "error": error}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
