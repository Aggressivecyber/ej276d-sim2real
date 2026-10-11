"""Local artificial-only launcher using the existing Windows resource guard."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


INPUT_PATHS = [
    "AGENTS.md",
    "R68_REPORT.md",
    "reviews/R68_DIRECTION_PREPARATION_001/charge_compatibility/REPORT.md",
    "reviews/R68_SOURCE_PROVENANCE_AUDIT_001/REPORT.md",
    "work/R67_EQUAL_COST_REPAIR_001/full_step.py",
    "work/R67_EQUAL_COST_REPAIR_001/mini_step.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/source_flow_core.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/objective_adapter/objective_adapter.py",
    "work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/distribution_loss.py",
    "work/R67_EQUAL_COST_REPAIR_001/pilot.py",
    "work/R67_EQUAL_COST_REPAIR_001/PROTOCOL.md",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/network.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/model.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/legacy_model.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/network_update_check/network_update_check.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/implementation_upgrade/CUDA_EQUIVALENCE.json",
    "work/R67_EQUAL_COST_REPAIR_001/full/TRAINING_RECEIPT.json",
    "work/R67_EQUAL_COST_REPAIR_001/mini/TRAINING_RECEIPT.json",
    "work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/R61_distribution_loss.py",
    "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/distribution_adapter/distribution_gradient.py",
    "reviews/R65_THROUGHPUT_OPTIMIZATION_001/agent_candidate/REVIEW.md",
    "tasks/R69_SAMPLED_FULL_GRADIENT_PILOT_001.md",
    "PROJECT_ACCEPTANCE.md",
    "R64_REPORT.md",
    "PROJECT_ACCEPTANCE.json",
    "work/R67_EQUAL_COST_REPAIR_001/RESULT.json",
    "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py",
]


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r69_existing_resource_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    native_guard = guard.apply(output)
    if native_guard["CPU_affinity_mask"] != 2 or native_guard["CPU_affinity_count"] != 1:
        raise RuntimeError("local native affinity must be logical CPU1 (mask2)")
    if not sys.flags.dont_write_bytecode:
        raise RuntimeError("launcher requires Python -B")

    started = time.perf_counter()
    utc_start = datetime.now(timezone.utc).isoformat()

    def binding(path, relative):
        data = path.read_bytes()
        return {"path": relative, "absolute_path": str(path), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    sources = [(root / relative, relative) for relative in INPUT_PATHS]
    sources.append((Path("E:/GAN/AGENTS.md"), "external:E:/GAN/AGENTS.md"))
    before = [binding(path, relative) for path, relative in sources]
    sys.path.insert(0, str(output))
    from check_sampled_blocks import run_checks
    result = run_checks()
    after = [binding(path, relative) for path, relative in sources]
    if before != after:
        raise RuntimeError("a reviewed static input changed during artificial checks")
    usage = guard.memory()
    if usage["peak_RSS_bytes"] > native_guard["RSS_hard_max_bytes"]:
        raise RuntimeError("local RSS bound exceeded")

    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write("ARTIFICIAL_CHECK_RECEIPT.json", result)
    resources = {
        "status": "PASS_NATIVE_ARTIFICIAL_ONLY",
        "started_utc": utc_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "guard_source": str(guard_path),
        "guard": native_guard,
        "memory": usage,
        "reviewed_inputs_count": len(before),
        "reviewed_inputs_unchanged_during_native_check": True,
        "check_count": result["checks_passed"],
        "scope": result["scope"],
        "environment": {name: os.environ.get(name) for name in
                        ("TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX",
                         "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "evidence_boundary": result["evidence_boundary"],
    }
    write("RESOURCE_RECEIPT.json", resources)
    artifacts = [binding(output / name, name) for name in
                 ("OPTIMIZATION_REVIEW.md", "sampled_blocks.py", "check_sampled_blocks.py",
                  "run_native_checks.py", "ARTIFICIAL_CHECK_RECEIPT.json", "RESOURCE_RECEIPT.json")]
    write("SHA256_MANIFEST.json", {
        "schema": "R69_static_review_and_artificial_helpers_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "reviewed_inputs": before,
        "binding_note": "Governance and the prospective R69 task are read-time snapshots, not declarations of permanent sealing.",
        "artifacts": artifacts,
        "inputs_unchanged_during_native_check": True,
    })
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"],
                      "elapsed_seconds": resources["elapsed_seconds"],
                      "CPU_affinity_mask": native_guard["CPU_affinity_mask"],
                      "peak_RSS_bytes": usage["peak_RSS_bytes"],
                      "manifest": str(output / "SHA256_MANIFEST.json")}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
