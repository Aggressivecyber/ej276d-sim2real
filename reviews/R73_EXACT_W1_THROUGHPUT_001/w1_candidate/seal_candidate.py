"""Hash only this candidate and the explicitly read source snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r73_seal_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(out)
    started = time.perf_counter()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    native = json.loads((out / "ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    checked = json.loads((out / "ATTEMPT_002_ARTIFICIAL_CHECK.json").read_text(encoding="utf-8"))
    actual_exit = int((out / "ATTEMPT_002_NATIVE_EXIT_CODE.txt").read_text(encoding="utf-8-sig").strip())
    assert checked["status"] == "PASS" and checked["result"]["pass_all"]
    assert native["native_exit_requested"] == actual_exit == 0
    assert native["peak_RSS_bytes"] <= 1073741824
    for name, expected in native["attempt_source_SHA256"].items():
        assert sha(out / name) == expected, "tested source changed: " + name
    inputs = ["AGENTS.md", "work/R72_FULL_WAVEFORM_FULL_TRAIN_001/full_waveform_loss.py",
              "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"]
    source = root / inputs[1]
    assert sha(source) == native["source_SHA256"]
    write("READ_INPUTS_SHA256.json", {"kind": "read_only_source_snapshot", "scientific_arrays_weights_read": False,
        "inputs": [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha(root / name)} for name in inputs]})
    resources.update(guard.memory())
    resources.update({"operation": "standard_library_source_hashing_only", "native_exit_requested": 0,
                      "elapsed_wall_seconds": time.perf_counter() - started, "CUDA_executed": False,
                      "scientific_arrays_weights_read": False})
    write("SEAL_RESOURCE_RECEIPT.json", resources)
    excluded = {"SHA256_MANIFEST.json", "ARTIFACT_COMPLETION_RECEIPT.json", "SEAL_EXIT_CODE.txt"}
    write("SHA256_MANIFEST.json", {"hash_algorithm": "SHA256", "artifacts": [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(out.iterdir()) if p.is_file() and p.name not in excluded]})
    write("ARTIFACT_COMPLETION_RECEIPT.json", {
        "status": "SINGLE_CANDIDATE_PREPARED_ARTIFICIAL_CPU_PASS",
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "candidate_SHA256": sha(out / "exact_uniform_w1.py"),
        "checker_SHA256": sha(out / "check_exact_uniform_w1.py"),
        "manifest_SHA256": sha(out / "SHA256_MANIFEST.json"),
        "source_SHA256": sha(source), "forward_bitwise_cases": 1601,
        "gradient_sides_compared": 3138, "Fraction_formula_cases": 1521,
        "native_exit_code": actual_exit, "native_CPU_affinity_mask": native["CPU_affinity_mask"],
        "native_threads": native["CPU_threads"], "peak_RSS_bytes": native["peak_RSS_bytes"],
        "actual_native_receipt": "ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json",
        "prior_attempt_preserved": True, "gradient_bitwise_equivalence_claimed": False,
        "CUDA_execution": False, "science_execution": False, "real_speedup_claimed": False,
        "scientific_arrays_weights_read": False, "root_frozen_files_modified": False,
        "remaining_root_work": ["independent_static_review", "registered_CUDA_artificial_checks",
                                "complete_objective_and_Adam_checks", "full_step_cost_verification"],
    })
    print(json.dumps({"status": "SEALED", "candidate_SHA256": sha(out / "exact_uniform_w1.py"),
                      "receipt": str(out / "ARTIFACT_COMPLETION_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
