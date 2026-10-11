"""Seal this helper's artifacts and current read-only source snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main() -> int:
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r72_seal_local_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(out)
    started = time.perf_counter()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(name, payload):
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sources = [
        "AGENTS.md",
        "work/R71_REAL_CHARGE_FULL_TRAIN_001/public_charge_head.py",
        "work/R71_REAL_CHARGE_FULL_TRAIN_001/common_response.py",
        "work/R71_REAL_CHARGE_FULL_TRAIN_001/numpy_reference.py",
        "work/R71_REAL_CHARGE_FULL_TRAIN_001/physical_operators.py",
        "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/source_flow_core.py",
        "work/R72_FULL_WAVEFORM_FULL_TRAIN_001/train_full.py",
        "work/R72_FULL_WAVEFORM_FULL_TRAIN_001/postcheck.py",
        "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py",
    ]
    entries = [{"path": relative, "bytes": (root / relative).stat().st_size,
                "sha256": sha(root / relative)} for relative in sources]
    write("READ_INPUTS_SHA256.json", {
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "binding_kind": "read_only_static_source_snapshot",
        "root_R72_sources_are_still_root_owned": True,
        "not_formal_approval_of_root_final_registration": True,
        "scientific_arrays_or_weights_read": False,
        "inputs": entries,
    })
    result = json.loads((out / "ARTIFICIAL_CHECK.json").read_text(encoding="utf-8"))
    native = json.loads((out / "NATIVE_RESOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    actual_exit = int((out / "NATIVE_EXIT_CODE.txt").read_text(encoding="utf-8-sig").strip())
    assert result["status"] == "PASS" and result["result"]["check_count"] == 38
    assert native["native_exit_code"] == actual_exit == 0
    assert native["peak_RSS_bytes"] <= 1073741824
    for filename in ("ATTEMPT_01_ARTIFICIAL_CHECK.json", "ATTEMPT_01_NATIVE_RESOURCE_RECEIPT.json",
                     "ATTEMPT_01_NATIVE_EXIT_CODE.txt", "PERMUTATION_DIAGNOSTIC.json"):
        assert (out / filename).is_file()
    resources.update(guard.memory())
    resources.update({"native_exit_requested": 0, "elapsed_wall_seconds": time.perf_counter() - started,
                      "operation": "standard_library_artifact_hashing_only",
                      "scientific_arrays_weights_read": False, "CUDA_executed": False})
    write("SEAL_RESOURCE_RECEIPT.json", resources)
    excluded = {"SHA256_MANIFEST.json", "ARTIFACT_COMPLETION_RECEIPT.json", "SEAL_EXIT_CODE.txt"}
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(out.iterdir()) if path.is_file() and path.name not in excluded]
    write("SHA256_MANIFEST.json", {"hash_algorithm": "SHA256", "artifacts": files})
    write("ARTIFACT_COMPLETION_RECEIPT.json", {
        "status": "CANDIDATE_PREPARED_AND_ARTIFICIAL_CPU_CHECKED",
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(out / "SHA256_MANIFEST.json"),
        "input_manifest_sha256": sha(out / "READ_INPUTS_SHA256.json"),
        "response_module_sha256": sha(out / "waveform_response.py"),
        "reference_module_sha256": sha(out / "waveform_numpy_reference.py"),
        "artificial_checker_sha256": sha(out / "check_waveform_response.py"),
        "native_CPU": {"exit_code": actual_exit, "checks": 38,
                       "CPU_affinity_mask": native["CPU_affinity_mask"],
                       "threads": native["CPU_threads"], "peak_RSS_bytes": native["peak_RSS_bytes"],
                       "receipt": "NATIVE_RESOURCE_RECEIPT.json"},
        "initial_byte_permutation_failure_preserved": True,
        "numerical_permutation_assertion_is_fixture_engineering_only": True,
        "CUDA_execution": False, "scientific_execution": False,
        "science_arrays_weights_read": False, "existing_frozen_or_root_files_modified": False,
        "entry_points": {"fixture_and_checks": "check_waveform_response.py:make_fixture/run_artificial",
                         "response": "waveform_response.py",
                         "reference": "waveform_numpy_reference.py:reference_numpy"},
        "not_proven": ["full_population_CUDA_memory_or_latency", "complete_objective_VJP_correctness",
                       "scientific_quality_or_same_treated_REAL_preservation", "universal_bitwise_permutation_equivariance",
                       "final_A_F_acceptance"],
    })
    print(json.dumps({"status": "SEALED", "module_sha256": sha(out / "waveform_response.py"),
                      "receipt": str(out / "ARTIFACT_COMPLETION_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
