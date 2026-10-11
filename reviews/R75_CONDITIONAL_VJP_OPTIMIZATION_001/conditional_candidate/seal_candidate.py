"""Seal pure-artificial conditional VJP evidence without importing Torch."""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    guard_path = root / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r75_seal_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    resources = guard.apply(out)
    work = root / "work/R74_EXACT_LOSS_FULL_TRAIN_001"
    targets = ("SOURCE_MANIFEST.json", "SEAL_NATIVE_RESOURCE_RECEIPT.json", "COMPLETION_RECEIPT.json")
    if any((out/name).exists() for name in targets):
        raise FileExistsError("do not overwrite a completed seal")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    passed = json.loads((out/"ATTEMPT_002_ARTIFICIAL_CHECK.json").read_text(encoding="utf-8"))
    receipt = json.loads((out/"ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    assert passed["status"] == "PASS" and passed["result"]["pass_all"]
    assert (out/"ATTEMPT_002_NATIVE_EXIT_CODE.txt").read_text(encoding="utf-8-sig").strip() == "0"
    assert receipt["CPU_affinity_mask"] == 2 and receipt["peak_RSS_bytes"] <= 1073741824
    for name, expected in receipt["attempt_source_SHA256"].items():
        assert digest(out/name) == expected, ("source changed after check", name)
    for name, expected in receipt["R74_source_SHA256"].items():
        assert digest(work/name) == expected, ("input changed after check", name)
    input_names = ("combined_loss.py", "full_waveform_loss.py", "exact_uniform_w1.py",
                   "check_combined_loss.py", "artificial_checks.py", "waveform_response.py",
                   "common_response.py", "public_charge_head.py", "physical_operators.py",
                   "precision_order_reference.py", "numpy_reference.py")
    artifacts = sorted(path for path in out.iterdir() if path.is_file())
    manifest = {"sealed_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "evidence_level": "CPU_artificial_only_not_CUDA_or_science",
                "source_SHA256": receipt["attempt_source_SHA256"],
                "input_source_SHA256": {str(work/name): digest(work/name) for name in input_names},
                "guard_SHA256": digest(guard_path),
                "artifact_SHA256": {path.name: digest(path) for path in artifacts},
                "native_artificial_exit": 0, "failed_attempts_retained": ["001"],
                "actual_science_or_CUDA_executed": False}
    write = lambda name, value: (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write("SOURCE_MANIFEST.json", manifest)
    resources.update(guard.memory())
    resources.update({"native_exit_requested": 0, "python_executable": sys.executable,
                      "sys_dont_write_bytecode": sys.dont_write_bytecode,
                      "scientific_arrays_weights_read": False, "CUDA_executed": False})
    write("SEAL_NATIVE_RESOURCE_RECEIPT.json", resources)
    completion = {"status": "SEALED_CPU_ARTIFICIAL_PASS", "native_artificial_exit": 0,
                  "manifest_SHA256": digest(out/"SOURCE_MANIFEST.json"),
                  "report_SHA256": digest(out/"REPORT.md"),
                  "artificial_receipt_SHA256": digest(out/"ATTEMPT_002_ARTIFICIAL_CHECK.json"),
                  "native_resource_receipt_SHA256": digest(out/"ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json"),
                  "seal_resource_receipt_SHA256": digest(out/"SEAL_NATIVE_RESOURCE_RECEIPT.json"),
                  "candidate_SHA256": digest(out/"conditional_vjp_loss.py"),
                  "CUDA_pass_claimed": False, "scientific_gain_claimed": False}
    write("COMPLETION_RECEIPT.json", completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
