"""Preserve the initial static seal and bind root-requested offline gate tests."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
spec = importlib.util.spec_from_file_location("_r65_delivery_resource", PREPARED / "r65_controller_common.py")
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
GUARD_PROJECT = ROOT / "delivery_guard_project"
(GUARD_PROJECT / "work" / local.STAGE).mkdir(parents=True, exist_ok=True)
owner = local.ResourceOwner(GUARD_PROJECT, "offline_delivery_seal")


def read(path):
    return json.loads(path.read_text("utf8"))


def write(path, value):
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


try:
    archive = ROOT / "static_seal_v1"
    assert not archive.exists(), "Initial static seal already archived; do not repeat finalization"
    prior = read(ROOT / "PREPARATION_RECEIPT.json")
    old_binding = read(PREPARED / "CONTROLLER_BINDING.json")
    result_path = ROOT / "synthetic_gate_checks" / "attempt_001" / "RESULT.json"
    result = read(result_path)
    assert result["status"] == "PASS" and result["passed"] == result["total"] == 23
    assert all(row["pass"] for row in result["checks"])
    assert result["SSH_connections"] == result["subprocess_Popen_calls"] == result["actual_scientific_array_or_weight_reads"] == 0
    assert all(local.sha(PREPARED / name) == digest for name, digest in result["production_controller_sources"].items())
    assert local.sha(ROOT / "synthetic_gate_checks.py") == result["synthetic_helper_SHA256"]
    assert all(local.sha(Path(row["path"])) == row["sha256"] for row in result["source_bindings"])
    archive.mkdir()
    for row in prior["files"]:
        source = PREPARED / row["name"]
        assert local.sha(source) == row["sha256"]
        dest = archive / "prepared" / row["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert local.sha(dest) == row["sha256"]
    for name in ("PREPARATION_RECEIPT.json", "SOURCE_BINDING.json", "AST_RECEIPT.json"):
        shutil.copy2(ROOT / name, archive / name)
        assert local.sha(archive / name) == local.sha(ROOT / name)
    write(archive / "ARCHIVE_RECEIPT.json", {"UTC": datetime.now(timezone.utc).isoformat(),
          "initial_PREPARATION_RECEIPT_SHA256": local.sha(archive / "PREPARATION_RECEIPT.json"),
          "all_original_prepared_files_preserved": True, "archive_only_own_preparation_files": True})

    path = PREPARED / "README.md"
    text = path.read_text("utf8")
    old = "This package is for root review. None of these controllers, SSH requests, file transfers, or scientific entries has been executed during preparation. Only the stdlib preparation helper parsed/compiled source and bound metadata under CPU1, Windows Job process commit1GiB, and an actual RSS1GiB watchdog. R65 training was still active when root assigned this preparation. A successful training endpoint must never be inferred from these static receipts."
    new = "This package is for root review. No operational controller entry point, SSH request, file transfer, supervisor spawn, or scientific entry has been executed during preparation. Root additionally requested offline synthetic metadata tests:23/23 passed by calling the standard-library metadata gate functions against newly created fake files, with synthetic PID observations and a forbidden subprocess launcher. All preparation/test processes used CPU1, Windows Job process commit1GiB, and an actual RSS1GiB watchdog. R65 training remained active during preparation. A successful real training endpoint must never be inferred from these receipts."
    assert old in text
    text = text.replace(old, new, 1)
    old2 = "Do not run multiple watchers or parallel manual launches. An exclusive local watcher marker prevents a second watcher."
    new2 = "Root must wait until the preparation agents have ended before starting the formal watcher, then account for this controller's actual single-CPU affinity and1GiB memory cap. The current implementation chooses the lowest available CPU bit; it does not automatically inherit root's separate CPU3 choice. Do not run multiple watchers or parallel manual launches. An exclusive local watcher marker prevents a second watcher."
    assert old2 in text
    text = text.replace(old2, new2, 1)
    text += "\n## Offline validation evidence\n\nThe outer `synthetic_gate_checks/attempt_001/RESULT.json` binds the final remote-gate/common Python SHA256 values and records23 passing cases. The fixture uses the actual frozen training writer's top-level key names, config schema and audit check names; every binary-looking path contains a small newly created non-scientific byte string. Valid metadata reaches numerical READY and each subsequent READY. Live worker/supervisor, nonzero native exit, bad nonce, absent50000/49999, missing eighth stream, bad/missing artifact SHA, changed artifact/frozen bytes and skipped numerical stage are refused before launch intent or subprocess. A negative A–C result completes reporting with no next entry; train/recovery/D/E/F launches are refused.\n\nActual `/proc` interpretation, PID lifecycle, SSH/SCP, supervisor spawn/admission, watcher timing, file-transfer publication/archive behavior, real saved-array audit,128/256 calculation, generation/evaluation/replay and rendered figures remain statically reviewed or unverified. No synthetic test is evidence of those runtime/scientific results. The earlier static-only seal is preserved under the outer `static_seal_v1/`.\n"
    path.write_text(text, encoding="utf8")
    files = [{"name": row["name"], "sha256": local.sha(PREPARED / row["name"]),
              "bytes": (PREPARED / row["name"]).stat().st_size} for row in old_binding["files"]]
    new_binding = dict(old_binding, files=files, offline_metadata_tests={
        "passed": 23, "total": 23, "RESULT_SHA256": local.sha(result_path),
        "operational_or_scientific_execution": False})
    write(PREPARED / "CONTROLLER_BINDING.json", new_binding)
    prior.update(status="PREPARED_AST_AND_OFFLINE_METADATA_PASS_OPERATIONAL_UNEXECUTED",
        UTC=datetime.now(timezone.utc).isoformat(), files=files + [{
            "name": "CONTROLLER_BINDING.json", "sha256": local.sha(PREPARED / "CONTROLLER_BINDING.json"),
            "bytes": (PREPARED / "CONTROLLER_BINDING.json").stat().st_size}],
        original_static_seal_preserved_at=str(archive),
        offline_metadata_tests={"path": str(result_path), "sha256": local.sha(result_path), "passed": 23, "total": 23,
                               "production_controller_sources": result["production_controller_sources"]},
        remote_metadata_function_branches_executed_only_on_new_synthetic_files=True,
        actual_operational_entrypoints_executed=False,
        only_static_or_unverified=result["only_static_or_unverified"],
        finalize_delivery_source_SHA256=local.sha(Path(__file__)))
    write(ROOT / "PREPARATION_RECEIPT.json", prior)
    local.verify_controller_bundle()
    proof = {"UTC": datetime.now(timezone.utc).isoformat(), "status": prior["status"],
        "PREPARATION_RECEIPT_SHA256": local.sha(ROOT / "PREPARATION_RECEIPT.json"),
        "CONTROLLER_BINDING_SHA256": local.sha(PREPARED / "CONTROLLER_BINDING.json"),
        "source_SHA256": {row["name"]: row["sha256"] for row in files},
        "synthetic_RESULT_SHA256": local.sha(result_path), "synthetic_passed": 23,
        "synthetic_native_receipt_SHA256": local.sha(ROOT / "SYNTHETIC_ATTEMPT_001.NATIVE_EXIT_RECEIPT.json"),
        "synthetic_resource_receipt_SHA256": local.sha(result_path.parent / "RESOURCE_RECEIPT.json"),
        "original_static_seal_preserved": True, "SSH_connections": 0, "science_launches": 0,
        "actual_scientific_array_or_weight_reads": 0, "root_metadata_or_Git_writes": 0}
    write(ROOT / "DELIVERY_RECEIPT.json", proof)
    owner.finish("DELIVERY_SEAL_COMPLETE")
    resource = read(owner.root / (owner.operation + ".RESOURCE_RECEIPT.json"))
    write(ROOT / "DELIVERY_RESOURCE_RECEIPT.json", resource)
    print(json.dumps({"status": proof["status"], "PREPARATION_RECEIPT_SHA256": proof["PREPARATION_RECEIPT_SHA256"],
                      "DELIVERY_RECEIPT_SHA256": local.sha(ROOT / "DELIVERY_RECEIPT.json"),
                      "CONTROLLER_BINDING_SHA256": proof["CONTROLLER_BINDING_SHA256"],
                      "source_SHA256": proof["source_SHA256"], "resource": resource}, allow_nan=False))
except BaseException as error:
    write(ROOT / "DELIVERY_FAILURE.json", {"UTC": datetime.now(timezone.utc).isoformat(), "error": repr(error)})
    owner.finish("DELIVERY_SEAL_FAILED", repr(error))
    raise
