"""Single R37 control-only byte seal; no array decoding or scientific imports."""
import hashlib
import json
import os
import pathlib
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
END = datetime.fromisoformat("2026-10-09T22:01:09.839435+00:00")
CAP = 512 * 1024**2
EXCLUDED = {"FINAL_MANIFEST.json", "FINAL_RECEIPT.json"}


def utc():
    return datetime.now(timezone.utc)


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(path):
    digest = hashlib.sha256()
    returned = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            returned += len(block)
            digest.update(block)
    assert returned == path.stat().st_size, str(path)
    return digest.hexdigest(), returned


def main():
    began = time.perf_counter()
    assert utc() < END
    assert not any((ROOT / n).exists() for n in EXCLUDED), "already sealed"
    history = load("RUN_HISTORY.json")
    assert len(history) == 7
    assert all(r["actual_worker_native_exit"] == 0 and r["launcher_native_exit"] == 0 for r in history)
    science_s = sum(r["elapsed_s"] for r in history)
    replay = load("REPLAY_RECEIPT.json")
    qa = load("FIGURE_QA.json")
    gate = load("GATE_METRICS.json")
    assert replay["rows"] == 2000 and replay["pass"]
    assert qa["figure_count"] == 11 and qa["pass"]
    assert gate["retain"] is False
    assert science_s < 1800
    resource = load("RESOURCE_RECEIPT.json")
    assert resource["counts"]["HGB_attempts"] == 10
    assert resource["counts"]["density_description_slots"] == 8
    assert resource["counts"]["density_EM_attempts"] == 7
    assert resource["actual_worker_RSS_max_observed_bytes"] < 4 * 1024**3
    assert resource["GPU_peak_allocated_bytes"] < 4 * 1024**3
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT) for p in files)
    stage_before = sum(p.stat().st_size for p in files)
    # Reserve a small fixed control margin before modifying the final resource record.
    assert stage_before + 512 * 1024 < CAP
    resource["delivery_assembly_native"] = {
        "tool": "30176f", "actual_PID": 24280, "observed_native_exit": 0,
        "UTC": "2026-10-09T21:11:19.270238+00:00", "control_only": True,
    }
    resource["seal_control"] = {
        "PID": os.getpid(), "begin_UTC": utc().isoformat(), "pure_byte_file_hashing": True,
        "scientific_array_decodes_or_fits_or_generation": 0,
        "member_file_count_before_two_self_reference_exclusions": len(files),
        "stage_bytes_before_final_resource_replacement": stage_before,
        "additional_final_control_reserved_bytes": 512 * 1024,
        "final_exact_sizes_and_hash_cost": "FINAL_RECEIPT.json and original seal stdout",
        "actual_native_exit_source": "external exec tool result returned after child completion",
    }
    final_resource_bytes = encode(resource)
    assert utc() < END
    (ROOT / "RESOURCE_RECEIPT.json").write_bytes(final_resource_bytes)

    members = []
    bytes_hashed = 0
    hash_started = time.perf_counter()
    for path in files:
        assert utc() < END
        digest, count = sha(path)
        bytes_hashed += count
        members.append({"path": path.relative_to(ROOT).as_posix(), "bytes": count, "sha256": digest})
    manifest = {
        "stage": ROOT.name, "UTC": utc().isoformat(), "PID": os.getpid(),
        "member_count": len(members), "member_bytes": bytes_hashed,
        "members": members, "excluded_self_references": sorted(EXCLUDED),
        "exclusion_reason": "manifest cannot hash itself; final receipt binds the frozen manifest",
        "originals_old_stages_unchanged": True,
        "scientific_arrays_decoded_for_sealing": 0,
        "migration_content_tests": 0,
    }
    manifest_bytes = encode(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    receipt = {
        "stage": ROOT.name, "status": "sealed", "PID": os.getpid(),
        "registered_UTC": "2026-10-09T20:41:09.839435+00:00",
        "executor_deadline_UTC": END.isoformat(), "seal_prewrite_UTC": utc().isoformat(),
        "manifest_path": "FINAL_MANIFEST.json", "manifest_sha256": manifest_sha,
        "manifest_bytes": len(manifest_bytes), "member_count": len(members), "member_bytes": bytes_hashed,
        "resource_sha256": hashlib.sha256(final_resource_bytes).hexdigest(),
        "resource_bytes": len(final_resource_bytes),
        "REPORT_sha256": next(m["sha256"] for m in members if m["path"] == "REPORT.md"),
        "science_elapsed_s": science_s, "science_budget_s": 1800,
        "scientific_all_native_exit_zero": True, "independent_complete_replay_rows": 2000,
        "replay_bitwise_pass": True, "figure_actual_QA_count": 11,
        "scientific_joint_gate_pass": False, "decision": gate["decision"],
        "scientific_gates": {k: v["pass"] for k, v in gate.items() if k.startswith("gate")},
        "RSS_two_query_gap_reason": "unknown", "project_goal_closed": False,
        "control_only_seal_hash_bytes_returned": bytes_hashed,
        "control_only_hash_elapsed_s": time.perf_counter() - hash_started,
        "control_only_elapsed_s_at_receipt_assembly": time.perf_counter() - began,
        "final_total_file_bytes_exact": "reported in original seal stdout after both files are written",
        "actual_last_filesystem_write_UTC": "reported in original seal stdout using final receipt filesystem mtime",
        "native_exit_evidence": "external exec tool result; not asserted in advance by this child",
        "no_further_stage_writes": True,
    }
    receipt_bytes = encode(receipt)
    final_total = bytes_hashed + len(manifest_bytes) + len(receipt_bytes)
    assert final_total < CAP and utc() < END
    with (ROOT / "FINAL_MANIFEST.json").open("xb") as f:
        f.write(manifest_bytes)
    with (ROOT / "FINAL_RECEIPT.json").open("xb") as f:
        f.write(receipt_bytes)
    last_write = datetime.fromtimestamp((ROOT / "FINAL_RECEIPT.json").stat().st_mtime, timezone.utc)
    assert last_write < END
    # Only stdout follows the final receipt; no stage file is written again.
    print(json.dumps({
        "PID": os.getpid(), "status": "sealed", "last_write_UTC": last_write.isoformat(),
        "manifest_sha256": manifest_sha, "manifest_bytes": len(manifest_bytes),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(), "receipt_bytes": len(receipt_bytes),
        "resource_sha256": receipt["resource_sha256"], "member_count": len(members),
        "all_file_count": len(members) + 2, "all_file_bytes": final_total,
        "output_cap_bytes": CAP, "science_s": science_s,
        "control_only_seal_elapsed_s": time.perf_counter() - began,
        "scientific_joint_gate_pass": False, "no_further_stage_writes": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
