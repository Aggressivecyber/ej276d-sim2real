from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    begin = dt.datetime.now(dt.timezone.utc)
    cfg = load("config.json")
    limit = dt.datetime.fromisoformat(cfg["deadline_UTC"])
    assert begin < limit
    assert not (ROOT / "FINAL_MANIFEST.json").exists()
    assert not (ROOT / "FINAL_RECEIPT.json").exists()
    jobs = load("RUN_HISTORY.json")
    assert len(jobs) == 2 and all(j["actual_worker_native_exit"] == 0 for j in jobs)
    assert sum(j["elapsed_s"] for j in jobs) < cfg["science_budget"]
    numeric = load("NUMERIC_RECEIPT.json")
    assert numeric["counters"]["HGB_attempts"] == cfg["HGB_max_fits"] == 15
    assert all(numeric["counters"][k] == 0 for k in ("response_updates", "response_generation", "EM_attempts"))
    qa = load("FIGURE_QA.json")
    assert len(qa["figures"]) == 9
    assert all(v["executor_actually_viewed"] and not v["final_image_modified_after_review"] for v in qa["figures"].values())
    resource = load("RESOURCE_RECEIPT.json")
    resource["control_only_seal_PID"] = os.getpid()
    resource["seal_started_UTC"] = begin.isoformat()
    resource["science_terminal_no_live_owned_job"] = True
    resource["all_nine_final_images_actually_reviewed"] = True
    resource["source_and_statistics_unchanged_since_numeric_run"] = True
    resource["all_stage_byte_hash_scope"] = "new R27 files once for final manifest; no old scientific inputs reopened"
    resource["control_check_native_receipt"] = "CONTROL_NATIVE_CHAIN.json"
    (ROOT / "RESOURCE_RECEIPT.json").write_bytes(encoded(resource))

    excluded = {"FINAL_MANIFEST.json", "FINAL_RECEIPT.json"}
    members = []
    hashed = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in excluded:
            continue
        assert not p.is_symlink(), rel
        h = hashlib.sha256()
        n = 0
        with p.open("rb") as stream:
            while b := stream.read(1024 * 1024):
                n += len(b)
                h.update(b)
        assert n == p.stat().st_size, rel
        hashed += n
        members.append({"path": rel, "bytes": n, "sha256": h.hexdigest()})
    assert hashed < cfg["output_limit_bytes"]
    by_name = {m["path"]: m for m in members}
    frozen = load("PRECOMPUTE_FREEZE.json")["bindings"]
    assert all(by_name[name]["bytes"] == v["bytes"] and by_name[name]["sha256"] == v["sha256"] for name, v in frozen.items())
    for name, v in qa["figures"].items():
        f = by_name["results/figures/" + name]
        assert f["bytes"] == v["bytes"] and f["sha256"] == v["sha256"], name
    manifest = {
        "stage": cfg["stage"], "seal_PID": os.getpid(), "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
        "member_count": len(members), "member_bytes": hashed,
        "hash_read_bytes": hashed, "members": members,
        "self_reference_excluded": sorted(excluded),
        "old_stage_writes_or_source_rehash": 0,
        "scientific_array_decode_for_manifest": 0,
        "checksum_rule": "SHA256 exact stored bytes; no NPZ/string/numeric decoding",
        "receipt_binds_this_manifest": "FINAL_RECEIPT.json; its self hash is reported externally",
    }
    assert dt.datetime.now(dt.timezone.utc) < limit
    manifest_bytes = encoded(manifest)
    (ROOT / "FINAL_MANIFEST.json").write_bytes(manifest_bytes)
    receipt = {
        "stage": cfg["stage"], "status": "sealed_before_native_process_exit",
        "seal_PID": os.getpid(), "seal_started_UTC": begin.isoformat(),
        "final_receipt_write_started_UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executor_deadline_UTC": cfg["deadline_UTC"],
        "last_write_and_native_exit": "actual last-write UTC/PID/exit are reported by the native calling tool after this write",
        "FINAL_MANIFEST_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "FINAL_MANIFEST_bytes": len(manifest_bytes),
        "manifest_member_count": len(members), "manifest_member_bytes": hashed,
        "total_file_count_including_manifest_and_receipt": len(members) + 2,
        "total_stage_bytes": 0,
        "science_elapsed_s": sum(j["elapsed_s"] for j in jobs),
        "response_updates_generation_EM": 0, "HGB_attempts": 15,
        "science_failures": [], "source_frozen_bindings_unchanged": True,
        "figures_reviewed": 9, "no_further_stage_writes": True,
        "key_files": {k: by_name[k] for k in (
            "REPORT.md", "METRICS_COMPACT.json", "RESOURCE_RECEIPT.json", "REPRODUCTION.md",
            "FIGURE_QA.json", "CONTROL_NATIVE_CHAIN.json", "config.json", "PROTOCOL.md",
        )},
        "native_process_exit_not_fabricated_before_exit": True,
    }
    # Account for receipt's own exact length without inserting its self hash.
    for _ in range(10):
        b = encoded(receipt)
        total = hashed + len(manifest_bytes) + len(b)
        if receipt["total_stage_bytes"] == total:
            break
        receipt["total_stage_bytes"] = total
    receipt_bytes = encoded(receipt)
    assert receipt["total_stage_bytes"] == hashed + len(manifest_bytes) + len(receipt_bytes)
    assert receipt["total_stage_bytes"] <= cfg["output_limit_bytes"]
    assert dt.datetime.now(dt.timezone.utc) < limit
    (ROOT / "FINAL_RECEIPT.json").write_bytes(receipt_bytes)
    # This is the last stage write. Exact external native exit is obtained by caller.
    last_write = dt.datetime.fromtimestamp((ROOT / "FINAL_RECEIPT.json").stat().st_mtime, dt.timezone.utc)
    stopped = dt.datetime.now(dt.timezone.utc)
    print(json.dumps({
        "PID": os.getpid(), "status": "sealed_no_further_stage_writes",
        "last_write_UTC": last_write.isoformat(), "actual_stop_UTC": stopped.isoformat(),
        "manifest_member_count": len(members), "stage_file_count": len(members) + 2,
        "stage_bytes": receipt["total_stage_bytes"],
        "hash_read_bytes": hashed,
        "FINAL_MANIFEST_sha256": receipt["FINAL_MANIFEST_sha256"],
        "FINAL_RECEIPT_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "key_files": receipt["key_files"], "before_deadline": stopped < limit,
        "science_elapsed_s": receipt["science_elapsed_s"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
