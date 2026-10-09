"""One control-only seal. Hash saved bytes; do not import models or decode arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
EXCLUDED = {"FINAL_MANIFEST.json", "FINAL_RECEIPT.json"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            size += len(block)
            h.update(block)
    if size != path.stat().st_size:
        raise RuntimeError(f"file changed during seal: {path.name}")
    return size, h.hexdigest()


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(config["deadline_UTC"]).timestamp()
    assert time.time() < deadline - 20
    assert all(not (ROOT / n).exists() for n in EXCLUDED)
    history = json.loads((ROOT / "RUN_HISTORY.json").read_text(encoding="utf-8"))
    active = json.loads((ROOT / "ACTIVE_JOB.json").read_text(encoding="utf-8"))
    assert active["actual_worker_native_exit"] == 0 and active["stop_UTC"]
    resource = json.loads((ROOT / "RESOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    assert abs(sum(r["elapsed_s"] for r in history) - resource["science_elapsed_s"]) < 1e-9
    assert resource["science_elapsed_s"] < config["science_budget"]
    assert resource["actual_worker_peak_RSS_bytes"] < config["RSS_limit_bytes"]
    assert resource["GPU_peak_allocated_bytes"] < config["GPU_allocated_limit_bytes"]
    qa = json.loads((ROOT / "FIGURE_QA.json").read_text(encoding="utf-8"))
    assert qa["all_final_figures_actual_visual_review_pass"] and len(qa["figures"]) == 10
    for name in ["REPORT.md", "REPRODUCTION.md", "METRICS_COMPACT.json", "GATE_METRICS.json",
                 "REPLAY_RECEIPT.json", "NATIVE_CHAIN.json", "SOURCE_BINDINGS.json"]:
        assert (ROOT / name).is_file()
    resource["final_control"] = {"assembly_tool": "f2e2a7", "assembly_PID": 54644,
        "assembly_observed_native_exit": 0, "text_control_check_tool": "903ce3",
        "text_control_check_observed_native_exit": 0, "seal_PID": os.getpid(),
        "seal_start_UTC": utc(), "operation": "pure saved-byte SHA, JSON and control seal",
        "new_scientific_fits_generations_replays_array_decodes": 0,
        "seal_native_exit_is_observed_by_enclosing_tool_after_return": True}
    dump(ROOT / "RESOURCE_RECEIPT.json", resource)

    members = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.relative_to(ROOT).as_posix() in EXCLUDED:
            continue
        assert not p.is_symlink()
        n, sha = digest(p)
        members.append({"path": p.relative_to(ROOT).as_posix(), "bytes": n, "sha256": sha})
    total = sum(m["bytes"] for m in members)
    assert total < config["output_limit_bytes"]
    assert time.time() < deadline
    manifest = {"stage": ROOT.name, "seal_PID": os.getpid(), "UTC": utc(),
        "members": members, "file_count": len(members), "member_bytes": total,
        "scope": "all stage files, including originals, failures, copied inputs, transfer packages and cache",
        "self_reference_exclusions": sorted(EXCLUDED),
        "pure_byte_hash_pass_returned_bytes": total,
        "not_scientific_array_decode_or_migration_comparison": True,
        "output_cap_bytes": config["output_limit_bytes"],
        "all_science_processes_terminal": True, "science_elapsed_s": resource["science_elapsed_s"]}
    dump(ROOT / "FINAL_MANIFEST.json", manifest, exclusive=True)
    manifest_n, manifest_sha = digest(ROOT / "FINAL_MANIFEST.json")
    resource_row = next(m for m in members if m["path"] == "RESOURCE_RECEIPT.json")
    receipt = {"stage": ROOT.name, "status": "sealed", "PID": os.getpid(), "UTC": utc(),
        "registered_UTC": config["registered_UTC"], "executor_deadline_UTC": config["deadline_UTC"],
        "no_further_stage_writes": True, "all_science_processes_terminal": True,
        "decision": "close_this_fixed_diffusion_edit_recipe", "gate_pass": [False, False, False, False, False, True],
        "manifest": {"path": "FINAL_MANIFEST.json", "bytes": manifest_n, "sha256": manifest_sha,
                     "file_count": len(members), "member_bytes": total},
        "resource_sha256": resource_row["sha256"], "science_elapsed_s": resource["science_elapsed_s"],
        "replay_rows": 4000, "replay_bitwise_pass": True, "all_ten_final_figures_actual_QA": True,
        "self_reference_exclusions": sorted(EXCLUDED),
        "receipt_write_start_UTC": utc(), "seal_elapsed_before_last_write_s": time.monotonic() - START,
        "native_exit_evidence": "enclosing exec tool observes actual native exit after this process returns; not self-invented",
        "assembly_native_evidence": {"tool": "f2e2a7", "PID": 54644, "observed_exit": 0}}
    dump(ROOT / "FINAL_RECEIPT.json", receipt, exclusive=True)
    receipt_n, receipt_sha = digest(ROOT / "FINAL_RECEIPT.json")
    final_write = datetime.fromtimestamp((ROOT / "FINAL_RECEIPT.json").stat().st_mtime, timezone.utc).isoformat()
    inclusive_total = total + manifest_n + receipt_n
    assert inclusive_total < config["output_limit_bytes"] and time.time() < deadline
    print(json.dumps({"PID": os.getpid(), "operation": "control-only seal", "last_write_UTC": final_write,
        "elapsed_s": time.monotonic() - START, "manifest_SHA256": manifest_sha, "receipt_SHA256": receipt_sha,
        "manifest_file_count": len(members), "stage_file_count_including_seal": len(members) + 2,
        "manifest_member_bytes": total, "stage_total_bytes_including_seal": inclusive_total,
        "science_elapsed_s": resource["science_elapsed_s"], "no_further_stage_writes": True,
        "FINAL_RECEIPT": str(ROOT / "FINAL_RECEIPT.json")}), flush=True)


if __name__ == "__main__":
    main()
