"""Standard-library remote controller payload; root must review before execution.

This file is sent through stdin, never installed over frozen stage sources.
It does not import scientific code or decode arrays/weights. Byte hashes are
checked only when a future root invocation requests a completed-stage gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

STAGE = "R65_FULL_EMPIRICAL_D_RESPONSE_001"
ARM = "FULL_EMPIRICAL_RESPONSE"
CONTROL = "FULL_EMPIRICAL_REAL_CONTROL"
REVIEW_SHA = "e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5"
REMOTE_ROOT = "/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC"
ENTRIES = ("train_final.py", "numerical_check.py", "generate_all.py",
           "evaluate_saved.py", "replay_all.py", "summarize_and_plot.py")
POST_ENTRIES = ENTRIES[1:]
ARTIFACTS = ("FINAL_FLOW.pt", "INITIALIZATION.json", "INITIAL_RANDOM_STATES.json",
             "FINAL_RANDOM_STATES.json", "LOSS_BY_UPDATE_PRIVATE.npz",
             "PRECISION_ORDER_BY_UPDATE_PRIVATE.npz", "REPLAY_BY_UPDATE_PRIVATE.npz",
             "DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz", "PARENT_EXPOSURES_PRIVATE.npz")
GRADIENT_COUNTS = {"combined_gradient_cast_events": 50000, "combined_clip_calls": 50000,
                   "optimizer_step_calls": 50000, "separate_D_or_C_clip_calls": 0,
                   "parameter_updates_between_D_passes": 0, "discarded_micro_D_backward_calls": 0,
                   "endpoint_mismatch_chunks": 0, "canonical_code_mismatch_chunks": 0,
                   "Box_free_mask_mismatch_chunks": 0}
TRUE_INTEGRITY = ("same_R63_initial_state", "seven_original_stream_hashes_equal_R63",
                  "seven_original_final_states_equal_R63", "C_parent_exposures_equal_R63",
                  "frozen_distribution_and_temporal_statistics_reused",
                  "discarded_micro_D_not_in_backward", "cast_clip_Adam_once_per_update",
                  "full_D_fresh_direction_binding_every_update", "D_endpoint_GY_once_per_update",
                  "all_original64_chunks_replayed", "all_endpoint_codes_free_masks_exact",
                  "D_then_C_CPU_float64_parameter_accumulation",
                  "source_and_REAL_masses_independently_preserved", "no_REAL_feature_cache",
                  "C_original_nine_call_four_piece_selection")
FALSE_INTEGRITY = ("new_statistics_fit", "new_normalization_fit", "new_noise_calibration_fit",
                   "new_valley_fit", "joint_D_C_noise_coupling_preserved")


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative_name(name):
    require(isinstance(name, str) and bool(name), "Empty relative path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and str(p) == name and all(
        part not in ("", ".", "..") and re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in p.parts), "Unsafe relative path: " + repr(name))
    return name


def under(root, name):
    path = root / relative_name(name)
    require(not path.is_symlink(), "Symlink cannot be a bound artifact: " + name)
    require(path.resolve().is_relative_to(root.resolve()), "Path leaves frozen stage")
    return path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(root, name):
    path = under(root, name)
    require(path.stat().st_size <= 8 * 1024 * 1024, "Unexpectedly large controller JSON: " + name)
    return json.loads(path.read_text("utf8"))


def process_record(pid):
    if not integer(pid) or pid <= 0:
        return {"PID": pid, "exists": False, "valid_PID": False}
    base = Path("/proc") / str(pid)
    if not base.exists():
        return {"PID": pid, "exists": False, "valid_PID": True}
    result = {"PID": pid, "exists": True, "valid_PID": True}
    try:
        result["argv"] = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf8", "replace")
        result["cwd"] = str((base / "cwd").resolve())
        result["status"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                            for line in (base / "status").read_text().splitlines()
                            if line.startswith(("State:", "PPid:", "VmRSS:", "VmHWM:", "Cpus_allowed_list:"))}
    except OSError as error:
        result["observation_error"] = repr(error)
    return result


def frozen_binding(root):
    require(str(root) == REMOTE_ROOT, "Wrong R65 remote root")
    review_path = under(root, "ROOT_IMPLEMENTATION_REVIEW.json")
    require(sha(review_path) == REVIEW_SHA, "Implementation review SHA changed")
    review = read(root, "ROOT_IMPLEMENTATION_REVIEW.json")
    rows = review["files"]
    require(isinstance(rows, list) and len(rows) == 112, "Frozen list is not exactly 112 files")
    require(len({row["name"] for row in rows}) == 112, "Duplicate frozen filename")
    for row in rows:
        path = under(root, row["name"])
        require(hex_sha(row["sha256"]) and integer(row["bytes"]), "Malformed frozen binding")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Frozen artifact changed: " + row["name"])
    cfg = read(root, "config.json")
    require(cfg["stage"] == STAGE and cfg["remote_root"] == REMOTE_ROOT and cfg["arm"] == ARM,
            "Config stage, remote root or arm differs")
    require(cfg["updates"] == 50000 and cfg["SOURCE_N"] == 6593 and cfg["REAL_N"] == 21047,
            "Frozen population/update definition differs")
    require(cfg["solver_steps"] == 128 and cfg["solver_check_steps"] == 256,
            "Original numerical resolutions changed")
    return cfg, {row["name"]: row for row in rows}


def native_success(root, entry):
    require(entry in ENTRIES, "Unregistered entry")
    native = read(root, entry + ".NATIVE_RECEIPT.json")
    worker = read(root, entry + ".WORKER_RECEIPT.json")
    identity = read(root, entry + ".SCIENCE_PID.json")
    ack = read(root, entry + ".RSS_ACK.json")
    pid = native.get("actual_worker_PID")
    supervisor = native.get("supervisor_PID")
    require(native.get("entry") == worker.get("entry") == identity.get("entry") == entry,
            "Native/worker/identity entry mismatch")
    require(integer(pid) and pid > 0 and integer(supervisor) and supervisor > 0,
            "Missing actual worker or supervisor identity")
    require(all(integer(v) and v == 0 for v in (
        native.get("actual_worker_native_exit"), native.get("launcher_native_exit"), worker.get("native_exit"))),
        "Actual worker/launcher native exit is not zero")
    require(native.get("termination_reason") is None and worker.get("error") is None,
            "Termination or worker failure was recorded")
    require(native.get("worker_self_receipt") == worker, "Supervisor-bound worker self receipt differs")
    require(native.get("handshake_before_imports") is True
            and identity.get("before_scientific_imports") is True
            and worker.get("supervisor_bound_real_PID") is True
            and ack.get("actual_PID_memory_query_success") is True,
            "Actual PID resource handshake was not completed")
    require(pid == worker.get("PID") == identity.get("PID") == ack.get("actual_PID")
            and supervisor == worker.get("parent_PID") == identity.get("parent_PID")
            == native.get("actual_worker_parent_PID") == ack.get("supervisor_PID"),
            "Actual PID/parent acknowledgement mismatch")
    require(isinstance(identity.get("nonce"), str) and bool(identity["nonce"])
            and identity["nonce"] == worker.get("nonce") == ack.get("nonce"), "Handshake nonce differs")
    require(identity.get("cwd") == str(root), "Worker cwd differs")
    require(not process_record(pid)["exists"] and not process_record(supervisor)["exists"],
            "Previous actual worker or supervisor still exists")
    cfg = read(root, "config_reporting.json" if entry == ENTRIES[-1] else "config.json")
    require(native.get("RSS_limit_bytes") == ack.get("RSS_limit_bytes") == cfg["RSS_limit_bytes"],
            "Native memory limit differs")
    for name in ("actual_worker_RSS_peak_bytes", "job_tree_RSS_peak_bytes"):
        value = native.get(name)
        require(integer(value) and 0 < value <= cfg["RSS_limit_bytes"], "Missing/excess native RSS: " + name)
    require(integer(worker.get("self_peak_RSS_bytes"))
            and 0 < worker["self_peak_RSS_bytes"] <= cfg["RSS_limit_bytes"], "Missing/excess worker peak RSS")
    return native


def training_complete(root, cfg):
    native = native_success(root, ENTRIES[0])
    receipt = read(root, ARM + "/TRAINING_RECEIPT.json")
    require(receipt.get("PID") == native["actual_worker_PID"] and receipt.get("arm") == ARM,
            "Training receipt is not bound to terminal actual worker")
    require(receipt.get("updates") == receipt.get("unique_final_weight_step") == 50000
            and receipt.get("selection") == "sole final50000"
            and receipt.get("new_response_training_runs") == 1, "Unique final50000 is incomplete")
    require(receipt.get("source_parents") == 6593 and receipt.get("REAL_parents") == 21047
            and receipt.get("actual_unique_SOURCE_REAL_reference_REAL_identity") == [6593, 21047, 21047],
            "Original parent population is incomplete")
    require(receipt.get("initial_state_SHA") == cfg["expected_R63_initial_state_SHA"]
            and hex_sha(receipt.get("final_state_SHA")), "Initialization/final state record differs")
    for key, value in cfg["expected_completed_update_counts"].items():
        actual = receipt["counter"].get(key)
        require(integer(actual) and actual == value, "Incomplete committed counter: " + key)
    require(receipt["counter"].get("response_fit_attempts") == 1, "More than one logical fit was recorded")
    for key in ("teacher_waveforms_constructed", "new_HGB_fits", "new_density_fits", "DEV_read", "EVAL_read", "CAL_decode"):
        require(receipt["counter"].get(key) == 0, "Unexpected training counter: " + key)
    for key in ("teacher_waveforms_constructed", "DEV_read", "EVAL_read", "CAL_decode"):
        require(receipt.get(key) == 0, "Unexpected training use: " + key)
    for key in ("R58_law_or_atoms_training_target", "h_PDG_network_inputs", "source_val_target_val_decoded",
                "compute_budget_enforced", "model_goal_complete"):
        require(receipt.get(key) is False, "Unexpected training assertion: " + key)
    require(receipt["full_D_SOURCE_parent_exposure_min_max"] == [50000, 50000]
            and receipt["full_D_REAL_parent_exposure_min_max"] == [50000, 50000], "Full-D exposure receipt incomplete")
    require(receipt.get("loss_columns") == cfg["loss_columns"] and len(cfg["loss_columns"]) == 32
            and receipt.get("precision_order_columns") == cfg["precision_columns"] and len(cfg["precision_columns"]) == 4
            and receipt.get("replay_columns") == cfg["replay_columns"] and len(cfg["replay_columns"]) == 12,
            "R65 saved-record schema differs")
    for key in ("loss_trace_rows", "precision_order_trace_rows", "replay_trace_rows", "discarded_D_trace_rows",
                "projection_direction_calls", "temporal_direction_calls", "full_SOURCE_epsilon_calls"):
        require(receipt.get(key) == 50000, "Incomplete R65 trace/draw count: " + key)
    discarded = receipt["discarded_D_columns"]
    require(isinstance(discarded, list) and len(discarded) == 7
            and set(discarded[:5]) == set(cfg["full_D_scalar_order"])
            and discarded[5:] == ["original_nine_total", "discarded_D_total"], "Discarded micro-D schema differs")
    for key, value in GRADIENT_COUNTS.items():
        require(receipt["gradient_execution_counts"].get(key) == value, "Gradient execution count differs: " + key)
    integrity = receipt["full_empirical_D_integrity"]
    require(all(integrity.get(key) is True for key in TRUE_INTEGRITY)
            and all(integrity.get(key) is False for key in FALSE_INTEGRITY), "R65 integrity metadata incomplete")
    require(receipt["PCG64_seeds"] == cfg["PCG64_seeds"]
            and set(receipt["stream_hashes"]) == set(cfg["PCG64_seeds"])
            and all(hex_sha(v) for v in receipt["stream_hashes"].values()), "Eight random-stream records incomplete")
    require(all(receipt["stream_hashes"][key] == value for key, value in cfg["expected_R63_stream_hashes"].items()),
            "Original seven stream hashes differ")
    frozen = receipt["frozen_binding"]
    for key, name in {"protocol_SHA": "PROTOCOL.md", "config_SHA": "config.json",
                      "input_binding_SHA": "INPUT_BINDING.json", "implementation_review_SHA": "ROOT_IMPLEMENTATION_REVIEW.json"}.items():
        require(frozen.get(key) == sha(under(root, name)), "Training frozen binding differs: " + key)
    require(frozen["implementation_review_SHA"] == REVIEW_SHA, "Training review SHA differs")
    suite = read(root, "ARTIFICIAL_SUITE_RECEIPT.json")
    require(suite.get("pass") is True and suite.get("implementation_review_SHA") == REVIEW_SHA
            and receipt.get("artificial_suite_SHA256") == sha(under(root, "ARTIFICIAL_SUITE_RECEIPT.json")),
            "Pre-TRAIN artificial pass not bound")
    require(set(receipt["artifact_sha256"]) == set(ARTIFACTS), "Complete original R65 artifact set missing")
    for name in ARTIFACTS:
        path = under(root, ARM + "/" + name)
        require(path.stat().st_size > 0 and hex_sha(receipt["artifact_sha256"][name])
                and sha(path) == receipt["artifact_sha256"][name], "Training artifact hash differs: " + name)
    require(receipt.get("unique_final_weight_SHA256") == receipt["artifact_sha256"]["FINAL_FLOW.pt"],
            "Unique final weight binding differs")
    attempts = receipt.get("physical_attempt_compute")
    require(isinstance(attempts, list) and len(attempts) >= 1 and isinstance(receipt.get("recovery_history"), list)
            and len(receipt["recovery_history"]) == len(attempts) - 1, "Physical attempt/recovery history missing")
    require(attempts[-1].get("PID") == receipt["PID"] and attempts[-1].get("attempt_id") == receipt["native_attempt_id"]
            and attempts[-1].get("last_completed_update") == 50000 and attempts[-1].get("counters_complete") is True,
            "Terminal physical attempt is not complete")
    return {"updates": 50000, "artifact_count": len(ARTIFACTS), "training_receipt_SHA256": sha(under(root, ARM + "/TRAINING_RECEIPT.json")),
            "worker_PID": receipt["PID"], "full_saved_array_audit_deferred_to_owned_numerical_entry": True}


def numerical_complete(root):
    native = native_success(root, "numerical_check.py")
    audit = read(root, "TRAINING_FROZEN_AUDIT.json")
    require(audit.get("pass") is True and audit.get("stage") == STAGE and audit.get("arm") == ARM
            and audit.get("all_updates_recorded") == 50000 and isinstance(audit.get("checks"), dict)
            and bool(audit["checks"]) and all(value is True for value in audit["checks"].values()), "R65 saved-record audit did not pass")
    require(audit.get("original_numerical_gate_and_A_F_thresholds_modified") is False,
            "Audit records modified thresholds")
    numeric = read(root, "TRAIN_SOLVER_CHECK.json")
    require(numeric.get("PID") == native["actual_worker_PID"] and numeric.get("both_pass") is True
            and numeric.get("DEV_read_or_generated") == 0 and numeric.get("new_response_updates") == 0
            and numeric.get("new_models") == 1 and numeric.get("historical_control_pass_reused") is True,
            "TRAIN numerical receipt incomplete")
    require(set(numeric.get("models", {})) == {ARM}, "TRAIN numerical model differs")
    model = numeric["models"][ARM]
    require(model.get("pass") is True and model.get("candidate_steps") == 128
            and model.get("reference_only_steps") == 256 and model.get("no_step_search") is True
            and model.get("TRAIN_SOURCE_REAL_rows") == [64, 64], "Original 128/256 TRAIN check incomplete")
    require(model.get("median_limit_LSB") == .25 and model.get("p95_limit_LSB") == 1.
            and model.get("relative_q_limit") == 1e-5, "Original numerical thresholds differ")
    require(0 <= model["RMS_median_LSB"] <= .25 and 0 <= model["RMS_p95_LSB"] <= 1.
            and 0 <= model["preADC_relative_q_max"] <= 1e-5, "Original numerical gate failed")
    return {"R65_training_audit_pass": True, "TRAIN_128_256_pass": True}


def generation_complete(root):
    native = native_success(root, "generate_all.py")
    record = read(root, "GENERATION_NUMERIC_RECEIPT.json")
    require(record.get("PID") == native["actual_worker_PID"] and record.get("status") == "complete",
            "Generation is not complete")
    counts = record["counters"]
    require([counts.get(k) for k in ("main_generation_rows", "repeat_noise_rows", "source_probe_rows", "extra_guide_probe_rows")]
            == [4000, 1024, 832, 128], "Original generation rows incomplete")
    require(all(counts.get(k) == 0 for k in ("response_fit_or_update", "HGB_attempts", "density_EM_attempts", "EVAL_read", "CAL_waveform_decode")),
            "Unexpected generation fits or data use")
    receipt = read(root, "GENERATION_RECEIPT.json")
    require(receipt.get("counter") == counts and receipt.get("all_generation_rows") == 5984,
            "Generation receipts disagree")
    for arm in (ARM, CONTROL):
        require(receipt["domains"][arm].get("rows") == 2000 and under(root, arm + "/generated.npz").stat().st_size > 0,
                "Required generated endpoint missing: " + arm)
    return {"generation_rows": 5984}


def evaluation_complete(root):
    native = native_success(root, "evaluate_saved.py")
    record = read(root, "EVALUATION_NUMERIC_RECEIPT.json")
    require(record.get("PID") == native["actual_worker_PID"] and record.get("status") == "complete", "Evaluation is incomplete")
    counts = record["counters"]
    require(counts.get("HGB_attempts") == 10 and counts.get("density_description_slots") == 16
            and integer(counts.get("density_EM_attempts")) and 0 <= counts["density_EM_attempts"] <= 16,
            "Frozen evaluator fit counts incomplete")
    require(all(counts.get(k) == 0 for k in ("EVAL_read", "CAL_waveform_decode", "response_updates")),
            "Unexpected evaluator data use or updates")
    # Native zero plus the terminal receipt binds the frozen writer, which writes
    # metrics before this receipt. Its potentially large aggregates need no parse.
    require(under(root, "metrics.json").stat().st_size > 0, "Saved metrics missing")
    return {"evaluation_complete": True}


def replay_complete(root):
    native = native_success(root, "replay_all.py")
    record = read(root, "REPLAY_RECEIPT.json")
    require(record.get("PID") == native["actual_worker_PID"] and record.get("rows") == 4000
            and record.get("pass") is True and record.get("independent_process") is True
            and record.get("fit_or_update") == 0 and record.get("portable_loader_executed") is True,
            "Independent replay did not pass")
    require(set(record.get("endpoints", {})) == {ARM, CONTROL}, "Replay endpoint set differs")
    for arm in (ARM, CONTROL):
        endpoint = record["endpoints"][arm]
        require(endpoint.get("rows") == 2000 and endpoint.get("bitwise_pass") is True
                and bool(endpoint.get("checks")) and all(v is True for v in endpoint["checks"].values()),
                "Bitwise replay failed for " + arm)
    return {"replay_rows": 4000, "bitwise_pass": True}


def reporting_complete(root):
    native = native_success(root, "summarize_and_plot.py")
    record = read(root, "SUMMARY_PLOT_NUMERIC_RECEIPT.json")
    require(record.get("PID") == native["actual_worker_PID"] and record.get("aggregate_and_plot_only") is True
            and all(record.get(k) == 0 for k in ("new_HGB", "new_EM", "new_neural_inference")),
            "Aggregate/report entry incomplete")
    gate = read(root, "GATE_METRICS.json")
    acceptance = read(root, "ACCEPTANCE_A_C.json")
    require(gate.get("primary") == ARM and ARM in acceptance and isinstance(gate.get("primary_A_C_pass"), bool),
            "Saved original acceptance result incomplete")
    require(all(k in acceptance[ARM] for k in ("A", "B", "C")), "A-C result missing")
    # Negative scientific gates are valid completed reports. This controller never
    # changes acceptance or starts separately registered D/E/F work.
    return {"report_complete": True, "primary_A_C_pass": gate["primary_A_C_pass"], "decision": gate.get("decision"),
            "scientific_model_goal_complete": False}


def completion_chain(root, cfg, through):
    require(through in ENTRIES, "Invalid terminal entry")
    results = {ENTRIES[0]: training_complete(root, cfg)}
    checks = (numerical_complete, generation_complete, evaluation_complete, replay_complete, reporting_complete)
    for entry, check in zip(POST_ENTRIES, checks):
        if ENTRIES.index(entry) <= ENTRIES.index(through):
            results[entry] = check(root)
    return results


def active_state(root):
    active = read(root, "REMOTE_ACTIVE_JOB.json")
    entry = active.get("entry")
    require(entry in ENTRIES, "Active entry is not in the R65 train/post sequence")
    worker = process_record(active.get("actual_worker_PID"))
    launcher = process_record(active.get("launcher_PID"))
    supervisor = process_record(active.get("supervisor_PID"))
    alive = any(p["exists"] for p in (worker, launcher, supervisor))
    return active, {"worker": worker, "launcher": launcher, "supervisor": supervisor}, alive


def observe(root):
    result = {"UTC": utc(), "stage": STAGE, "root": str(root), "mode": "observe",
              "controller_never_starts_or_recovers_training": True}
    cfg, files = frozen_binding(root)
    result.update(frozen_file_count=len(files), implementation_review_SHA256=REVIEW_SHA)
    active, processes, alive = active_state(root)
    result.update(active=active, processes=processes)
    if alive:
        result.update(state="RUNNING_OR_FINALIZING", ready_entry=None)
        return result
    if active.get("actual_worker_native_exit") is None:
        result.update(state="ATTENTION", ready_entry=None, reason="All recorded PIDs gone but no actual native exit")
        return result
    if active.get("actual_worker_native_exit") != 0 or active.get("termination_reason") is not None:
        result.update(state="ATTENTION", ready_entry=None, reason="Active attempt failed; no automatic retry or recovery")
        return result
    result["completed_chain"] = completion_chain(root, cfg, active["entry"])
    index = ENTRIES.index(active["entry"])
    result.update(state="COMPLETE" if index == len(ENTRIES) - 1 else "READY",
                  ready_entry=None if index == len(ENTRIES) - 1 else ENTRIES[index + 1])
    return result


def launch(root, entry):
    require(entry in POST_ENTRIES, "Only the five registered post entries may be launched")
    before = observe(root)
    require(before["state"] == "READY" and before["ready_entry"] == entry,
            "The true predecessor has not completed successfully")
    cfg, _ = frozen_binding(root)
    supervisor_name = "linux_supervise_reporting.py" if entry == ENTRIES[-1] else "linux_supervise.py"
    launch_cfg = read(root, "config_reporting.json" if entry == ENTRIES[-1] else "config.json")
    require(entry in launch_cfg["allowed_scientific_entries"], "Entry absent from frozen supervisor config")
    for suffix in (".used", ".NATIVE_RECEIPT.json", ".SCIENCE_PID.json"):
        require(not under(root, entry + suffix).exists(), "Entry already attempted; root must inspect existing evidence")
    marker = under(root, "LAUNCH_" + entry + ".json")
    record = {"UTC": utc(), "entry": entry, "controller": "R65 root-reviewed post controller",
              "implementation_review_SHA256": REVIEW_SHA, "predecessor": before["active"]["entry"],
              "status": "INTENT_RECORDED_BEFORE_SPAWN", "automatic_retry": False,
              "completed_chain": before["completed_chain"]}
    with marker.open("x", encoding="utf8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # No training/source code is imported here. The original frozen supervisor
    # performs the actual machine admission and owned-worker handshake.
    cpus = sorted(os.sched_getaffinity(0))[:cfg["CPU_threads"]]
    require(len(cpus) == cfg["CPU_threads"], "Fewer than the registered CPU cores are available")
    os.sched_setaffinity(0, cpus)
    with under(root, entry + ".supervisor.stdout.log").open("xb") as out, under(root, entry + ".supervisor.stderr.log").open("xb") as err:
        process = subprocess.Popen([cfg["server_python"], "-B", str(root / supervisor_name), entry],
                                   cwd=root, stdout=out, stderr=err, env=os.environ.copy(), start_new_session=True)
    record.update(status="SUPERVISOR_SPAWNED_COMPLETION_UNCONFIRMED", supervisor_PID=process.pid, spawned_UTC=utc())
    # Keep the exclusive intent marker as the immutable duplicate-submission lock.
    with under(root, "LAUNCH_" + entry + ".SPAWN.json").open("x", encoding="utf8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    time.sleep(1.)
    record["supervisor_process"] = process_record(process.pid)
    try:
        current = read(root, "REMOTE_ACTIVE_JOB.json")
        record["active_after_spawn"] = current
        record["active_entry_changed"] = current.get("entry") == entry
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        record["active_observation_error"] = repr(error)
    record["instruction"] = "Observe actual worker/handshake/native receipt; spawn alone is not success."
    return record


def manifest(root, entry):
    require(entry in ENTRIES, "Invalid fetch terminal entry")
    status = observe(root)
    require(status["state"] in ("READY", "COMPLETE") and status["active"]["entry"] == entry,
            "Fetch requires requested true terminal entry and no active process")
    rows = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in ("cache", "__pycache__", ".git") for part in rel.parts) or path.suffix == ".pyc":
            continue
        require(not path.is_symlink(), "Fetch refuses stage symlink: " + str(rel))
        if not path.is_file():
            continue
        name = relative_name(rel.as_posix())
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    require(rows, "Empty remote manifest")
    return {"UTC": utc(), "stage": STAGE, "root": str(root), "terminal_entry": entry,
            "implementation_review_SHA256": REVIEW_SHA, "verified_terminal": status, "files": rows,
            "excluded_directories": ["cache", "__pycache__", ".git"], "download_started": False}


def main(request):
    require(os.name == "posix" and sys.dont_write_bytecode, "Remote payload requires Linux python -B")
    require(request.get("review_SHA256") == REVIEW_SHA and request.get("remote_root") == REMOTE_ROOT,
            "Request is not bound to the frozen R65 root")
    root = Path(REMOTE_ROOT)
    mode = request.get("mode")
    if mode == "observe":
        return observe(root)
    if mode == "launch":
        return launch(root, request.get("entry"))
    if mode == "manifest":
        return manifest(root, request.get("entry"))
    raise RuntimeError("Unsupported remote controller mode")


if __name__ == "__main__":
    try:
        result = main(REQUEST)  # REQUEST is a JSON literal injected by the local controller.
        print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
    except BaseException as error:
        print(json.dumps({"ok": False, "UTC": utc(), "error_type": type(error).__name__, "error": str(error),
                          "no_automatic_retry_or_recovery": True}, allow_nan=False), flush=True)
        raise SystemExit(2)
