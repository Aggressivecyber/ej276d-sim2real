"""Root-authorized offline metadata gate tests; no SSH or scientific execution.

All stage files below are newly created JSON or visibly fake byte fixtures. The
production Linux process reader and subprocess launcher are replaced by explicit
synthetic functions. Scientific source is parsed only to check writer key names.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
PROJECT = ROOT.parents[2]
ACTUAL_STAGE = PROJECT / "work" / "R65_FULL_EMPIRICAL_D_RESPONSE_001"
ATTEMPT = ROOT / "synthetic_gate_checks" / "attempt_001"


def load_definitions(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if os.name != "nt" or not sys.dont_write_bytecode:
    raise RuntimeError("Offline synthetic tests require Windows python -B")
if ATTEMPT.exists():
    raise RuntimeError("Attempt already exists; preserve it and explicitly version any repair")
ARTIFICIAL_PROJECT = ATTEMPT / "resource_guard_project"
(ARTIFICIAL_PROJECT / "work" / "R65_FULL_EMPIRICAL_D_RESPONSE_001").mkdir(parents=True)
# Loading these own stdlib definitions neither runs their __main__ entry nor
# opens any actual project data. Resource guard starts before all metadata reads.
local = load_definitions(PREPARED / "r65_controller_common.py", "_r65_synthetic_local")
owner = local.ResourceOwner(ARTIFICIAL_PROJECT, "offline_synthetic_metadata_cases")
remote = load_definitions(PREPARED / "r65_remote_gate.py", "_r65_synthetic_remote")
STARTED = datetime.now(timezone.utc).isoformat()
results = []
forbidden_launch_calls = []
original_popen = remote.subprocess.Popen
original_remote_root = remote.REMOTE_ROOT
original_review_sha = remote.REVIEW_SHA
original_process_record = remote.process_record
LIVE = set()


def forbidden_popen(*args, **kwargs):
    forbidden_launch_calls.append({"args_repr": repr(args), "kwargs_keys": sorted(kwargs)})
    raise AssertionError("A synthetic metadata test reached subprocess.Popen")


def synthetic_process(pid):
    return {"PID": pid, "valid_PID": isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
            "exists": pid in LIVE, "synthetic": True}


def write(path, value):
    assert path.resolve().is_relative_to(ATTEMPT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def fake_bytes(path, label):
    assert path.resolve().is_relative_to(ATTEMPT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("SYNTHETIC NON-SCIENTIFIC BYTE FIXTURE; NOT AN ARRAY OR WEIGHT; " + label + "\n").encode())


def read(path):
    return json.loads(path.read_text("utf8"))


def mutate_json(path, function):
    value = read(path)
    function(value)
    write(path, value)


def source_constants(path):
    value = {}
    for node in ast.parse(path.read_text("utf8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return value


def native_fixture(root, entry, index, persist):
    pid, supervisor = 990100 + index, 990200 + index
    nonce = "synthetic_nonce_" + str(index)
    worker = {"PID": pid, "parent_PID": supervisor, "nonce": nonce, "entry": entry,
              "native_exit": 0, "error": None, "elapsed_s": 1., "self_peak_RSS_bytes": 1024 * 1024,
              "supervisor_bound_real_PID": True, "UTC": STARTED}
    identity = {"PID": pid, "parent_PID": supervisor, "nonce": nonce, "entry": entry,
                "UTC": STARTED, "cwd": str(root), "argv": ["synthetic_owned_worker.py", entry],
                "before_scientific_imports": True, "python": "SYNTHETIC"}
    ack = {"actual_PID": pid, "nonce": nonce, "actual_PID_memory_query_success": True,
           "supervisor_PID": supervisor, "RSS_limit_bytes": frozen_cfg["RSS_limit_bytes"], "UTC": STARTED}
    native = {"entry": entry, "supervisor_PID": supervisor, "actual_worker_PID": pid,
              "actual_worker_parent_PID": supervisor, "launcher_PID": pid,
              "actual_worker_native_exit": 0, "launcher_native_exit": 0, "termination_reason": None,
              "handshake_before_imports": True, "RSS_limit_bytes": frozen_cfg["RSS_limit_bytes"],
              "actual_worker_RSS_peak_bytes": 1024 * 1024, "job_tree_RSS_peak_bytes": 1024 * 1024,
              "worker_self_receipt": worker, "start_UTC": STARTED, "stop_UTC": STARTED, "elapsed_s": 1.}
    if persist:
        for suffix, row in ((".NATIVE_RECEIPT.json", native), (".WORKER_RECEIPT.json", worker),
                            (".SCIENCE_PID.json", identity), (".RSS_ACK.json", ack)):
            write(root / (entry + suffix), row)
    return native


def make_case(name, terminal="train_final.py", negative_AC=False):
    root = ATTEMPT / "cases" / name
    root.mkdir(parents=True)
    config = copy.deepcopy(frozen_cfg)
    config["remote_root"] = str(root)
    # Reproduce exactly the 112 names, but every file is a new small fake byte
    # object except the minimal controller JSON documents below. No original
    # binary is copied, opened or decoded.
    for row in frozen_manifest:
        fake_bytes(root / row["name"], row["name"])
    write(root / "config.json", config)
    reporting = copy.deepcopy(config)
    reporting["allowed_scientific_entries"] = ["summarize_and_plot.py"]
    reporting["GPU_scientific_entries"] = []
    write(root / "config_reporting.json", reporting)
    write(root / "INPUT_BINDING.json", {"inputs": {}, "synthetic": True})
    files = [{"name": row["name"], "sha256": local.sha(root / row["name"]),
              "bytes": (root / row["name"]).stat().st_size} for row in frozen_manifest]
    write(root / "ROOT_IMPLEMENTATION_REVIEW.json", {"files": files, "synthetic": True})
    remote.REMOTE_ROOT = str(root)
    remote.REVIEW_SHA = local.sha(root / "ROOT_IMPLEMENTATION_REVIEW.json")
    LIVE.clear()
    native = {entry: native_fixture(root, entry, index, index <= remote.ENTRIES.index(terminal))
              for index, entry in enumerate(remote.ENTRIES)}
    suite = {"pass": True, "implementation_review_SHA": remote.REVIEW_SHA, "synthetic": True}
    write(root / "ARTIFICIAL_SUITE_RECEIPT.json", suite)
    arm = root / remote.ARM
    arm.mkdir()
    for name0 in remote.ARTIFACTS:
        fake_bytes(arm / name0, name0)
    counter = dict(config["expected_completed_update_counts"])
    counter.update(response_fit_attempts=1, **{key: 0 for key in audit_constants["ZERO_COUNTERS"]})
    streams = dict(config["expected_R63_stream_hashes"])
    streams["full_SOURCE_epsilon"] = "8" * 64
    integrity = {key: True for key in remote.TRUE_INTEGRITY}
    integrity.update({key: False for key in remote.FALSE_INTEGRITY})
    integrity["sole_change"] = config["sole_scientific_change"]
    receipt = {"PID": native["train_final.py"]["actual_worker_PID"], "UTC": STARTED, "arm": remote.ARM,
        "updates": 50000, "unique_final_weight_step": 50000, "selection": "sole final50000",
        "new_response_training_runs": 1, "source_parents": 6593, "REAL_parents": 21047,
        "actual_unique_SOURCE_REAL_reference_REAL_identity": [6593, 21047, 21047],
        "initial_state_SHA": config["expected_R63_initial_state_SHA"], "final_state_SHA": "f" * 64,
        "counter": counter, "teacher_waveforms_constructed": 0, "DEV_read": 0, "EVAL_read": 0, "CAL_decode": 0,
        "R58_law_or_atoms_training_target": False, "h_PDG_network_inputs": False,
        "source_val_target_val_decoded": False, "compute_budget_enforced": False, "model_goal_complete": False,
        "full_D_SOURCE_parent_exposure_min_max": [50000, 50000], "full_D_REAL_parent_exposure_min_max": [50000, 50000],
        "loss_columns": config["loss_columns"], "precision_order_columns": config["precision_columns"],
        "replay_columns": config["replay_columns"], "loss_trace_rows": 50000, "precision_order_trace_rows": 50000,
        "replay_trace_rows": 50000, "discarded_D_trace_rows": 50000, "projection_direction_calls": 50000,
        "temporal_direction_calls": 50000, "full_SOURCE_epsilon_calls": 50000,
        "discarded_D_columns": config["full_D_scalar_order"] + ["original_nine_total", "discarded_D_total"],
        "gradient_execution_counts": dict(remote.GRADIENT_COUNTS), "full_empirical_D_integrity": integrity,
        "PCG64_seeds": config["PCG64_seeds"], "stream_hashes": streams,
        "frozen_binding": {key: local.sha(root / filename) for key, filename in {
            "protocol_SHA": "PROTOCOL.md", "config_SHA": "config.json", "input_binding_SHA": "INPUT_BINDING.json",
            "implementation_review_SHA": "ROOT_IMPLEMENTATION_REVIEW.json"}.items()},
        "artificial_suite_SHA256": local.sha(root / "ARTIFICIAL_SUITE_RECEIPT.json"),
        "artifact_sha256": {name0: local.sha(arm / name0) for name0 in remote.ARTIFACTS},
        "native_attempt_id": "synthetic_attempt_0", "logical_run_id": "synthetic_run",
        "recovery_history": [], "physical_attempt_compute": [{"PID": native["train_final.py"]["actual_worker_PID"],
            "attempt_id": "synthetic_attempt_0", "last_completed_update": 50000, "counters_complete": True}]}
    receipt["unique_final_weight_SHA256"] = receipt["artifact_sha256"]["FINAL_FLOW.pt"]
    assert set(receipt).issubset(training_writer_keys), sorted(set(receipt) - training_writer_keys)
    write(arm / "TRAINING_RECEIPT.json", receipt)

    audit = {"pass": True, "stage": remote.STAGE, "arm": remote.ARM, "all_updates_recorded": 50000,
             "checks": {key: True for key in audit_check_names}, "original_numerical_gate_and_A_F_thresholds_modified": False}
    write(root / "TRAINING_FROZEN_AUDIT.json", audit)
    model = {"pass": True, "candidate_steps": 128, "reference_only_steps": 256, "no_step_search": True,
             "TRAIN_SOURCE_REAL_rows": [64, 64], "median_limit_LSB": .25, "p95_limit_LSB": 1., "relative_q_limit": 1e-5,
             "RMS_median_LSB": .1, "RMS_p95_LSB": .5, "preADC_relative_q_max": 1e-6}
    write(root / "TRAIN_SOLVER_CHECK.json", {"PID": native["numerical_check.py"]["actual_worker_PID"],
          "both_pass": True, "DEV_read_or_generated": 0, "new_response_updates": 0,
          "new_models": 1, "historical_control_pass_reused": True, "models": {remote.ARM: model}})
    generation_counts = {"source_main_rows": 2000, "REAL_control_rows": 2000, "main_generation_rows": 4000,
        "repeat_noise_rows": 1024, "source_probe_rows": 832, "extra_guide_probe_rows": 128,
        "response_fit_or_update": 0, "HGB_attempts": 0, "density_EM_attempts": 0, "EVAL_read": 0, "CAL_waveform_decode": 0}
    write(root / "GENERATION_NUMERIC_RECEIPT.json", {"PID": native["generate_all.py"]["actual_worker_PID"],
          "status": "complete", "counters": generation_counts})
    write(root / "GENERATION_RECEIPT.json", {"counter": generation_counts, "all_generation_rows": 5984,
          "domains": {remote.ARM: {"rows": 2000}, remote.CONTROL: {"rows": 2000}}})
    for name0 in (remote.ARM, remote.CONTROL):
        fake_bytes(root / name0 / "generated.npz", name0 + " generated placeholder")
    evaluation_counts = {"HGB_attempts": 10, "density_description_slots": 16, "density_EM_attempts": 16,
                         "EVAL_read": 0, "CAL_waveform_decode": 0, "response_updates": 0}
    write(root / "EVALUATION_NUMERIC_RECEIPT.json", {"PID": native["evaluate_saved.py"]["actual_worker_PID"],
          "status": "complete", "counters": evaluation_counts})
    write(root / "metrics.json", {"status": "complete", "synthetic": True})
    write(root / "REPLAY_RECEIPT.json", {"PID": native["replay_all.py"]["actual_worker_PID"], "rows": 4000,
        "pass": True, "independent_process": True, "fit_or_update": 0, "portable_loader_executed": True,
        "endpoints": {name0: {"rows": 2000, "bitwise_pass": True,
                      "checks": {"pre_ADC_V": True, "waveforms_V": True}} for name0 in (remote.ARM, remote.CONTROL)}})
    write(root / "SUMMARY_PLOT_NUMERIC_RECEIPT.json", {"PID": native["summarize_and_plot.py"]["actual_worker_PID"],
          "aggregate_and_plot_only": True, "new_HGB": 0, "new_EM": 0, "new_neural_inference": 0})
    write(root / "GATE_METRICS.json", {"primary": remote.ARM, "primary_A_C_pass": not negative_AC,
          "decision": "close_fixed_R65_full_empirical_D_response" if negative_AC else "primary_pending_separately_registered_D_E"})
    write(root / "ACCEPTANCE_A_C.json", {remote.ARM: {key: {"pass": not negative_AC} for key in ("A", "B", "C")}})
    write(root / "REMOTE_ACTIVE_JOB.json", native[terminal])
    return root


def record(name, condition, detail):
    row = {"name": name, "pass": bool(condition), "detail": detail}
    results.append(row)
    write(ATTEMPT / "PROGRESS.json", {"checks": results, "subprocess_Popen_calls": len(forbidden_launch_calls)})
    if not condition:
        raise AssertionError(row)


def expect_launch_refusal(name, mutation, active_native_mutation=None):
    root = make_case(name)
    if mutation:
        mutation(root)
    if active_native_mutation:
        mutate_json(root / "REMOTE_ACTIVE_JOB.json", active_native_mutation)
    before = len(forbidden_launch_calls)
    error = None
    try:
        remote.launch(root, "numerical_check.py")
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    record(name, error is not None and len(forbidden_launch_calls) == before
           and not (root / "LAUNCH_numerical_check.py.json").exists(),
           {"refused": error, "no_remote_spawn": len(forbidden_launch_calls) == before, "no_launch_intent_written": True})


try:
    local.verify_controller_bundle()
    binding = read(ROOT / "SOURCE_BINDING.json")
    input_sources = []
    for name in ("config.json", "train_final.py", "r65_training_audit.py", "linux_supervise.py", "linux_owned_worker.py"):
        path = ACTUAL_STAGE / name
        expected = next(row for row in binding["sources"] if Path(row["path"]) == path)
        assert local.sha(path) == expected["sha256"]
        input_sources.append(expected)
    frozen_cfg = read(ACTUAL_STAGE / "config.json")
    frozen_manifest = read(PREPARED / "FROZEN_IMPLEMENTATION_MANIFEST.json")["files"]
    audit_constants = source_constants(ACTUAL_STAGE / "r65_training_audit.py")
    training_tree = ast.parse((ACTUAL_STAGE / "train_final.py").read_text("utf8"))
    nodes = [node for node in ast.walk(training_tree) if isinstance(node, ast.Dict)
             and any(isinstance(key, ast.Constant) and key.value == "unique_final_weight_step" for key in node.keys)]
    assert len(nodes) == 1
    training_writer_keys = {key.value for key in nodes[0].keys if isinstance(key, ast.Constant)}
    audit_tree = ast.parse((ACTUAL_STAGE / "r65_training_audit.py").read_text("utf8"))
    audit_check_names = sorted({node.slice.value for node in ast.walk(audit_tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "checks"
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)})
    assert len(audit_check_names) >= 30
    remote.subprocess.Popen = forbidden_popen
    remote.process_record = synthetic_process

    root = make_case("valid_completed_TRAIN")
    observed = remote.observe(root)
    record("valid_completed_TRAIN_ready_numerical", observed["state"] == "READY"
           and observed["ready_entry"] == "numerical_check.py" and set(observed["completed_chain"]) == {"train_final.py"}, observed)
    # Intermediate transitions exercise every actual field gate independently.
    for terminal, expected in (("numerical_check.py", "generate_all.py"), ("generate_all.py", "evaluate_saved.py"),
                               ("evaluate_saved.py", "replay_all.py"), ("replay_all.py", "summarize_and_plot.py")):
        root = make_case("valid_" + terminal.replace(".py", ""), terminal=terminal)
        observed = remote.observe(root)
        record("valid_" + terminal + "_next_" + expected, observed["state"] == "READY" and observed["ready_entry"] == expected,
               {"state": observed["state"], "ready_entry": observed["ready_entry"], "completed_chain": list(observed["completed_chain"])})

    expect_launch_refusal("live_actual_worker", lambda root: LIVE.add(990100))
    expect_launch_refusal("live_supervisor", lambda root: LIVE.add(990200))
    expect_launch_refusal("native_nonzero", lambda root: mutate_json(root / "train_final.py.NATIVE_RECEIPT.json",
        lambda row: row.update(actual_worker_native_exit=1)), lambda row: row.update(actual_worker_native_exit=1))
    expect_launch_refusal("wrong_nonce", lambda root: mutate_json(root / "train_final.py.RSS_ACK.json", lambda row: row.update(nonce="wrong")))
    expect_launch_refusal("missing_50000_field", lambda root: mutate_json(root / remote.ARM / "TRAINING_RECEIPT.json", lambda row: row.pop("updates")))
    expect_launch_refusal("only_49999_updates", lambda root: mutate_json(root / remote.ARM / "TRAINING_RECEIPT.json", lambda row: row.update(updates=49999)))
    expect_launch_refusal("missing_eighth_stream", lambda root: mutate_json(root / remote.ARM / "TRAINING_RECEIPT.json",
        lambda row: row["stream_hashes"].pop("full_SOURCE_epsilon")))
    expect_launch_refusal("wrong_artifact_SHA", lambda root: mutate_json(root / remote.ARM / "TRAINING_RECEIPT.json",
        lambda row: row["artifact_sha256"].update({"LOSS_BY_UPDATE_PRIVATE.npz": "0" * 64})))
    expect_launch_refusal("mutated_artifact_bytes", lambda root: fake_bytes(root / remote.ARM / "LOSS_BY_UPDATE_PRIVATE.npz", "different synthetic bytes"))
    expect_launch_refusal("missing_artifact_record", lambda root: mutate_json(root / remote.ARM / "TRAINING_RECEIPT.json",
        lambda row: row["artifact_sha256"].pop("REPLAY_BY_UPDATE_PRIVATE.npz")))
    expect_launch_refusal("frozen_file_changed", lambda root: fake_bytes(root / "network.py", "different frozen stub bytes"))

    root = make_case("cannot_skip_numerical")
    error = None
    try:
        remote.launch(root, "generate_all.py")
    except RuntimeError as exc:
        error = str(exc)
    record("cannot_skip_numerical", error is not None and not (root / "LAUNCH_generate_all.py.json").exists(), {"refused": error})

    root = make_case("negative_A_C_completed_report", terminal="summarize_and_plot.py", negative_AC=True)
    observed = remote.observe(root)
    summary = observed["completed_chain"]["summarize_and_plot.py"]
    record("negative_A_C_completed_report", observed["state"] == "COMPLETE" and observed["ready_entry"] is None
           and summary["primary_A_C_pass"] is False and summary["scientific_model_goal_complete"] is False,
           {"state": observed["state"], "ready_entry": observed["ready_entry"], "summary": summary})
    for forbidden_entry in ("train_final.py", "recover_train.py", "gate_D.py", "gate_E.py", "gate_F.py"):
        error = None
        try:
            remote.launch(root, forbidden_entry)
        except RuntimeError as exc:
            error = str(exc)
        record("forbidden_entry_" + forbidden_entry, error is not None, {"refused": error})
    assert not forbidden_launch_calls
    assert all(local.sha(Path(row["path"])) == row["sha256"] for row in input_sources)
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "status": "PASS", "checks": results,
        "passed": sum(row["pass"] for row in results), "total": len(results), "source_bindings": input_sources,
        "production_controller_sources": {name: local.sha(PREPARED / name) for name in ("r65_remote_gate.py", "r65_controller_common.py")},
        "synthetic_helper_SHA256": local.sha(Path(__file__)), "synthetic_only": True,
        "actual_training_writer_top_level_keys_checked": sorted(training_writer_keys),
        "fixture_keys_are_subset_of_frozen_writer_keys": True, "audit_check_names_from_frozen_source": audit_check_names,
        "actual_operational_entrypoints_executed": False, "remote_function_metadata_branches_executed_offline": True,
        "mocked": ["REMOTE_ROOT", "REVIEW_SHA for synthetic 112-file byte manifest", "process_record", "subprocess.Popen forbidden sentinel"],
        "only_static_or_unverified": ["Linux /proc reader and actual PID lifecycle", "SSH/SCP transport and timeouts",
            "actual supervisor spawn/machine admission/resource handshake", "local watcher timed loop and stop-file behavior",
            "local fetch archive/publication/remote-stability behavior", "real TRAIN completion and saved-array audit",
            "original 128/256 scientific calculation", "actual generation/evaluation/replay", "scientific A-F values and rendered figure quality"],
        "SSH_connections": 0, "subprocess_Popen_calls": 0, "uploads_or_downloads": 0, "science_launches": 0,
        "actual_scientific_array_or_weight_reads": 0, "root_metadata_or_Git_writes": 0,
        "all_real_sources_unchanged": True, "all_fixture_files_within_exclusive_directory": True}
    write(ATTEMPT / "RESULT.json", result)
    owner.finish("OFFLINE_SYNTHETIC_PASS")
    resource_path = owner.root / (owner.operation + ".RESOURCE_RECEIPT.json")
    write(ATTEMPT / "RESOURCE_RECEIPT.json", read(resource_path))
    print(json.dumps({"status": "PASS", "passed": result["passed"], "total": result["total"],
                      "RESULT_SHA256": local.sha(ATTEMPT / "RESULT.json"), "resource": read(resource_path),
                      "SSH_connections": 0, "subprocess_Popen_calls": 0}, allow_nan=False))
except BaseException as error:
    write(ATTEMPT / "FAILURE.json", {"UTC": datetime.now(timezone.utc).isoformat(), "error": repr(error), "checks": results,
          "forbidden_launch_calls": forbidden_launch_calls, "actual_scientific_array_or_weight_reads": 0, "SSH_connections": 0})
    owner.finish("OFFLINE_SYNTHETIC_FAILED", repr(error))
    raise
finally:
    remote.subprocess.Popen = original_popen
    remote.process_record = original_process_record
    remote.REMOTE_ROOT = original_remote_root
    remote.REVIEW_SHA = original_review_sha
