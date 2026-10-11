"""Build a draft R65 same-run request from saved JSON evidence only.

This program never queries a process, loads a checkpoint/tensor, removes a used
marker, installs a request in the remote stage, or launches a worker.  External
native/archive observations are checked for consistency, not re-performed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from datetime import datetime

VERSION = 1
STAGE = "R65_FULL_EMPIRICAL_D_RESPONSE_001"
REMOTE_ROOT = "/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC"
RUN_ID = STAGE + "@" + REMOTE_ROOT
ARM = "FULL_EMPIRICAL_RESPONSE"
ENTRY = "train_final.py"
TRAINER_SHA = "54834b0fd73ee08a420096e791be6038c2193acfb22c91c81563e059b64c67f3"
FROZEN_HASHES = {
    "protocol_SHA": "324b8e925cbc1b65578b3e36614693ae6cfb5561d0b3e379293ceab7aa4e256b",
    "config_SHA": "1aacb74b0cd443d0ea74fe898fd4f7aabe40f2082fa52e0bbb6579d54e1d2caf",
    "input_binding_SHA": "b5db5dd9efa348ecc7a8a7fff651ae5acb7a7a57b2d779d19abd28b2cda0910e",
    "implementation_review_SHA": "e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5",
}
UNIQUE_UPDATES = 50000
PER_UPDATE = {
    "new_response_updates": 1, "SOURCE_training_rows": 320,
    "REAL_reference_rows": 320, "REAL_identity_rows": 64,
    "full_D_SOURCE_law_rows": 6593, "full_D_REAL_law_rows": 21047,
    "scientific_neural_forward_rows": 13570, "scientific_neural_forward_calls": 209,
    "D_parameter_VJP_calls": 104, "C_parameter_reverse_calls": 1,
    "full_D_endpoint_reverse_calls": 1, "full_D_objective_calls": 1,
    "replay_chunks_checked": 104,
}
PHYSICAL_MAP = {
    "new_response_updates": "optimizer_step_calls",
    "SOURCE_training_rows": "C_SOURCE_objective_rows",
    "REAL_reference_rows": "C_REAL_reference_objective_rows",
    "REAL_identity_rows": "C_REAL_identity_objective_rows",
    "full_D_SOURCE_law_rows": "full_D_SOURCE_law_rows",
    "full_D_REAL_law_rows": "full_D_REAL_law_rows",
    "scientific_neural_forward_rows": "neural_forward_rows",
    "scientific_neural_forward_calls": "neural_forward_calls",
    "D_parameter_VJP_calls": "D_parameter_VJP_calls",
    "C_parameter_reverse_calls": "C_parameter_reverse_calls",
    "full_D_endpoint_reverse_calls": "full_D_endpoint_reverse_calls",
    "full_D_objective_calls": "full_D_objective_calls",
    "replay_chunks_checked": "endpoint_replay_checked_chunks",
}
PHYSICAL_KEYS = (
    "neural_forward_rows", "neural_forward_calls", "full_D_objective_calls",
    "full_D_endpoint_reverse_calls", "D_parameter_VJP_calls", "C_objective_calls",
    "C_parameter_reverse_calls", "parameter_cast_calls", "gradient_clip_calls",
    "optimizer_step_calls", "D_source_noise_rows", "C_SOURCE_noise_rows",
    "C_identity_noise_rows", "full_window_direction_draw_calls",
    "temporal_direction_draw_calls", "endpoint_replay_checked_chunks",
    "full_D_SOURCE_law_rows", "full_D_REAL_law_rows", "C_SOURCE_objective_rows",
    "C_REAL_reference_objective_rows", "C_REAL_identity_objective_rows",
)
FINAL_NAMES = (
    "LOSS_BY_UPDATE_PRIVATE.npz", "PRECISION_ORDER_BY_UPDATE_PRIVATE.npz",
    "REPLAY_BY_UPDATE_PRIVATE.npz", "DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz",
    "PARENT_EXPOSURES_PRIVATE.npz", "FINAL_FLOW.pt", "FINAL_TORCH_RNG.pt",
    "FINAL_RANDOM_STATES.json", "TRAINING_RECEIPT.json",
)
MUTABLE_PATHS = (
    ENTRY + ".used", ENTRY + ".stdout.log", ENTRY + ".stderr.log",
    ENTRY + ".SCIENCE_PID.json", ENTRY + ".RSS_ACK.json",
    ENTRY + ".WORKER_RECEIPT.json", ENTRY + ".NATIVE_RECEIPT.json",
    ENTRY + ".START_RESOURCE.json", "REMOTE_ACTIVE_JOB.json",
    "REMOTE_RUN_HISTORY.json", "LAUNCH_RECEIPT.json",
    "RECOVERY_REQUEST.json", "RECOVERY_CHECKPOINT.json",
    "RECOVERY_CHECKPOINT_PENDING.json", "cache/RECOVERY_SAME_RUN.pt",
    "cache/RECOVERY_SAME_RUN.tmp.pt", "PHYSICAL_PROGRESS.json",
    "TRAINING_FAILURE.json", "REPLAY_MISMATCH.json", "TRAIN_STARTED.json",
    "TRAIN_PROGRESS.json", "ARTIFICIAL_SUITE_RECEIPT.json",
    ARM + "/ENVIRONMENT.json", ARM + "/INITIALIZATION.json",
    ARM + "/INITIAL_RANDOM_STATES.json", ARM + "/TRAINING_TRACE.json",
) + tuple(ARM + "/" + name for name in FINAL_NAMES)
PUBLICATIONS = {
    "publication_committed": ("RECOVERY_CHECKPOINT.json", "committed"),
    "publication_pending": (
        "RECOVERY_CHECKPOINT_PENDING.json",
        "durable_pending_atomic_checkpoint_publication"),
}
REQUEST_KEYS = {
    "logical_run_id", "prior_actual_PID_gone", "prior_native_attempt_preserved",
    "frozen_binding", "checkpoint_publication_receipt_path",
    "checkpoint_publication_receipt_SHA256", "committed_update",
    "checkpoint_creator_attempt_id", "checkpoint_SHA256",
    "preserved_physical_attempt_compute", "previous_native_attempt_id",
    "prior_recovery_history", "preserved_finalization_outputs",
}
MAX_JSON_BYTES = 16 * 1024 ** 2


class EvidenceError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def identity(value):
    return isinstance(value, str) and bool(value) and len(value) <= 128 and all(
        c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)


def utc(value):
    need(isinstance(value, str), "UTC must be a timezone-aware string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("invalid UTC") from exc
    need(result.tzinfo is not None, "naive time is not evidence")
    return result


def relative(value):
    need(isinstance(value, str) and "\\" not in value, "relative paths must use slash")
    path = PurePosixPath(value)
    need(bool(value) and not path.is_absolute() and str(path) == value
         and ".." not in path.parts and "." not in path.parts
         and ":" not in value, "unsafe relative path")
    return value


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw.decode("utf8"), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(EvidenceError("non-finite JSON number")))


class Documents:
    def __init__(self, manifest_path):
        manifest_path = Path(manifest_path).resolve(strict=True)
        need(manifest_path.suffix == ".json", "manifest must be JSON")
        self.root = manifest_path.parent
        self.manifest, self.manifest_sha = self._read(manifest_path)
        need(self.manifest["schema_version"] == VERSION, "unknown bundle schema")
        need(self.manifest["evidence_kind"] in ("root_saved_evidence", "artificial_saved_record"),
             "unknown evidence kind")
        self.values = {}
        self.hashes = {}
        for role, ref in self.manifest["references"].items():
            need(isinstance(role, str) and role, "invalid evidence role")
            local = relative(ref["path"])
            path = self.root / local
            need(path.suffix == ".json", "only JSON documents may be opened")
            for parent in (path,) + tuple(path.parents):
                if parent == self.root:
                    break
                need(not parent.is_symlink(), "symlink evidence refused")
            resolved = path.resolve(strict=True)
            need(self.root in resolved.parents, "evidence escaped bundle")
            value, sha = self._read(resolved)
            need(digest(ref["sha256"]) and sha == ref["sha256"], "saved document SHA mismatch: " + role)
            self.values[role] = value
            self.hashes[role] = sha

    @staticmethod
    def _read(path):
        need(path.stat().st_size <= MAX_JSON_BYTES, "JSON document exceeds bound")
        raw = path.read_bytes()
        need(len(raw) <= MAX_JSON_BYTES, "JSON document grew past bound")
        return strict_json(raw), hashlib.sha256(raw).hexdigest()

    def get(self, role):
        need(role in self.values, "missing saved evidence: " + role)
        return self.values[role]

    def sha(self, role):
        self.get(role)
        return self.hashes[role]


def verify_frozen(documents):
    manifest = documents.manifest
    need(manifest["logical_run_id"] == RUN_ID, "wrong logical run")
    binding = manifest["frozen_binding"]
    need(set(binding) == set(FROZEN_HASHES) | {"environment"}, "incomplete frozen binding")
    need(all(binding[key] == value for key, value in FROZEN_HASHES.items()),
         "frozen protocol/config/input/review hash changed")
    need(isinstance(binding["environment"], dict) and binding["environment"], "missing frozen environment")
    proof = documents.get("frozen_verification")
    need(proof["logical_run_id"] == RUN_ID and proof["frozen_binding"] == binding,
         "frozen revalidation binding mismatch")
    need(proof["trainer_SHA256"] == TRAINER_SHA, "trainer changed")
    for key in ("all_reviewed_source_hashes_verified", "all_scientific_input_hashes_verified",
                "environment_exactly_matches_bound_environment",
                "no_scientific_parameters_changed"):
        need(proof[key] is True, "missing root revalidation: " + key)
    need(proof["verification_scope"] == "saved_root_observation_no_training_advance",
         "unexpected revalidation scope")
    utc(proof["UTC"])
    suite = documents.get("artificial_suite")
    need(suite["pass"] is True and suite["implementation_review_SHA"] == FROZEN_HASHES["implementation_review_SHA"],
         "artificial suite missing or bound to another implementation")
    return binding


def verify_native(documents, prefix, attempt_id):
    hs = documents.get(prefix + "science_pid")
    ack = documents.get(prefix + "rss_ack")
    native = documents.get(prefix + "native_receipt")
    death = documents.get(prefix + "death_confirmation")
    worker, supervisor = hs["PID"], hs["parent_PID"]
    need(integer(worker, 1) and integer(supervisor, 1) and worker != supervisor, "bad saved PID identity")
    need(hs["nonce"] == attempt_id and hs["entry"] == ENTRY and hs["cwd"] == REMOTE_ROOT
         and hs["before_scientific_imports"] is True, "science handshake identity mismatch")
    need(ack["actual_PID"] == worker and ack["supervisor_PID"] == supervisor
         and ack["nonce"] == attempt_id and ack["actual_PID_memory_query_success"] is True,
         "RSS acknowledgement identity mismatch")
    need(native["entry"] == ENTRY and native["actual_worker_PID"] == worker
         and native["supervisor_PID"] == supervisor and native["launcher_PID"] == worker
         and native["handshake_before_imports"] is True, "native receipt identity mismatch")
    need(type(native["actual_worker_native_exit"]) is int
         and native["launcher_native_exit"] == native["actual_worker_native_exit"],
         "native wait returncode required; absence is not native zero")
    if "worker_self_receipt" in native:
        saved = native["worker_self_receipt"]
        need(saved["PID"] == worker and saved["parent_PID"] == supervisor
             and saved["nonce"] == attempt_id and saved["entry"] == ENTRY
             and saved["native_exit"] == native["actual_worker_native_exit"],
             "embedded worker receipt contradicts supervisor")
    need(death["logical_run_id"] == RUN_ID and death["attempt_id"] == attempt_id
         and death["actual_worker_PID"] == worker and death["supervisor_PID"] == supervisor,
         "PID-death evidence identity mismatch")
    for key in ("same_host_and_process_namespace_verified", "saved_handshake_and_native_identity_verified",
                "actual_worker_PID_absent", "supervisor_PID_absent",
                "no_matching_attempt_or_owned_descendants_alive"):
        need(death[key] is True, "death observation missing: " + key)
    need(death["native_receipt_SHA256"] == documents.sha(prefix + "native_receipt"),
         "death observation does not bind native receipt")
    need(utc(death["UTC"]) >= utc(native["stop_UTC"]) >= utc(hs["UTC"]),
         "death/exit chronology contradiction")
    need(isinstance(death["collector"], str) and death["collector"], "root evidence collector missing")
    return worker, supervisor, death


def verify_archive(documents, prefix, attempt_id, death):
    inv = documents.get(prefix + "archive_inventory")
    need(inv["logical_run_id"] == RUN_ID and inv["attempt_id"] == attempt_id, "archive identity mismatch")
    expected_root = "native_attempts/" + attempt_id
    need(inv["archive_root"] == expected_root, "wrong archive root")
    for key in ("all_stage_files_except_native_attempts_inventory_closed",
                "all_existing_files_copied_and_rehashed", "prior_attempt_archives_preserved",
                "independent_byte_copies_no_hardlinks_or_symlinks"):
        need(inv[key] is True, "archive closure missing: " + key)
    need(utc(inv["UTC"]) >= utc(death["UTC"]), "archive must follow confirmed death")
    files = inv["files"]
    need(isinstance(files, dict) and set(MUTABLE_PATHS) <= set(files), "mutable inventory has unchecked paths")
    for name, record in files.items():
        relative(name)
        need(not name.startswith("native_attempts/"), "recursive archive inventory refused")
        need(type(record["exists"]) is bool, "inventory existence must be explicit")
        if record["exists"]:
            archive_name = relative(record["archive_relative_path"])
            need(archive_name == expected_root + "/stage/" + name, "archive path mismatch")
            need(digest(record["source_sha256"]) and record["source_sha256"] == record["archive_sha256"],
                 "archive byte hash mismatch")
            need(integer(record["source_bytes"]) and record["archive_bytes"] == record["source_bytes"],
                 "archive byte length mismatch")
            need(record["regular_file"] is True and record["source_symlink"] is False,
                 "non-regular archive source refused")
    mandatory = (ENTRY + ".used", ENTRY + ".stdout.log", ENTRY + ".stderr.log",
                 ENTRY + ".SCIENCE_PID.json", ENTRY + ".RSS_ACK.json",
                 ENTRY + ".NATIVE_RECEIPT.json", ENTRY + ".START_RESOURCE.json",
                 "REMOTE_ACTIVE_JOB.json", "REMOTE_RUN_HISTORY.json")
    need(all(files[name]["exists"] for name in mandatory), "native/log/used history missing")
    for role, name in (("science_pid", ENTRY + ".SCIENCE_PID.json"),
                       ("rss_ack", ENTRY + ".RSS_ACK.json"),
                       ("native_receipt", ENTRY + ".NATIVE_RECEIPT.json")):
        need(files[name]["source_sha256"] == documents.sha(prefix + role), "native evidence not archived: " + role)
    return files, inv


def physical_record(documents, prefix, descriptor, pid):
    snapshot = documents.get(prefix + "physical_snapshot")
    kind = descriptor["physical_source_kind"]
    start = descriptor["resumed_from_update"]
    need(integer(start) and start <= UNIQUE_UPDATES and start % 500 == 0, "invalid resumed baseline")
    baseline_only = False
    if kind in ("PHYSICAL_PROGRESS", "TRAINING_FAILURE"):
        need(snapshot["logical_run_id"] == RUN_ID and snapshot["native_attempt_id"] == descriptor["attempt_id"]
             and snapshot["PID"] == pid, "physical snapshot identity mismatch")
        raw_end = snapshot["last_completed_update"] if kind == "PHYSICAL_PROGRESS" else snapshot["updates"]
        current = snapshot["current_update"]
        ops = snapshot["known_completed_physical_operations_this_attempt"]
        counter = snapshot["physical_schema_counter_lower_bound"]
        phase = snapshot["phase"]
        if (start > 0 and raw_end == 0 and all(value == 0 for value in counter.values())
                and phase in ("startup_before_artificial_checks", "artificial_passed_before_TRAIN_decode")):
            # The incoming immutable request proves this logical baseline.  It does
            # not prove that the failed worker restored it or made zero later work.
            end, baseline_only = start, True
        else:
            end = raw_end
            need(integer(current) and end <= current <= min(UNIQUE_UPDATES, end + 1),
                 "physical current/completed update mismatch")
            if kind == "PHYSICAL_PROGRESS":
                need(snapshot["attempt_start_committed_update"] == start, "snapshot resume baseline mismatch")
        observed = snapshot["UTC"]
    elif kind == "CHECKPOINT_METADATA":
        need(snapshot["scope"] == "root_saved_checkpoint_metadata_no_training_advance",
             "checkpoint metadata provenance missing")
        need(snapshot["logical_run_id"] == RUN_ID and snapshot["native_attempt_id"] == descriptor["attempt_id"]
             and snapshot["checkpoint_transaction_version"] == 1
             and snapshot["attempt_start_update"] == start, "checkpoint metadata identity mismatch")
        need(digest(snapshot["checkpoint_SHA256"]) and snapshot["all_fields_after_completed_optimizer_and_log_commit"] is True,
             "checkpoint metadata lacks transaction binding")
        end = snapshot["committed_update"]
        ops = snapshot["physical_this_attempt"]
        counter = {key: ops[value] for key, value in PHYSICAL_MAP.items()}
        observed = snapshot["publication_UTC"]
        phase = "saved_checkpoint_metadata"
    else:
        raise EvidenceError("unsupported physical snapshot source")
    utc(observed)
    need(integer(end) and start <= end <= UNIQUE_UPDATES, "invalid physical completed range")
    need(set(ops) == set(PHYSICAL_KEYS) and all(integer(value) for value in ops.values()),
         "physical operation keys/counts missing")
    need(set(counter) == set(PER_UPDATE) and all(integer(value) for value in counter.values()),
         "physical schema counters incomplete")
    need(all(counter[key] == ops[value] for key, value in PHYSICAL_MAP.items()),
         "physical schema mapping contradicts saved operations")
    adam = counter["new_response_updates"]
    need(adam >= end - start, "known Adam calls below known completed updates")
    need(all(counter[key] >= adam * value for key, value in PER_UPDATE.items()),
         "physical operations below known Adam-step minimum")
    # Before endpoint, a progress/failure/checkpoint snapshot cannot attest the
    # unseen interval to process death.  50000 is a conservative protocol bound,
    # not a runtime/exposure estimate or invented missing operation count.
    complete = end == UNIQUE_UPDATES
    intervals = [] if complete else [{
        "first_unpersisted_update": end + 1, "last_possible_update": UNIQUE_UPDATES,
        "reason": "Saved evidence is a lower bound; operations after its timestamp through confirmed process death are unknown.",
        "after_saved_snapshot_UTC": observed,
    }]
    return {
        "attempt_id": descriptor["attempt_id"], "PID": pid,
        "resumed_from_update": start, "last_completed_update": end,
        "completed_operations": copy.deepcopy(ops), "counter": copy.deepcopy(counter),
        "counters_complete": complete, "unknown_intervals": intervals,
        "known_uncommitted_optimizer_steps": adam - (end - start),
        "exact_completed_operation_counts": complete,
        "unknown_after_crash_fields": [] if complete else list(PHYSICAL_KEYS),
        "prior_native_attempts_separately_preserved": True,
        "evidence_source_kind": kind, "evidence_SHA256": documents.sha(prefix + "physical_snapshot"),
        "evidence_UTC": observed, "snapshot_phase": phase,
        "logical_baseline_from_incoming_request_before_restore": baseline_only,
        "unknown_nontraining_finalize_or_startup_work_not_in_these_counters": True,
    }


def validate_saved_documents(documents):
    binding = verify_frozen(documents)
    attempts = documents.manifest["attempts"]
    need(isinstance(attempts, list) and attempts, "no prior attempts")
    need(len(attempts) <= 1000, "unbounded attempt list")
    records, history, ids, inventories = [], [], [], []
    latest_death = None
    for number, descriptor in enumerate(attempts):
        ident = descriptor["attempt_id"]
        need(identity(ident) and ident not in ids, "invalid/duplicate attempt ID")
        need(descriptor["resumed_from_update"] == 0 if number == 0
             else descriptor["resumed_from_update"] >= 500, "invalid first or resumed attempt baseline")
        prefix = "attempt/" + ident + "/"
        worker, supervisor, death = verify_native(documents, prefix, ident)
        files, inventory = verify_archive(documents, prefix, ident, death)
        if number:
            old = documents.get(prefix + "previous_recovery_request")
            need(set(old) == REQUEST_KEYS, "old request interface/fields changed")
            need(files["RECOVERY_REQUEST.json"]["exists"]
                 and files["RECOVERY_REQUEST.json"]["source_sha256"] == documents.sha(prefix + "previous_recovery_request"),
                 "used recovery request not archived")
            need(old["logical_run_id"] == RUN_ID and old["frozen_binding"] == binding
                 and old["prior_actual_PID_gone"] is True and old["prior_native_attempt_preserved"] is True,
                 "old request run/binding/death mismatch")
            need(old["preserved_physical_attempt_compute"] == records
                 and old["prior_recovery_history"] == history, "immutable prior recovery chain changed")
            need(old["previous_native_attempt_id"] == ids[-1]
                 and old["checkpoint_creator_attempt_id"] in ids
                 and old["committed_update"] == descriptor["resumed_from_update"]
                 and old["committed_update"] <= records[-1]["last_completed_update"],
                 "old request creator/adjacency/checkpoint mismatch")
            need(old["checkpoint_publication_receipt_path"] in {item[0] for item in PUBLICATIONS.values()}
                 and digest(old["checkpoint_publication_receipt_SHA256"]) and digest(old["checkpoint_SHA256"]),
                 "old request transaction digest missing")
            prior_files = inventories[-1][0]
            selected_name = old["checkpoint_publication_receipt_path"]
            need(prior_files[selected_name]["exists"]
                 and prior_files[selected_name]["source_sha256"] == old["checkpoint_publication_receipt_SHA256"]
                 and prior_files["cache/RECOVERY_SAME_RUN.pt"]["exists"]
                 and prior_files["cache/RECOVERY_SAME_RUN.pt"]["source_sha256"] == old["checkpoint_SHA256"],
                 "old request SHA transaction not closed by predecessor archive")
            prior_publication = documents.get(prefix + "prior_selected_publication")
            need(documents.sha(prefix + "prior_selected_publication") == old["checkpoint_publication_receipt_SHA256"]
                 and prior_publication["logical_run_id"] == RUN_ID
                 and prior_publication["frozen_binding"] == binding
                 and prior_publication["checkpoint_SHA256"] == old["checkpoint_SHA256"]
                 and prior_publication["committed_update"] == old["committed_update"]
                 and prior_publication["native_attempt_id"] == old["checkpoint_creator_attempt_id"]
                 and prior_publication["transaction_version"] == 1
                 and prior_publication["checkpoint_path"] == "cache/RECOVERY_SAME_RUN.pt"
                 and prior_publication["all_fields_after_completed_optimizer_and_log_commit"] is True
                 and prior_publication["publication_status"] == (
                     "committed" if selected_name == "RECOVERY_CHECKPOINT.json"
                     else "durable_pending_atomic_checkpoint_publication"),
                 "old selected transaction receipt contradicts old request")
            expected_old_final = [
                {"name": name, "sha256": prior_files[ARM + "/" + name]["source_sha256"],
                 "archive_relative_path": prior_files[ARM + "/" + name]["archive_relative_path"]}
                for name in FINAL_NAMES if prior_files[ARM + "/" + name]["exists"]]
            need(old["preserved_finalization_outputs"] == expected_old_final,
                 "old finalization preservation evidence changed")
            history.append({
                "from_attempt_id": ids[-1], "to_attempt_id": ident,
                "checkpoint_creator_attempt_id": old["checkpoint_creator_attempt_id"],
                "checkpoint_update": old["committed_update"], "checkpoint_sha256": old["checkpoint_SHA256"],
                "checkpoint_publication_receipt_path": old["checkpoint_publication_receipt_path"],
                "checkpoint_publication_receipt_SHA256": old["checkpoint_publication_receipt_SHA256"],
                "bindings_verified": True, "previous_PID_confirmed_dead": True,
            })
        else:
            need(not files["RECOVERY_REQUEST.json"]["exists"], "initial attempt unexpectedly used a recovery request")
        record = physical_record(documents, prefix, descriptor, worker)
        snapshot_kind = descriptor["physical_source_kind"]
        if snapshot_kind in ("PHYSICAL_PROGRESS", "TRAINING_FAILURE"):
            snap_name = snapshot_kind + ".json"
            need(files[snap_name]["exists"]
                 and files[snap_name]["source_sha256"] == documents.sha(prefix + "physical_snapshot"),
                 "physical snapshot not archived")
        else:
            cp = documents.get(prefix + "physical_snapshot")
            need(files["cache/RECOVERY_SAME_RUN.pt"]["exists"]
                 and files["cache/RECOVERY_SAME_RUN.pt"]["source_sha256"] == cp["checkpoint_SHA256"],
                 "saved checkpoint metadata not tied to archived checkpoint")
        need(utc(record["evidence_UTC"]) <= utc(death["UTC"]), "physical snapshot is after death observation")
        records.append(record)
        ids.append(ident)
        inventories.append((files, inventory))
        latest_death = death
    latest_files, latest_inventory = inventories[-1]
    need(utc(documents.get("frozen_verification")["UTC"]) >= utc(latest_death["UTC"]),
         "frozen revalidation must follow latest confirmed death")
    need(latest_files["ARTIFICIAL_SUITE_RECEIPT.json"]["exists"]
         and latest_files["ARTIFICIAL_SUITE_RECEIPT.json"]["source_sha256"] == documents.sha("artificial_suite"),
         "bound artificial suite not archived")
    need(all(latest_files[name]["exists"] for name in (
        "TRAIN_STARTED.json", ARM + "/ENVIRONMENT.json", ARM + "/INITIALIZATION.json",
        ARM + "/INITIAL_RANDOM_STATES.json")), "same-run initialization evidence missing")
    observation = documents.get("checkpoint_observation")
    need(observation["logical_run_id"] == RUN_ID and observation["path"] == "cache/RECOVERY_SAME_RUN.pt",
         "only the published checkpoint target is eligible")
    for key in ("exists", "regular_file", "observed_after_all_prior_processes_dead",
                "hash_observed_without_checkpoint_deserialization", "no_tmp_promotion_performed"):
        need(observation[key] is True, "checkpoint observation missing: " + key)
    need(observation["symlink"] is False and digest(observation["sha256"])
         and integer(observation["bytes"], 1), "checkpoint target has no valid byte observation")
    need(utc(observation["UTC"]) >= utc(latest_death["UTC"]), "checkpoint hash observed before death")
    target_record = latest_files["cache/RECOVERY_SAME_RUN.pt"]
    need(target_record["exists"] and target_record["source_sha256"] == observation["sha256"]
         and target_record["source_bytes"] == observation["bytes"], "checkpoint target not preserved exactly")
    eligible = []
    for role, (name, status) in PUBLICATIONS.items():
        exists = latest_files[name]["exists"]
        need((role in documents.values) == exists, "publication existence/evidence contradiction")
        if not exists:
            continue
        publication = documents.get(role)
        need(latest_files[name]["source_sha256"] == documents.sha(role), "publication bytes not archived")
        need(publication["logical_run_id"] == RUN_ID and publication["frozen_binding"] == binding
             and publication["transaction_version"] == 1 and publication["publication_status"] == status
             and publication["checkpoint_path"] == "cache/RECOVERY_SAME_RUN.pt"
             and publication["all_fields_after_completed_optimizer_and_log_commit"] is True,
             "invalid publication receipt")
        step = publication["committed_update"]
        need(integer(step, 500) and step <= UNIQUE_UPDATES and step % 500 == 0
             and digest(publication["checkpoint_SHA256"]) and publication["native_attempt_id"] in ids,
             "invalid publication checkpoint/creator")
        need(integer(publication["bytes"], 1), "invalid checkpoint publication size")
        if publication["checkpoint_SHA256"] == observation["sha256"]:
            need(publication["bytes"] == observation["bytes"], "matching SHA with conflicting size")
            eligible.append((role, name, publication))
    need(bool(eligible), "No valid already-published checkpoint: do not promote tmp or manufacture a SHA receipt")
    first = eligible[0][2]
    need(all((entry[2]["committed_update"], entry[2]["native_attempt_id"], entry[2]["checkpoint_SHA256"])
             == (first["committed_update"], first["native_attempt_id"], first["checkpoint_SHA256"])
             for entry in eligible), "matching receipts contradict checkpoint identity")
    role, publication_name, publication = eligible[0]  # COMMITTED preferred when both match.
    step = publication["committed_update"]
    creator_index = ids.index(publication["native_attempt_id"])
    need(step <= records[-1]["last_completed_update"]
         and records[creator_index]["resumed_from_update"] < step <= records[creator_index]["last_completed_update"],
         "checkpoint step is unsupported by saved creator/adjacent-attempt evidence")
    preserved_final = []
    for name in FINAL_NAMES:
        record = latest_files[ARM + "/" + name]
        if record["exists"]:
            preserved_final.append({"name": name, "sha256": record["source_sha256"],
                                    "archive_relative_path": record["archive_relative_path"]})
    request = {
        "logical_run_id": RUN_ID, "prior_actual_PID_gone": True,
        "prior_native_attempt_preserved": True, "frozen_binding": copy.deepcopy(binding),
        "checkpoint_publication_receipt_path": publication_name,
        "checkpoint_publication_receipt_SHA256": documents.sha(role),
        "committed_update": step, "checkpoint_creator_attempt_id": publication["native_attempt_id"],
        "checkpoint_SHA256": publication["checkpoint_SHA256"],
        "preserved_physical_attempt_compute": records, "previous_native_attempt_id": ids[-1],
        "prior_recovery_history": history, "preserved_finalization_outputs": preserved_final,
    }
    need(set(request) == REQUEST_KEYS, "internal request interface drift")
    summary = {
        "status": "saved_evidence_consistent_draft_for_root_review",
        "evidence_kind": documents.manifest["evidence_kind"],
        "logical_run_id": RUN_ID, "committed_update": step,
        "prior_attempt_count": len(records), "prior_recovery_count": len(history),
        "checkpoint_creator_attempt_id": publication["native_attempt_id"],
        "previous_native_attempt_id": ids[-1], "selected_publication_path": publication_name,
        "next_native_attempt_id": None,
        "known_physical_counter_lower_bounds": {
            key: sum(record["counter"][key] for record in records) for key in PER_UPDATE},
        "physical_counter_totals_exact": all(record["counters_complete"] for record in records),
        "unknown_intervals": [{"attempt_id": record["attempt_id"], **interval}
                              for record in records for interval in record["unknown_intervals"]],
        "logical_checkpoint_update_is_not_physical_work_total": True,
        "finalization_outputs_preserved": len(preserved_final),
        "saved_evidence_document_sha256": copy.deepcopy(documents.hashes),
        "saved_evidence_manifest_sha256": documents.manifest_sha,
        "native_recovery_validation_performed": False, "direct_PID_queries": 0,
        "checkpoint_tensor_reads": 0, "scientific_array_reads": 0,
        "native_launches": 0, "process_terminations": 0, "used_marker_removals": 0,
        "external_observation_limit": "Root-supplied native/PID/archive/hash observations are checked for internal consistency; this constructor does not repeat or authenticate their underlying collection.",
        "deferred_to_frozen_trainer": "Checkpoint tensor/model/optimizer/RNG/log content is validated by the frozen trainer when root explicitly performs same-run recovery; no native restoration has been tested here.",
    }
    return request, summary


def build(manifest_path, output_dir):
    documents = Documents(manifest_path)
    request, summary = validate_saved_documents(documents)
    output_dir = Path(output_dir).resolve()
    if os.name == "nt":
        need(output_dir.drive.upper() in ("E:", "D:"), "outputs must be on E or D")
    need(not output_dir.exists(), "output directory already exists; no overwrite permitted")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("RECOVERY_REQUEST.DRAFT.json", request), ("EVIDENCE_VALIDATION.json", summary)):
        with (output_dir / name).open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from local_resource_guard import apply, memory
    guard = apply(Path(__file__).resolve().parent)
    summary = build(args.evidence, args.output)
    print(json.dumps({"status": summary["status"], "resources": guard, "memory": memory(),
                      "native_recovery_validation_performed": False}, allow_nan=False))


if __name__ == "__main__":
    main()
