"""Standard-library artificial saved-record checks. No native recovery test."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from local_resource_guard import apply, memory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "selfcheck_saved_records" / "attempt_001"
if OUT.exists():
    raise RuntimeError("Refuse to overwrite a prior selfcheck")
OUT.mkdir(parents=True)
RESOURCES = apply(OUT)
START = time.monotonic()
violations = []

def audit(event, args):
    if event in ("subprocess.Popen", "os.system", "socket.connect", "socket.connect_ex"):
        violations.append(event)
        raise RuntimeError("Forbidden process/network operation")
    if event == "open":
        name = args[0]
        if isinstance(name, (str, bytes)):
            path = Path(name).resolve()
            if path.suffix.lower() in (".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle"):
                violations.append(str(path))
                raise RuntimeError("Scientific/checkpoint-like file open forbidden, including artificial placeholders")
            mode = args[1]
            flags = args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (1 | 2 | 64 | 512 | 1024))
            if writing and OUT not in path.parents:
                violations.append(str(path))
                raise RuntimeError("Write escaped artificial selfcheck directory")

sys.addaudithook(audit)
import build_recovery_request as b

def raw(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf8")

def sha(value):
    return hashlib.sha256(raw(value)).hexdigest()

def t(minute=0, second=0):
    return "2026-10-11T04:%02d:%02d+00:00" % (minute, second)

ENV = {"python": "artificial-only", "device": "cuda:0", "CPU_threads": 12,
       "no_actual_environment_verified_by_this_fixture": True}
BINDING = dict(b.FROZEN_HASHES, environment=ENV)

def counters(n):
    counter = {key: n * value for key, value in b.PER_UPDATE.items()}
    operations = dict.fromkeys(b.PHYSICAL_KEYS, 0)
    for key, op in b.PHYSICAL_MAP.items():
        operations[op] = counter[key]
    operations.update(C_objective_calls=n, parameter_cast_calls=n, gradient_clip_calls=n,
                      D_source_noise_rows=n * 6593, C_SOURCE_noise_rows=n * 320,
                      C_identity_noise_rows=n * 64, full_window_direction_draw_calls=n,
                      temporal_direction_draw_calls=n)
    return operations, counter

def archive_entry(attempt_id, name, content_sha=None, size=None):
    value = ("artificial metadata descriptor for " + name).encode()
    size = len(value) if size is None else size
    content_sha = hashlib.sha256(value).hexdigest() if content_sha is None else content_sha
    return {"exists": True, "archive_relative_path": "native_attempts/" + attempt_id + "/stage/" + name,
            "source_sha256": content_sha, "archive_sha256": content_sha,
            "source_bytes": size, "archive_bytes": size, "regular_file": True, "source_symlink": False}

def put_attempt(case, ident, n=500, start=0, minute=0, pid=10101, kind="PHYSICAL_PROGRESS"):
    prefix = "attempt/" + ident + "/"
    operations, counter = counters(n - start)
    docs = case["documents"]
    docs[prefix + "science_pid"] = {
        "PID": pid, "parent_PID": pid - 1, "nonce": ident, "entry": b.ENTRY,
        "UTC": t(minute), "cwd": b.REMOTE_ROOT, "argv": ["artificial"],
        "before_scientific_imports": True, "python": "artificial-only"}
    docs[prefix + "rss_ack"] = {
        "actual_PID": pid, "supervisor_PID": pid - 1, "nonce": ident,
        "actual_PID_memory_query_success": True}
    docs[prefix + "native_receipt"] = {
        "entry": b.ENTRY, "actual_worker_PID": pid, "supervisor_PID": pid - 1,
        "launcher_PID": pid, "handshake_before_imports": True,
        "actual_worker_native_exit": -9, "launcher_native_exit": -9,
        "stop_UTC": t(minute, 50)}
    docs[prefix + "death_confirmation"] = {
        "logical_run_id": b.RUN_ID, "attempt_id": ident, "actual_worker_PID": pid,
        "supervisor_PID": pid - 1, "same_host_and_process_namespace_verified": True,
        "saved_handshake_and_native_identity_verified": True, "actual_worker_PID_absent": True,
        "supervisor_PID_absent": True, "no_matching_attempt_or_owned_descendants_alive": True,
        "native_receipt_SHA256": "not-refreshed", "collector": "ARTIFICIAL_SAVED_RECORD",
        "UTC": t(minute, 55)}
    snapshot = {
        "UTC": t(minute, 40), "PID": pid, "logical_run_id": b.RUN_ID, "native_attempt_id": ident,
        "phase": "completed_update_committed", "current_update": n, "last_completed_update": n,
        "last_checkpoint_update": n, "attempt_start_committed_update": start,
        "known_completed_physical_operations_this_attempt": operations,
        "physical_schema_counter_lower_bound": counter}
    docs[prefix + "physical_snapshot"] = snapshot
    files = {name: {"exists": False} for name in b.MUTABLE_PATHS}
    for name in (b.ENTRY + ".used", b.ENTRY + ".stdout.log", b.ENTRY + ".stderr.log",
                 b.ENTRY + ".SCIENCE_PID.json", b.ENTRY + ".RSS_ACK.json",
                 b.ENTRY + ".NATIVE_RECEIPT.json", b.ENTRY + ".START_RESOURCE.json",
                 "REMOTE_ACTIVE_JOB.json", "REMOTE_RUN_HISTORY.json", "TRAIN_STARTED.json",
                 "ARTIFICIAL_SUITE_RECEIPT.json", "RECOVERY_CHECKPOINT.json",
                 "cache/RECOVERY_SAME_RUN.pt", "PHYSICAL_PROGRESS.json",
                 b.ARM + "/ENVIRONMENT.json", b.ARM + "/INITIALIZATION.json",
                 b.ARM + "/INITIAL_RANDOM_STATES.json"):
        files[name] = archive_entry(ident, name)
    docs[prefix + "archive_inventory"] = {
        "logical_run_id": b.RUN_ID, "attempt_id": ident, "archive_root": "native_attempts/" + ident,
        "all_stage_files_except_native_attempts_inventory_closed": True,
        "all_existing_files_copied_and_rehashed": True, "prior_attempt_archives_preserved": True,
        "independent_byte_copies_no_hardlinks_or_symlinks": True, "UTC": t(minute, 58), "files": files}
    case["manifest"]["attempts"].append(
        {"attempt_id": ident, "resumed_from_update": start, "physical_source_kind": kind})
    return prefix

def base(n=500):
    case = {"manifest": {"schema_version": 1, "evidence_kind": "artificial_saved_record",
                         "logical_run_id": b.RUN_ID, "frozen_binding": BINDING,
                         "attempts": [], "references": {}},
            "documents": {}}
    d = case["documents"]
    d["frozen_verification"] = {
        "logical_run_id": b.RUN_ID, "frozen_binding": BINDING, "trainer_SHA256": b.TRAINER_SHA,
        "all_reviewed_source_hashes_verified": True, "all_scientific_input_hashes_verified": True,
        "environment_exactly_matches_bound_environment": True, "no_scientific_parameters_changed": True,
        "verification_scope": "saved_root_observation_no_training_advance", "UTC": t(4)}
    d["artificial_suite"] = {"pass": True, "implementation_review_SHA": b.FROZEN_HASHES["implementation_review_SHA"],
                             "fixture_does_not_represent_actual_suite_results": True}
    d["publication_committed"] = {
        "UTC": t(0, 35), "committed_update": n, "native_attempt_id": "fake_a",
        "logical_run_id": b.RUN_ID, "transaction_version": 1,
        "checkpoint_path": "cache/RECOVERY_SAME_RUN.pt", "bytes": 4096,
        "checkpoint_SHA256": "a" * 64, "frozen_binding": BINDING,
        "all_fields_after_completed_optimizer_and_log_commit": True, "publication_status": "committed"}
    d["checkpoint_observation"] = {
        "logical_run_id": b.RUN_ID, "path": "cache/RECOVERY_SAME_RUN.pt",
        "exists": True, "regular_file": True, "observed_after_all_prior_processes_dead": True,
        "hash_observed_without_checkpoint_deserialization": True, "no_tmp_promotion_performed": True,
        "symlink": False, "sha256": "a" * 64, "bytes": 4096, "UTC": t(3)}
    put_attempt(case, "fake_a", n)
    return case

def refresh(case):
    d = case["documents"]
    for descriptor in case["manifest"]["attempts"]:
        ident = descriptor["attempt_id"]
        prefix = "attempt/" + ident + "/"
        d[prefix + "death_confirmation"]["native_receipt_SHA256"] = sha(d[prefix + "native_receipt"])
        files = d[prefix + "archive_inventory"]["files"]
        pairs = {"science_pid": b.ENTRY + ".SCIENCE_PID.json", "rss_ack": b.ENTRY + ".RSS_ACK.json",
                 "native_receipt": b.ENTRY + ".NATIVE_RECEIPT.json"}
        if descriptor["physical_source_kind"] in ("PHYSICAL_PROGRESS", "TRAINING_FAILURE"):
            pairs["physical_snapshot"] = descriptor["physical_source_kind"] + ".json"
        if prefix + "previous_recovery_request" in d:
            pairs["previous_recovery_request"] = "RECOVERY_REQUEST.json"
        for role, name in pairs.items():
            files[name] = archive_entry(ident, name, sha(d[prefix + role]), len(raw(d[prefix + role])))
        files["ARTIFICIAL_SUITE_RECEIPT.json"] = archive_entry(
            ident, "ARTIFICIAL_SUITE_RECEIPT.json", sha(d["artificial_suite"]), len(raw(d["artificial_suite"])))
        for role, (name, _) in b.PUBLICATIONS.items():
            if role in d:
                files[name] = archive_entry(ident, name, sha(d[role]), len(raw(d[role])))
            else:
                files[name] = {"exists": False}
        files["cache/RECOVERY_SAME_RUN.pt"] = archive_entry(
            ident, "cache/RECOVERY_SAME_RUN.pt", d["checkpoint_observation"]["sha256"],
            d["checkpoint_observation"]["bytes"])
    return case

def materialize(case, label):
    folder = OUT / "fixtures" / label
    folder.mkdir(parents=True)
    manifest = copy.deepcopy(case["manifest"])
    manifest["references"] = {}
    for role, value in case["documents"].items():
        name = role.replace("/", "__") + ".json"
        data = raw(value)
        (folder / name).write_bytes(data)
        manifest["references"][role] = {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
    manifest_path = folder / "EVIDENCE_BUNDLE.json"
    manifest_path.write_bytes(raw(manifest))
    return manifest_path

results = []
def run(label, case, accepted, check=None, expected_error=None, already_refreshed=False):
    if not already_refreshed:
        refresh(case)
    path = materialize(case, label)
    try:
        docs = b.Documents(path)
        request, summary = b.validate_saved_documents(docs)
        if not accepted:
            raise AssertionError("Invalid evidence was accepted: " + label)
        if check:
            check(request, summary)
        output = OUT / "drafts" / label
        emitted = b.build(path, output)
        reread = json.loads((output / "RECOVERY_REQUEST.DRAFT.json").read_text("utf8"))
        assert reread == request and emitted == summary
        result = {"case": label, "expected_acceptance": True, "accepted": True, "pass": True,
                  "selected_publication": request["checkpoint_publication_receipt_path"],
                  "committed_update": request["committed_update"]}
    except b.EvidenceError as exc:
        if accepted:
            raise
        if expected_error:
            assert expected_error in str(exc), (label, str(exc), expected_error)
        result = {"case": label, "expected_acceptance": False, "accepted": False,
                  "pass": True, "rejection": str(exc)}
    results.append(result)
    return path

def equals(value, expected):
    assert value == expected, (value, expected)

run("committed_target", base(), True,
    lambda r,s: (equals(r["committed_update"],500),equals(s["physical_counter_totals_exact"],False),
                 equals(s["known_physical_counter_lower_bounds"]["new_response_updates"],500)))
case = base()
case["documents"]["publication_pending"] = case["documents"].pop("publication_committed")
case["documents"]["publication_pending"]["publication_status"] = b.PUBLICATIONS["publication_pending"][1]
run("pending_after_target_replace", case, True,
    lambda r,s: equals(r["checkpoint_publication_receipt_path"],"RECOVERY_CHECKPOINT_PENDING.json"))
case = base()
case["documents"]["publication_pending"] = dict(case["documents"]["publication_committed"],
    publication_status=b.PUBLICATIONS["publication_pending"][1], committed_update=1000, checkpoint_SHA256="b"*64)
run("old_committed_before_target_replace", case, True,
    lambda r,s: equals(r["checkpoint_publication_receipt_path"],"RECOVERY_CHECKPOINT.json"))
case = base()
case["documents"]["publication_pending"] = dict(case["documents"]["publication_committed"],
    publication_status=b.PUBLICATIONS["publication_pending"][1])
run("both_receipts_match_prefer_committed",case,True,
    lambda r,s: equals(r["checkpoint_publication_receipt_path"],"RECOVERY_CHECKPOINT.json"))
case = base(50000)
files = case["documents"]["attempt/fake_a/archive_inventory"]["files"]
for name in b.FINAL_NAMES:
    files[b.ARM+"/"+name] = archive_entry("fake_a",b.ARM+"/"+name)
run("endpoint_partial_finalization_all_nine_files_preserved",case,True,
    lambda r,s:(equals(len(r["preserved_finalization_outputs"]),9),
                equals(s["physical_counter_totals_exact"],True),equals(r["committed_update"],50000)))
case = base()
snapshot = case["documents"]["attempt/fake_a/physical_snapshot"]
snapshot["known_completed_physical_operations_this_attempt"],snapshot["physical_schema_counter_lower_bound"] = counters(501)
snapshot["updates"] = snapshot.pop("last_completed_update")
snapshot["current_update"] = 501
snapshot["phase"] = "original_C_and_combined_update"
case["manifest"]["attempts"][0]["physical_source_kind"] = "TRAINING_FAILURE"
case["documents"]["attempt/fake_a/archive_inventory"]["files"]["PHYSICAL_PROGRESS.json"] = {"exists":False}
run("known_uncommitted_adam_is_physical_only",case,True,
    lambda r,s:(equals(r["preserved_physical_attempt_compute"][0]["known_uncommitted_optimizer_steps"],1),
                equals(r["committed_update"],500),equals(s["known_physical_counter_lower_bounds"]["new_response_updates"],501)))
case = base()
ops,_ = counters(500)
case["manifest"]["attempts"][0]["physical_source_kind"] = "CHECKPOINT_METADATA"
case["documents"]["attempt/fake_a/physical_snapshot"] = {
    "scope":"root_saved_checkpoint_metadata_no_training_advance","logical_run_id":b.RUN_ID,
    "native_attempt_id":"fake_a","checkpoint_transaction_version":1,"attempt_start_update":0,
    "checkpoint_SHA256":"a"*64,"all_fields_after_completed_optimizer_and_log_commit":True,
    "committed_update":500,"physical_this_attempt":ops,"publication_UTC":t(0,35)}
run("saved_checkpoint_metadata_counter_provenance",case,True)

seed = refresh(base())
seed_path = materialize(seed,"chain_seed")
old_request,_ = b.validate_saved_documents(b.Documents(seed_path))
chain = copy.deepcopy(seed)
prefix = put_attempt(chain,"fake_b",500,500,1,20202)
chain["documents"][prefix+"previous_recovery_request"] = old_request
chain["documents"][prefix+"prior_selected_publication"] = copy.deepcopy(chain["documents"]["publication_committed"])
snapshot = chain["documents"][prefix+"physical_snapshot"]
snapshot.update(last_completed_update=0,current_update=0,last_checkpoint_update=0,
                attempt_start_committed_update=0,phase="artificial_passed_before_TRAIN_decode")
run("creator_is_ancestor_failed_resume_before_restore",chain,True,
    lambda r,s:(equals(r["checkpoint_creator_attempt_id"],"fake_a"),equals(r["previous_native_attempt_id"],"fake_b"),
                equals(len(r["prior_recovery_history"]),1),
                equals(r["preserved_physical_attempt_compute"][-1]["logical_baseline_from_incoming_request_before_restore"],True)))

case = base()
case["documents"]["checkpoint_observation"]["exists"] = False
run("no_first_500_published_checkpoint",case,False,expected_error="checkpoint observation missing: exists")
case = base()
case["documents"].pop("publication_committed")
run("missing_both_transaction_receipts",case,False,expected_error="No valid already-published checkpoint")
case = base()
case["documents"]["checkpoint_observation"]["path"] = "cache/RECOVERY_SAME_RUN.tmp.pt"
run("tmp_must_not_be_promoted",case,False,expected_error="only the published checkpoint target")
case = base()
case["documents"]["publication_committed"]["checkpoint_SHA256"] = "b"*64
run("target_matches_no_saved_receipt",case,False,expected_error="No valid already-published checkpoint")
case = base()
case["documents"]["attempt/fake_a/death_confirmation"]["actual_worker_PID_absent"] = False
run("actual_worker_still_present",case,False,expected_error="death observation missing")
case = base()
case["documents"]["attempt/fake_a/native_receipt"]["actual_worker_native_exit"] = None
case["documents"]["attempt/fake_a/native_receipt"]["launcher_native_exit"] = None
run("PID_absence_is_not_native_zero",case,False,expected_error="native wait returncode required")
for field,value in (("nonce","stale_attempt"),("PID",77777),("native_exit",0)):
    case = base()
    worker = {"PID":10101,"parent_PID":10100,"nonce":"fake_a","entry":b.ENTRY,"native_exit":-9}
    worker[field] = value
    case["documents"]["attempt/fake_a/native_receipt"]["worker_self_receipt"] = worker
    run("stale_worker_receipt_"+field,case,False,expected_error="embedded worker receipt contradicts supervisor")
case = base()
case["documents"]["attempt/fake_a/archive_inventory"]["files"].pop(b.ENTRY+".used")
run("used_marker_inventory_gap",case,False,expected_error="mutable inventory has unchecked paths")
case = base()
case["documents"]["attempt/fake_a/archive_inventory"]["files"][b.ENTRY+".stdout.log"]["archive_sha256"] = "b"*64
run("truncated_or_mismatched_log_archive",case,False,expected_error="archive byte hash mismatch")
case = base(50000)
record = archive_entry("fake_a",b.ARM+"/FINAL_FLOW.pt")
record["archive_bytes"] += 1
case["documents"]["attempt/fake_a/archive_inventory"]["files"][b.ARM+"/FINAL_FLOW.pt"] = record
run("existing_endpoint_not_preserved_exactly",case,False,expected_error="archive byte length mismatch")
case = base()
case["documents"]["publication_committed"]["committed_update"] = 501
run("not_a_registered_checkpoint_boundary",case,False,expected_error="invalid publication checkpoint")
case = base()
case["documents"]["publication_committed"]["committed_update"] = 1000
run("checkpoint_ahead_of_available_physical_evidence",case,False,
    expected_error="checkpoint step is unsupported")
case = base()
case["documents"]["attempt/fake_a/physical_snapshot"]["physical_schema_counter_lower_bound"]["new_response_updates"] = 501
run("physical_counter_mapping_contradiction",case,False,expected_error="physical schema mapping contradicts")
case = base()
case["documents"]["frozen_verification"]["trainer_SHA256"] = "b"*64
run("changed_frozen_driver",case,False,expected_error="trainer changed")
case = base()
case["documents"]["artificial_suite"]["implementation_review_SHA"] = "b"*64
run("artificial_suite_from_other_code",case,False,expected_error="artificial suite missing or bound")
case = copy.deepcopy(chain)
case["documents"]["attempt/fake_b/previous_recovery_request"]["preserved_physical_attempt_compute"][0]["counter"]["new_response_updates"] += 1
run("old_physical_history_must_not_be_rewritten",case,False,expected_error="immutable prior recovery chain changed")
case = refresh(copy.deepcopy(chain))
case["documents"]["attempt/fake_b/archive_inventory"]["files"]["RECOVERY_REQUEST.json"] = {"exists":False}
run("old_request_must_be_archived_before_overwrite",case,False,
    expected_error="used recovery request not archived",already_refreshed=True)

bad_path = materialize(refresh(base()),"json_digest_tamper")
bad_manifest = json.loads(bad_path.read_text("utf8"))
bad_manifest["references"]["artificial_suite"]["sha256"] = "b"*64
bad_path.write_bytes(raw(bad_manifest))
try:
    b.Documents(bad_path)
    raise AssertionError("Tampered document digest accepted")
except b.EvidenceError as exc:
    assert "saved document SHA mismatch" in str(exc)
    results.append({"case":"saved_JSON_byte_digest_tamper","expected_acceptance":False,
                    "accepted":False,"pass":True,"rejection":str(exc)})

# Static source comparison only; importing or executing the trainer is forbidden.
trainer = ROOT.parents[2] / "work" / b.STAGE / "train_final.py"
source_bytes = trainer.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == b.TRAINER_SHA
tree = ast.parse(source_bytes)
request_keys = {
    node.slice.value for node in ast.walk(tree)
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
    and node.value.id=="request" and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)}
assert request_keys == b.REQUEST_KEYS
parsed_builder = ast.parse((ROOT/"build_recovery_request.py").read_text("utf8"))
imports = {node.names[0].name for node in ast.walk(parsed_builder) if isinstance(node,ast.Import)}
assert not imports & {"torch","numpy","scipy","subprocess","socket"}
source_text = source_bytes.decode()
publication_slice = source_text[source_text.index("    def checkpoint_save():"):source_text.index("    for step in range")]
assert (publication_slice.index("os.fsync(stream.fileno())")
        < publication_slice.index("write('RECOVERY_CHECKPOINT_PENDING.json'")
        < publication_slice.index("os.replace(tmp,target)")
        < publication_slice.index("sync_directory(CACHE)")
        < publication_slice.index("write('RECOVERY_CHECKPOINT.json'"))

report = {
    "status":"standard_library_artificial_saved_record_selfcheck_passed",
    "pass":all(result["pass"] for result in results),"case_count":len(results),
    "positive_cases":sum(result["accepted"] for result in results),
    "negative_cases":sum(not result["accepted"] for result in results),
    "cases":results,
    "static_checks":{"frozen_driver_SHA_matches":True,"exact_13_request_fields_match_driver_AST":True,
                     "checkpoint_publication_order_matches_frozen_source":True,
                     "constructor_no_scientific_or_process_network_imports":True},
    "resources":RESOURCES,"memory":memory(),"elapsed_s":time.monotonic()-START,
    "guard_violations":violations,
    "actual_native_recovery_validated":False,"remote_process_queries":0,
    "checkpoint_file_reads":0,"scientific_array_reads":0,"scientific_model_invocations":0,
    "fake_checkpoint_files_created":0,"native_launches":0,"shared_source_edits":0,
    "meaning":"All native/process/archive/checkpoint facts in these fixtures are explicitly invented metadata. Only constructor refusal/acceptance and source-interface consistency were tested."}
(OUT/"SELF_CHECK.json").write_bytes(raw(report))
print(json.dumps({key:report[key] for key in ("pass","case_count","positive_cases","negative_cases","memory","elapsed_s","actual_native_recovery_validated")}))
