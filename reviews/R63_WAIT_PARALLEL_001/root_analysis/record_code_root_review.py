"""Record the root's source review of prepared D statistics; run no science."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / "reviews/R63_WAIT_PARALLEL_001/validation_prep"
OUT = Path(__file__).resolve().parent / "CODE_ROOT_REVIEW.json"
assert ROOT.drive.upper() in ("E:", "D:")
assert not OUT.exists(), "Preserve earlier review receipts."

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt = read(PREP / "CODE_PREPARATION_RECEIPT.json")
selftest = read(PREP / "ARTIFICIAL_SELFTEST.json")
assert receipt["selftest"]["all_passed"]
assert selftest["status"] == "passed"
assert selftest["tests_run"] == 20 and not selftest["errors"] and not selftest["failures"]
assert len(selftest["tests"]) == 20
assert all(item["status"] == "passed" for item in selftest["tests"])
checks = []
for name, expected in receipt["hashes_sha256"].items():
    actual = sha(PREP / name)
    assert actual == expected, name
    checks.append({"path": str(PREP / name), "sha256": actual, "receipt_match": True})
assert sha(PREP / "report.json") == receipt["immutable_report"]["expected_sha256"]
acceptance = read(ROOT / "PROJECT_ACCEPTANCE.json")
assert sha(ROOT / "PROJECT_ACCEPTANCE.md") == acceptance["document_sha256"]
assert acceptance["utility"]["PSD_MAE_relative_to_REAL_only_point_max"] == -0.05
assert acceptance["utility"]["PSD_MAE_relative_degradation_vs_full_REAL_max"] == 0.05
assert acceptance["utility"]["logQ_RMSE_relative_degradation_vs_budget_REAL_max"] == 0.05

result = {
    "UTC": datetime.now(timezone.utc).isoformat(),
    "status": "source_review_accepted_as_unwired_code_preparation",
    "reviewer": "root",
    "scope": "D statistics adapter source, artificial selftest source, and existing native test receipt",
    "independent_test_rerun": False,
    "reason_not_rerun": "The recorded native tests passed and their source/output hashes match; review found no code change or unresolved test failure.",
    "reviewed_contracts": [
        "D1/D4/D5 point gates use inclusive metric bounds without a tolerance; D2/D3 CI upper bounds are strictly below zero.",
        "D2 forms the relative MAE difference separately in every paired resample, not by scaling a CI using the point estimate.",
        "All 8 packets use the same whole-group bootstrap draws; group repetitions preserve event weights for unequal group sizes.",
        "The independent test expands all 1000 resamples into repeated event indices and checks each arm and both estimators.",
        "A zero reference or any invalid required CI replicate is undefined and cannot pass; no replicate filtering or denominator floor.",
        "Both Ridge and HGB must satisfy every gate; row order, group binding, missing packets, nonfinite values and derived overflows are rejected.",
        "The caller must explicitly provide a bool design declaration, with no default; this declaration alone proves neither data lineage nor study freezing.",
    ],
    "verified_files": checks,
    "immutable_readiness_report_sha256": sha(PREP / "report.json"),
    "preparation_receipt_sha256": sha(PREP / "CODE_PREPARATION_RECEIPT.json"),
    "original_acceptance_json_sha256": sha(ROOT / "PROJECT_ACCEPTANCE.json"),
    "original_acceptance_md_sha256": sha(ROOT / "PROJECT_ACCEPTANCE.md"),
    "observed_artificial_test_count": 20,
    "observed_artificial_peak_working_set_bytes": receipt["execution"]["observed_peak_working_set_bytes"],
    "scope_boundaries": {
        "actual_sample_identities_verified": False,
        "actual_prediction_packets_available": False,
        "wired_into_frozen_D_protocol": False,
        "D_or_E_science_registered_or_run": False,
        "scientific_arrays_read": False,
        "new_model_forwards_gradients_fits_updates": 0,
        "R63_frozen_science_changed": False,
        "actual_D_acceptance_established": False,
    },
    "remaining_before_use": "Register and verify the frozen D population, row identities/groups, source-parent selection, same candidate weight, target definitions, estimators, and REAL consumption; then bind actual predictions to this adapter.",
    "model_goal_complete": False,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
read(OUT)
print(json.dumps({"status": result["status"], "verified_files": len(checks), "output": str(OUT)}))
