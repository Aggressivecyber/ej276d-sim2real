"""Stdlib-only static audit sealing; does not import or execute candidate code."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CAND = ROOT / "reviews/R74_EXACT_LOSS_FULL_TRAIN_001/evaluation_candidate"
GUARD = ROOT / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
GUARD_SHA = "01caa0405c4e24a555cdd701a3435ebc3917339db9cd900a3a4f3e415c855d41"
R77_CONFIG_SHA = "7ff3520f0bae83f1fed18d84afea8440d0790b5ba3e24e7c01d694632f7b93a7"
COMPLETION_SHA = "c05bf4fcb6c17c765c1319dd2a478381bdced8aba935fb4eb46e25161ddd5f8e"
OUTPUT_MANIFEST_SHA = "f97f13915deb83a01a1b6bd7288b1621585583f4dbff57e0fd34df6446448003"
MODULES = (
    "evaluation_io.py", "full_response_adapter.py",
    "generate_full_response.py", "replay_full_response.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    target = OUT / name
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n", encoding="utf-8")


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    start = time.monotonic()
    if sha(GUARD) != GUARD_SHA:
        raise RuntimeError("existing guard SHA changed")
    spec = importlib.util.spec_from_file_location("local_static_resource_guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    limits = guard.apply(OUT)
    if limits["CPU_affinity_mask"] != 2:
        raise RuntimeError("native launcher must inherit affinity mask2")
    if any(k in sys.modules for k in ("torch", "numpy", "scipy")):
        raise RuntimeError("static audit must remain stdlib-only")
    checked = {}
    try:
        receipt = load(CAND / "COMPLETION_RECEIPT.json")
        manifest = load(CAND / "OUTPUT_SHA256.json")
        assert sha(CAND / "COMPLETION_RECEIPT.json") == COMPLETION_SHA
        assert sha(CAND / "SEAL_RECEIPT.json") == COMPLETION_SHA
        assert sha(CAND / "OUTPUT_SHA256.json") == OUTPUT_MANIFEST_SHA
        assert receipt["OUTPUT_manifest_SHA256"] == OUTPUT_MANIFEST_SHA
        assert receipt["status"] == "COMPLETE_STATIC_AND_ARTIFICIAL_ONLY"
        assert receipt["source_locked_after_this_receipt"] is True
        for name, row in manifest["files"].items():
            path = CAND / name
            assert sha(path) == row["sha256"], name + " sealed SHA mismatch"
            assert path.stat().st_size == row["bytes"], name + " byte count mismatch"
        trees = {}
        for name in MODULES:
            path = CAND / name
            assert sha(path) == receipt["module_SHA256"][name]
            text = path.read_text(encoding="utf-8-sig")
            trees[name] = ast.parse(text, filename=str(path))
            compile(trees[name], str(path), "exec", dont_inherit=True)
        checked["sealed_four_modules_SHA_and_AST_compile"] = True
        checked["complete_B_output_manifest_SHA_and_sizes"] = True
        checked["B_seal_and_completion_bytes_equal"] = True

        current = load(ROOT / "CURRENT_ACCEPTANCE.json")
        v5 = load(ROOT / "PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json")
        assert current["version"] == v5["version"] == 5
        assert sha(ROOT / current["document"]) == current["document_sha256"]
        assert sha(ROOT / current["registry"]) == current["registry_sha256"]
        assert v5["document_sha256"] == current["document_sha256"]
        checked["current_v5_document_registry_binding"] = True

        r77_path = ROOT / "work/R77_FIXED25000_FULL_DATA_TRAIN_001/config.json"
        config = load(r77_path)
        assert sha(r77_path) == R77_CONFIG_SHA
        assert config["stage"] == "R77_FIXED25000_FULL_DATA_TRAIN_001"
        assert config["updates"] == 25000
        assert config["SOURCE_N"] == 6593 and config["REAL_N"] == 21047
        assert config["generation_chunk"] == 256 and config["Q_head_frozen"] is True
        assert config["shape_network"] == [482, 128, 128, 480]
        assert config["shape_dtype"] == "float32" and config["physical_dtype"] == "float64"
        checked["prospective_R77_config_SHA_endpoint_and_population"] = True
        exposures = {
            "SOURCE_main_exposures": 6593 * 25000,
            "REAL_main_exposures": 21047 * 25000,
            "full_neural_firstpass_rows": 27640 * 25000,
            "full_parameter_VJP_rows": 27640 * 25000,
        }
        assert exposures == {
            "SOURCE_main_exposures": 164825000,
            "REAL_main_exposures": 526175000,
            "full_neural_firstpass_rows": 691000000,
            "full_parameter_VJP_rows": 691000000,
        }
        template = load(CAND / "evaluation_config.example.json")
        assert template["execution_enabled"] is False
        assert template["training"]["config"]["sha256"] == R77_CONFIG_SHA
        checked["template_disabled_and_binds_R77_config"] = True

        run = next(n for n in trees["replay_full_response.py"].body
                   if isinstance(n, ast.FunctionDef) and n.name == "run")
        loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                    and any(isinstance(c, ast.Call) and
                            isinstance(c.func, ast.Attribute) and c.func.attr == "infer"
                            for c in ast.walk(n)))
        receipt_reads = [n for n in ast.walk(run) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "read_text"
                         and any(isinstance(c, ast.Constant) and
                                 c.value == "GENERATION_RECEIPT.json" for c in ast.walk(n))]
        assert len(receipt_reads) == 1
        assert loop.end_lineno < receipt_reads[0].lineno
        checked["replay_both_domain_infer_loop_precedes_generation_receipt_read"] = True
        generated_tree = trees["generate_full_response.py"]
        literals = {n.value for n in ast.walk(generated_tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "new_BASE_control_inference" in literals
        assert "same_treated_REAL_control_inference_rows" in literals
        assert "new_control_inference" not in literals
        checked["final_BASE_and_same_treated_REAL_counter_names"] = True

        B_artificial = load(CAND / "ARTIFICIAL_SCHEMA_CHECKS.json")
        assert receipt["artificial_check_count"] == 31
        assert receipt["training_metadata_rejection_cases"] == 20
        assert receipt["actual_artificial_native"]["native_exit"] == 0
        checked["B_artificial_evidence_sealed_not_rerun"] = True

        inputs = [ROOT / "AGENTS.md", ROOT / "CURRENT_ACCEPTANCE.json",
                  ROOT / "PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md",
                  ROOT / "PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json",
                  ROOT / "PROJECT_ACCEPTANCE.json",
                  ROOT / "tasks/R77_FIXED25000_FULL_DATA_TRAIN_001.md",
                  r77_path,
                  ROOT / "work/R77_FIXED25000_FULL_DATA_TRAIN_001/train_full.py",
                  ROOT / "work/R77_FIXED25000_FULL_DATA_TRAIN_001/postcheck.py",
                  CAND / "COMPLETION_RECEIPT.json", CAND / "SEAL_RECEIPT.json",
                  CAND / "OUTPUT_SHA256.json"]
        inputs += [CAND / name for name in manifest["files"]]
        # Bytes are read for hashing only. No candidate or training module is imported.
        input_manifest = {
            "UTC": utc(), "evidence_level": "STATIC_ONLY",
            "files": {str(p.relative_to(ROOT)).replace("\\", "/"): {
                "SHA256": sha(p), "bytes": p.stat().st_size} for p in inputs},
            "current_acceptance_version": 5,
            "guard_SHA256": GUARD_SHA,
            "report_SHA256": sha(OUT / "REPORT.md"),
            "sealer_SHA256": sha(__file__),
            "original_arrays_or_weights_read": 0,
        }
        write("INPUT_MANIFEST.json", input_manifest)
        resource = {"UTC": utc(), "limits": limits, "memory": guard.memory(),
                    "elapsed_seconds": time.monotonic() - start,
                    "native_entry": "python -B seal_static_review.py",
                    "actual_native_exit_confirmation": "NATIVE_TOOL_LEDGER.json from launcher",
                    "Torch_imported": False, "NumPy_imported": False,
                    "candidate_modules_imported_or_executed": 0,
                    "scientific_arrays_opened": 0, "weights_opened": 0,
                    "forward_or_training_or_fitting": 0, "SSH_or_Git": 0,
                    "all_new_paths_E": True}
        assert resource["memory"]["peak_RSS_bytes"] <= 1024 ** 3
        write("NATIVE_RESOURCE_RECEIPT.json", resource)
        write("STATIC_CHECKS.json", {
            "UTC": utc(), "pass_all": all(checked.values()), "checks": checked,
            "check_count": len(checked), "R77_config_SHA256": R77_CONFIG_SHA,
            "R77_expected_completed_exposures": exposures,
            "B_artificial_summary_check_count": receipt["artificial_check_count"],
            "B_artificial_metadata_rejection_cases": receipt["training_metadata_rejection_cases"],
            "B_artificial_rerun_by_this_reviewer": False,
            "scientific4000_replay": "unknown_not_executed_by_this_review",
            "A_to_F_pass": None,
            "evaluator_must_use_current_canonical_output_bin": "evaluation_output_bin_raw",
            "root_actual_R77_completion_and_future_registration_required": True,
        })
        # Recheck sealed production source after all reads.
        assert all(sha(CAND / n) == receipt["module_SHA256"][n] for n in MODULES)
        print(json.dumps({"pass_all": True, "check_count": len(checked),
                          "resource": resource, "input_manifest_SHA256": sha(OUT / "INPUT_MANIFEST.json")},
                         ensure_ascii=False, allow_nan=False))
    except BaseException as exc:
        import traceback
        write("STATIC_CHECK_FAILURE.json", {"UTC": utc(), "error": repr(exc),
              "traceback": traceback.format_exc(), "completed_checks": checked,
              "limits": limits, "memory": guard.memory(), "scientific_arrays_opened": 0})
        raise


if __name__ == "__main__":
    main()
