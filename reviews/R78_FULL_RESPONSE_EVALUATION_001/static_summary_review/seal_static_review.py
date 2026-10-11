"""Stdlib-only source/receipt sealing, run only after root's freeze notice."""
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
CAND = ROOT / "reviews/R78_FULL_RESPONSE_EVALUATION_001/root_summary_candidate"
GUARD = ROOT / "reviews/R65_WAIT_ANALYSIS_001/recovery_procedure/local_resource_guard.py"
GUARD_SHA = "01caa0405c4e24a555cdd701a3435ebc3917339db9cd900a3a4f3e415c855d41"
SOURCE_NAMES = ("acceptance_gates.py", "evidence_math.py", "summarize_and_plot.py",
                "check_acceptance_gates.py", "check_legacy_statistical_rules.py",
                "check_evidence_math.py", "check_native_receipt_rules.py")
KNOWN_SHAS = {
    "acceptance_gates.py": "4a472c53dea6fe9d85754e18aabd6e8f403f5201c1eac779bbeed0a2ab2c3c59",
    "evidence_math.py": "9cca79d3b9925d23262b9803a7ca4f5c27904fc8ce14a7025ef62cd818a4151c",
    "summarize_and_plot.py": "8d58b19ebd7e4441502ec2118a75cbc18ac2c3a5fb444e366cd1803abe9b3c46",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                            allow_nan=False) + "\n", encoding="utf-8")


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    started = time.monotonic()
    if sha(GUARD) != GUARD_SHA:
        raise RuntimeError("existing local guard changed")
    spec = importlib.util.spec_from_file_location("static_summary_resource_guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    limits = guard.apply(OUT)
    assert limits["CPU_affinity_mask"] == 2
    assert not any(name in sys.modules for name in ("torch", "numpy", "scipy"))
    checked = {}
    try:
        freeze = load(OUT / "ROOT_FREEZE_NOTICE.json")
        assert freeze["root_authorized_final_source_binding"] is True
        assert freeze["reviewed_production_source_SHA256"] == KNOWN_SHAS
        source_hashes = {}
        for name in SOURCE_NAMES:
            path = CAND / name
            source_hashes[name] = sha(path)
            if name in KNOWN_SHAS:
                assert source_hashes[name] == KNOWN_SHAS[name], name + " changed since repaired-code review"
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            compile(tree, str(path), "exec", dont_inherit=True)
        checked["seven_sources_AST_compile_only"] = True
        checked["three_reviewed_production_SHA_unchanged"] = True

        gate = load(CAND / "ARTIFICIAL_GATE_CHECK.json")
        legacy = load(CAND / "LEGACY_RULE_REGRESSION.json")
        evidence = load(CAND / "ARTIFICIAL_EVIDENCE_CHECK.json")
        native_rules = load(CAND / "ARTIFICIAL_NATIVE_RULES.json")
        assert gate["status"] == "PASS_ARTIFICIAL_ONLY" and gate["checks"] == 60
        assert legacy["status"] == "PASS_ARTIFICIAL_LEGACY_REGRESSION" and legacy["scenarios"] == 14
        assert evidence["status"] == "PASS_ARTIFICIAL_EVIDENCE_ONLY" and evidence["checks"] == 15
        for name, value in gate["source_SHA256"].items():
            if name in source_hashes:
                assert source_hashes[name] == value
        for name, value in evidence["source_SHA256"].items():
            assert source_hashes[name] == value
        assert legacy["new_source_SHA256"] == source_hashes["acceptance_gates.py"]
        checked["root_60_gate_14_legacy_15_evidence_artifact_source_bindings"] = True
        assert native_rules["status"] == "PASS_ARTIFICIAL_NATIVE_RULES" and native_rules["checks"] == 30
        assert native_rules["source_SHA256"] == source_hashes["summarize_and_plot.py"]
        checked["root_30_native_rule_artifact_source_binding"] = True

        old_path = ROOT / "work/R65_FULL_EMPIRICAL_D_RESPONSE_001/summarize_and_plot.py"
        assert sha(old_path) == legacy["legacy_source_SHA256"]
        old = old_path.read_text(encoding="utf-8")
        fragment = old[old.index("gates={};primary="):old.index("write('GATE_METRICS.json',gate)")]
        assert hashlib.sha256(fragment.encode()).hexdigest() == legacy["executed_fragment_SHA256"]
        compile(fragment, str(old_path) + "[static only]", "exec", dont_inherit=True)
        checked["actual_frozen_F65_source_fragment_SHA_compile_no_exec"] = True

        current = load(ROOT / "CURRENT_ACCEPTANCE.json")
        assert current["version"] == 5
        assert sha(ROOT / current["document"]) == current["document_sha256"]
        assert sha(ROOT / current["registry"]) == current["registry_sha256"]
        checked["current_REAL_charge_v5_document_registry_binding"] = True
        r77_path = ROOT / "work/R77_FIXED25000_FULL_DATA_TRAIN_001/config.json"
        assert sha(r77_path) == "7ff3520f0bae83f1fed18d84afea8440d0790b5ba3e24e7c01d694632f7b93a7"
        cfg = load(r77_path)
        assert cfg["updates"] == 25000 and cfg["SOURCE_N"] == 6593 and cfg["REAL_N"] == 21047
        checked["R77_prospective_25000_all_parent_config_binding"] = True

        paths = [CAND / n for n in SOURCE_NAMES] + [CAND / "ARTIFICIAL_GATE_CHECK.json",
                CAND / "LEGACY_RULE_REGRESSION.json", CAND / "ARTIFICIAL_EVIDENCE_CHECK.json",
                CAND / "ARTIFICIAL_NATIVE_RULES.json",
                old_path, ROOT / "CURRENT_ACCEPTANCE.json", ROOT / current["document"],
                ROOT / current["registry"], r77_path,
                ROOT / "reviews/R74_EXACT_LOSS_FULL_TRAIN_001/evaluation_candidate/evaluation_io.py",
                ROOT / "reviews/R78_FULL_RESPONSE_EVALUATION_001/static_adapter_review/COMPLETION_RECEIPT.json"]
        manifest = {
            "UTC": utc(), "evidence_level": "STATIC_ONLY",
            "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): {
                "SHA256": sha(p), "bytes": p.stat().st_size} for p in paths},
            "production_source_SHA256": {n: source_hashes[n] for n in SOURCE_NAMES[:3]},
            "source_AST_compile_only_no_import_execution": True,
            "root_freeze_notice_SHA256": sha(OUT / "ROOT_FREEZE_NOTICE.json"),
            "initial_review_SHA256": sha(OUT / "INITIAL_REVIEW.md"),
            "report_SHA256": sha(OUT / "REPORT.md"), "sealer_SHA256": sha(__file__),
            "guard_SHA256": GUARD_SHA, "scientific_arrays_or_weights_read": 0,
        }
        write("INPUT_MANIFEST.json", manifest)
        resource = {
            "UTC": utc(), "limits": limits, "memory": guard.memory(),
            "elapsed_seconds": time.monotonic() - started,
            "native_entry": "python -B seal_static_review.py",
            "candidate_and_checker_execution": 0, "Torch_NumPy_SciPy_import": False,
            "original_arrays_or_weights_opened": 0, "model_forward_training_or_fit": 0,
            "SSH_or_Git": 0, "all_new_paths_E": True,
            "actual_exit_confirmation": "NATIVE_TOOL_LEDGER.json from launcher",
        }
        assert resource["memory"]["peak_RSS_bytes"] <= 1024 ** 3
        write("NATIVE_RESOURCE_RECEIPT.json", resource)
        assert all(sha(CAND / n) == v for n, v in source_hashes.items())
        write("STATIC_CHECKS.json", {
            "UTC": utc(), "pass_all": all(checked.values()), "checks": checked,
            "check_count": len(checked), "reviewed_source_SHA256": source_hashes,
            "initial_CDF_and_CUDA_peak_findings_resolved_in_reviewed_source": True,
            "root_artificial_checkers_rerun_by_this_reviewer": False,
            "scientific_A_to_F_result": "unknown_not_executed_by_this_review",
            "visual_render_quality": "not_reviewed", "D_E_F_success_claim": False,
        })
        print(json.dumps({"pass_all": True, "check_count": len(checked), "resource": resource,
                          "input_manifest_SHA256": sha(OUT / "INPUT_MANIFEST.json")}, allow_nan=False))
    except BaseException as exc:
        import traceback
        write("STATIC_CHECK_FAILURE.json", {"UTC": utc(), "error": repr(exc),
              "traceback": traceback.format_exc(), "checks": checked,
              "limits": limits, "memory": guard.memory(), "scientific_arrays_opened": 0})
        raise


if __name__ == "__main__":
    main()
