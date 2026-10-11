
"""Receipt-only registered v5 aggregation and plotting.

Consumes completed formal evaluator JSON and existing generation/replay/TRAIN
receipts. This module never opens NPZ/PT, reads waveforms, or recomputes Q, ADC,
identity, CDF, KS/W1, migration matrices, or zero-baseline features. Plotting uses
the original renderer and its already saved derived moment packet.
"""
from __future__ import annotations
import os
import json
import math
import time
import traceback
from pathlib import Path
from evaluation_io import load_config, owned_runtime, require, sha_file, utc, write_json
from acceptance_gates import evaluate, finite


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_receipt_bound(row, label):
    path = Path(row["path"])
    require(path.is_absolute() and sha_file(path) == row["sha256"], label + " JSON binding changed")
    return read(path)


def native_record_pass(row, entry, c):
    rss = row.get("actual_worker_RSS_peak_bytes")
    gpu = row.get("GPU_process_memory_peak_bytes")
    cuda = entry != "evaluate_saved.py"
    gpu_ok = (finite(gpu) and 0 < gpu <= c["GPU_limit_bytes"]) if cuda else (
        gpu is None or finite(gpu) and 0 <= gpu <= c["GPU_limit_bytes"])
    return bool(row.get("entry") == entry and type(row.get("actual_worker_native_exit")) is int and
        row["actual_worker_native_exit"] == 0 and row.get("termination_reason") is None and
        row.get("handshake_before_imports") is True and type(row.get("actual_worker_PID")) is int and
        row["actual_worker_PID"] > 0 and finite(rss) and 0 < rss <= c["RSS_limit_bytes"] and gpu_ok)


def native_receipts(root, c):
    result = {}
    for entry in ["generate_full_response.py", "replay_full_response.py", "evaluate_saved.py"]:
        path = root / (entry + ".NATIVE_RECEIPT.json")
        row = read(path)
        result[entry] = {
            "pass": native_record_pass(row, entry, c),
            "actual_worker_PID": row.get("actual_worker_PID"),
            "native_exit": row.get("actual_worker_native_exit"),
            "peak_RSS_bytes": row.get("actual_worker_RSS_peak_bytes"),
            "peak_GPU_bytes": row.get("GPU_process_memory_peak_bytes"),
            "receipt_SHA256": sha_file(path),
        }
    return result


def run(root):
    start = time.monotonic()
    root = Path(root).resolve()
    c = read(root / "config_post.json")
    cfg_path = Path(c["evaluation_adapter_config"])
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    ac, binding = load_config(cfg_path)
    require(Path(ac["output_root"]).resolve() == root, "summary output root mismatch")
    require(c["SOURCE_arm"] == ac["SOURCE_arm"] and c["REAL_arm"] == ac["REAL_arm"] and
            c["acceptance_version"] == 5, "summary prospective domains/acceptance mismatch")
    handshake = owned_runtime(ac, "summarize_and_plot.py", root)
    require(not (root / "SUMMARY_STARTED.json").exists(), "summary already attempted; preserve old result")
    write_json(root / "SUMMARY_STARTED.json", {
        "UTC": utc(), "PID": os.getpid(), "config": binding,
        "scope": "receipt-only aggregation; no independent numerical recalculation",
    }, exclusive=True)
    evidence = None
    try:
        formal = read(root / "EVALUATION_NUMERIC_RECEIPT.json")
        require(formal.get("status") == "complete", "formal evaluation receipt incomplete")
        m = read_receipt_bound(formal["metrics_file"], "formal metrics")
        exported = read_receipt_bound(formal["gate_evidence_file"], "formal gate evidence")
        require(m.get("status") == "complete" and exported.get("status") == "complete" and
                exported.get("schema") == "R78_FORMAL_EVALUATION_GATE_EVIDENCE_V1",
                "formal evaluation/export incomplete")
        generated = read(root / "GENERATION_RECEIPT.json")
        replay = read_receipt_bound(exported["replay_receipt_binding"], "prescribed independent replay")
        require(generated.get("status") == "complete5984_generation_only",
                "complete population generation absent")
        require(m["configuration_binding"] == exported["configuration_binding"] ==
                generated["configuration_binding"] == replay["configuration_binding"] == binding,
                "configuration changed")
        counters = m["counters"]
        require(counters["HGB_attempts"] == counters["HGB_completed_slots"] == 10 and
                counters["density_description_slots"] == counters["density_completed_slots"] == 16 and
                counters["density_EM_attempts"] <= 16 and counters["response_updates"] == 0 and
                counters["response_model_forward_rows_in_evaluation"] == 0,
                "fixed formal evaluation fit accounting changed")
        require(counters["EVAL_read"] == counters["CAL_waveform_decode"] == 0,
                "formal evaluation scope changed")
        require([generated["counter"][key] for key in [
            "source_main_rows", "REAL_control_rows", "repeat_noise_rows",
            "source_probe_rows", "extra_guide_probe_rows"]] == [2000, 2000, 1024, 832, 128],
            "generation rows differ")
        current = m["current_training"]
        admission = formal["training_admission"]
        require(admission == current["numeric_admission"] == generated["training_admission"] ==
                replay["training_admission"] and admission["completed_updates"] ==
                current["completed_updates"] == current["unique_selected_endpoint"] == c["updates"],
                "formal completed training endpoint differs")
        numeric = current["numerical_reference"]
        independent = (set(numeric.get("results", {})) == {"SOURCE", "REAL"} and
                       all(row.get("pass_gate") is True for row in numeric["results"].values()) and
                       admission.get("numeric_gate_recomputed") is True)
        native = native_receipts(root, ac)
        model_binding = generated["model_and_runtime_binding"]
        same_binding = (
            model_binding == replay["model_and_runtime_binding"] and
            model_binding["model_files"] == ac["model"] and
            model_binding["runtime_sources"] == ac["runtime_sources"] and
            model_binding.get("same_SOURCE_REAL_public_path") is True and
            model_binding.get("Q_state_equal_after_container_load") is True and
            model_binding.get("shape_buffers_equal_bound_statistics") is True)
        evidence = {
            "training_admission": admission,
            "training_records_revalidated": admission.get("numeric_gate_recomputed") is True,
            "training_records_revalidated_provenance": "completed formal evaluator JSON-only preflight; not repeated by summary",
            "independent_numpy_reference_pass": bool(independent),
            "numerical_reference": numeric,
            "population": exported["population"], "charge": exported["charge"],
            "zero_baseline_counts": exported["zero_baseline_counts"], "replay": replay,
            "prior_native_entries_pass": all(row["pass"] for row in native.values()),
            "prior_native_entries": native,
            "same_public_operator_and_bindings": bool(same_binding),
            "v5_charge_descriptions_complete": exported["mass"].get("pass") is True,
            "charge_mass_formal_evaluation_evidence": exported["mass"],
            "actual_generation_counter": generated["counter"], "evaluation_counter": counters,
            "unknown_fields": exported.get("unknown_fields", []),
            "no_new_neural_inference": True, "no_new_HGB_or_EM_fits": True,
            "source_array_read_in_summary": False,
            "Q_ADC_identity_CDF_migration_recomputed_in_summary": False,
            "independent_root_recalculation": False,
            "gate_evidence_provenance": formal["gate_evidence_file"],
            "legacy_numeric_gate_recomputed_key_scope": "original formal preflight's receipt threshold validation only; summary copies the already validated result",
        }
        write_json(root / "ROOT_AGGREGATE_EVIDENCE.json", safe(evidence))
        write_json(root / "CHARGE_IMPLEMENTATION_AUDIT.json", safe({
            "population": evidence["population"], "charge": evidence["charge"],
            "mass": exported["mass"], "provenance": "formal evaluator/export and prescribed replay receipts",
            "independent_root_recalculation": False,
        }))
        arm = c["SOURCE_arm"]
        zeros = evidence["zero_baseline_counts"]
        if any(type(zeros.get(name)) is not int for name in ["BASE", arm]):
            write_json(root / "SUMMARY_INCOMPLETE.json", {
                "status": "unknown", "unknown_fields": exported.get("unknown_fields", []),
                "reason": "required zero-baseline count unavailable in formal evaluator; no fabricated gate pass",
                "primary_A_C_pass": None,
            })
            raise ValueError("formal zero_baseline_counts unknown; original gate cannot be decided")
        gate, acceptance = evaluate(c, m, evidence)
        write_json(root / "GATE_METRICS.json", safe(gate))
        write_json(root / "ACCEPTANCE_A_C.json", safe(acceptance))
        compact = {
            "status": m["status"], "gate": gate, "acceptance_A_C": acceptance, "counters": counters,
            "training": admission, "C2ST": m["C2ST"],
            "overall_domain_vs_REAL": {name: value["overall"] for name, value in m["domain_vs_REAL"].items()},
            "PSD_main_vs_REAL": m["PSD_main"]["vs_REAL"],
            "energy_conditional_width": m["energy_conditional_width"],
            "source_composition": m["source_composition"],
            "source_association_actual_preADC_input": m["source_association_actual_preADC_input"],
            "boundary_differences": m["boundary_differences"], "guide_probe": m["guide_probe"],
            "explicit_constraint_projection": m["explicit_constraint_projection"],
            "numerical_implementation": evidence["charge"],
            "replay": {"rows": replay.get("rows"), "pass": replay.get("pass")},
            "charge_diagnostic_file": "CHARGE_DISTRIBUTION_DIAGNOSTICS.json",
            "receipt_only_aggregation": True, "independent_root_recalculation": False,
            "unknown_fields": exported.get("unknown_fields", []),
            "physical_incident_energy_calibration": "unknown",
            "true_missing_region_event_rate": "unknown", "D_E_evaluated": False,
            "private_event_arrays_not_in_Git": True,
        }
        write_json(root / "METRICS_COMPACT.json", safe(compact))
        from render_full_response import run as render
        figures = render(root, c, m, gate, acceptance, evidence)
        receipt = {
            "UTC": utc(), "PID": os.getpid(), "elapsed_s": time.monotonic() - start,
            "primary_A_C_pass": gate["primary_A_C_pass"],
            "new_HGB": 0, "new_EM": 0, "new_neural_inference": 0,
            "aggregate_and_plot_only": True, "receipt_only_aggregation": True,
            "source_array_read_for_independent_identity_and_charge_check": False,
            "Q_ADC_identity_CDF_migration_recomputed_in_summary": False,
            "independent_root_recalculation": False,
            "existing_derived_moment_packet_used_only_by_original_renderer": True,
            "resource_handshake": handshake, "figure_files": figures,
            "rendered_files_require_root_visual_QA": True,
            "formal_evaluation_receipt": {
                "path": str(root / "EVALUATION_NUMERIC_RECEIPT.json"),
                "sha256": sha_file(root / "EVALUATION_NUMERIC_RECEIPT.json")},
            "formal_gate_evidence": formal["gate_evidence_file"],
            "unknown_fields": exported.get("unknown_fields", []),
        }
        write_json(root / "SUMMARY_PLOT_NUMERIC_RECEIPT.json", receipt)
        print(json.dumps(receipt, allow_nan=False), flush=True)
        return receipt
    except BaseException as exc:
        write_json(root / "SUMMARY_FAILURE.json", {
            "UTC": utc(), "PID": os.getpid(), "error": repr(exc),
            "traceback": traceback.format_exc(), "elapsed_s": time.monotonic() - start,
            "scope": "receipt-only aggregation; no new scientific calculation",
            "unknown_fields": [] if evidence is None else evidence.get("unknown_fields", []),
        })
        raise


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
