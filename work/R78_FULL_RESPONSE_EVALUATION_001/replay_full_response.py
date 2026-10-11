"""Independent actual4000 replay. Saved outputs are never inference inputs."""
from __future__ import annotations

from pathlib import Path
import os
import time
import traceback

from evaluation_io import (array_sha, check_payload, config_argument, exact, identity_payload,
                           load_config, main_inputs, merge_columns, preflight_training,
                           require, sha_file, utc, validate_main, write_json)
from full_response_adapter import FullResponseAdapter


def compare_payload(expected, saved):
    require(set(expected) == set(saved), "saved full-chain field set changed")
    return {key: exact(expected[key], saved[key]) for key in expected}


def compare_after_full_compute(computed, paths, loader):
    """Testable barrier: this helper is called only after both domains exist."""
    require(set(computed) == set(paths) and len(computed) == 2,
            "complete both main domains before opening any saved candidate output")
    return {name: compare_payload(payload, loader(paths[name]))
            for name, payload in computed.items()}


def run(config_path):
    start = time.monotonic()
    c, config_binding = load_config(config_path)
    root = Path(c["output_root"])
    require(not (root/"REPLAY_STARTED.json").exists(), "replay already attempted")
    admission = preflight_training(c)
    write_json(root/"REPLAY_STARTED.json",
               {"UTC": utc(), "PID": os.getpid(), "config": config_binding,
                "training_admission": admission}, exclusive=True)
    computed = {}
    rows = {}
    try:
        # New process and newly loaded complete Q+shape/statistics. Original
        # arrays are the sole input; neither saved Qstar nor prepared is loaded.
        adapter = FullResponseAdapter(c, "replay_full_response.py")
        actual, metadata, reference = main_inputs(c)
        axes = validate_main(actual, metadata, reference, adapter.edges)
        for domain, name, raw in [("SOURCE", c["SOURCE_arm"], actual["source_waveforms_V"]),
                                  ("REAL", c["REAL_arm"], reference["reference_V"])]:
            result, description = adapter.infer(raw, actual["white"], actual["z"], actual["style"])
            expected = check_payload(merge_columns(
                result, identity_payload(domain, actual, metadata, reference, axes)), 2000)
            computed[name] = expected
            rows[name] = {"rows": 2000, "inference": description,
                          "pre_ADC_SHA256": array_sha(expected["pre_ADC_V"]),
                          "canonical_pipeline_SHA256": array_sha(expected["waveforms_V"]),
                          "Qstar_SHA256": array_sha(expected["predicted_target_Q_Vns"]),
                          "strict_integer_code_SHA256": array_sha(expected["ADC_integer_codes"])}
        require(sum(v["rows"] for v in rows.values()) == 4000 and
                adapter.counts["Q_head_forward_rows"] == 4000 and
                adapter.counts["shape_forward_rows"] == 4000 and
                adapter.counts["Box_capture_rows"] == 4000 and
                adapter.counts["Box_capture_calls"] == 16, "actual4000 replay counters failed")
        # Barrier: both2000 responses are entirely computed before even reading
        # GENERATION_RECEIPT, file bytes, or saved candidate arrays.
        generated = __import__("json").loads((root/"GENERATION_RECEIPT.json").read_text("utf-8"))
        require(generated.get("status") == "complete5984_generation_only" and
                generated.get("configuration_binding") == config_binding and
                generated.get("model_and_runtime_binding") == adapter.bindings,
                "generation/replay configuration or weight binding differs")
        paths = {name: root/name/"generated.npz" for name in computed}

        def load_saved(path):
            name = path.parent.name
            require(sha_file(path) == generated["output_files"][name]["sha256"],
                    "saved candidate file changed since generation")
            import numpy as np
            with np.load(path, allow_pickle=False) as packet:
                return {key: packet[key].copy() for key in packet.files}

        checks = compare_after_full_compute(computed, paths, load_saved)
        for name in rows:
            rows[name].update(checks=checks[name], bitwise_pass=all(checks[name].values()))
        receipt = {"UTC": utc(), "PID": os.getpid(), "elapsed_s": time.monotonic()-start,
                   "rows": 4000, "pass": all(v["bitwise_pass"] for v in rows.values()),
                   "endpoints": rows, "independent_process": True,
                   "complete_Q_head_shape_lifting_noise_Box_ADC_replayed": True,
                   "both_domains_computed_before_any_candidate_output_read": True,
                   "saved_Qstar_or_prepared_used_as_input": False,
                   "same_original_order_chunk256_and_partial_batches": True,
                   "no_permutation_or_other_chunk_byte_invariance_claim": True,
                   "strict_integer_codes_and_actual_canonical_voltage_checked_separately": True,
                   "original_SOURCE_population_and_reference_REAL_roles_preserved": True,
                   "new_fit_or_update": 0, "HGB_or_EM_fit": 0,
                   "training_admission": admission, "model_and_runtime_binding": adapter.bindings,
                   "configuration_binding": config_binding, "actual_response_counters": adapter.counts,
                   "resource_handshake": adapter.resource_handshake,
                   "final_A_to_F_pass": None}
        write_json(root/"REPLAY_RECEIPT.json", receipt)
        require(receipt["pass"], "complete held-out4000 bitwise replay failed")
        return receipt
    except BaseException as exc:
        write_json(root/"REPLAY_FAILURE.json", {
            "UTC": utc(), "PID": os.getpid(), "error": repr(exc),
            "traceback": traceback.format_exc(), "completed_rows": sum(v["rows"] for v in rows.values()),
            "endpoints": rows, "completed_rows_scope": "entire completed main-domain dictionaries only",
            "completed_batch_response_counters": adapter.counts if "adapter" in locals() else None,
            "incomplete_current_batch_attempted_rows": "unknown",
            "candidate_quality_conclusion": None})
        raise


if __name__ == "__main__":
    path = config_argument("replay_full_response.py", Path(__file__).parent)
    print(__import__("json").dumps(run(path), allow_nan=False))
