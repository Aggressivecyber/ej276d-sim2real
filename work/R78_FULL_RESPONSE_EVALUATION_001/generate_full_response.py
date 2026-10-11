"""Future registered held-out generation: exact original 5984 response rows."""
from __future__ import annotations

from pathlib import Path
import os
import time
import traceback

import numpy as np

from evaluation_io import (array_sha, check_payload, config_argument, feature_Q_PSD,
                           identity_payload, load_config, main_inputs, merge_columns,
                           preflight_training, q_window, raw_bins, read_npz_bound, require,
                           save_output, summary, utc, validate_aux, validate_main, write_json)
from full_response_adapter import FullResponseAdapter


def describe_rows(data, original_bin):
    fields = ["input_variance100_V2", "lifted_baseline_variance", "noise_amplitude_squared",
              "noise_kappa2", "innovation_variance100_actual_V2", "initial_variance100_actual_V2",
              "Q_relative_error_f64", "Q_relative_error_f32",
              "projection_RMS_V", "constraint_projection_maxabs_V",
              "constraint_projection_rail_sample_count",
              "pre_constraint_out_of_range_sample_count", "pre_constraint_charge_relative_error"]
    n = len(data["Qstar"])
    by = []
    for b in range(-1, 9):
        use = original_bin == b
        by.append({"original_bin_raw": b, "N": int(use.sum()),
                   "fields": {k: summary(data[k][use]) for k in fields},
                   "range_projection_N": int(data["range_projection_active"][use].sum()),
                   "noise_completion_triggered_N": int(data["noise_completion_triggered"][use].sum())})
    return {"N": n, "fields": {k: summary(data[k]) for k in fields}, "by_original_bin": by,
            "range_projection_N": int(data["range_projection_active"].sum()),
            "range_projection_fraction": float(data["range_projection_active"].mean()),
            "noise_completion_triggered_N": int(data["noise_completion_triggered"].sum()),
            "initial_variance_semantics": "lifted input plus Qstar-conditioned innovation",
            "missing_physical_signal_electronic_split": "unknown",
            "internal_projection_not_ADC_clipping": True}


def probe_description(a, output, main_output):
    reference = main_output["evaluation_canonical_V"][a["reference_parent_index"]]
    post = output["evaluation_canonical_V"]
    rms = np.sqrt(np.mean((post-reference)**2, axis=1))
    rows = {}
    for tag in np.unique(a["probe_kind"]):
        use = a["probe_kind"] == tag
        rows[str(tag)] = {"N": int(use.sum()),
                         "waveform_RMS_vs_correct_reference_V": summary(rms[use])}
    same = a["probe_kind"] == "same_input_same_noise"
    return {"rows": rows, "new_generated_rows": len(post),
            "same_input_same_noise_bitwise_observed":
                bool(np.array_equal(output["waveforms_V"][same],
                                    main_output["waveforms_V"][a["source_parent_index"][same]])),
            "bitwise_scope": "observed prescribed layouts only; no permutation/batch invariance claim",
            "corrected_parent_semantics_reused": True,
            "physical_final_branch_split": "unknown"}


def joint_description(a, output):
    parent = a["source_parent_index"].reshape(128, 8)
    original = a["source_waveforms_V"].astype(np.float64)
    innovation = output["projected_colored_innovation_V"]
    remainder = output["pre_ADC_V"].astype(np.float64)-original-innovation
    pieces = np.stack([original, innovation, remainder]).reshape(3, 128, 8, 480)
    centered = pieces-pieces.mean(2, keepdims=True)
    covariance = np.einsum("ignt,jgnt->gijt", centered, centered)/7.
    total = output["pre_ADC_V"].reshape(128, 8, 480).astype(np.float64).var(1, ddof=1)
    return {"parent_count": 128, "draws_each": 8, "actual_rows": 1024,
            "source_parent_union": int(len(np.unique(parent[:, 0]))),
            "variance_denominator": 7, "new_independent_source_parents": 0,
            "component_order": ["complete_original_input", "projected_Qstar_colored_innovation",
                                "Q_lifting_shape_Box_and_final_f32"],
            "cross_terms_retained": True,
            "additive_algebraic_covariance_closure_maxabs_V2":
                float(np.max(np.abs(covariance.sum((1, 2))-total))),
            "physical_final_signal_electronic_split": "unknown",
            "full_components_reconstructible_from_saved_input_innovation_pre32": True}


def guide_inputs(source, original_inside):
    pick = np.flatnonzero(original_inside)[:128]
    require(len(pick) == 128, "original supported guide128 missing")
    j = np.arange(250, dtype=np.float64)
    perturb = np.sin(2*np.pi*j/249.)
    perturb -= perturb.mean()
    perturb *= .001/np.sqrt(np.mean(perturb**2))
    delta = np.zeros((128, 480), dtype=np.float64)
    delta[:, 150:400] = perturb
    changed = (source[pick].astype(np.float64)+delta).astype(np.float32)
    return pick, changed, delta


def run(config_path):
    start = time.monotonic()
    c, config_binding = load_config(config_path)
    root = Path(c["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/"GENERATION_STARTED.json").exists(), "generation already attempted; use new registered output")
    admission = preflight_training(c)  # Complete endpoint before creating or decoding scientific output.
    write_json(root/"GENERATION_STARTED.json",
               {"UTC": utc(), "PID": os.getpid(), "config": config_binding,
                "training_admission": admission}, exclusive=True)
    counter = {"source_main_rows": 0, "REAL_control_rows": 0, "repeat_noise_rows": 0,
               "source_probe_rows": 0, "extra_guide_probe_rows": 0,
               "response_fit_or_update": 0, "HGB_attempts": 0, "density_EM_attempts": 0,
               "final_EVAL_read": 0, "untreated_REAL_target_decoded_here": 0}
    try:
        adapter = FullResponseAdapter(c, "generate_full_response.py")
        actual, metadata, reference = main_inputs(c)
        axes = validate_main(actual, metadata, reference, adapter.edges)
        domains = {}
        files = {}
        descriptions = {}
        source_arm, real_arm = c["SOURCE_arm"], c["REAL_arm"]
        source_output, sr = adapter.infer(actual["source_waveforms_V"], actual["white"],
                                          actual["z"], actual["style"])
        source_identity = identity_payload("SOURCE", actual, metadata, reference, axes)
        source_payload = check_payload(merge_columns(source_output, source_identity), 2000)
        files[source_arm] = save_output(root/source_arm/"generated.npz", source_payload)
        counter["source_main_rows"] = 2000
        domains[source_arm] = {**sr, "input_role": "original_SOURCE2000",
                               "all_source_ID_order_weights_and_composition_preserved": True,
                               "TRAIN_support_outside_N": int((~axes["inside_support"]).sum()),
                               "full_response_applied_N": 2000, "BASE_fallback_N": 0,
                               "legacy_nominal_Q_is_separate_from_window_Q": True}
        descriptions[source_arm] = describe_rows(source_output, axes["support_bin"])

        # Same public path. reference_* is processed; real_* is comparator only.
        real_output, rr = adapter.infer(reference["reference_V"], actual["white"],
                                        actual["z"], actual["style"])
        real_identity = identity_payload("REAL", actual, metadata, reference, axes)
        real_payload = check_payload(merge_columns(real_output, real_identity), 2000)
        files[real_arm] = save_output(root/real_arm/"generated.npz", real_payload)
        counter["REAL_control_rows"] = 2000
        domains[real_arm] = {**rr, "input_role": "original_reference_REAL_TRAIN_REFERENCE2000",
                             "untreated_comparison_role": "original_real_REAL_DEV2000",
                             "reference_ID_order_groups_preserved": True,
                             "same_actual_white_z_style_as_SOURCE": True,
                             "no_G4_composition_or_PDG_assignment": True,
                             "full_response_applied_N": 2000, "BASE_fallback_N": 0}
        descriptions[real_arm] = describe_rows(real_output,
                                               raw_bins(q_window(reference["reference_V"]), adapter.edges))
        out = root/"results"
        probes = read_npz_bound(c["inputs"]["probes"], "probes")
        validate_aux(probes, 832, actual, "probes")
        pstyle = actual["style"][probes["noise_origin_parent_index"]]
        probe_output, pr = adapter.infer(probes["source_waveforms_V"], probes["white"], probes["z"], pstyle)
        probe_parent = {k: probes[k] for k in [
            "source_parent_index", "source_parent_ids", "noise_origin_parent_index",
            "reference_parent_index", "source_waveform_modified_counterfactual", "probe_kind"]}
        # Original random payloads are retained as well as their bound input file SHA.
        probe_payload = merge_columns(probe_output, probe_parent,
                                      {"actual_input_V": probes["source_waveforms_V"],
                                       "actual_white": probes["white"], "actual_z": probes["z"],
                                       "actual_style_unused": pstyle,
                                       "original_parent_weight": np.full(832, 1/2000., dtype=np.float64)})
        files["probes"] = save_output(out/(source_arm+"_SOURCE_PROBES_PRIVATE.npz"), probe_payload)
        counter["source_probe_rows"] = 832
        probe_summary = probe_description(probes, probe_output, source_output)
        probe_summary["inference"] = pr
        write_json(root/"SOURCE_PROBES_SUMMARY.json", {source_arm: probe_summary})

        joint = read_npz_bound(c["inputs"]["joint8"], "joint8")
        validate_aux(joint, 1024, actual, "joint8")
        joint_output, jr = adapter.infer(joint["source_waveforms_V"], joint["white"],
                                        joint["z"], joint["style"])
        joint_payload = merge_columns(joint_output, joint,
                                      {"actual_input_V": joint["source_waveforms_V"],
                                       "original_parent_weight": np.full(1024, 1/2000., dtype=np.float64),
                                       "within_parent_draw_weight": np.full(1024, 1/8., dtype=np.float64)})
        files["joint8"] = save_output(out/(source_arm+"_joint8_REPEAT_PRIVATE.npz"), joint_payload)
        counter["repeat_noise_rows"] = 1024
        js = joint_description(joint, joint_output)
        js["inference"] = jr
        write_json(root/"JOINT8_SUMMARY.json", {source_arm: {"groups": {"joint8": js}}})

        pick, changed, delta = guide_inputs(actual["source_waveforms_V"], axes["inside_support"])
        require(np.array_equal(pick, axes["guide_pick"]), "guide original support selection changed")
        guide_output, gr = adapter.infer(changed, actual["white"][pick],
                                         actual["z"][pick], actual["style"][pick])
        guide_payload = merge_columns(guide_output, {
            "parent_index": pick, "source_ids": actual["source_ids"][pick],
            "original_guide_V": actual["source_waveforms_V"][pick],
            "actual_perturbed_guide_f32_V": changed, "designed_perturbation_f64_V": delta,
            "original_output_preADC_V": source_output["pre_ADC_V"][pick],
            "fixed_source_Qbin": axes["support_bin"][pick],
            "actual_white": actual["white"][pick], "actual_z": actual["z"][pick],
            "actual_style_unused": actual["style"][pick],
            "original_parent_weight": np.full(128, 1/2000., dtype=np.float64)})
        files["guide"] = save_output(out/(source_arm+"_GUIDE_PERTURBATION_PRIVATE.npz"), guide_payload)
        counter["extra_guide_probe_rows"] = 128
        dv = guide_output["pre_ADC_V"].astype(np.float64)-source_output["pre_ADC_V"][pick].astype(np.float64)
        originalQ, originalPSD = feature_Q_PSD(actual["source_waveforms_V"][pick])
        changedQ, changedPSD = feature_Q_PSD(changed)
        gs = {"rows": 128, "selection": "first128 original TRAIN-supported source order",
              "full_Q_head_and_shape_recomputed_for_actual_f32_perturbation": True,
              "designed_segment_mean_V": float(delta[:, 150:400].mean()),
              "designed_segment_RMS_V": float(np.sqrt(np.mean(delta[:, 150:400]**2))),
              "actual_f32_Qtotal_mVns_change": summary(changedQ-originalQ),
              "actual_f32_PSD_change": summary(changedPSD-originalPSD),
              "preADC_output_RMS_V": summary(np.sqrt(np.mean(dv*dv, axis=1))),
              "output_nonidentical_fraction": float(np.any(
                  guide_output["pre_ADC_V"] != source_output["pre_ADC_V"][pick], axis=1).mean()),
              "ADC_LSB_V": 1/16383., "inference": gr,
              "gate_not_computed_here": True}
        write_json(root/"GUIDE_PERTURBATION_SUMMARY.json", {source_arm: gs})
        write_json(root/"GUIDE_PERTURBATION_COMPLETE_SUMMARY.json", {source_arm: gs})
        require([counter[k] for k in ["source_main_rows", "REAL_control_rows",
                                     "repeat_noise_rows", "source_probe_rows",
                                     "extra_guide_probe_rows"]] == [2000, 2000, 1024, 832, 128],
                "full5984 row accounting failed")
        require(adapter.counts["Q_head_forward_rows"] == 5984 and
                adapter.counts["shape_forward_rows"] == 5984 and
                adapter.counts["Box_capture_rows"] == 5984 and
                adapter.counts["Box_capture_calls"] == 25, "actual full response counters failed")
        write_json(root/"INITIAL_NOISE_DESCRIPTIONS.json", descriptions)
        write_json(root/"CONSTRAINT_PROJECTION_SUMMARY.json", descriptions)
        write_json(root/"REAL_CONTROL_INPUT_BINDING.json", {
            "reference_role": "REAL_packet.reference_V/reference_ids/reference_groups",
            "comparison_role": "REAL_packet.real_V/real_ids/real_groups",
            "REAL_packet": c["inputs"]["REAL_packet"], "random_file": c["inputs"]["actual"],
            "reference_waveform_SHA256": array_sha(reference["reference_V"]),
            "reference_ids_SHA256": array_sha(reference["reference_ids"]),
            "actual_random_array_SHA256": {k: array_sha(actual[k]) for k in ["z", "white", "style"]},
            "rows": 2000, "unused_random_inputs": ["z", "style"],
            "RAW_identity_expected": False, "new_REAL_acquisition": 0})
        receipt = {"UTC": utc(), "PID": os.getpid(), "elapsed_s": time.monotonic()-start,
                   "status": "complete5984_generation_only", "domains": domains, "counter": counter,
                   "all_generation_rows": 5984, "actual_response_counters": adapter.counts,
                   "training_admission": admission, "model_and_runtime_binding": adapter.bindings,
                   "input_binding": c["inputs"], "configuration_binding": config_binding,
                   "output_files": files, "resource_handshake": adapter.resource_handshake,
                   "new_BASE_control_inference": 0, "same_treated_REAL_control_inference_rows": 2000,
                   "controls_not_opened_here": True,
                   "main_SOURCE2000_and_reference_REAL2000_original_order": True,
                   "final_A_to_F_pass": None, "independent_heldout_replay_complete": False}
        write_json(root/"GENERATION_RECEIPT.json", receipt)
        write_json(root/"GENERATION_NUMERIC_RECEIPT.json",
                   {"UTC": utc(), "PID": os.getpid(), "status": "complete",
                    "counters": counter, "actual_response_counters": adapter.counts})
        return receipt
    except BaseException as exc:
        write_json(root/"GENERATION_FAILURE.json", {
            "UTC": utc(), "PID": os.getpid(), "error": repr(exc),
            "traceback": traceback.format_exc(), "counter": counter,
            "counter_scope": "fully completed and saved populations only",
            "completed_batch_response_counters": adapter.counts if "adapter" in locals() else None,
            "incomplete_current_batch_attempted_rows": "unknown",
            "candidate_quality_conclusion": None, "elapsed_s": time.monotonic()-start})
        raise


if __name__ == "__main__":
    path = config_argument("generate_full_response.py", Path(__file__).parent)
    print(__import__("json").dumps(run(path), allow_nan=False))
