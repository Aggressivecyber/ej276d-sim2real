"""R65 saved-record audit. No model load, inference, fit, or loss recomputation.

Only the five-D/four-C recording semantics change. Numerical/acceptance
thresholds belong to the inherited callers and are not defined by this module.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import evaluation as ev

STAGE = "R65_FULL_EMPIRICAL_D_RESPONSE_001"
ARM = "FULL_EMPIRICAL_RESPONSE"
UPDATES, SOURCE_N, REAL_N = 50000, 6593, 21047
INITIAL_SHA = "4a51097e257d4b35976452982aa9a84f48e321d425e0ba3859e8afe8a7bd239b"
D_TERMS = ["full_window","coordinate_RMS","PSD","noise","temporal_joint"]
C_TERMS = ["source_order","source_valley","REAL_identity","box_correction"]
LOSS_COLUMNS = ["total_loss","full_window","PSD","source_order","source_valley","noise","REAL_identity","box_correction","coordinate_RMS","temporal_joint","learning_rate","gradient_norm","max_relative_q_f64","max_relative_q_f32","SOURCE_global_range_projection_fraction","REAL_range_projection_fraction","SOURCE_global_projection_RMS_V","REAL_projection_RMS_V","overall_PSD_W1","Q1_Q7_mean_PSD_W1","Q3_h_one_soft_valley","Q4_h_one_soft_valley","D_total","C_total","full_D_SOURCE_range_projection_fraction","full_D_SOURCE_projection_RMS_V","full_D_max_relative_q_f64","full_D_max_relative_q_f32","C_max_relative_q_f64","C_max_relative_q_f32","D_parameter_gradient_L2","C_parameter_gradient_L2"]
PRECISION_COLUMNS = ["original_order_loss","omitted_order_loss","original_nonzero_pair_mass","resolved_pair_fraction"]
REPLAY_COLUMNS = ["full_SOURCE_firstpass_rows","full_SOURCE_replay_rows","replay_chunks","endpoint_bitwise_pass_chunks","canonical_code_pass_chunks","Box_free_mask_pass_chunks","full_D_objective_calls","full_D_endpoint_reverse_calls","D_parameter_VJP_calls","C_microbatch_forward_calls","C_parameter_reverse_calls","optimizer_updates_completed"]
REPLAY_ROW = [6593,6593,104,104,104,104,1,1,104,1,1,1]
EXPECTED_COUNTS = {"new_response_updates":50000,"SOURCE_training_rows":16000000,"REAL_reference_rows":16000000,"REAL_identity_rows":3200000,"full_D_SOURCE_law_rows":329650000,"full_D_REAL_law_rows":1052350000,"scientific_neural_forward_rows":678500000,"scientific_neural_forward_calls":10450000,"D_parameter_VJP_calls":5200000,"C_parameter_reverse_calls":50000,"full_D_endpoint_reverse_calls":50000,"full_D_objective_calls":50000,"replay_chunks_checked":5200000}
GRADIENT_COUNTS = {"combined_gradient_cast_events":50000,"combined_clip_calls":50000,"optimizer_step_calls":50000,"separate_D_or_C_clip_calls":0,"parameter_updates_between_D_passes":0,"discarded_micro_D_backward_calls":0,"endpoint_mismatch_chunks":0,"canonical_code_mismatch_chunks":0,"Box_free_mask_mismatch_chunks":0}
OLD_STREAMS = ["source_rows","REAL_reference_rows","REAL_identity_rows","source_epsilon","identity_epsilon","projection_directions","baseline_projection_directions"]
NEW_STREAM = "full_SOURCE_epsilon"
ARTIFACTS = ["DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz", "FINAL_FLOW.pt","INITIALIZATION.json","INITIAL_RANDOM_STATES.json","FINAL_RANDOM_STATES.json","LOSS_BY_UPDATE_PRIVATE.npz","PRECISION_ORDER_BY_UPDATE_PRIVATE.npz","REPLAY_BY_UPDATE_PRIVATE.npz","PARENT_EXPOSURES_PRIVATE.npz"]
C_ARRAYS = ["source_counts", "REAL_reference_counts", "REAL_identity_counts",
            "source_bin0", "REAL_bin0", "source_ids"]
TRUE_INTEGRITY = [
    "seven_original_stream_hashes_equal_R63",
    "seven_original_final_states_equal_R63", "C_parent_exposures_equal_R63",
    "frozen_distribution_and_temporal_statistics_reused",
    "discarded_micro_D_not_in_backward", "cast_clip_Adam_once_per_update",
]
FALSE_INTEGRITY = [
    "new_statistics_fit", "joint_D_C_noise_coupling_preserved",
    "new_normalization_fit", "new_noise_calibration_fit", "new_valley_fit",
]
ZERO_COUNTERS = ["teacher_waveforms_constructed", "new_HGB_fits",
                 "new_density_fits", "DEV_read", "EVAL_read", "CAL_decode"]
FROZEN_CONFIG_KEYS = [
    "model_seed", "updates", "batch_per_region", "identity_batch",
    "region_categories", "lr", "lr_initial", "lr_final", "warmup_updates",
    "betas", "gradient_clip", "projection_directions",
    "box_projection_iterations", "solver_steps", "solver_check_steps",
    "generation_chunk", "SOURCE_N", "REAL_N", "PSD_loss_scale",
    "valley_delta", "valley_smoothing", "feature_names",
    "feature_scale_floor", "noise_scale_floor", "loss_weights",
    "source_order_rule", "temporal_feature_count", "temporal_lags",
    "temporal_band_bin_count", "temporal_band_count",
    "temporal_projection_directions", "temporal_scale_floor",
    "temporal_power_floor", "coordinate_RMS_pooling",
]


def _read(path):
    return json.loads(Path(path).read_text("utf8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _npz(path, names):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in names}


def _table(path, columns, width, kind):
    data = _npz(path, ["values", "columns"])
    values = data["values"]
    _require(data["columns"].tolist() == columns,
             str(path) + ": column names/order differ from frozen schema")
    _require(values.shape == (UPDATES, width)
             and values.dtype.kind == kind and values.dtype.itemsize == 8,
             str(path) + ": wrong shape or dtype")
    _require(bool(np.isfinite(values).all()), str(path) + ": nonfinite record")
    return values


def _physical(receipt):
    """Validate persisted lower bounds; never invent missing crash work."""
    attempts = receipt["physical_attempt_compute"]
    history = receipt["recovery_history"]
    _require(isinstance(attempts, list) and attempts, "no physical attempt record")
    _require(isinstance(history, list) and len(history) == len(attempts) - 1,
             "one recovery record is required between consecutive attempts")
    known = {key: 0 for key in EXPECTED_COUNTS}
    ids, unknown = [], []
    for number, attempt in enumerate(attempts):
        ident = attempt["attempt_id"]
        start, end = attempt["resumed_from_update"], attempt["last_completed_update"]
        _require(isinstance(ident, str) and bool(ident) and ident not in ids,
                 "physical attempt IDs must be unique strings")
        _require(_int(start) and _int(end) and 0 <= start <= end <= UPDATES,
                 "invalid physical attempt completed-update range")
        _require(start == 0 if number == 0 else start % 500 == 0,
                 "recovery must start at a committed registered snapshot")
        ids.append(ident)
        counter, intervals = attempt["counter"], attempt["unknown_intervals"]
        complete = attempt["counters_complete"]
        _require(type(complete) is bool and isinstance(intervals, list)
                 and complete == (len(intervals) == 0),
                 "physical completeness and unknown intervals disagree")
        _require(all(key in counter and _int(counter[key]) for key in known),
                 "physical counters must contain all frozen nonnegative integer keys")
        _require(counter["new_response_updates"] == end - start,
                 "physical update counter must start at zero for each attempt")
        for key, final in EXPECTED_COUNTS.items():
            _require(counter[key] >= (end - start) * (final // UPDATES),
                     "physical counter below its known completed-update work: " + key)
            known[key] += counter[key]
        for interval in intervals:
            first, last = interval["first_unpersisted_update"], interval["last_possible_update"]
            _require(_int(first) and _int(last) and start < first <= last <= UPDATES
                     and isinstance(interval["reason"], str) and bool(interval["reason"]),
                     "unknown physical interval is missing honest bounds/reason")
            unknown.append(dict(attempt_id=ident, **interval))
        if number:
            recovery, previous = history[number - 1], attempts[number - 1]
            _require(recovery["from_attempt_id"] == previous["attempt_id"]
                     and recovery["to_attempt_id"] == ident
                     and recovery["checkpoint_update"] == start
                     and start <= previous["last_completed_update"]
                     and _hex(recovery["checkpoint_sha256"])
                     and recovery["bindings_verified"] is True
                     and recovery["previous_PID_confirmed_dead"] is True,
                     "recovery provenance or checkpoint boundary mismatch")
    _require(attempts[-1]["last_completed_update"] == UPDATES,
             "last physical attempt did not complete the unique endpoint")
    _require(all(known[k] >= v for k, v in EXPECTED_COUNTS.items()),
             "known physical work cannot undercount the final committed path")
    if len(attempts) == 1:
        _require(attempts[0]["counters_complete"] is True and known == EXPECTED_COUNTS,
                 "an uninterrupted endpoint must have exactly the frozen physical totals")
    exact = all(a["counters_complete"] for a in attempts)
    return {
        "attempt_count": len(attempts), "recovery_count": len(history),
        "known_physical_counter_lower_bounds": known,
        "physical_counter_totals_exact": exact,
        "physical_counter_totals": known if exact else None,
        "unknown_intervals": unknown,
        "logical_completed_counter": {k: receipt["counter"][k] for k in EXPECTED_COUNTS},
        "logical_and_physical_work_separated": True,
        "recovery_snapshot_deserialized_again": False,
        "native_exit_codes_inferred_from_training_receipt": False,
    }


def _run_audit(root, post_cfg, receipt):
    root = Path(root)
    training_root = Path(post_cfg.get("historical_training_root", root))
    cfg = _read(training_root / "config.json")
    p = training_root / cfg["arm"]
    checks = {}
    _require(cfg["stage"] == STAGE and cfg["arm"] == ARM
             and post_cfg["arm"] == ARM and receipt["arm"] == ARM,
             "R65 audit invoked on the wrong stage/arm")
    _require(cfg["updates"] == UPDATES and cfg["SOURCE_N"] == SOURCE_N
             and cfg["REAL_N"] == REAL_N,
             "frozen unique endpoint/population changed")
    _require(cfg["loss_columns"] == LOSS_COLUMNS
             and cfg["precision_columns"] == PRECISION_COLUMNS
             and cfg["replay_columns"] == REPLAY_COLUMNS
             and cfg["expected_replay_row"] == REPLAY_ROW
             and cfg["expected_completed_update_counts"] == EXPECTED_COUNTS
             and cfg["full_D_scalar_order"] == D_TERMS
             and cfg["C_scalar_order"] == C_TERMS,
             "runtime config differs from the frozen R65 recording schema")

    frozen = receipt["frozen_binding"]
    frozen_paths = {
        "protocol_SHA": "PROTOCOL.md", "config_SHA": "config.json",
        "input_binding_SHA": "INPUT_BINDING.json",
        "implementation_review_SHA": "ROOT_IMPLEMENTATION_REVIEW.json",
    }
    checks["same_protocol_config_inputs_and_implementation_bindings"] = all(
        _hex(frozen[key]) and _sha(training_root / name) == frozen[key]
        for key, name in frozen_paths.items())
    review = _read(training_root / "ROOT_IMPLEMENTATION_REVIEW.json")
    _require(isinstance(review["files"], list) and review["files"],
             "implementation binding has no file records")
    checks["all_implementation_review_file_bytes_preserved"] = all(
        (training_root / entry["name"]).resolve().is_relative_to(training_root.resolve())
        and _hex(entry["sha256"]) and _sha(training_root / entry["name"]) == entry["sha256"]
        for entry in review["files"])
    environment = frozen["environment"]
    checks["frozen_original_environment_and_training_mode"] = (
        all(environment[k] == cfg["expected_R33_environment"][k]
            for k in ["python", "numpy", "torch", "scipy", "device", "GPU"])
        and environment["model_training_mode"] is True
        and environment["MHA_fastpath"] is False and environment["TF32"] is False
        and environment["deterministic_algorithms"] is True
        and environment["CPU_threads"] == receipt["CPU_threads"])
    checks["pre_TRAIN_artificial_receipt_file_bound"] = (
        _hex(receipt["artificial_suite_SHA256"])
        and _sha(training_root / "ARTIFICIAL_SUITE_RECEIPT.json") == receipt["artificial_suite_SHA256"])
    binding = _read(training_root / "INPUT_BINDING.json")
    inputs = cfg["input_paths"]
    checks["registered_input_binding_keys"] = set(binding["inputs"]) == set(inputs)
    input_checks = {}
    for name, entry in binding["inputs"].items():
        path = Path(inputs[name])
        input_checks[name] = (str(path) == entry["path"]
                              and path.stat().st_size == entry["bytes"]
                              and _sha(path) == entry["sha256"])
    checks["all_frozen_input_file_hashes_match"] = all(input_checks.values())
    historical = _read(inputs["R63_training_receipt"])
    old_cfg = _read(inputs["R63_config"])
    old_rng1 = _read(inputs["R63_final_random_states"])
    old_root = Path(cfg["historical_R63_root"])
    checks["unchanged_original_formula_optimizer_and_numerical_parameters"] = all(
        cfg[k] == old_cfg[k] for k in FROZEN_CONFIG_KEYS)
    checks["nine_outer_weights_one"] = (set(cfg["loss_weights"]) == set(D_TERMS + C_TERMS)
                                        and all(v == 1. for v in cfg["loss_weights"].values()))

    artifact_hashes = {name: _sha(p / name) for name in ARTIFACTS}
    checks["saved_artifact_bytes_bound_by_training_receipt"] = all(
        receipt["artifact_sha256"].get(name) == digest
        for name, digest in artifact_hashes.items())
    init = _read(p / "INITIALIZATION.json")
    rng0, rng1 = _read(p / "INITIAL_RANDOM_STATES.json"), _read(p / "FINAL_RANDOM_STATES.json")
    losses = _table(p / "LOSS_BY_UPDATE_PRIVATE.npz", LOSS_COLUMNS, 32, "f")
    precision = _table(p / "PRECISION_ORDER_BY_UPDATE_PRIVATE.npz", PRECISION_COLUMNS, 4, "f")
    replay = _table(p / "REPLAY_BY_UPDATE_PRIVATE.npz", REPLAY_COLUMNS, 12, "i")
    discarded_columns = receipt["discarded_D_columns"]
    _require(isinstance(discarded_columns, list) and len(discarded_columns) == 7
             and len(set(discarded_columns[:5])) == 5
             and set(discarded_columns[:5]) == set(D_TERMS)
             and discarded_columns[5:] == ["original_nine_total", "discarded_D_total"],
             "discarded micro-D record does not name all five original D pieces")
    discarded = _table(p / "DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz",
                        discarded_columns, 7, "f")
    col = {name: losses[:, i] for i, name in enumerate(LOSS_COLUMNS)}
    checks["discarded_micro_D_finite_nonnegative_and_original_nine_decomposition"] = bool(
        receipt["discarded_D_trace_rows"] == UPDATES and np.all(discarded >= 0)
        and np.allclose(discarded[:, 6], discarded[:, :5].sum(1),
                        rtol=1e-12, atol=1e-12)
        and np.allclose(discarded[:, 5], discarded[:, 6] + col["C_total"],
                        rtol=1e-12, atol=1e-12))
    exposures = _npz(p / "PARENT_EXPOSURES_PRIVATE.npz",
                     C_ARRAYS + ["full_D_source_counts", "full_D_REAL_counts"])
    old_exposures = _npz(inputs["R63_parent_exposures"], C_ARRAYS + ["source_original_psd"])
    c_equal = {name: bool(exposures[name].dtype == old_exposures[name].dtype
                         and np.array_equal(exposures[name], old_exposures[name]))
               for name in C_ARRAYS}
    checks["all_original_C_parent_arrays_and_exposures_equal_R63"] = all(c_equal.values())
    checks["source_REAL_bin_and_parent_lengths"] = (
        exposures["source_bin0"].shape == (SOURCE_N,)
        and exposures["REAL_bin0"].shape == (REAL_N,)
        and len(exposures["source_ids"]) == SOURCE_N)
    c_totals = [("source_counts", SOURCE_N, 16000000),
                ("REAL_reference_counts", REAL_N, 16000000),
                ("REAL_identity_counts", REAL_N, 3200000)]
    checks["positive_original_C_exposures_and_totals"] = all(
        exposures[k].shape == (n,) and exposures[k].dtype.kind in "iu"
        and np.all(exposures[k] > 0) and int(exposures[k].sum()) == total
        for k, n, total in c_totals)
    checks["exact_original_C_stratified_draw_totals"] = all(
        int(exposures[count][exposures[bins] == region].sum()) == 1600000
        for count, bins in [("source_counts", "source_bin0"),
                            ("REAL_reference_counts", "REAL_bin0")]
        for region in range(-1, 9))
    checks["full_D_every_parent_once_per_committed_update"] = all(
        exposures[k].shape == (n,) and exposures[k].dtype.kind in "iu"
        and np.all(exposures[k] == UPDATES)
        and int(exposures[k].sum()) == n * UPDATES
        for k, n in [("full_D_source_counts", SOURCE_N), ("full_D_REAL_counts", REAL_N)])
    actual_unique = [int(np.count_nonzero(exposures[k])) for k, _, _ in c_totals]
    checks["unique_parent_counts_from_saved_exposures"] = (
        actual_unique == receipt["actual_unique_SOURCE_REAL_reference_REAL_identity"]
        == [SOURCE_N, REAL_N, REAL_N])

    checks["sole_original_initialization_and_unique_final50000"] = (
        receipt["updates"] == UPDATES and receipt["selection"] == "sole final50000"
        and receipt["new_response_training_runs"] == 1
        and receipt["model_seed"] == cfg["model_seed"] == init["seed"] == 20261010590
        and receipt["initial_state_SHA"] == init["state_tensor_SHA"]
        == historical["initial_state_SHA"] == cfg["expected_R63_initial_state_SHA"] == INITIAL_SHA
        and init["loaded_previous_trained_response"] is False
        and _hex(receipt["final_state_SHA"]))
    checks["no_teacher_labels_or_validation_decode"] = (
        receipt["teacher_waveforms_constructed"] == 0
        and receipt["R58_law_or_atoms_training_target"] is False
        and receipt["h_PDG_network_inputs"] is False
        and receipt["source_val_target_val_decoded"] is False
        and all(receipt[k] == 0 for k in ["DEV_read", "EVAL_read", "CAL_decode"])
        and all(receipt["counter"][k] == 0 for k in ZERO_COUNTERS))
    checks["completed_update_counters_equal_frozen_schema"] = all(
        _int(receipt["counter"][k]) and receipt["counter"][k] == value
        for k, value in EXPECTED_COUNTS.items())
    checks["all104_chunk_replays_and_single_reverse_counts_every_update"] = bool(
        np.array_equal(replay, np.broadcast_to(np.asarray(REPLAY_ROW, dtype=np.int64), replay.shape)))
    checks["one_combined_cast_clip_Adam_and_no_discarded_D_backward"] = all(
        _int(receipt["gradient_execution_counts"][k])
        and receipt["gradient_execution_counts"][k] == value
        for k, value in GRADIENT_COUNTS.items())

    names = set(OLD_STREAMS + [NEW_STREAM])
    seeds = cfg["PCG64_seeds"]
    checks["all_eight_seed_chain_and_state_names_present"] = (
        set(seeds) == set(receipt["PCG64_seeds"]) == set(receipt["stream_hashes"])
        == set(rng0) == set(rng1) == names
        and receipt["PCG64_seeds"] == seeds
        and all(_hex(value) for value in receipt["stream_hashes"].values()))
    checks["eight_initial_PCG64_states_match_frozen_seeds"] = all(
        rng0[name] == np.random.PCG64(seed).state for name, seed in seeds.items())
    checks["original_seven_seed_chain_and_final_states_equal_bound_R63"] = (
        {k: seeds[k] for k in OLD_STREAMS} == historical["PCG64_seeds"] == old_cfg["PCG64_seeds"]
        and {k: receipt["stream_hashes"][k] for k in OLD_STREAMS}
        == historical["stream_hashes"] == cfg["expected_R63_stream_hashes"]
        and {k: rng1[k] for k in OLD_STREAMS} == old_rng1
        and isinstance(receipt["stream_hash_algorithm"], str)
        and bool(receipt["stream_hash_algorithm"]))
    checks["fresh_full_parent_eighth_stream_and_current_directions"] = (
        seeds[NEW_STREAM] == 20261011650
        and cfg["new_stream"] == {"name": NEW_STREAM, "seed": 20261011650,
                                  "draw_per_update": "standard_normal((6593,480)).astype(<f4)"}
        and rng1[NEW_STREAM]["bit_generator"] == "PCG64"
        and rng1[NEW_STREAM] != rng0[NEW_STREAM]
        and all(receipt[k] == UPDATES for k in [
            "projection_direction_calls", "temporal_direction_calls", "full_SOURCE_epsilon_calls"]))
    integrity = receipt["full_empirical_D_integrity"]
    checks["R65_integrity_flags_match_declared_algorithm"] = (
        all(integrity[k] is True for k in TRUE_INTEGRITY)
        and all(integrity[k] is False for k in FALSE_INTEGRITY)
        and cfg["complete_parent_D"] is True
        and cfg["original_C_microbatches_per_update"] == 1
        and cfg["neural_replay_chunk"] == 64
        and cfg["original_D_C_random_coupling_preserved"] is False
        and cfg["new_statistics_fit"] is False
        and cfg["new_REAL_feature_cache"] is False
        and cfg["extracted_C_implementation"] is False
        and cfg["sampled_parameter_VJP"] is False
        and cfg["model_training_mode"] is True
        and cfg["MHA_fastpath_enabled"] is False)

    for name, terms in [("D_total", D_TERMS), ("C_total", C_TERMS),
                        ("total_loss", D_TERMS + C_TERMS)]:
        checks[name + "_ordered_component_sum"] = bool(np.allclose(
            col[name], sum(col[k] for k in terms), rtol=1e-12, atol=1e-12))
    checks["total_is_D_plus_C"] = bool(np.allclose(
        col["total_loss"], col["D_total"] + col["C_total"], rtol=1e-12, atol=1e-12))
    checks["nine_losses_and_norms_nonnegative"] = all(np.all(col[k] >= 0) for k in
        D_TERMS + C_TERMS + ["D_total", "C_total", "gradient_norm",
                            "D_parameter_gradient_L2", "C_parameter_gradient_L2"])
    checks["original_full_D_PSD_half_sum_scale"] = bool(np.allclose(
        col["PSD"], .5 * (col["overall_PSD_W1"] + col["Q1_Q7_mean_PSD_W1"])
        / cfg["PSD_loss_scale"], rtol=1e-12, atol=1e-12))
    learning_rates = []
    for step in range(1, UPDATES + 1):
        if step <= cfg["warmup_updates"]:
            value = cfg["lr_initial"] + (cfg["lr"] - cfg["lr_initial"]) * (step - 1) / (cfg["warmup_updates"] - 1)
        else:
            fraction = (step - cfg["warmup_updates"]) / (UPDATES - cfg["warmup_updates"])
            value = cfg["lr_final"] + .5 * (cfg["lr"] - cfg["lr_final"]) * (1 + math.cos(math.pi * fraction))
        learning_rates.append(value)
    checks["unchanged_original_learning_rate_trace"] = bool(np.allclose(
        col["learning_rate"], learning_rates, rtol=1e-12, atol=1e-12))
    for suffix in ["f32", "f64"]:
        overall, d_q, c_q = (col["max_relative_q_" + suffix],
                            col["full_D_max_relative_q_" + suffix], col["C_max_relative_q_" + suffix])
        checks["original_charge_bound_and_complete_trace_" + suffix] = bool(
            np.all(overall >= 0) and np.all(d_q >= 0) and np.all(c_q >= 0)
            and np.array_equal(overall, np.maximum(d_q, c_q))
            and float(overall.max()) <= 1e-5
            and np.isfinite(receipt["max_output_relative_q_error_" + suffix])
            and 0 <= receipt["max_output_relative_q_error_" + suffix] <= 1e-5
            and np.isclose(receipt["max_output_relative_q_error_" + suffix],
                           float(overall.max()), rtol=1e-12, atol=1e-12))
    fraction_fields = ["SOURCE_global_range_projection_fraction", "REAL_range_projection_fraction",
                       "full_D_SOURCE_range_projection_fraction"]
    rms_fields = ["SOURCE_global_projection_RMS_V", "REAL_projection_RMS_V",
                  "full_D_SOURCE_projection_RMS_V"]
    checks["separate_full_D_and_original_C_projection_records_valid"] = bool(
        all(np.all((col[k] >= 0) & (col[k] <= 1)) for k in fraction_fields)
        and all(np.all(col[k] >= 0) for k in rms_fields))
    checks["original_C_precision_order_trace"] = bool(
        receipt["precision_order_columns"] == PRECISION_COLUMNS
        and receipt["precision_order_trace_rows"] == UPDATES
        and np.all(precision >= 0) and np.all(precision[:, 2] > 0)
        and np.all(precision[:, 2] <= 1 + 1e-12)
        and np.all(precision[:, 3] <= 1 + 1e-12)
        and np.allclose(precision[:, 0], col["source_order"] + precision[:, 1],
                        rtol=1e-10, atol=1e-10))

    precision_names = ["SOURCE_PSD_PRECISION_PRIVATE.npz", "SOURCE_ORDER_PRECISION_DEFINITION.json"]
    checks["original_source_order_files_copied_byte_identically"] = all(
        _sha(training_root / name) == _sha(old_root / name) for name in precision_names)
    meta = _read(training_root / "SOURCE_ORDER_PRECISION_DEFINITION.json")
    intervals = _npz(training_root / "SOURCE_PSD_PRECISION_PRIVATE.npz",
                     ["source_ids", "source_original_psd", "PSD_interval_low", "PSD_interval_high"])
    lo, hi, original_psd = (intervals["PSD_interval_low"], intervals["PSD_interval_high"],
                            intervals["source_original_psd"])
    checks["original_order_rule_intervals_and_denominator_preserved"] = bool(
        meta["interval_rule"] == receipt["source_order_rule"] == cfg["source_order_rule"]
        and meta["all_original_denominator_pairs_retained"] is True
        and meta["new_threshold_fit"] is False
        and np.array_equal(intervals["source_ids"], exposures["source_ids"])
        and np.array_equal(original_psd, old_exposures["source_original_psd"])
        and lo.shape == hi.shape == original_psd.shape == (SOURCE_N,)
        and np.isfinite(lo).all() and np.isfinite(hi).all()
        and np.all(lo <= original_psd) and np.all(original_psd <= hi))
    constant_files = {
        "UNIT_CHARGE_CONSTANTS.npz": "constants_path",
        "NOISE_CALIBRATION.npz": "noise_calibration_path",
        "TRAIN_DISTRIBUTION_CONSTANTS.npz": "distribution_statistics",
        "TRAIN_TEMPORAL_CONSTANTS.npz": "temporal_statistics",
    }
    constant_hashes = {name: _sha(training_root / name) for name in constant_files}
    checks["all_original_constant_copies_byte_identical_no_refit_claim"] = all(
        constant_hashes[name] == binding["inputs"][key]["sha256"]
        for name, key in constant_files.items())
    temporal = _npz(training_root / "TRAIN_TEMPORAL_CONSTANTS.npz",
                     ["temporal_center", "temporal_scale", "temporal_REAL_counts"])
    counts = _npz(training_root / "TRAIN_DISTRIBUTION_CONSTANTS.npz", ["REAL_counts"])["REAL_counts"]
    tm = _read(training_root / "TRAIN_TEMPORAL_CONSTANTS.json")
    center, scale = temporal["temporal_center"], temporal["temporal_scale"]
    checks["original_temporal_metadata_shape_scale_and_population"] = bool(
        _sha(training_root / "TRAIN_TEMPORAL_CONSTANTS.json")
        == _sha(old_root / "TRAIN_TEMPORAL_CONSTANTS.json")
        and center.shape == scale.shape == (10, 31)
        and np.isfinite(center).all() and np.isfinite(scale).all()
        and np.all(scale >= np.asarray(cfg["temporal_scale_floor"]))
        and np.array_equal(center, tm["center"]) and np.array_equal(scale, tm["scale"])
        and np.array_equal(temporal["temporal_REAL_counts"], counts)
        and int(counts.sum()) == tm["fitted_REAL_TRAIN_rows"] == REAL_N
        and constant_hashes["TRAIN_TEMPORAL_CONSTANTS.npz"] == tm["new_statistics_SHA256"]
        and constant_hashes["TRAIN_DISTRIBUTION_CONSTANTS.npz"] == tm["old_statistics_SHA256"]
        and tm["fitted_before_first_update"] is True
        and tm["scientific_response_fit"] is False and tm["teacher_fit"] is False)
    checks["GPU_and_CPU_training_receipt_within_original_caps"] = (
        all(_int(receipt[k]) and receipt[k] <= cfg["GPU_limit_bytes"]
            for k in ["GPU_allocated_peak_bytes", "GPU_reserved_peak_bytes"])
        and _int(receipt["CPU_threads"]) and 1 <= receipt["CPU_threads"] <= cfg["CPU_threads"] <= 12)
    checks["no_compute_budget_or_model_completion_claim"] = (
        receipt["compute_budget_enforced"] is False and receipt["model_goal_complete"] is False)
    if "endpoint_weight_paths" in post_cfg:
        checks["numerical_gate_weight_is_bound_unique_final_file"] = (
            _sha(Path(post_cfg["endpoint_weight_paths"][ARM])) == artifact_hashes["FINAL_FLOW.pt"])
    physical = _physical(receipt)
    checks["physical_attempt_and_recovery_records_valid"] = True
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "pass": all(checks.values()), "checks": checks, "stage": STAGE,
        "training_root": str(training_root), "arm": ARM,
        "all_updates_recorded": len(losses), "loss_columns": LOSS_COLUMNS,
        "precision_trace_columns": PRECISION_COLUMNS, "replay_columns": REPLAY_COLUMNS,
        "discarded_micro_D_columns": discarded_columns,
        "discarded_micro_D_trace_rows": len(discarded),
        "discarded_micro_D_SOURCE_REAL_forward_occurrences": [receipt["counter"]["SOURCE_training_rows"], receipt["counter"]["REAL_reference_rows"]],
        "discarded_micro_D_backward_calls": receipt["gradient_execution_counts"]["discarded_micro_D_backward_calls"],
        "weight_file_SHA256": artifact_hashes["FINAL_FLOW.pt"],
        "artifact_SHA256": artifact_hashes, "frozen_input_hash_checks": input_checks,
        "R63_C_parent_array_checks": c_equal, "constant_copy_SHA256": constant_hashes,
        "constraint_loss_trace_summary": {k: ev.summary(col[k]) for k in fraction_fields + rms_fields},
        "D_C_loss_trace_summary": {k: ev.summary(col[k]) for k in ["D_total", "C_total"]},
        "precision_trace_summary": {k: ev.summary(precision[:, i]) for i, k in enumerate(PRECISION_COLUMNS)},
        "original_minus_resolved_minus_omitted_maxabs":
            float(np.max(np.abs(precision[:, 0] - col["source_order"] - precision[:, 1]))),
        "source_interval_width_summary": ev.summary(hi - lo),
        "physical_compute": physical,
        "TRAIN_summary_measure": "five full finite-parent D terms at each update's fresh noise/directions; four original single-microbatch C terms",
        "full_D_every_original_parent_exposure_verified": checks["full_D_every_parent_once_per_committed_update"],
        "registered_full_D_measure": "separate equal original SOURCE and REAL parent laws",
        "C_realized_exposures_uniform": False,
        "registered_C_measure": "original equal-parent measure in expectation via unchanged regional masses",
        "joint_D_C_stochastic_coupling_identical_to_R63": False,
        "original_seven_streams_compared_to_bound_actual_R63_chains": True,
        "original_seven_streams_independently_reconstructed_again": False,
        "eighth_stream_independently_reconstructed": False,
        "final_tensor_hash_independently_recomputed": False,
        "single_training_seed": True, "independent_training_seed_variability_estimated": False,
        "saved_records_only_no_new_scientific_forward_or_fit": True,
        "native_exit_and_tree_RSS_evidence": "root worker/supervisor receipts; not inferred here",
        "original_numerical_gate_and_A_F_thresholds_modified": False,
    }


def audit_saved_records(root, cfg, receipt):
    """Return a fail-closed audit instead of hiding malformed/missing records."""
    try:
        return _run_audit(root, cfg, receipt)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        return {
            "pass": False, "checks": {"complete_well_formed_R65_records": False},
            "stage": STAGE, "error_type": type(error).__name__, "error": str(error),
            "saved_records_only_no_new_scientific_forward_or_fit": True,
            "original_numerical_gate_and_A_F_thresholds_modified": False,
        }
