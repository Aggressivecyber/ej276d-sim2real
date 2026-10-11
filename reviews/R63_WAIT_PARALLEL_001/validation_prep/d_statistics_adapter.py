"""Pure NumPy implementation of the frozen D1-D6 statistical decision rules.

This module performs no I/O, model fitting, prediction, generation, or selection.
The caller supplies manufactured arrays in the included selftests. Connection to
real D evaluation and verification of the actual frozen design are NOT provided.

Call ``evaluate_d_statistics(truth=..., estimates=...,
frozen_input_design_valid=True)`` only after independently validating the design.
The flag is mandatory and has no default. Omission is a Python TypeError; False
or a non-bool declaration returns undefined checks that cannot pass. Each truth
and prediction packet must separately bind ``row_ids`` and ``group_ids`` to its
one-dimensional ``psd`` and ``logq`` values. Every packet must match the truth's
row order and groups. Bindings are caller declarations, not provenance proofs.

Required estimator keys: Ridge, HGB. Required arm keys for each estimator:
candidate, real_only_1024, original_sim_1024, full_real. The latter three names
mean REAL-only at 1024, REAL1024 + original SIM, and current full REAL-only.
The API does not establish training counts, source parents, or model freezing;
those obligations are included in the external frozen-input-design declaration.

All arms and estimators share exactly 1000 resamples of whole caller-bound groups
using PCG64(seed=2026100853). Whole-group multiplicity weights each event in that
group, including unequal group sizes. Quantiles use NumPy method='linear'. Any
invalid bootstrap replicate makes its CI undefined; no replicate is discarded.
Zero reference errors are undefined, including a zero-over-zero comparison.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np


ESTIMATORS = ("Ridge", "HGB")
ARMS = ("candidate", "real_only_1024", "original_sim_1024", "full_real")
CHECKS = ("D1", "D2", "D3", "D4", "D5")
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 2026100853
CI_PROBABILITIES = (0.025, 0.975)


class InputBindingError(ValueError):
    """A required packet, value, or row/group binding is not valid."""


def _undefined(reason, **details):
    return {"status": "undefined", "defined": False, "pass": False,
            "reason": reason, **details}


def _decision(passed, **details):
    passed = bool(passed)
    return {"status": "passed" if passed else "failed", "defined": True,
            "pass": passed, **details}


def _undefined_report(reason, declaration):
    return {
        "schema_version": "D-statistics-preparation-1",
        "scope": "statistics_only_external_design_not_verified",
        "frozen_input_design_valid": declaration if type(declaration) is bool else None,
        "input_binding": _undefined(reason),
        "bootstrap": None,
        "estimators": {name: {"point_metrics": None,
                              "checks": {check: _undefined(reason) for check in CHECKS}}
                       for name in ESTIMATORS},
        "D6": _undefined(reason),
    }


def _identities(values, name):
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise InputBindingError(f"{name}: masked identities are missing")
    try:
        objects = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise InputBindingError(f"{name}: identities cannot be read") from exc
    if objects.ndim != 1 or objects.size == 0:
        raise InputBindingError(f"{name}: nonempty one-dimensional identities required")
    items = objects.tolist()
    if all(isinstance(value, str) and len(value) > 0 for value in items):
        return np.asarray(items, dtype=np.str_)
    if all(isinstance(value, (int, np.integer)) and
           not isinstance(value, (bool, np.bool_)) for value in items):
        try:
            return np.asarray(items, dtype=np.int64)
        except (ValueError, OverflowError) as exc:
            raise InputBindingError(f"{name}: integer identity outside int64") from exc
    raise InputBindingError(f"{name}: use homogeneous nonempty strings or int64 identities")


def _numbers(values, name, count):
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise InputBindingError(f"{name}: masked numeric values are missing")
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf":
            raise InputBindingError(f"{name}: real numeric values required")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputBindingError(f"{name}: real numeric values required") from exc
    if result.ndim != 1 or result.size != count:
        raise InputBindingError(f"{name}: one value per bound row required")
    if not np.all(np.isfinite(result)):
        raise InputBindingError(f"{name}: all values must be finite")
    return result


def _packet(packet, name):
    if not isinstance(packet, Mapping):
        raise InputBindingError(f"{name}: a packet mapping is required")
    required = ("row_ids", "group_ids", "psd", "logq")
    missing = [key for key in required if key not in packet]
    if missing:
        raise InputBindingError(f"{name}: missing fields {missing}")
    rows = _identities(packet["row_ids"], f"{name}.row_ids")
    groups = _identities(packet["group_ids"], f"{name}.group_ids")
    if groups.size != rows.size:
        raise InputBindingError(f"{name}: row/group lengths differ")
    if np.unique(rows).size != rows.size:
        raise InputBindingError(f"{name}: duplicate row identities")
    return {"row_ids": rows, "group_ids": groups,
            "psd": _numbers(packet["psd"], f"{name}.psd", rows.size),
            "logq": _numbers(packet["logq"], f"{name}.logq", rows.size)}


def _same_identities(left, right):
    return left.dtype.kind == right.dtype.kind and np.array_equal(left, right)


def _identity_sha256(values):
    encoded = json.dumps({"kind": "int64" if values.dtype.kind == "i" else "string",
                          "values": values.tolist()}, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _group_weighted_means(group_error_sums, group_sizes, multiplicities):
    """Return event means after repeating whole groups; no equal-group averaging."""
    sums = np.asarray(group_error_sums, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    counts = np.asarray(multiplicities, dtype=np.float64)
    if sizes.ndim != 1 or counts.shape != sizes.shape or sums.shape[-1:] != sizes.shape:
        raise ValueError("group sums, sizes, and multiplicities must align")
    if (not np.all(np.isfinite(sums)) or not np.all(np.isfinite(sizes)) or
            not np.all(np.isfinite(counts)) or np.any(sizes <= 0) or
            np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("finite positive group sizes and nonnegative integer draws required")
    denominator = float(np.sum(sizes * counts))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("group resample must contain events")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.sum(sums * counts, axis=-1) / denominator


def _rmse(errors):
    scale = float(np.max(np.abs(errors)))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(scale * np.sqrt(np.mean(np.square(errors / scale))))


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _relative_point(candidate, reference, threshold, metric):
    detail = {"metric": metric, "candidate": _json_number(candidate),
              "reference": _json_number(reference), "relative_threshold": threshold,
              "comparison": "candidate <= (1 + relative_threshold) * reference",
              "strict": False, "difference": None, "relative_difference": None,
              "comparison_lhs": _json_number(candidate), "comparison_rhs": None}
    if not np.isfinite(candidate) or not np.isfinite(reference) or reference <= 0:
        return _undefined("nonfinite metric or zero reference denominator", **detail)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = float(np.float64(candidate) - np.float64(reference))
        relative = float(np.float64(difference) / np.float64(reference))
        bound = float((1.0 + threshold) * np.float64(reference))
    detail.update(difference=_json_number(difference), relative_difference=_json_number(relative),
                  comparison_rhs=_json_number(bound))
    if not np.all(np.isfinite([difference, relative, bound])):
        return _undefined("nonfinite derived point comparison", **detail)
    # This equivalent raw-metric comparison is inclusive at a represented 5% bound.
    # No epsilon or tolerance widens the scientific threshold.
    return _decision(candidate <= bound, **detail)


def _bootstrap_ci(candidate, reference, *, relative, point_difference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    bad_denominator = (~np.isfinite(reference) | (reference <= 0)) if relative else np.zeros(
        reference.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = candidate - reference
        if relative:
            values = values / reference
    invalid = (~np.isfinite(candidate) | ~np.isfinite(reference) |
               ~np.isfinite(values) | bad_denominator)
    valid_count = int(np.count_nonzero(~invalid))
    detail = {
        "estimand": "relative_PSD_MAE_difference" if relative else "absolute_PSD_MAE_difference",
        "point_difference": _json_number(point_difference),
        "formula": "(MAE_candidate[b] - MAE_reference[b]) / MAE_reference[b]" if relative
                   else "MAE_candidate[b] - MAE_reference[b]",
        "confidence_interval": None, "quantile_probabilities": list(CI_PROBABILITIES),
        "quantile_method": "linear", "comparison": "CI_upper < 0", "strict": True,
        "requested_replicates": BOOTSTRAP_REPEATS, "valid_replicates": valid_count,
        "invalid_replicates": int(np.count_nonzero(invalid)),
        "denominator_failures": int(np.count_nonzero(bad_denominator)),
        "invalid_replicate_indices": np.flatnonzero(invalid).tolist(),
        "replicate_values": [None if bad else float(value) for value, bad in zip(values, invalid)],
        "shared_draw_reference": "bootstrap.draws_sha256",
    }
    if candidate.shape != (BOOTSTRAP_REPEATS,) or reference.shape != (BOOTSTRAP_REPEATS,):
        return _undefined("incorrect bootstrap replicate count", **detail)
    if not np.isfinite(point_difference):
        return _undefined("nonfinite or undefined point comparison", **detail)
    if valid_count != BOOTSTRAP_REPEATS:
        return _undefined("at least one invalid bootstrap replicate; CI not computed", **detail)
    interval = np.quantile(values, CI_PROBABILITIES, method="linear")
    if not np.all(np.isfinite(interval)):
        return _undefined("nonfinite percentile confidence interval", **detail)
    detail["confidence_interval"] = interval.tolist()
    return _decision(interval[1] < 0.0, **detail)


def evaluate_d_statistics(*, truth, estimates, frozen_input_design_valid):
    """Evaluate the fixed D gates from explicitly bound, same-order arrays.

    The returned report is JSON-safe (nonfinite quantities become None), contains
    all bootstrap replicates, and does not mutate the inputs. Missing fields,
    identity mismatches, and nonfinite input values produce undefined checks.
    ``frozen_input_design_valid`` must be a Python bool and explicitly True.
    """
    if type(frozen_input_design_valid) is not bool or not frozen_input_design_valid:
        return _undefined_report("frozen input design was not explicitly declared valid by caller",
                                 frozen_input_design_valid)
    try:
        actual = _packet(truth, "truth")
        if not isinstance(estimates, Mapping) or set(estimates) != set(ESTIMATORS):
            raise InputBindingError("estimates must contain exactly Ridge and HGB")
        packets = {}
        for estimator in ESTIMATORS:
            if not isinstance(estimates[estimator], Mapping) or set(estimates[estimator]) != set(ARMS):
                raise InputBindingError(f"{estimator}: exactly the four primary arms are required")
            for arm in ARMS:
                label = f"{estimator}.{arm}"
                packet = _packet(estimates[estimator][arm], label)
                for key in ("row_ids", "group_ids"):
                    if not _same_identities(packet[key], actual[key]):
                        raise InputBindingError(f"{label}.{key}: same-order truth binding mismatch")
                packets[(estimator, arm)] = packet
        groups, inverse, group_sizes = np.unique(actual["group_ids"], return_inverse=True,
                                                 return_counts=True)
        error_sums, point_metrics = [], {name: {} for name in ESTIMATORS}
        for estimator in ESTIMATORS:
            for arm in ARMS:
                packet = packets[(estimator, arm)]
                with np.errstate(over="ignore", invalid="ignore"):
                    psd_errors = np.abs(packet["psd"] - actual["psd"])
                    logq_errors = packet["logq"] - actual["logq"]
                    sums = np.bincount(inverse, weights=psd_errors, minlength=groups.size)
                if not (np.all(np.isfinite(psd_errors)) and np.all(np.isfinite(logq_errors)) and
                        np.all(np.isfinite(sums))):
                    raise InputBindingError(f"{estimator}.{arm}: nonfinite derived errors or group sums")
                error_sums.append(sums)
                with np.errstate(over="ignore", invalid="ignore"):
                    mae = float(np.sum(sums) / actual["row_ids"].size)
                point_metrics[estimator][arm] = {
                    "psd_mae": _json_number(mae), "logq_rmse": _json_number(_rmse(logq_errors))}
        error_sums = np.asarray(error_sums, dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws_hash = hashlib.sha256()
        bootstrap_mae = np.empty((BOOTSTRAP_REPEATS, len(ESTIMATORS) * len(ARMS)))
        for index in range(BOOTSTRAP_REPEATS):
            draws = rng.integers(0, groups.size, size=groups.size, dtype=np.int64)
            draws_hash.update(np.asarray(draws, dtype="<i8").tobytes(order="C"))
            multiplicities = np.bincount(draws, minlength=groups.size)
            bootstrap_mae[index] = _group_weighted_means(error_sums, group_sizes, multiplicities)
    except (InputBindingError, ValueError, TypeError, OverflowError) as exc:
        return _undefined_report(str(exc), frozen_input_design_valid)

    estimator_reports = {}
    for estimator_index, estimator in enumerate(ESTIMATORS):
        raw = point_metrics[estimator]
        # Missing/nonfinite derived metrics use NaN only internally, never in JSON.
        c, r, sim, full = [float(raw[arm]["psd_mae"]) if raw[arm]["psd_mae"] is not None else np.nan
                           for arm in ARMS]
        cq, rq = [float(raw[arm]["logq_rmse"]) if raw[arm]["logq_rmse"] is not None else np.nan
                  for arm in ("candidate", "real_only_1024")]
        column = estimator_index * len(ARMS)
        bc, br, bs = (bootstrap_mae[:, column + offset] for offset in range(3))
        d1 = _relative_point(c, r, -0.05, "PSD_MAE_candidate_vs_REAL1024")
        checks = {
            "D1": d1,
            "D2": _bootstrap_ci(bc, br, relative=True,
                                 point_difference=d1["relative_difference"]
                                 if d1["relative_difference"] is not None else np.nan),
            "D3": _bootstrap_ci(bc, bs, relative=False, point_difference=c - sim),
            "D4": _relative_point(c, full, 0.05, "PSD_MAE_candidate_vs_REAL_FULL"),
            "D5": _relative_point(cq, rq, 0.05, "logQ_RMSE_candidate_vs_REAL1024"),
        }
        estimator_reports[estimator] = {"point_metrics": raw, "checks": checks,
            "bootstrap_psd_mae": {
                arm: [_json_number(value) for value in bootstrap_mae[:, column + offset]]
                for offset, arm in enumerate(ARMS)}}
    all_checks = [(name, check, estimator_reports[name]["checks"][check])
                  for name in ESTIMATORS for check in CHECKS]
    joint_detail = {"rule": "all D1..D5 for BOTH Ridge and HGB AND frozen_input_design_valid",
                    "frozen_input_design_valid": True,
                    "design_declaration_source": "caller; actual design not verified by this adapter",
                    "undefined_checks": [f"{name}.{key}" for name, key, item in all_checks
                                         if not item["defined"]],
                    "failed_defined_checks": [f"{name}.{key}" for name, key, item in all_checks
                                              if item["defined"] and not item["pass"]]}
    joint = _undefined("at least one required check is undefined", **joint_detail) if joint_detail[
        "undefined_checks"] else _decision(all(item["pass"] for _, _, item in all_checks), **joint_detail)
    return {
        "schema_version": "D-statistics-preparation-1",
        "scope": "statistics_only_external_design_not_verified",
        "frozen_input_design_valid": True,
        "input_binding": {"status": "matched", "rows": int(actual["row_ids"].size),
                          "groups": int(groups.size), "all_eight_prediction_packets_matched": True,
                          "row_ids_sha256": _identity_sha256(actual["row_ids"]),
                          "group_ids_sha256": _identity_sha256(actual["group_ids"]),
                          "truth_psd_f64le_sha256": _array_sha256(actual["psd"]),
                          "truth_logq_f64le_sha256": _array_sha256(actual["logq"]),
                          "prediction_f64le_sha256": {
                              name: {arm: {target: _array_sha256(packets[(name, arm)][target])
                                           for target in ("psd", "logq")} for arm in ARMS}
                              for name in ESTIMATORS},
                          "provenance_verified": False},
        "bootstrap": {"unit": "whole caller-bound group", "repeats": BOOTSTRAP_REPEATS,
                      "seed": BOOTSTRAP_SEED, "rng": "numpy.random.Generator(PCG64)",
                      "draw_method": "integers(0, n_groups, size=n_groups, dtype=int64) once per replicate",
                      "group_order": groups.tolist(), "group_sizes": group_sizes.tolist(),
                      "draws_sha256": draws_hash.hexdigest(),
                      "draw_hash_encoding": "concatenated replicate draws as little-endian signed int64",
                      "shared_across_all_estimators_and_arms": True,
                      "quantile_method": "linear", "quantile_probabilities": list(CI_PROBABILITIES),
                      "invalid_replicate_policy": "any invalid replicate makes that CI undefined; no filtering"},
        "estimators": estimator_reports, "D6": joint,
    }
