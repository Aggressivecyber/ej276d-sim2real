"""Pure v5 SOURCE/charge diagnostics; no file read, RNG, fit, or model import."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
import evaluation_primitives as ev
from common_math_primitives import distances
from evaluation_io import feature_Q_PSD, q_window, require

LABELS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, "invalid"]
MIGRATIONS = {"input_to_Qstar": ("input_Q0", "target_Qstar"),
              "Qstar_to_preADC": ("target_Qstar", "preADC_window"),
              "preADC_to_canonical": ("preADC_window", "canonical_evaluation")}


def raw_bins(q_mVns, edges):
    q = np.asarray(q_mVns, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    require(q.ndim == 1 and edges.shape == (9,) and np.isfinite(edges).all() and
            (np.diff(edges) > 0).all(), "charge diagnostic dimensions/edges invalid")
    legal = np.isfinite(q) & (q > 0)
    result = np.full(len(q), -2, dtype=np.int64)  # -2 is invalid, never a support bin.
    z = np.log(q[legal])
    bins = np.searchsorted(edges, z, side="right") - 1
    bins[z == edges[-1]] = 7
    result[legal] = bins
    return result


def original_uniform_weights(n, supplied=None):
    require(n > 0, "empty original population")
    expected = np.full(n, 1/n, dtype=np.float64)
    weights = expected if supplied is None else np.asarray(supplied)
    require(weights.dtype == np.float64 and weights.shape == (n,) and
            np.array_equal(weights, expected),
            "original equal-weight parents required; no reweighting or renormalization")
    return weights


def migration(a, b, weights):
    require(a.shape == b.shape == weights.shape, "migration parent alignment changed")
    indices = []
    for bins in [a, b]:
        require(np.isin(bins, [-2, -1, *range(9)]).all(), "unmapped migration category")
        indices.append(np.where(bins == -2, 10, bins+1))
    counts = np.zeros((11, 11), dtype=np.int64)
    mass = np.zeros((11, 11), dtype=np.float64)
    np.add.at(counts, tuple(indices), 1)
    np.add.at(mass, tuple(indices), weights)
    return {"labels": LABELS, "count_matrix": counts.tolist(),
            "weight_matrix": mass.tolist(), "count_total": int(counts.sum()),
            "weight_total": float(mass.sum()),
            "all_original_parents_including_outside_and_invalid": True}


def axis_description(q, edges, weights, REAL_q=None, REAL_weights=None):
    q = np.asarray(q, dtype=np.float64)
    original_uniform_weights(len(q), weights)
    bins = raw_bins(q, edges)
    invalid = bins == -2
    indices = np.where(invalid, 10, bins+1)
    counts = np.bincount(indices, minlength=11)
    mass = np.bincount(indices, weights=weights, minlength=11)
    outside = (bins == -1) | (bins == 8)
    cdf = {"unique_Q_mVns": [], "cumulative_original_weight": [],
           "defined": False, "unknown_reason": None}
    if not invalid.any():
        values, inverse = np.unique(q, return_inverse=True)
        jumps = np.bincount(inverse, weights=weights, minlength=len(values))
        cdf.update(unique_Q_mVns=values.tolist(),
                   cumulative_original_weight=np.cumsum(jumps).tolist(), defined=True)
    else:
        cdf["unknown_reason"] = "nonpositive_or_nonfinite_original_charge_retained; no rows deleted"
    versus = {"KS": None, "W1": None, "unknown_reason": None}
    if REAL_q is not None:
        REAL_q = np.asarray(REAL_q, dtype=np.float64)
        original_uniform_weights(len(REAL_q), REAL_weights)
        bad_REAL = (raw_bins(REAL_q, edges) == -2).any()
        if invalid.any() or bad_REAL:
            versus["unknown_reason"] = "candidate_or_REAL_target_invalid; no deletion_or_reweighting"
        else:
            versus.update(distances(q, REAL_q))
    else:
        versus["unknown_reason"] = "REAL comparison not supplied"
    return {"rows": len(q), "units": "mV ns", "labels": LABELS,
            "counts": counts.tolist(), "weights": mass.tolist(),
            "original_weight_sum": float(weights.sum()),
            "outside_support_count": int(outside.sum()),
            "outside_support_weight": float(weights[outside].sum()),
            "invalid_count": int(invalid.sum()), "invalid_weight": float(weights[invalid].sum()),
            "Q_summary": ev.summary(q), "empirical_CDF": cdf, "vs_REAL": versus,
            "invalid_not_folded_into_above": True}, bins


def charge_diagnostics(payloads, raw_inputs, REAL_V, edges, roles=None):
    """All original2000 rows per arm, against untreated REAL DEV target.

    Required canonical_evaluation Q is original evaluation.features mean-centered
    Q on the strict common canonical voltage, in mV ns. It never takes a saved
    Qstar, pipeline-QADC, source bin, or valid/QC subset as the canonical axis.
    """
    REAL_q, _ = feature_Q_PSD(REAL_V)
    rw = original_uniform_weights(len(REAL_q))
    ref_axis, _ = axis_description(REAL_q, edges, rw, REAL_q, rw)
    result = {"domains": {}, "REAL_reference": {
        "role": "untreated_REAL_DEV_target", "rows": len(REAL_q),
        "original_weight_sum": float(rw.sum()), "units": "mV ns",
        "empirical_CDF": ref_axis["empirical_CDF"], "counts": ref_axis["counts"],
        "weights": ref_axis["weights"], "invalid_count": ref_axis["invalid_count"],
        "invalid_weight": ref_axis["invalid_weight"],
        "outside_support_count": ref_axis["outside_support_count"],
        "outside_support_weight": ref_axis["outside_support_weight"],
        "axes": {"canonical_evaluation": ref_axis}},
        "required_axes": ["input_Q0", "target_Qstar", "preADC_window", "canonical_evaluation"],
        "required_migrations": list(MIGRATIONS),
        "same_original_parent_measure": True, "new_Q_acceptance_threshold": None}
    for name, p in payloads.items():
        n = len(raw_inputs[name])
        require(n == len(p["pre_ADC_V"]) == len(p["waveforms_V"]), "charge population changed")
        weights = original_uniform_weights(n, p.get("original_weight"))
        canonical_q, _ = feature_Q_PSD(p["evaluation_canonical_V"])
        axes_q = {
            "input_Q0": 1000.*q_window(raw_inputs[name]),
            "target_Qstar": 1000.*p["predicted_target_Q_Vns"],
            "preADC_window": 1000.*q_window(p["pre_ADC_V"]),
            "canonical_evaluation": canonical_q,
            "canonical_pipeline_window": 1000.*q_window(p["waveforms_V"]),
            "recorded_pipeline_Q0": 1000.*p["Q0"],
            "recorded_pipeline_QADC": 1000.*p["QADC"]}
        axes = {}; bins = {}
        for key, q in axes_q.items():
            axes[key], bins[key] = axis_description(q, edges, weights, REAL_q, rw)
        result["domains"][name] = {
            "rows": n, "role": None if roles is None else roles[name],
            "original_weight_sum": float(weights.sum()), "axes": axes,
            "migrations": {key: migration(bins[a], bins[b], weights)
                           for key, (a, b) in MIGRATIONS.items()},
            "saved_pipeline_bins_are_separate_from_evaluation_output_bins": True}
    return result


def REAL_identity(raw, payload):
    n = len(raw)
    require(n == len(payload["pre_ADC_V"]), "REAL identity parent population changed")
    pre = payload["pre_ADC_V"].astype(np.float64)
    canonical = payload["evaluation_canonical_V"]
    original = np.asarray(raw, dtype=np.float64)
    change_pre = pre-original
    return {"rows": n, "paired_original_reference_rows": True,
            "input_role": "original_TRAIN_REFERENCE", "same_public_response_as_SOURCE": True,
            "preADC_RMS_difference_V": ev.summary(np.sqrt(np.mean(change_pre**2, axis=1))),
            "canonical_RMS_difference_from_original_raw_V": ev.summary(
                np.sqrt(np.mean((canonical-original)**2, axis=1))),
            "input_Q_Vns": ev.summary(q_window(original)),
            "predicted_Qstar_minus_input_Q_Vns": ev.summary(
                payload["predicted_target_Q_Vns"]-q_window(original)),
            "identity_is_descriptive_not_an_added_gate": True}


def source_tables(domains, actual_source_bins, reference_source_bins, comp, valley, edges):
    # Original fixed mother groups and denominators remain passed-in inputs.
    result = {}
    for name, d in domains.items():
        if name == "REAL": continue
        result[name] = {}
        output_bins = raw_bins(d["features"]["Qtotal_mVns"], edges)
        for category in ["pure_EE", "pure_had", "mixed"]:
            result[name][category] = {}
            for mode, bins in [
                ("actual_NN_source_Q_preADC", actual_source_bins),
                ("historical_SOURCE_CLEAN_Q_postADC", reference_source_bins),
                ("observed_output_Q", output_bins)]:
                table = []
                for i in range(8):
                    population = (comp == category) & (bins == i)
                    use = population & d["valid"]
                    psd = d["features"]["PSD"][use]; definition = valley["bins"][i]
                    table.append({"bin_id_1based": i+1, "source_count": int(population.sum()),
                        "N": int(use.sum()), "invalid_count": int((population & ~d["valid"]).sum()),
                        "output_PSD": ev.summary(psd),
                        "valley_fraction": float(np.mean((psd >= definition["low"]) &
                            (psd <= definition["high"]))) if len(psd) and definition["valid"] else None})
                result[name][category][mode] = table
    return {"all_bins_and_compositions": result,
        "primary_pure_had_actual_input_source_Q_bins3_4": {
            n: {"bin"+str(i+1): result[n]["pure_had"]["actual_NN_source_Q_preADC"][i]
                for i in [2, 3]} for n in result},
        "historical_postADC_reference_Q_bins3_4": {
            n: {"bin"+str(i+1): result[n]["pure_had"]["historical_SOURCE_CLEAN_Q_postADC"][i]
                for i in [2, 3]} for n in result},
        "valley_not_particle_error_truth": True, "main_population_reweighted": False}


def association(domains, comp, pdg, edges):
    source, out = domains["SOURCE_CLEAN"], domains["TRANSLATED"]
    sq, sp = source["features"]["Qtotal_mVns"], source["features"]["PSD"]
    oq, op = out["features"]["Qtotal_mVns"], out["features"]["PSD"]
    good = source["valid"] & out["valid"]
    # Preserve the historical source-axis formula. Current output uses the
    # inclusive final edge and a distinct invalid category.
    sourcebin = np.searchsorted(edges, np.log(np.maximum(sq, 1e-30)), side="right")-1
    outbin = raw_bins(oq, edges)
    source_invalid = ~np.isfinite(sq) | (sq <= 0)
    sourcebin = np.where(source_invalid, -2, sourcebin)
    outbin = np.where(out["valid"], outbin, -2)
    def corr(a, b, mask):
        use = mask & np.isfinite(a) & np.isfinite(b)
        return {"n": int(use.sum()), "Spearman": float(spearmanr(a[use], b[use]).statistic)
            if use.sum() > 2 and len(np.unique(a[use])) > 1 and len(np.unique(b[use])) > 1 else None}
    rows = {}
    for category, mask in [("all", np.ones(len(sq), dtype=np.bool_))] + [
        (str(n), comp == n) for n in np.unique(comp)] + [
        ("PDG_inferred_"+str(n), pdg == n) for n in np.unique(pdg)]:
        bins = []
        for i in range(8):
            use = mask & good & (sourcebin == i)
            bins.append({"source_bin_1based": i+1, "N": int(use.sum()),
                         "source_PSD": ev.summary(sp[use]), "output_PSD": ev.summary(op[use]),
                         "output_minus_source_PSD": ev.summary((op-sp)[use]),
                         "PSD_ranking": corr(sp, op, use)})
        rows[category] = {"source_n": int(mask.sum()), "joint_QC_n": int((mask & good).sum()),
            "source_Q_to_output_Q": corr(sq, oq, mask & good),
            "source_PSD_to_output_PSD": corr(sp, op, mask & good),
            "logQ_change": ev.summary(np.log(np.maximum(oq[mask & good], 1e-30)/sq[mask & good])),
            "bins": bins}
    table = migration(sourcebin, outbin, original_uniform_weights(len(sq)))
    return {"composition_and_PDG": rows,
        "migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid": table["count_matrix"],
        "migration_labels": table["labels"], "migration_original_weight_matrix": table["weight_matrix"],
        "source_axis_historical_formula_preserved": True, "output_final_edge_included": True,
        "no_microscopic_particle_truth_claim": True}


def boundary_metrics(domains, out=None):
    data = {}
    for name, d in domains.items():
        if name in ["SOURCE_CLEAN", "SOURCE_INPUT"]: continue
        a = d["V"]
        data[name] = {key: a[:, j]-a[:, i] for key, (i, j) in {
            "first_99_100": (99, 100), "first_149_150": (149, 150),
            "first_399_400": (399, 400)}.items()}
        for i in [99, 100, 149, 150, 399, 400]:
            data[name]["second_center_"+str(i)] = a[:, i-1]-2*a[:, i]+a[:, i+1]
    results = {}
    for name, values in data.items():
        results[name] = {}
        for key, a in values.items():
            target = data["REAL"][key]
            good = np.isfinite(a) & domains[name]["valid"]
            rg = np.isfinite(target) & domains["REAL"]["valid"]
            results[name][key] = {"candidate": ev.summary(a[good]), "REAL": ev.summary(target[rg]),
                "W1_V": float(wasserstein_distance(a[good], target[rg])) if good.any() and rg.any() else None}
    if out is not None:
        np.savez_compressed(out/"BOUNDARY_DIFFERENCES_PRIVATE.npz",
            **{name+"__"+key: a for name, values in data.items() for key, a in values.items()})
    return {"fixed_locations_only": True, "native_voltage_V": True,
            "first_definition": "v[j]-v[i]", "second_definition": "v[i-1]-2*v[i]+v[i+1]",
            "ADC_LSB_V": 1/16383, "results": results}


# Exact static extraction from the original legacy_model.py ADC; retained only
# for the historical SOURCE_CLEAN mother denominators207/182.
def legacy_source_adc(x):
    import numpy as np
    v=np.asarray(x,dtype=np.float32);clipped=np.count_nonzero((v<-.5)|(v>.5))
    out=(-.5+np.rint((np.clip(v,-.5,.5)+.5)*16383)/16383).astype(np.float32)
    return out,{'clip_count':int(clipped),'pre_ADC_finite':bool(np.isfinite(v).all()),'bits':14,'Vrange':[-.5,.5]}
