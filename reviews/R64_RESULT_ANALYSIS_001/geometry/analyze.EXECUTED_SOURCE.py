"""Read only completed R64 geometry products under an owned CPU/RSS guard."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
WORK = PROJECT / "work" / "R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001"
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(TEMP)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
if not sys.dont_write_bytecode:
    raise RuntimeError("Launch with python -B")
START = time.monotonic()
LIMIT = 1024**3
MODE = sys.argv[1] if len(sys.argv) == 2 else "inspect"
if MODE not in ("inspect", "analyze"):
    raise ValueError("Allowed modes: inspect, analyze")
if os.name != "nt":
    raise RuntimeError("This result-analysis resource owner requires Windows")
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
ptr = ctypes.c_size_t
k32.GetCurrentProcess.restype = wintypes.HANDLE
process = k32.GetCurrentProcess()
k32.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ptr), ctypes.POINTER(ptr))
pmask, smask = ptr(), ptr()
if not k32.GetProcessAffinityMask(process, ctypes.byref(pmask), ctypes.byref(smask)):
    raise ctypes.WinError(ctypes.get_last_error())
one_cpu = pmask.value & -pmask.value
k32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ptr)
if not k32.SetProcessAffinityMask(process, one_cpu):
    raise ctypes.WinError(ctypes.get_last_error())


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ptr), ("MaximumWorkingSetSize", ptr),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ptr), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ptr),
                ("JobMemoryLimit", ptr), ("PeakProcessMemoryUsed", ptr), ("PeakJobMemoryUsed", ptr)]


class MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ptr),
                ("WorkingSetSize", ptr), ("QuotaPeakPagedPoolUsage", ptr), ("QuotaPagedPoolUsage", ptr),
                ("QuotaPeakNonPagedPoolUsage", ptr), ("QuotaNonPagedPoolUsage", ptr), ("PagefileUsage", ptr),
                ("PeakPagefileUsage", ptr), ("PrivateUsage", ptr)]


psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(MEMORY_COUNTERS), wintypes.DWORD)


def memory():
    m = MEMORY_COUNTERS()
    m.cb = ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(m), m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"RSS_bytes": int(m.WorkingSetSize), "peak_RSS_bytes": int(m.PeakWorkingSetSize),
            "private_bytes": int(m.PrivateUsage), "peak_commit_bytes": int(m.PeakPagefileUsage)}


k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
job = k32.CreateJobObjectW(None, None)
limit = EXTENDED_LIMIT()
limit.BasicLimitInformation.LimitFlags = 0x100
limit.ProcessMemoryLimit = LIMIT
if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
    raise ctypes.WinError(ctypes.get_last_error())
if not k32.AssignProcessToJobObject(job, process):
    raise ctypes.WinError(ctypes.get_last_error())
stop = threading.Event()


def watch():
    while not stop.wait(.05):
        m = memory()
        if m["peak_RSS_bytes"] > LIMIT:
            (ROOT / (MODE + ".RESOURCE_FAILURE.json")).write_text(json.dumps(m), encoding="utf8")
            os._exit(91)


threading.Thread(target=watch, daemon=True).start()

import numpy as np


def write_json(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


SPECS = {
    "geometry_summary": WORK / "ENDPOINT_GEOMETRY_SUMMARY.json",
    "geometry_private": WORK / "ENDPOINT_GEOMETRY_PRIVATE.npz",
    "supporting_summary": WORK / "SUPPORTING_SUMMARY.json",
    "full_reference_loss": WORK / "FULL_REFERENCE_LOSS.json",
    "completed_transfer_receipt": WORK / "GEOMETRY_EARLY_FETCH_RECEIPT.json",
    "protocol": PROJECT / "tasks" / "R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001.md",
}


def inputs():
    records = [{"id": key, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for key, path in SPECS.items()]
    old = ROOT / "INPUT_BINDING.json"
    if old.exists():
        before = json.loads(old.read_text("utf8"))["files"]
        if before != records:
            raise RuntimeError("Completed input file changed between bounded reads")
    else:
        write_json("INPUT_BINDING.json", {"UTC": datetime.now(timezone.utc).isoformat(), "files": records,
                                         "scope": "four completed geometry products plus original protocol and normal-transfer receipt"})
    loaded = {key: json.loads(path.read_text("utf-8-sig")) for key, path in SPECS.items() if path.suffix == ".json"}
    loaded["protocol"] = SPECS["protocol"].read_text("utf-8-sig")
    return records, loaded


def tree(value, depth=0):
    if isinstance(value, dict):
        return {k: tree(v, depth+1) for k, v in value.items()}
    if isinstance(value, list):
        return {"type": "list", "N": len(value), "first": tree(value[0], depth+1) if value else None}
    return value if depth <= 3 else type(value).__name__


def inspect(loaded, archive):
    geo = loaded["geometry_summary"]
    result = {
        "geometry_keys": list(geo), "geometry_population": geo["population"],
        "geometry_weighted_terms": geo["original_SOURCE_mass_weighted_regional_terms"],
        "geometry_weighted_REAL_floor": geo["matched_N_REAL_pair_weighted_regional_terms"],
        "same_treated_REAL_weighted_terms": geo.get("same_treated_REAL_original_SOURCE_mass_weighted_regional_terms"),
        "charge_bound_aggregation": geo["charge_bound_aggregation"],
        "temporal_names": geo["temporal_names"],
        "rows": [],
        "support_structure": tree(loaded["supporting_summary"]),
        "full_reference_loss": loaded["full_reference_loss"],
        "npz_key_count": len(archive.files),
        "npz_first_keys": archive.files[:35],
        "npz_last_keys": archive.files[-15:],
    }
    for row in geo["rows"]:
        result["rows"].append({
            "scope": row["scope"], "SOURCE_N": row["SOURCE_N"], "REAL_N": row["REAL_N"],
            "source_terms": row["SOURCE_vs_complete_REAL"]["terms"],
            "control_terms": row.get("same_treated_REAL_vs_complete_REAL", {}).get("terms"),
            "floor_terms": {key: {k: v[k] for k in ("mean", "min", "max")} for key, v in row["matched_N_REAL_pair_floor"]["terms"].items()},
            "source_views": row["SOURCE_vs_complete_REAL"]["views"],
            "charge_bound": row.get("charge_mean_bound"),
        })
    write_json("INSPECTION.json", result)
    printed = {k: v for k, v in result.items() if k not in ("rows", "geometry_weighted_REAL_floor")}
    printed["rows_compact"] = [{k: v for k, v in row.items() if k not in ("source_views", "charge_bound")} for row in result["rows"]]
    print(json.dumps(printed, ensure_ascii=False))


def analyze(loaded, archive):
    geo, support, full_loss = (loaded[k] for k in ("geometry_summary", "supporting_summary", "full_reference_loss"))
    terms = ("full_window", "coordinate_RMS", "temporal_joint")
    views = ("V480", "U480", "DU479", "D2U478", "temporal31")
    region_axis = geo["population"]["region_axis"]
    scopes = ["all"] + ["region" + str(r) for r in region_axis]
    if [row["scope"] for row in geo["rows"]] != scopes or [row["scope"] for row in support["rows"]] != scopes:
        raise ValueError("All eleven original scopes must remain in frozen order")
    mass = np.asarray(geo["population"]["source_mass"], dtype=np.float64)
    if not np.array_equal(np.asarray(geo["population"]["source_counts"]), archive["source_original_region_counts"]):
        raise ValueError("SOURCE count binding mismatch")
    if not np.array_equal(mass, archive["source_original_region_mass"]):
        raise ValueError("SOURCE mass binding mismatch")

    def label(scope):
        if scope == "all":
            return "all"
        raw = int(scope[6:])
        return "below" if raw == -1 else "above" if raw == 8 else "Q" + str(raw + 1)

    def array(key):
        value = np.asarray(archive[key], dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid saved nonnegative W1 array: " + key)
        return value

    def descriptive_floor(values):
        if len(values) != 32:
            raise ValueError("Require all 32 completed REAL pairs")
        return {"values": np.asarray(values).tolist(), "min": float(np.min(values)), "mean": float(np.mean(values)),
                "max": float(np.max(values)), "replicates": 32, "confidence_interval": False}

    def compare(s, c, floor):
        f = descriptive_floor(floor)
        return {"SOURCE": float(s), "same_treated_REAL": float(c), "REAL_pair": f,
                "SOURCE_minus_REAL_pair_max": float(s-f["max"]), "SOURCE_minus_control": float(s-c),
                "SOURCE_over_REAL_pair_max": float(s/f["max"]) if f["max"] else None,
                "SOURCE_above_all_32_REAL_pairs": bool(s > f["max"]), "SOURCE_above_control": bool(s > c),
                "control_below_all_32_REAL_pairs": bool(c < f["min"]),
                "descriptive_comparison_only": True}

    region_analysis, bound_rows, region_contributions, reconstruction_errors = [], [], [], []
    weighted_by_view = {}
    for scope_index, row in enumerate(geo["rows"]):
        rec = {"scope": row["scope"], "label": label(row["scope"]), "SOURCE_N": row["SOURCE_N"],
               "REAL_N": row["REAL_N"], "terms": {}, "views": {}}
        for term in terms:
            floor = row["matched_N_REAL_pair_floor"]["terms"][term]["values"]
            rec["terms"][term] = compare(row["SOURCE_vs_complete_REAL"]["terms"][term],
                                        row["same_treated_REAL_vs_complete_REAL"]["terms"][term], floor)
        reconstructed_views = []
        for v, view in enumerate(views):
            prefix = row["scope"] + "__" + view
            w = array("SOURCE_REAL__" + prefix + "__coordinate_W1")
            sw = array("SOURCE_REAL__" + prefix + "__projection_W1")
            control = array("REAL_control_REAL__" + prefix + "__coordinate_W1")
            floor = array("REAL_pair_floor__" + prefix + "__coordinate_W1")
            floor_sw = array("REAL_pair_floor__" + prefix + "__projection_W1")
            raw = array("SOURCE_REAL__" + prefix + "__unstandardized_coordinate_W1")
            if floor.shape != (32, len(w)) or floor_sw.shape != (32, 64):
                raise ValueError("Coordinate or fixed projection shape mismatch")
            d, rms = len(w), float(np.sqrt(np.mean(w*w)))
            mean, sliced = float(w.mean()), float(sw.mean())
            saved = row["SOURCE_vs_complete_REAL"]["views"][v]
            if saved["view"] != view:
                raise ValueError("Saved view order changed")
            reconstruction_errors.extend(abs(actual-saved[key]) for key, actual in
                                         (("coordinate_mean", mean), ("coordinate_RMS", rms), ("sliced_mean", sliced)))
            reconstructed_views.append((mean, rms, sliced))
            above_floor = w > floor.max(axis=0)
            above_both = above_floor & (w > control)
            order = np.argsort(-w, kind="stable")
            top = []
            for j in order[:10]:
                top.append({"coordinate_index_zero_based": int(j), "feature_name": geo["temporal_names"][j] if view == "temporal31" else None,
                            "SOURCE_standardized_W1": float(w[j]), "same_treated_REAL_standardized_W1": float(control[j]),
                            "REAL_pair_max": float(floor[:,j].max()), "REAL_pair_mean": float(floor[:,j].mean()),
                            "SOURCE_unstandardized_W1": float(raw[j]), "above_all_32_REAL_pairs_and_control": bool(above_both[j])})
            rec["views"][view] = {"coordinate_count": d, "coordinate_mean": mean, "coordinate_RMS": rms, "sliced_mean": sliced,
                                  "coordinates_above_all_32_REAL_pairs": int(above_floor.sum()),
                                  "coordinates_above_all_32_REAL_pairs_and_control": int(above_both.sum()),
                                  "top_10_by_standardized_W1": top}
            if scope_index:
                m = mass[scope_index-1]
                if view not in weighted_by_view:
                    weighted_by_view[view] = {"source_w": np.zeros(d), "control_w": np.zeros(d), "floor_w": np.zeros((32,d)),
                                              "coordinate_allocation": np.zeros(d), "marginal_contribution": 0., "RMS_contribution": 0.,
                                              "sliced_contribution": 0., "regional_coordinate_allocations": [], "regional_total_contributions": []}
                dst = weighted_by_view[view]
                dst["source_w"] += m*w
                dst["control_w"] += m*control
                dst["floor_w"] += m*floor
                if view == "temporal31":
                    allocation = m*w/(2*d)
                    marginal_part, rms_part, sliced_part = m*mean/2, 0., m*sliced/2
                else:
                    marginal_part, rms_part, sliced_part = m*mean/8, m*rms/4, m*sliced/8
                    allocation = m*w/(8*d)
                    if rms:
                        allocation += m*w*w/(4*d*rms)
                dst["coordinate_allocation"] += allocation
                dst["marginal_contribution"] += marginal_part
                dst["RMS_contribution"] += rms_part
                dst["sliced_contribution"] += sliced_part
                dst["regional_coordinate_allocations"].append(allocation)
                dst["regional_total_contributions"].append(float(marginal_part+rms_part+sliced_part))
        reconstructed_terms = [np.mean([.5*(x[0]+x[2]) for x in reconstructed_views[:4]]),
                               np.mean([x[1] for x in reconstructed_views[:4]]),
                               .5*(reconstructed_views[4][0]+reconstructed_views[4][2])]
        reconstruction_errors.extend(abs(reconstructed_terms[j]-row["SOURCE_vs_complete_REAL"]["terms"][t]) for j,t in enumerate(terms))
        if scope_index:
            b = dict(row["charge_mean_bound"])
            b["label"] = label(row["scope"])
            b["weighted_output_V_bound"] = float(mass[scope_index-1]*b["output_LB_V480_two_term_contribution"])
            b["weighted_observed_V_coordinates"] = float(mass[scope_index-1]*b["observed_V480_two_coordinate_contributions"])
            b["weighted_source_form_V_bound"] = float(mass[scope_index-1]*b["source_LB_V480_two_term_contribution"])
            b["charge_statistics"] = row["charge_statistics"]
            bound_rows.append(b)
            contribution = {t: float(mass[scope_index-1]*row["SOURCE_vs_complete_REAL"]["terms"][t]) for t in terms}
            region_contributions.append({"scope": row["scope"], "label": label(row["scope"]), "source_mass": float(mass[scope_index-1]),
                                         "contributions": contribution, "fractions_of_weighted_term":
                                         {t: contribution[t]/geo["original_SOURCE_mass_weighted_regional_terms"][t] for t in terms}})
        region_analysis.append(rec)

    weighted_comparison = {t: compare(geo["original_SOURCE_mass_weighted_regional_terms"][t],
                                     geo["same_treated_REAL_original_SOURCE_mass_weighted_regional_terms"][t],
                                     geo["matched_N_REAL_pair_weighted_regional_terms"][t]["values"]) for t in terms}
    B = sum(x["weighted_output_V_bound"] for x in bound_rows)
    Bs = sum(x["weighted_source_form_V_bound"] for x in bound_rows)
    V = sum(x["weighted_observed_V_coordinates"] for x in bound_rows)
    allviews = geo["charge_bound_aggregation"]["observed_full_window_plus_RMS"]
    bound = {"output_V_bound": B, "source_form_V_bound": Bs, "observed_V_coordinate_contributions": V,
             "observed_all_views_full_window_plus_coordinate_RMS": allviews,
             "output_bound_fraction_V_only": B/V if V else None, "source_bound_fraction_V_only": Bs/V if V else None,
             "output_bound_fraction_all_views": B/allviews if allviews else None, "source_bound_fraction_all_views": Bs/allviews if allviews else None,
             "original_saved_all_views_fraction": geo["charge_bound_aggregation"]["output_lower_bound_fraction_of_observed_full_window_plus_RMS"],
             "all_nominal_checks_pass": geo["charge_bound_aggregation"]["all_nominal_inequality_checks_pass"],
             "regions": bound_rows, "causal_fraction_or_uniform_irreducibility_claim": False}

    view_reports, temporal_coordinates = {}, []
    for view, x in weighted_by_view.items():
        contrib = x["coordinate_allocation"]
        regional = np.stack(x["regional_coordinate_allocations"])
        top = []
        for j in np.argsort(-contrib, kind="stable")[:10]:
            best = np.argsort(-regional[:,j], kind="stable")[:3]
            top.append({"coordinate_index_zero_based": int(j), "feature_name": geo["temporal_names"][j] if view == "temporal31" else None,
                        "weighted_coordinate_allocation": float(contrib[j]), "source_mass_weighted_coordinate_W1": float(x["source_w"][j]),
                        "control_mass_weighted_coordinate_W1": float(x["control_w"][j]), "weighted_REAL_pair_max": float(x["floor_w"][:,j].max()),
                        "main_original_region_allocations": [{"label": label(scopes[k+1]), "allocation": float(regional[k,j])} for k in best]})
        view_reports[view] = {"coordinate_marginal_objective_contribution": float(x["marginal_contribution"]),
                              "coordinate_RMS_objective_contribution": float(x["RMS_contribution"]),
                              "sliced_objective_contribution": float(x["sliced_contribution"]),
                              "total_own_view_contribution": float(x["marginal_contribution"]+x["RMS_contribution"]+x["sliced_contribution"]),
                              "regional_total_contributions": x["regional_total_contributions"],
                              "source_mass_weighted_coordinate_W1": x["source_w"].tolist(),
                              "same_control_mass_weighted_coordinate_W1": x["control_w"].tolist(),
                              "weighted_REAL_pair_coordinate_min": x["floor_w"].min(axis=0).tolist(),
                              "weighted_REAL_pair_coordinate_mean": x["floor_w"].mean(axis=0).tolist(),
                              "weighted_REAL_pair_coordinate_max": x["floor_w"].max(axis=0).tolist(),
                              "coordinate_objective_allocation": contrib.tolist(),
                              "coordinate_allocation_sum": float(contrib.sum()), "top_10_by_coordinate_objective_allocation": top,
                              "allocation_definition": "For four waveform views: mS*W_j/(8*d) + mS*W_j^2/(4*d*RMS), zero second part if RMS=0. For temporal: mS*W_j/(2*31). Sum regions. RMS squared-W shares are an exact scalar allocation, not a causal/gradient attribution."}
        if view == "temporal31":
            for j,name in enumerate(geo["temporal_names"]):
                temporal_coordinates.append({"index": j, "name": name, "group": "RMS" if j == 0 else "ACF_lags1_20" if j <= 20 else "power_fractions_10_bands",
                                             "SOURCE_mass_weighted_W1": float(x["source_w"][j]), "control_mass_weighted_W1": float(x["control_w"][j]),
                                             "REAL_pair_weighted_min": float(x["floor_w"][:,j].min()), "REAL_pair_weighted_mean": float(x["floor_w"][:,j].mean()),
                                             "REAL_pair_weighted_max": float(x["floor_w"][:,j].max()),
                                             "above_all_32_weighted_REAL_pairs_and_control": bool(x["source_w"][j] > x["floor_w"][:,j].max() and x["source_w"][j] > x["control_w"][j]),
                                             "marginal_contribution_to_temporal_joint": float(contrib[j]),
                                             "fraction_of_total_temporal_joint": float(contrib[j]/geo["original_SOURCE_mass_weighted_regional_terms"]["temporal_joint"])})
    temporal_groups = {g: {"marginal_contribution": sum(t["marginal_contribution_to_temporal_joint"] for t in temporal_coordinates if t["group"] == g),
                           "coordinate_count": sum(t["group"] == g for t in temporal_coordinates),
                           "above_all_saved_pairs_and_control_count": sum(t["group"] == g and t["above_all_32_weighted_REAL_pairs_and_control"] for t in temporal_coordinates)}
                       for g in ("RMS", "ACF_lags1_20", "power_fractions_10_bands")}
    support_rows = []
    for row in support["rows"]:
        r = dict(row)
        r["label"] = label(row["scope"])
        support_rows.append(r)
    byscope = {r["scope"]: r for r in support["rows"]}
    q17 = float(np.mean([byscope["region"+str(k)]["SOURCE_canonical_vs_REAL_canonical_PSD_W1"] for k in range(7)]))
    controlq17 = float(np.mean([byscope["region"+str(k)]["REAL_control_vs_REAL_canonical_PSD_W1"] for k in range(7)]))
    psdreconstructed = .5*(byscope["all"]["SOURCE_canonical_vs_REAL_canonical_PSD_W1"]+q17)/.01
    numerical_agreement = {t: {"geometry_NumPy": geo["original_SOURCE_mass_weighted_regional_terms"][t], "full_reference_CUDA": full_loss["components"][t],
                               "CUDA_minus_NumPy": full_loss["components"][t]-geo["original_SOURCE_mass_weighted_regional_terms"][t]} for t in terms}
    summary = {"status": "COMPLETED_GEOMETRY_RESULT_ANALYSIS_NO_MACRO_GRADIENT_RESULTS_READ", "UTC": datetime.now(timezone.utc).isoformat(),
               "population": geo["population"], "all_11_scopes_retained": True, "regional_and_global_statistics_kept_separate": True,
               "weighted_three_term_comparison": weighted_comparison, "scope_comparisons": region_analysis,
               "source_mass_region_contributions": region_contributions, "charge_bound": bound, "view_contributions": view_reports,
               "temporal_all_31_coordinates": temporal_coordinates, "temporal_group_marginal_contributions": temporal_groups,
               "supporting": {"rows": support_rows, "projection": support["projection"], "boundary_order": support["boundary_order"],
                              "boundary_units_in_input_metadata": "not declared; raw saved values retained without rescaling",
                              "SOURCE_canonical_PSD_Q1_Q7_equal_mean": q17, "control_PSD_Q1_Q7_equal_mean": controlq17,
                              "PSD_loss_reconstructed_from_support": psdreconstructed,
                              "PSD_loss_reconstruction_difference_from_CUDA": psdreconstructed-full_loss["components"]["PSD"]},
               "full_reference_CUDA_loss": full_loss, "NumPy_geometry_vs_CUDA_loss": numerical_agreement,
               "saved_array_summary_reconstruction_max_abs_error": max(reconstruction_errors),
               "scope": {"raw_waveforms_or_weights_read": 0, "new_network_calls": 0, "new_training_or_fits": 0, "macro_gradients_read": 0,
                         "new_DEV_EVAL_CAL_reads": 0, "source_population_or_acceptance_changes": 0, "new_method_selected": None},
               "interpretation_boundaries": ["32 REAL pairs are empirical descriptive reference values, not confidence intervals or acceptance thresholds.",
                   "SOURCE-to-complete-REAL and two finite REAL draws have different sampling designs; same-treated REAL is a separate full-population control.",
                   "The V bound is nominal float64 and output-specific. Its fraction is not a causal contribution or uniform model-independent impossibility proof.",
                   "Coordinate rankings are descriptive orderings of existing statistics; no multiple-comparison significance claim is made.",
                   "The all scope pools regional-standardized parents; it is not the SOURCE-mass weighted regional objective.",
                   "D excludes four original C terms. No gradient outcome is analyzed here. R63 original A-F and R60 negative conflict conclusions remain unchanged."]}
    write_json("SUMMARY.json", summary)
    printed = {"charge_bound": {k:v for k,v in bound.items() if k != "regions"},
               "weighted_three_term": weighted_comparison, "region_contributions": region_contributions,
               "view_contributions": {k:{n:z for n,z in v.items() if n in ("coordinate_marginal_objective_contribution", "coordinate_RMS_objective_contribution", "sliced_objective_contribution", "total_own_view_contribution", "top_10_by_coordinate_objective_allocation")} for k,v in view_reports.items()},
               "temporal_31": temporal_coordinates, "temporal_groups": temporal_groups,
               "support_all": support_rows[0], "support_PSD_and_ranks_all_regions": [{k:v for k,v in r.items() if k != "fixed_boundaries"} for r in support_rows],
               "projection": support["projection"], "SOURCE_PSD_Q1_Q7_mean": q17, "control_PSD_Q1_Q7_mean": controlq17,
               "saved_summary_max_error": max(reconstruction_errors), "NumPy_CUDA": numerical_agreement}
    write_json("ANALYSIS_DIGEST.json", printed)
    print(json.dumps({"status": summary["status"], "charge_bound": printed["charge_bound"], "region_contributions": region_contributions,
                      "weighted_three_term": {k:{n:z for n,z in v.items() if n != "REAL_pair"} | {"REAL_pair_min_mean_max": [v["REAL_pair"][t] for t in ("min","mean","max")]} for k,v in weighted_comparison.items()},
                      "views": {k:{n:z for n,z in v.items() if n != "top_10_by_coordinate_objective_allocation"} for k,v in printed["view_contributions"].items()},
                      "top_coordinates": {k:v["top_10_by_coordinate_objective_allocation"][:5] for k,v in view_reports.items()},
                      "top_temporal": sorted(temporal_coordinates, key=lambda z:-z["marginal_contribution_to_temporal_joint"])[:10],
                      "temporal_groups": temporal_groups, "support_all": support_rows[0], "projection": support["projection"],
                      "SOURCE_PSD_Q1_Q7_mean": q17, "control_PSD_Q1_Q7_mean": controlq17,
                      "saved_summary_max_error": max(reconstruction_errors), "NumPy_CUDA": numerical_agreement}, ensure_ascii=False))


try:
    before, loaded = inputs()
    with np.load(SPECS["geometry_private"], allow_pickle=False) as archive:
        if MODE == "inspect":
            inspect(loaded, archive)
        else:
            analyze(loaded, archive)
    for record in before:
        if sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Completed input changed during read")
    m = memory()
    if m["peak_RSS_bytes"] > LIMIT:
        raise RuntimeError("Result analysis resource limit exceeded")
    receipt = {"status": "COMPLETED_GEOMETRY_" + MODE.upper() + "_FINISHED", "UTC": datetime.now(timezone.utc).isoformat(),
               "PID": os.getpid(), "elapsed_seconds": time.monotonic()-START,
               "input_hashes_unchanged": True, "files": before,
               "resource": {"CPU_affinity_mask": one_cpu, "CPU_count": 1, "process_commit_limit_bytes": LIMIT,
                            "RSS_watchdog_limit_bytes": LIMIT, "RSS_watchdog_poll_seconds": .05,
                            "Python_B": bool(sys.dont_write_bytecode), "TEMP": str(TEMP), **m},
               "scope": {"network_calls": 0, "training": 0, "new_distribution_fits": 0,
                         "waveform_or_weight_files_opened": 0, "new_DEV_EVAL_CAL_reads": 0,
                         "macro_results_read": 0, "root_metadata_or_Git_writes": 0}}
    write_json(MODE + ".RESOURCE_RECEIPT.json", receipt)
finally:
    stop.set()
