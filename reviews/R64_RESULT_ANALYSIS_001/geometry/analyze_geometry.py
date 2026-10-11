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
if MODE not in ("inspect", "analyze", "report"):
    raise ValueError("Allowed modes: inspect, analyze, report")
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


def report():
    s = json.loads((ROOT / "SUMMARY.json").read_text("utf8"))
    b = s["charge_bound"]
    terms = ("full_window", "coordinate_RMS", "temporal_joint")
    def f(x):
        return "null" if x is None else format(x, ".6g")
    def pct(x):
        return "null" if x is None else format(100*x, ".4f") + "%"
    def rank(x):
        return "null" if x is None else format(x, ".10g")
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] +
                         ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])
    above = s["source_mass_region_contributions"][-1]
    lines = ["# R64 已完成几何结果的独立分析", "",
             "本分析使用已完成并正常回收的四个几何产物和原协议；未读取仍生成的宏梯度结果、原始波形、权重或新评价。全部6593 SOURCE、21047 REAL和十个原区域保留。来源及内容哈希见 [INPUT_BINDING.json](INPUT_BINDING.json)，完整派生量见 [SUMMARY.json](SUMMARY.json)。", "",
             "**主要结果。** 全窗与坐标 RMS 在全体及全部十区均超过相应32对 REAL 的最大值和同处理 REAL；时序项仅在 all、below、Q6、above 呈现这一模式。电荷名义下界占 V-only 坐标贡献的 " + pct(b["output_bound_fraction_V_only"]) + "，占全部四视图 full_window+coordinate_RMS 的 " + pct(b["output_bound_fraction_all_views"]) + "。这量化当前下界的紧致程度，不是残差的因果归属比例。", "",
             "## 1. 下界的两种分母", "",
             "沿用原权重：B_V=ΣmS·(LBmean/8+LBRMS/4)；V_obs=ΣmS·(V坐标W1均值/8+V坐标W1 RMS/4)。allview 分母还含 V 的切片项及 U、DU、D2 的相应项，不能与 V-only 混称。", "",
             table(["量", "数值"], [["输出特定名义 B_V", f(b["output_V_bound"])], ["SOURCE 电荷形式 B_V", f(b["source_form_V_bound"])],
                   ["V-only 坐标贡献 V_obs", f(b["observed_V_coordinate_contributions"])], ["full_window+coordinate_RMS", f(b["observed_all_views_full_window_plus_coordinate_RMS"])],
                   ["输出 / SOURCE形式：V-only占比", pct(b["output_bound_fraction_V_only"])+" / "+pct(b["source_bound_fraction_V_only"])],
                   ["输出 / SOURCE形式：allview占比", pct(b["output_bound_fraction_all_views"])+" / "+pct(b["source_bound_fraction_all_views"])] ]), "",
             "全部原名义不等式检查通过。下面逐区保留原 mS 与加权贡献；占比用该区自身分母计算，全体聚合采用先加权求和再取比。", "",
             table(["原区", "mS", "加权B_V", "加权V_obs", "本区V-only占比", "本区allview占比"],
                   [[x["label"], f(x["original_SOURCE_region_mass"]), f(x["weighted_output_V_bound"]), f(x["weighted_observed_V_coordinates"]),
                     pct(x["output_LB_fraction_of_V480_two_coordinate_contributions"]), pct(x["output_LB_fraction_of_full_window_plus_coordinate_RMS"])] for x in b["regions"]]), "",
             "逐区 signed 电荷均值差、原/投影后/共同ADC电荷摘要、实际 epsq 和 Erep 已完整保留在 SUMMARY。SOURCE形式与输出形式相近不等于 ADC 误差为零。名义浮点下界不能覆盖形状、切片或时序的全部机制，不能证明统一不可约残差或原 A–F 不可行；观测值减下界也不是已证明可实现的改进量。", "",
             "## 2. 三项几何量与32对 REAL", "",
             "先列原 SOURCE 质量加权的十区目标。REAL 栏是32对同量、有放回抽样的完整经验范围，不是置信区间；同处理 REAL 是完整处理后人口对原 canonical REAL 的单独对照。", "",
             table(["几何项", "SOURCE", "同处理REAL", "REAL对[min,max]", "SOURCE / REAL对max"],
                   [[t, f(c["SOURCE"]), f(c["same_treated_REAL"]), "["+f(c["REAL_pair"]["min"])+", "+f(c["REAL_pair"]["max"])+"]", f(c["SOURCE_over_REAL_pair_max"])]
                    for t,c in s["weighted_three_term_comparison"].items()]), "",
             "下表每个单元为 **SOURCE / 同处理REAL / 32对REAL最大值**；`↑`仅表示 SOURCE 严格超过这32个数值及对照，是描述性次序。各范围全部32值、min/mean/max和差值保存在 JSON。all 是汇集各父原区域标准化后的整体法则，不等于上一表的区域加权目标。", "",
             table(["范围（SOURCE/REAL N）", "full_window", "coordinate_RMS", "temporal_joint"],
                   [[r["label"]+" ("+str(r["SOURCE_N"])+"/"+str(r["REAL_N"])+")"] +
                    [" / ".join(f(r["terms"][t][k]) for k in ("SOURCE", "same_treated_REAL"))+" / "+f(r["terms"][t]["REAL_pair"]["max"])+
                     (" ↑" if r["terms"][t]["SOURCE_above_all_32_REAL_pairs"] and r["terms"][t]["SOURCE_above_control"] else "") for t in terms]
                    for r in s["scope_comparisons"]]), "",
             "同处理 REAL 在全部11范围的三项值均低于各自32对 REAL 的最小值。该模式与两种比较的人数、完整参照及配对依赖结构共同解释；不能叫作“优于REAL”。SOURCE 对完整 REAL 与两份有限 REAL 抽样不构成完全相同的两样本设计。Q1–Q5、Q7、Q8 的时序项没有超过全部32对，保留这一未分离结果，不宣称等效。PSD/noise 没有此 geometry 模块的同一32对参照。", "",
             "## 3. 原区域贡献与已有坐标定位", "",
             "above 保留 " + pct(above["source_mass"]) + " 的原 SOURCE 质量，贡献 full_window 的 " + pct(above["fractions_of_weighted_term"]["full_window"]) +
             "、coordinate_RMS 的 " + pct(above["fractions_of_weighted_term"]["coordinate_RMS"]) + "。它不是删除或重权的依据，标量贡献也不能证明梯度冲突。", "",
             table(["原区", "full_window贡献", "coordinate_RMS贡献", "temporal_joint贡献"],
                   [[r["label"]]+[f(r["contributions"][t])+" ("+pct(r["fractions_of_weighted_term"][t])+")" for t in terms] for r in s["source_mass_region_contributions"]]), "",
             "四视图贡献分解如下，已按原全窗/坐标 RMS 外权重计入，不额外归一化。", "",
             table(["视图", "坐标均值贡献", "坐标RMS贡献", "切片贡献", "合计"],
                   [[v]+[f(s["view_contributions"][v][k]) for k in ("coordinate_marginal_objective_contribution", "coordinate_RMS_objective_contribution", "sliced_objective_contribution", "total_own_view_contribution")]
                    for v in ("V480", "U480", "DU479", "D2U478")]), "",
             "为定位数值贡献，将区域 RMS 标量按 W_j² 份额分配到坐标：mS·W_j/(8d)+mS·W_j²/(4d·RMS)，再按区域求和。它严格加回坐标标量贡献，是描述性分配，不是梯度或因果贡献。下表列每个已有 U/DU/D2 视图数值最大的5个坐标；索引明确指保存数组的零起始索引。", "",
             table(["视图[索引]", "坐标标量分配", "ΣmS·W1", "同处理REAL ΣmS·W1", "加权REAL对max", "主要原区"],
                   [[v+"["+str(x["coordinate_index_zero_based"])+"]", f(x["weighted_coordinate_allocation"]), f(x["source_mass_weighted_coordinate_W1"]),
                     f(x["control_mass_weighted_coordinate_W1"]), f(x["weighted_REAL_pair_max"]), x["main_original_region_allocations"][0]["label"]]
                    for v in ("U480", "DU479", "D2U478") for x in s["view_contributions"][v]["top_10_by_coordinate_objective_allocation"][:5]]), "",
             "主要数值位置集中在114–118附近；D2U的148坐标也进入前5。V480首位为118。完整每坐标加权向量、所有原区的各视图前10名与超过已保存参照的坐标数均保留在 SUMMARY。排序不提供多重比较显著性，也不能仅凭周期/峰位外观识别某个网络层。", "",
             "## 4. 31维基线时序", "",
             "temporal_joint="+f(s["view_contributions"]["temporal31"]["coordinate_marginal_objective_contribution"])+"（31坐标边缘半权）+"+
             f(s["view_contributions"]["temporal31"]["sliced_objective_contribution"])+"（64方向切片半权）。按原质量汇总的31个边缘中，RMS_V、ACF_lag1/2/3/7、第一频带六项超过各自全部32对加权参照及同处理REAL；其余25项保留未分离结果。切片部分不能唯一分摊为某个坐标的贡献。", "",
             table(["原特征组", "边缘半权贡献", "坐标数", "超过全部参照及控制的数量"],
                   [[g, f(x["marginal_contribution"]), x["coordinate_count"], x["above_all_saved_pairs_and_control_count"]] for g,x in s["temporal_group_marginal_contributions"].items()]), "",
             "按边缘标量贡献排序的前10项如下；包括未超过全部参照的项。全部31项的逐项数值均在 JSON。", "",
             table(["时序坐标", "ΣmS·W1", "对照", "加权REAL对max", "对temporal_joint边缘贡献", "占temporal_joint", "超过全部参照/控制"],
                   [[t["name"], f(t["SOURCE_mass_weighted_W1"]), f(t["control_mass_weighted_W1"]), f(t["REAL_pair_weighted_max"]),
                     f(t["marginal_contribution_to_temporal_joint"]), pct(t["fraction_of_total_temporal_joint"]), "是" if t["above_all_32_weighted_REAL_pairs_and_control"] else "否"]
                    for t in sorted(s["temporal_all_31_coordinates"], key=lambda x:-x["marginal_contribution_to_temporal_joint"])[:10]]), "",
             "## 5. PSD、边界、源关联与投影描述", "",
             "SOURCE总体 canonical PSD W1="+f(s["supporting"]["rows"][0]["SOURCE_canonical_vs_REAL_canonical_PSD_W1"])+"，共同Q1–Q7等区均值="+
             f(s["supporting"]["SOURCE_canonical_PSD_Q1_Q7_equal_mean"])+"；同处理REAL分别为"+f(s["supporting"]["rows"][0]["REAL_control_vs_REAL_canonical_PSD_W1"])+" / "+
             f(s["supporting"]["control_PSD_Q1_Q7_equal_mean"])+"。总体数值不能替代条件分布；这些是冻结TRAIN直接端点描述，不重新判定历史DEV的原门。", "",
             table(["原范围", "SOURCE PSD W1", "同处理REAL PSD W1", "原Q–输出Q秩", "原PSD–输出PSD秩"],
                   [[r["label"], f(r["SOURCE_canonical_vs_REAL_canonical_PSD_W1"]), f(r["REAL_control_vs_REAL_canonical_PSD_W1"]),
                     rank(r["source_rank"]["original_Q_vs_output_Q"]), rank(r["source_rank"]["original_PSD_vs_output_PSD"])] for r in s["supporting"]["rows"]]), "",
             "Q3/Q4在共同Q1–Q7中的PSD W1分别为0.0423696/0.0404048，原PSD–输出PSD秩为-0.0348924/-0.136927；这些区域描述保留，不能由总体PSD W1较小而省略，也不据此改写原验收。", "",
             "下表是 all 范围九个固定边界的原摘要字段值；原文件没有边界数值的单位元数据，此处不自行换算或标成LSB。各十区的全部九项也在 JSON 中保留。", "",
             table(["固定边界", "SOURCE_raw字段", "SOURCE_canonical字段", "REAL_control_canonical字段"],
                   [[name]+[f(s["supporting"]["rows"][0]["fixed_boundaries"][state][name]) for state in ("SOURCE_raw", "SOURCE_canonical", "REAL_control_canonical")]
                    for name in s["supporting"]["boundary_order"]]), "",
             "SOURCE raw endpoint 超范围事件为830/6593，活跃电压轨采样点23374；同处理REAL为0/21047、0点。SOURCE投影修正RMS均值="+
             f(s["supporting"]["projection"]["SOURCE"]["projection_RMS"]["mean"])+"，preADC float32表示相对电荷误差最大="+
             f(s["supporting"]["projection"]["SOURCE"]["preADC_relative_Q_error_f32"]["max"])+"，canonical绝对电荷误差均值="+
             f(s["supporting"]["projection"]["SOURCE"]["canonical_absolute_Q_error"]["mean"])+"。保留f64/f32两条电荷误差及实际ADC偏移，不能以较小电荷误差宣称投影未改变波形。", "",
             "## 6. 数值边界与可以支持的后续问题", "",
             "从私有几何数组独立重构全部11范围的原视图归约，最大绝对差为"+f(s["saved_array_summary_reconstruction_max_abs_error"])+"。NumPy描述几何与CUDA full F的full_window/coordinate_RMS/temporal_joint之差分别为"+
             ", ".join(f(s["NumPy_geometry_vs_CUDA_loss"][t]["CUDA_minus_NumPy"]) for t in terms)+"；两路径明确分别标识，不声称跨设备逐位等同。CUDA五D值如下，仅是损失值，尚未在本分析读取梯度。", "",
             table(["D分量", "完整有限参考CUDA值"], [[k,f(v)] for k,v in s["full_reference_CUDA_loss"]["components"].items()]+[["total_D",f(s["full_reference_CUDA_loss"]["total_D"])]]), "",
             "现有证据支持后续继续核对：已有U/DU/D2残差为何集中在这些坐标与above原区；残余RMS/短滞后/第一频带如何对应同时保留的区域与同处理REAL；V均值下界与可实现的局部参数方向如何关联。宏梯度完成后再联合经验均值偏离、散度及端点/参数两空间证据。这里不预选新方法、不增大批量、不增加特征、不删除或重权区域、不做新拟合；不将标量主导改写成九项梯度冲突，也不修改R60阴性结论或原A–F。", "",
             "[原协议](../../../tasks/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001.md) · [完整分析JSON](SUMMARY.json) · [来源绑定](INPUT_BINDING.json) · [绑定及资源回执](BINDING_RECEIPT.json)", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


try:
    before, loaded = inputs()
    if MODE == "report":
        report()
    else:
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
    if MODE == "report":
        manifests = []
        for name in ("REPORT.md", "SUMMARY.json", "INPUT_BINDING.json", "analyze.EXECUTED_SOURCE.py", "analyze_geometry.py",
                     "inspect.RESOURCE_RECEIPT.json", "inspect.NATIVE_EXIT_RECEIPT.json", "analyze.RESOURCE_RECEIPT.json",
                     "analyze.NATIVE_EXIT_RECEIPT.json", "report.RESOURCE_RECEIPT.json"):
            path = ROOT / name
            manifests.append({"file": name, "bytes": path.stat().st_size, "sha256": sha(path)})
        resources = {mode: json.loads((ROOT / (mode + ".RESOURCE_RECEIPT.json")).read_text("utf8"))["resource"] for mode in ("inspect", "analyze", "report_initial", "report")}
        numeric_exit = json.loads((ROOT / "analyze.NATIVE_EXIT_RECEIPT.json").read_text("utf8"))["native_exit_code"]
        if numeric_exit != 0:
            raise RuntimeError("Numeric analysis lacks a successful native receipt")
        sealed = {"status": "GEOMETRY_ANALYSIS_REPORT_AND_JSON_BOUND", "UTC": datetime.now(timezone.utc).isoformat(),
                  "sources": before, "all_inputs_unchanged": True, "outputs": manifests, "resources_by_bounded_process": resources,
                  "executed_numeric_analysis_source": "analyze.EXECUTED_SOURCE.py", "native_numeric_process_exit": numeric_exit,
                  "report_native_exit_receipt": "report.NATIVE_EXIT_RECEIPT.json",
                  "scope": receipt["scope"], "frozen_acceptance_or_R60_results_modified": False,
                  "next_method_selected": None, "macro_gradient_results_read": False}
        write_json("BINDING_RECEIPT.json", sealed)
        again = json.loads((ROOT / "BINDING_RECEIPT.json").read_text("utf8"))
        for output in again["outputs"]:
            if sha(ROOT / output["file"]) != output["sha256"]:
                raise RuntimeError("Output manifest reopened hash mismatch")
finally:
    stop.set()
