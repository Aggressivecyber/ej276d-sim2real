"""Artificial-only checks.  Run with Python -B; writes only beside this script."""
from __future__ import annotations

import ast
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
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(TEMP)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
LIMIT = 1024**3
START = time.monotonic()

if os.name != "nt":
    raise RuntimeError("This preparation receipt runner uses an explicit Windows CPU/RSS guard")
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
k32.GetCurrentProcess.restype = wintypes.HANDLE
process = k32.GetCurrentProcess()
ptr = ctypes.c_size_t
k32.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ptr), ctypes.POINTER(ptr))
process_mask, system_mask = ptr(), ptr()
if not k32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
    raise ctypes.WinError(ctypes.get_last_error())
one_cpu_mask = process_mask.value & -process_mask.value
k32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ptr)
if not k32.SetProcessAffinityMask(process, one_cpu_mask):
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
limits = EXTENDED_LIMIT()
limits.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
limits.ProcessMemoryLimit = LIMIT
if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
    raise ctypes.WinError(ctypes.get_last_error())
if not k32.AssignProcessToJobObject(job, process):
    raise ctypes.WinError(ctypes.get_last_error())
stop_guard = threading.Event()


def watch_memory():
    while not stop_guard.wait(.05):
        m = memory()
        if m["peak_RSS_bytes"] > LIMIT:
            (ROOT / "RESOURCE_LIMIT_FAILURE.json").write_text(json.dumps(m), encoding="utf8")
            os._exit(91)


threading.Thread(target=watch_memory, daemon=True).start()

import numpy as np
import torch
import endpoint_geometry as eg

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

PROJECT = ROOT.parents[2]
R62 = PROJECT / "work" / "R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001"
R63 = PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
CHECKS = []


def check(name, condition, **evidence):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append({"name": name, "pass": True, **evidence})


def extract_functions(path, names, globals_):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {n.name for n in selected} != set(names):
        raise AssertionError("source helper extraction")
    for node in selected:
        node.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec")
    space = dict(globals_)
    exec(code, space)
    return space


def expected_rejection(name, callback):
    try:
        callback()
    except (ValueError, FloatingPointError):
        check(name, True)
    else:
        raise AssertionError(name + ": did not reject")


rng = np.random.Generator(np.random.PCG64(20261011649))
legacy = extract_functions(R62 / "residual_math.py", {"features", "exact_w1_columns"}, {"np": np})
r63temp = extract_functions(R63 / "distribution_loss.py", {"temporal_features"}, {"torch": torch})
r63adc = extract_functions(R63 / "source_flow_core.py", {"canonical_adc_torch", "canonical_REAL_torch", "q_torch"}, {"torch": torch})
x = rng.normal(0, .01, (17, 480))
ref = rng.normal(0, .01, (23, 480))
check("R62_four_pure_features_identical", all(np.array_equal(a, b) for a, b in zip(eg.feature_vectors(x), legacy["features"](x))))
w_new = eg.empirical_w1_columns(x, ref)
w_old = legacy["exact_w1_columns"](x, ref)
check("R62_unequal_population_W1_identical", np.array_equal(w_new, w_old), max_abs_difference=float(np.max(np.abs(w_new - w_old))))
check("unequal_sizes_same_empirical_law_zero", float(eg.empirical_w1_columns([0., 2.], [0., 0., 2., 2.])[0]) == 0.)
check("duplicate_occurrences_keep_mass", np.isclose(eg.empirical_w1_columns([0., 0., 2.], [0., 2.])[0], 1/3, atol=1e-15))
check("unequal_population_known_W1", np.isclose(eg.empirical_w1_columns([0., 2.], [1., 1., 3.])[0], 1.))
t = np.arange(100)
special = np.zeros((4, 480))
special[1, :100] = .01*np.sin(2*np.pi*10*t/100)
special[2, :100] = .005*(-1.)**t
special[3, :100] = 1e-12*np.sin(2*np.pi*3*t/100)
temp_input = np.r_[x, special]
actual31 = eg.temporal_features(temp_input)
with torch.no_grad():
    original31 = r63temp["temporal_features"](torch.from_numpy(temp_input), 1e-20).numpy()
check("R63_temporal31_CPU_formula_parity", np.allclose(actual31, original31, rtol=1e-12, atol=2e-14), max_abs_difference=float(np.max(np.abs(actual31 - original31))))
check("R63_zero_power_floor_is_preserved", actual31[-4, 0] == 1e-10 and np.count_nonzero(actual31[-4, 1:]) == 0)
check("R63_Nyquist_and_interior_multiplicity", np.isclose(actual31[-3, 22], 1.) and np.isclose(actual31[-2, 30], 1.))
check("R63_below_power_floor_not_renormalized", actual31[-1, 0] == 1e-10 and 0 < actual31[-1, 21:].sum() < 1)
adc_input = np.r_[x, np.full((1, 480), -.6), np.full((1, 480), .6)]
with torch.no_grad():
    orig_gen = r63adc["canonical_adc_torch"](torch.from_numpy(adc_input), False).numpy()
    orig_real = r63adc["canonical_REAL_torch"](torch.from_numpy(adc_input)).numpy()
check("generated_two_stage_ADC_CPU_parity", np.array_equal(eg.canonical_from_pre_adc(adc_input), orig_gen))
check("REAL_single_canonical_code_CPU_parity", np.array_equal(eg.canonical_real(adc_input), orig_real))
check("original_code_charge_CPU_parity", np.allclose(eg.q_np(x), r63adc["q_torch"](torch.from_numpy(x)).numpy(), rtol=0, atol=2e-15))

# Saturate the mean bound at one max-|a| coordinate, using exact binary values.
yr = np.zeros((3, 480)); rr = np.zeros((5, 480)); yr[:, 0] = .125
zy, zr = yr.astype(np.float32), rr.astype(np.float32)
w = eg.empirical_w1_columns(zy, zr)
bound = eg.charge_mean_bound(yr, rr, eg.q_np(yr), np.zeros(480), np.ones(480), zy, zr, w)
check("nonzero_charge_mean_bound_tight_fixture", bound["Dout_q"] == .375 and np.isclose(bound["output_specific_LB_coordinate_mean"], w.mean()) and bound["nominal_inequality_checks_pass"])
yr2 = np.tile(eg.CHARGE_A/256., (3, 1)); zy2 = yr2.astype(np.float32)
w2 = eg.empirical_w1_columns(zy2, zr)
bound2 = eg.charge_mean_bound(yr2, rr, eg.q_np(yr2), np.zeros(480), np.ones(480), zy2, zr, w2)
check("nonzero_charge_RMS_bound_tight_fixture", np.isclose(bound2["output_specific_LB_coordinate_RMS"], np.sqrt(np.mean(w2*w2)), rtol=1e-14) and bound2["nominal_inequality_checks_pass"])
error_bound = eg.charge_mean_bound(yr, rr, np.zeros(3), np.zeros(480), np.ones(480), zy, zr, w)
check("actual_postADC_charge_error_is_subtracted", error_bound["actual_postADC_total_mean_absolute_charge_error_q"] == .375 and error_bound["source_charge_form_LB_coordinate_mean"] == 0)
erased_rep = eg.charge_mean_bound(yr, rr, eg.q_np(yr), np.zeros(480), np.ones(480), np.zeros_like(zy), zr, np.zeros(480))
check("actual_representation_residual_prevents_false_bound", erased_rep["Erep_q"] == .375 and erased_rep["output_specific_LB_coordinate_mean"] == 0 and erased_rep["nominal_inequality_checks_pass"])
unit_bound = eg.charge_mean_bound(yr*1000, rr*1000, eg.q_np(yr)*1000, np.zeros(480), np.full(480, 1000.), zy, zr, w)
check("consistent_physical_unit_rescaling_leaves_bound_invariant", np.isclose(unit_bound["output_specific_LB_coordinate_mean"], bound["output_specific_LB_coordinate_mean"], atol=1e-16) and np.isclose(unit_bound["output_specific_LB_coordinate_RMS"], bound["output_specific_LB_coordinate_RMS"], atol=1e-16))


def artificial_arguments(same_law=False, nrep=3):
    sc = np.asarray([2, 3, 2, 4, 2, 3, 2, 3, 2, 4])
    rc = sc.copy() if same_law else np.asarray([3, 4, 5, 3, 4, 5, 3, 4, 5, 3])
    sb = np.repeat(np.asarray(eg.REGIONS), sc)
    rb = np.repeat(np.asarray(eg.REGIONS), rc)
    source = rng.normal(0., .003, (int(sc.sum()), 480)).astype(np.float32)
    source[:, 100:400] += np.float32(.03)
    real = source.copy() if same_law else rng.normal(0., .003, (int(rc.sum()), 480)).astype(np.float32)
    if not same_law:
        real[:, 100:400] += np.float32(.02)
    pre = source.astype(np.float64)
    # REAL and generated ADC need not agree on arbitrary float32 values.  For a
    # same-law fixture, choose original float32 code representatives with the
    # two paths already equal, then bind all three given arrays explicitly.
    if same_law:
        source = eg.canonical_real(source).astype(np.float32)
        real = source.copy()
        pre = source.astype(np.float64)
    stats = {"source_counts": sc, "REAL_counts": rc, "temporal_REAL_counts": rc.copy()}
    for name, dim in zip(eg.VIEW_NAMES, eg.DIMENSIONS):
        stem = "temporal" if name == "temporal31" else name
        stats[stem + "_center"] = np.zeros((10, dim))
        stats[stem + "_scale"] = np.ones((10, dim))
    dirs = []
    for dim in eg.DIMENSIONS:
        p = rng.normal(size=(dim, 64))
        dirs.append((p/np.linalg.norm(p, axis=0, keepdims=True)).astype(np.float32))
    matched = {}
    for scope in eg.SCOPES:
        si = eg._indices_for_scope(sb, scope)
        ri = eg._indices_for_scope(rb, scope)
        matched[scope] = {k: rng.choice(ri, size=(nrep, len(si)), replace=True).astype(np.int64) for k in ("A", "B")}
    return dict(source_out_preADC=pre, source_canonical=eg.canonical_from_pre_adc(pre), canonical_REAL=eg.canonical_real(real),
                source_raw=source, REAL_raw=real, source_bins=sb, REAL_bins=rb, source_charge=eg.q_np(source),
                statistics=stats, directions=dirs[:4], temporal_directions=dirs[4], matched_real_indices=matched,
                REAL_control_canonical=eg.canonical_real(real), expected_population=(len(source), len(real)), expected_reference_replicates=nrep)


args = artificial_arguments()
before_source = args["source_raw"].tobytes()
summary, private = eg.analyze_endpoint_geometry(**args)
check("array_only_complete_artificial_call", len(summary["rows"]) == 11 and len(summary["rows"][0]["SOURCE_vs_complete_REAL"]["views"]) == 5)
check("original_source_mass_and_parent_rows_preserved", args["source_raw"].tobytes() == before_source and np.array_equal(private["source_original_region_counts"], args["statistics"]["source_counts"]) and np.isclose(private["source_original_region_mass"].sum(), 1.))
check("old_and_new_objective_coefficients_and_original_mass", np.allclose(private["SOURCE_weighted_regional_terms"], (private["SOURCE_full_terms_by_scope"][1:]*private["source_original_region_mass"][:, None]).sum(0), rtol=0, atol=1e-15) and summary["charge_bound_aggregation"]["coefficients"] == {"V480_coordinate_mean_in_full_window": .125, "V480_coordinate_RMS_in_coordinate_RMS": .25})
check("all_ten_nominal_charge_bounds_checked", summary["charge_bound_aggregation"]["all_nominal_inequality_checks_pass"] and all(row["charge_mean_bound"]["Dout_q"] > 0 for row in summary["rows"][1:]))
check("matched_floor_keeps_supplied_repeats_and_scopes", all(np.array_equal(private["REAL_pair_floor__" + scope + "__A_indices"], args["matched_real_indices"][scope]["A"]) for scope in eg.SCOPES))
check("same_treated_REAL_control_stays_separate", max(summary["same_treated_REAL_original_SOURCE_mass_weighted_regional_terms"].values()) == 0.)
json.dumps(summary, allow_nan=False)
check("summary_serializable_without_nonfinite_values", True)
expected_rejection("wrong_q_reporting_units_rejected", lambda: eg.analyze_endpoint_geometry(**{**args, "source_charge": args["source_charge"]*1000}))
expected_rejection("clipped_network_bins_rejected", lambda: eg.analyze_endpoint_geometry(**{**args, "source_bins": np.clip(args["source_bins"], 0, 7)}))
wrong_draw = {k: {a: v.copy() for a, v in d.items()} for k, d in args["matched_real_indices"].items()}
wrong_draw["region-1"]["A"][0, 0] = int(np.flatnonzero(args["REAL_bins"] == 8)[0])
expected_rejection("REAL_draw_outside_original_raw_bin_rejected", lambda: eg.analyze_endpoint_geometry(**{**args, "matched_real_indices": wrong_draw}))
expected_rejection("incorrect_canonical_source_rejected", lambda: eg.analyze_endpoint_geometry(**{**args, "source_canonical": args["source_canonical"] + 1/16383}))
same_args = artificial_arguments(same_law=True, nrep=32)
same, same_private = eg.analyze_endpoint_geometry(**same_args)
check("complete_same_law_zero_all_views_and_charge_bounds", np.max(same_private["SOURCE_full_terms_by_scope"]) == 0 and same["charge_bound_aggregation"]["output_specific_V480_lower_bound_full_window_plus_RMS"] == 0.)
check("all_32_matched_REAL_replicates_retained", same_private["REAL_pair_floor_terms_by_scope"].shape == (11, 32, 3))
core_tree = ast.parse((ROOT / "endpoint_geometry.py").read_text(encoding="utf8"))
imports = [n.names[0].name for n in core_tree.body if isinstance(n, ast.Import)]
check("pure_module_only_numpy_import_and_no_IO_or_RNG_calls", imports == ["numpy"] and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"open", "exec", "eval"} for n in ast.walk(core_tree)))

end_memory = memory()
check("local_peak_RSS_under_1GiB", end_memory["peak_RSS_bytes"] <= LIMIT, peak_RSS_bytes=end_memory["peak_RSS_bytes"])
check("CPU_affinity_exactly_one_core", one_cpu_mask.bit_count() == 1)
source_paths = [R62 / n for n in ("diagnose_train.py", "residual_analysis.py", "residual_math.py", "traced_response.py")]
source_paths += [R63 / n for n in ("R61_distribution_loss.py", "distribution_loss.py", "source_flow_core.py", "noise_core.py", "config.json")]
source_paths += [PROJECT / "reviews" / "R63_ROOT_DIRECTION_001.json", PROJECT / "reviews" / "R63_WAIT_PARALLEL_001" / "root_analysis" / "CHARGE_LOSS_BOUND_STATIC.json", PROJECT / "reviews" / "R63_WAIT_PARALLEL_001" / "train_evidence" / "report.json"]
binding = {
    "UTC": datetime.now(timezone.utc).isoformat(), "scientific_array_or_weight_reads": 0,
    "source_files": [{"path": str(p), "SHA256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    "pure_helper_reuse": "R62 features and unequal-size empirical-W1 quantile algorithm; matched artificial outputs checked. R63 temporal31 and ADC functions checked by source AST extraction on artificial values only.",
    "endpoint_semantic_override": "Direct source_flow_core.endpoint projected output, per parent instruction; R62 traced_solve source read for integration awareness but not used to define Y.",
    "root_review_design_input": "next_action only; initial source-text tool output incidentally expanded other report fields; none used as scientific input",
    "local_artifact_SHA256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ("endpoint_geometry.py", "artificial_check.py")},
}
report = {
    "UTC": datetime.now(timezone.utc).isoformat(), "status": "all_artificial_checks_passed",
    "checks": CHECKS, "check_count": len(CHECKS), "synthetic_seed": 20261011649,
    "synthetic_main_population": list(args["expected_population"]),
    "synthetic_same_law_population": list(same_args["expected_population"]),
    "wall_seconds": time.monotonic() - START,
    "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "cwd": str(ROOT)},
    "resource": {"CPU_affinity_mask": one_cpu_mask, "CPU_affinity_core_count": 1, "torch_intraop_threads": torch.get_num_threads(),
                 "torch_interop_threads": torch.get_num_interop_threads(), "Windows_job_process_commit_limit_bytes": LIMIT,
                 "RSS_watchdog_limit_bytes": LIMIT, "RSS_watchdog_poll_seconds": .05, **end_memory,
                 "TEMP": str(TEMP), "Python_B": bool(sys.dont_write_bytecode)},
    "scope": {"scientific_arrays_decoded": 0, "scientific_weights_read": 0, "scientific_forward_rows": 0,
              "scientific_gradient_calls": 0, "artificial_gradient_calls": 0, "optimizer_updates": 0,
              "DEV_split_reads": 0, "EVAL_split_reads": 0, "CAL_split_reads": 0, "new_scientific_registration": False},
    "interpretation": "Only synthetic checks passed. No R64 scientific statistics or bound value has been measured.",
}
for filename, value in (("ARTIFICIAL_CHECK.json", report), ("SOURCE_BINDING.json", binding)):
    (ROOT / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
stop_guard.set()
print(json.dumps({"status": report["status"], "checks": len(CHECKS), "wall_seconds": report["wall_seconds"], "peak_RSS_bytes": end_memory["peak_RSS_bytes"]}))
