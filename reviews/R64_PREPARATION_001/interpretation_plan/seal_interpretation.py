"""Bounded metadata-only validation. Never imports or runs scientific sources."""
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
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
if not sys.dont_write_bytecode:
    raise RuntimeError("Launch with python -B")
START = time.monotonic()
LIMIT = 1024**3
if os.name != "nt":
    raise RuntimeError("This metadata receipt owner is Windows-only")

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
            (ROOT / "RESOURCE_FAILURE.json").write_text(json.dumps(m), encoding="utf8")
            os._exit(91)


threading.Thread(target=watch, daemon=True).start()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(text, names):
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != set(names):
        raise ValueError("Missing explicitly bound source constant")
    return values


try:
    binding = json.loads((ROOT / "SOURCE_BINDING.json").read_text("utf-8-sig"))
    sources = {}
    source_checks = []
    for record in binding["allowed_sources"]:
        path = Path(record["path"])
        actual = sha(path)
        if actual != record["sha256"]:
            raise RuntimeError("Sealed source drift: " + record["id"])
        sources[record["id"]] = path.read_text("utf-8-sig")
        source_checks.append({"id": record["id"], "sha256": actual, "unchanged": True})
    table = json.loads((ROOT / "DECISION_TABLE.json").read_text("utf8"))
    report = (ROOT / "INTERPRETATION_PLAN.txt").read_text("utf8")
    if not report.strip() or "\ufffd" in report:
        raise ValueError("Empty report or malformed UTF-8 replacement text")
    ids = [row["id"] for row in table["decision_rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate decision row ID")
    if table["selected_next_method"] is not None or table["selected_decision_rows"] or table["all_observed_R64_values"] is not None:
        raise ValueError("Pre-outcome plan contains a selected outcome/method")
    if table["new_acceptance_thresholds"] or table["existing_acceptance_changed"]:
        raise ValueError("Plan must not modify acceptance")
    for row in table["decision_rows"]:
        if not set(row["source_ids"]).issubset(sources):
            raise ValueError("Unbound decision source")
    cfg = json.loads(sources["R64_CONFIG"])
    stat = constants(sources["GRADIENT_STATISTICS_SOURCE"], ("D_NAMES", "MACROS", "BOOTSTRAPS", "DECOMPOSITION_ATOL", "DECOMPOSITION_RTOL"))
    geom = constants(sources["GEOMETRY_SOURCE"], ("TERM_NAMES", "NOMINAL_CHECK_ATOL"))
    fixed = table["fixed_comparison"]
    numerical = table["existing_numerical_rules_for_reference_only"]
    if fixed["D"] != list(stat["D_NAMES"]) or fixed["D"] != cfg["gradient_components"]:
        raise ValueError("Distribution component transcription mismatch")
    if fixed["geometry_floor_terms"] != list(geom["TERM_NAMES"]):
        raise ValueError("Geometry floor coverage transcription mismatch")
    if fixed["macro_pairs"] != stat["MACROS"] or stat["MACROS"] != cfg["paired_macro_batches"]:
        raise ValueError("Macro count transcription mismatch")
    if fixed["bootstrap"]["replicates"] != stat["BOOTSTRAPS"] or stat["BOOTSTRAPS"] != cfg["paired_gradient_bootstrap_replicates"]:
        raise ValueError("Bootstrap count transcription mismatch")
    if numerical["geometry_scaled_atol"] != geom["NOMINAL_CHECK_ATOL"] or geom["NOMINAL_CHECK_ATOL"] != cfg["charge_bound_nominal_scaled_atol"]:
        raise ValueError("Bound numerical-rule transcription mismatch")
    if numerical["gradient_decomposition_atol"] != stat["DECOMPOSITION_ATOL"] or numerical["gradient_decomposition_rtol"] != stat["DECOMPOSITION_RTOL"]:
        raise ValueError("Gradient numerical-rule transcription mismatch")
    for record in binding["allowed_sources"]:
        if sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Sealed source changed during metadata validation")
    files = []
    for name in ("SOURCE_BINDING.json", "INTERPRETATION_PLAN.txt", "DECISION_TABLE.json", "seal_interpretation.py"):
        path = ROOT / name
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    m = memory()
    if m["peak_RSS_bytes"] > LIMIT:
        raise RuntimeError("Metadata resource limit exceeded")
    receipt = {
        "status": "PREOUTCOME_INTERPRETATION_PLAN_SEALED",
        "UTC": datetime.now(timezone.utc).isoformat(), "PID": os.getpid(),
        "decision_rows": len(ids), "sources_bound": len(sources),
        "all_source_hashes_unchanged": True, "source_checks": source_checks,
        "reported_constants_verified_by_AST_and_config_only": {**stat, **geom},
        "scientific_source_modules_imported_or_executed": 0,
        "scientific_arrays_weights_or_R64_results_opened": 0,
        "DEV_EVAL_CAL_reads": 0, "training_updates": 0,
        "root_metadata_or_Git_writes": 0, "new_literature_sources": 0,
        "selected_next_method": None, "selected_decision_rows": [],
        "elapsed_seconds": time.monotonic() - START,
        "resource_scope": "This metadata-only seal process; no scientific computation was run",
        "resource": {"CPU_affinity_mask": one_cpu, "CPU_count": 1,
                     "process_commit_limit_bytes": LIMIT, "RSS_watchdog_limit_bytes": LIMIT,
                     "RSS_watchdog_poll_seconds": .05, "Python_B": bool(sys.dont_write_bytecode),
                     "TEMP": str(TEMP), **m},
        "files": files,
    }
    (ROOT / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
    reopened = json.loads((ROOT / "PREPARATION_RECEIPT.json").read_text("utf8"))
    for entry in reopened["files"]:
        if sha(ROOT / entry["file"]) != entry["sha256"]:
            raise RuntimeError("Reopened own-file manifest mismatch")
    print(json.dumps({"status": receipt["status"], "decision_rows": len(ids), "sources_unchanged": len(sources),
                      "peak_RSS_bytes": m["peak_RSS_bytes"], "elapsed_seconds": receipt["elapsed_seconds"],
                      "preparation_receipt_sha256": sha(ROOT / "PREPARATION_RECEIPT.json")}))
finally:
    stop.set()
