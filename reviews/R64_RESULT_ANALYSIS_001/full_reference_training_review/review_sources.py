"""Bounded read-only evidence inspection and local memo sealing; no scientific execution."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
COPY = ROOT.parent / "completed_summary_copy"
R63 = PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
R64 = PROJECT / "work" / "R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001"
MODE = sys.argv[1] if len(sys.argv) > 1 else "schema"
if MODE not in ("schema", "excerpts", "values", "seal"):
    raise ValueError("Only schema, excerpts, values, seal are supported")
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(TEMP)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
if not sys.dont_write_bytecode or os.name != "nt":
    raise RuntimeError("Launch Windows metadata reader using python -B")
START = time.monotonic()
LIMIT = 1024**3
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

SPECS = {
    "copy_receipt": COPY / "COPY_RECEIPT.json",
    "native_receipt": COPY / "diagnose_train.py.NATIVE_RECEIPT.json",
    "parameter_summary": COPY / "PARAMETER_GRADIENT_SUMMARY.json",
    "endpoint_summary": COPY / "ENDPOINT_GRADIENT_SUMMARY.json",
    "result": COPY / "RESULT.json",
    "r63_report": PROJECT / "R63_REPORT.md",
    "r64_protocol": PROJECT / "tasks" / "R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001.md",
    "r63_train": R63 / "train_final.py",
    "r63_config": R63 / "config.json",
    "r63_distribution": R63 / "distribution_loss.py",
    "r63_r61_distribution": R63 / "R61_distribution_loss.py",
    "r63_source_flow": R63 / "source_flow_core.py",
    "r63_network": R63 / "network.py",
    "r63_noise": R63 / "noise_core.py",
    "r63_precision": R63 / "precision_order.py",
    "r64_diagnose": R64 / "diagnose_train.py",
    "r64_config": R64 / "config.json",
    "r64_adapter": R64 / "distribution_adapter" / "distribution_gradient.py",
}
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()
def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
def summary_tree(value, depth=0):
    if isinstance(value, dict):
        if depth >= 4:
            return {"dict_keys": list(value)}
        return {k: summary_tree(v, depth+1) for k, v in value.items()}
    if isinstance(value, list):
        return {"list_N": len(value), "first": summary_tree(value[0], depth+1) if value else None}
    return value
def source_excerpt(text, patterns, before=3, after=7):
    lines = text.splitlines()
    chosen = set()
    for i, line in enumerate(lines):
        if any(re.search(p, line, re.I) for p in patterns):
            chosen.update(range(max(0, i-before), min(len(lines), i+after+1)))
    return "\n".join(f"{i+1}: {lines[i]}" for i in sorted(chosen))

try:
    manifest = [{"id": k, "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for k,p in SPECS.items()]
    binding = ROOT / "INPUT_BINDING.json"
    if binding.exists():
        if json.loads(binding.read_text("utf8"))["files"] != manifest:
            raise RuntimeError("Sealed input changed between reads")
    else:
        write_json("INPUT_BINDING.json", {"UTC": datetime.now(timezone.utc).isoformat(), "files": manifest,
                   "scope": "five sealed completed R64 JSON summaries or receipts, original R63 report/source/config and R64 protocol/source/config only"})
    loaded = {k: json.loads(p.read_text("utf-8-sig")) if p.suffix == ".json" else p.read_text("utf-8-sig") for k,p in SPECS.items()}
    if MODE == "schema":
        output = {k: summary_tree(v) for k,v in loaded.items() if isinstance(v, dict)}
        write_json("SCHEMA.json", output)
        print(json.dumps(output, ensure_ascii=False))
    elif MODE == "excerpts":
        patterns = {
            "r63_train": ["source_order", "source_valley", "identity", "box_correction", "backward", "scaler", "noise", "raw", "optimizer", "batch_size", "zero_grad"],
            "r63_distribution": ["def ", "source_order", "source_valley", "identity", "box_correction", "raw", "detach", "mass", "losses", "order", "valley"],
            "r63_r61_distribution": ["source_order", "source_valley", "identity", "box_correction", "def loss", "def forward", "def training", "weight", "mass", "order", "valley", "raw"],
            "r63_source_flow": ["box", "project", "canonical", "round", "straight", "STE", "raw"],
            "r63_network": ["class ", "def forward", "Dropout", "BatchNorm", "LayerNorm", "noise", "rand", "return"],
            "r63_noise": ["def ", "rand", "index", "seed"],
            "r63_precision": ["def ", "round", "detach", "STE", "out"],
            "r64_diagnose": ["full_reference", "replay", "surrogate", "endpoint_gradient", "GY", "G_Y", "autograd", "backward", "register_hook", "noise", "cache", "chunks", "chunk", "max_memory", "peak", "NATIVE"],
            "r64_adapter": ["def endpoint", "def objective", "mass", "weights", "source", "group", "region", "mS"],
        }
        output = {k: source_excerpt(loaded[k], ps) for k,ps in patterns.items()}
        output["r63_report"] = loaded["r63_report"]
        output["r64_protocol"] = loaded["r64_protocol"]
        (ROOT / "SOURCE_EXCERPTS.txt").write_text("\n\n".join(f"FILE: {SPECS[k]}\n{v}" for k,v in output.items()), encoding="utf8")
        print("Wrote SOURCE_EXCERPTS.txt. Use explicit file ids and line ranges for bounded console reads.")
        for k,v in output.items():
            print(k, len(v))
    elif MODE == "values":
        if len(sys.argv) != 3:
            raise ValueError("values mode expects own REQUEST.json basename")
        req_path = ROOT / sys.argv[2]
        if req_path.parent != ROOT or req_path.suffix != ".json":
            raise ValueError("Request must be a JSON file directly under this review")
        request = json.loads(req_path.read_text("utf8"))
        output = {}
        for item in request:
            key = item["id"]
            value = loaded[key]
            if "path" in item:
                for part in item["path"]:
                    value = value[part]
            if "lines" in item:
                start,end = item["lines"]
                value = "\n".join(f"{i+1}: {line}" for i,line in enumerate(value.splitlines()) if start <= i+1 <= end)
            output[item.get("label", key)] = value
        outname = req_path.stem + ".OUTPUT.json"
        write_json(outname, output)
        print(json.dumps(output, ensure_ascii=False))
    else:
        memo = ROOT / "REVIEW.md"
        data = ROOT / "REVIEW.json"
        if not memo.exists() or not data.exists():
            raise RuntimeError("Memo and structured review must exist before sealing")
        review = json.loads(data.read_text("utf8"))
        if review["selected_method"] is not None or review["scientific_runs"] != 0 or review["parameter_updates"] != 0:
            raise RuntimeError("This task may not select or execute a scientific method")
        files = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(ROOT.iterdir())
                 if p.is_file() and p.name not in ("BINDING_RECEIPT.json", "seal.RESOURCE_RECEIPT.json", "seal.NATIVE_RECEIPT.json", "seal.stdout.log", "seal.stderr.log")]
        write_json("BINDING_RECEIPT.json", {"status": "INDEPENDENT_FEASIBILITY_REVIEW_ONLY", "UTC": datetime.now(timezone.utc).isoformat(),
                  "input_files": manifest, "own_files": files, "source_hashes_unchanged": True, "selected_method": None,
                  "scientific_runs": 0, "parameter_updates": 0, "new_waveform_or_weight_reads": 0, "new_DEV_EVAL_CAL_reads": 0,
                  "root_metadata_or_Git_writes": 0, "scope": "Symbolic derivation and evidence/source inspection only"})
        print(json.dumps({"status": "SEALED", "memo_sha256": sha(memo), "review_sha256": sha(data), "receipt_sha256": sha(ROOT / "BINDING_RECEIPT.json")}, ensure_ascii=False))
    for row in manifest:
        if sha(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("Input source changed during bounded read")
    m = memory()
    if m["peak_RSS_bytes"] > LIMIT:
        raise RuntimeError("Resource guard exceeded")
    write_json(MODE + ".RESOURCE_RECEIPT.json", {"status": "COMPLETE", "UTC": datetime.now(timezone.utc).isoformat(),
               "PID": os.getpid(), "elapsed_seconds": time.monotonic()-START, "source_hashes_unchanged": True,
               "CPU_count": 1, "CPU_affinity_mask": one_cpu, "process_commit_limit_bytes": LIMIT, "RSS_watchdog_limit_bytes": LIMIT,
               "RSS_watchdog_poll_seconds": .05, "Python_B": bool(sys.dont_write_bytecode), "TEMP": str(TEMP), **m,
               "science_calculations": 0, "network_calls": 0, "new_waveform_or_weight_reads": 0, "root_metadata_or_Git_writes": 0})
finally:
    stop.set()
