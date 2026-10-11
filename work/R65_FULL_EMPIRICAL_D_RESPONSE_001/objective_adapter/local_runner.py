"""Single-CPU, 1-GiB peak-RSS-enforced Windows runner for synthetic checks."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
LIMIT = 1024**3
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
args = parser.parse_args()
ATTEMPT = Path(args.attempt).resolve()
if ROOT.drive.upper() not in ("E:", "D:") or not ATTEMPT.is_relative_to(ROOT / "runs"):
    raise RuntimeError("Local outputs must stay under this E/D adapter's runs directory")
if os.name != "nt":
    raise RuntimeError("This local resource guard is Windows only; use portable run_check on server")
ATTEMPT.mkdir(parents=True, exist_ok=False)
cache = ATTEMPT / "runtime_cache"
cache.mkdir()
for key in ("TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "MPLCONFIGDIR",
            "NUMBA_CACHE_DIR", "CUDA_CACHE_PATH", "HF_HOME", "PYTHONPYCACHEPREFIX"):
    target = cache / key.lower()
    target.mkdir()
    os.environ[key] = str(target)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
handle = kernel.GetCurrentProcess()
kernel.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
if not kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system_mask)):
    raise ctypes.WinError(ctypes.get_last_error())
if not kernel.SetProcessAffinityMask(handle, mask.value & -mask.value):
    raise ctypes.WinError(ctypes.get_last_error())
kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system_mask))
assert mask.value.bit_count() == 1
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]

def memory():
    p = PMC()
    p.cb = ctypes.sizeof(p)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(p), p.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return p

def dump(name, value):
    (ATTEMPT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")

def guard():
    while True:
        try:
            peak = memory().PeakWorkingSetSize
            if peak > LIMIT:
                dump("RESOURCE_STOP.json", dict(PID=os.getpid(), peak_RSS_bytes=peak,
                     RSS_limit_bytes=LIMIT, reason="peak_RSS_limit_exceeded", planned_exit=91))
                os._exit(91)
        except Exception as exc:
            dump("RESOURCE_STOP.json", dict(PID=os.getpid(), reason="RSS_probe_failed",
                                           error=str(exc), planned_exit=92))
            os._exit(92)
        time.sleep(.02)
threading.Thread(target=guard, daemon=True).start()
START = time.monotonic()
started = datetime.now(timezone.utc).isoformat()
dump("RUN_STARTED.json", dict(PID=os.getpid(), started_utc=started,
     CPU_affinity_count=mask.value.bit_count(), RSS_limit_bytes=LIMIT,
     guard_interval_seconds=.02, scientific_inputs_read=0))
exit_code = 1
try:
    manifest = json.loads((ROOT / "PREEXECUTION_BINDING.json").read_text("utf-8"))
    for item in manifest["files"]:
        data = (ROOT / item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("Pre-execution binding mismatch: " + item["file"])
        target = ATTEMPT / "input_snapshot" / item["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (ATTEMPT / "input_snapshot/PREEXECUTION_BINDING.json").write_bytes(
        (ROOT / "PREEXECUTION_BINDING.json").read_bytes())
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    spec = importlib.util.spec_from_file_location("_r65_portable_checks", ROOT / "portable_objective_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_check("cpu", objective_adapter_root=ROOT)
    result["resource"] = dict(PID=os.getpid(), CPU_affinity_count=mask.value.bit_count(),
        peak_RSS_bytes=memory().PeakWorkingSetSize, RSS_limit_bytes=LIMIT,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    dump("ARTIFICIAL_CPU.json", result)
    exit_code = 0
except Exception as exc:
    partial = getattr(exc, "receipt", None)
    if partial is None:
        partial = dict(status="LOCAL_RUNNER_FAILED", error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc())
    dump("ARTIFICIAL_CPU.json", partial)
    print(json.dumps({"status": partial["status"], "error": str(exc)}, ensure_ascii=False), flush=True)
finally:
    resource = dict(PID=os.getpid(), CPU_affinity_count=mask.value.bit_count(),
        peak_RSS_bytes=memory().PeakWorkingSetSize, RSS_limit_bytes=LIMIT,
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-START, planned_exit_code=exit_code,
        actual_native_exit_known=False, scientific_arrays_opened=0,
        scientific_weights_loaded=0, scientific_optimizer_updates=0)
    dump("SELF_PROCESS_RECEIPT.json", resource)
    print(json.dumps(resource), flush=True)
sys.exit(exit_code)

