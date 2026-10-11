"""Single-core, bounded-memory guard for this Windows artificial-only proof.

No numerical library is imported here. Call enter_guard() first. The job object
limits private commit to 900 MiB and the hard working-set maximum to 768 MiB;
the independently recorded Windows PeakWorkingSetSize must remain < 1 GiB.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import time

RSS_LIMIT_BYTES = 1 << 30
PRIVATE_COMMIT_LIMIT_BYTES = 900 << 20
HARD_WORKING_SET_MAX_BYTES = 768 << 20
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "GOTO_NUM_THREADS",
    "TBB_NUM_THREADS", "RAYON_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
CACHE_ENV_NAMES = (
    "TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX",
)
_START = time.perf_counter()
_JOB = None
_PROC = None
_K32 = None
_PMEM = None
_STATE = {}
_OPENED = set()
_BLOCKED = []


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IOCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits), ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _check(result, label):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error(), label)


def enter_guard(directory: Path):
    global _JOB, _PROC, _K32, _PMEM, _STATE
    directory = directory.resolve()
    if os.name != "nt":
        raise RuntimeError("This preparation runner is bound to the inspected Windows runtime.")
    if directory.drive.upper() not in ("E:", "D:") or Path.cwd().resolve() != directory:
        raise RuntimeError("Working directory must be this preparation directory on E:/D:.")
    if not sys.dont_write_bytecode:
        raise RuntimeError("Run Python with -B.")
    for key in THREAD_ENV_NAMES:
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key} must equal 1 before numerical imports.")
    caches = {}
    for key in CACHE_ENV_NAMES:
        value = Path(os.environ.get(key, ""))
        if not value.is_absolute() or value.drive.upper() not in ("E:", "D:"):
            raise RuntimeError(f"{key} must point to E:/D:.")
        value.mkdir(parents=True, exist_ok=True)
        caches[key] = str(value.resolve())
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    proc = k32.GetCurrentProcess()
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    _check(k32.GetProcessAffinityMask(proc, ctypes.byref(mask), ctypes.byref(system)),
           "GetProcessAffinityMask")
    before = int(mask.value)
    chosen = before & -before
    if not chosen:
        raise RuntimeError("Process has no available CPU affinity bit.")
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    _check(k32.SetProcessAffinityMask(proc, chosen), "SetProcessAffinityMask")
    k32.SetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD
    ]
    # HARDWS_MIN_DISABLE | HARDWS_MAX_ENABLE.
    _check(k32.SetProcessWorkingSetSizeEx(
        proc, 16 << 20, HARD_WORKING_SET_MAX_BYTES, 0x2 | 0x4
    ), "SetProcessWorkingSetSizeEx")
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    _check(job, "CreateJobObjectW")
    limits = _ExtendedLimits()
    # Limit private commit; terminate children on job-handle close.
    limits.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
    limits.ProcessMemoryLimit = PRIVATE_COMMIT_LIMIT_BYTES
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    _check(k32.SetInformationJobObject(
        job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
    ), "SetInformationJobObject")
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _check(k32.AssignProcessToJobObject(job, proc), "AssignProcessToJobObject")
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    pmem = psapi.GetProcessMemoryInfo
    pmem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
    pmem.restype = wintypes.BOOL
    _JOB, _PROC, _K32, _PMEM = job, proc, k32, pmem
    _STATE = {
        "pid": os.getpid(),
        "single_cpu_affinity_mask_decimal": chosen,
        "original_process_affinity_mask_decimal": before,
        "cpu_affinity_count": chosen.bit_count(),
        "thread_environment": {k: os.environ[k] for k in THREAD_ENV_NAMES},
        "cache_environment": caches,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "rss_limit_bytes_exclusive": RSS_LIMIT_BYTES,
        "hard_working_set_max_bytes": HARD_WORKING_SET_MAX_BYTES,
        "job_private_commit_limit_bytes": PRIVATE_COMMIT_LIMIT_BYTES,
        "windows_job_memory_limit_applied": True,
        "hard_working_set_limit_applied": True,
        "no_child_processes_requested": True,
    }
    root = directory
    parents = {str((root.parent / name).resolve()).casefold()
               for name in ("report.json", "C1_IMPLEMENTATION_NOTES.json")}

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() in (".npy", ".npz", ".pt", ".pth", ".safetensors", ".h5", ".hdf5"):
            _BLOCKED.append(str(p))
            raise RuntimeError("Scientific array/checkpoint file access is forbidden.")
        if p == root or root in p.parents or str(p).casefold() in parents:
            _OPENED.add(str(p))

    sys.addaudithook(audit)
    return snapshot()


def snapshot():
    if _PROC is None:
        raise RuntimeError("Resource guard was not entered.")
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    _check(_PMEM(_PROC, ctypes.byref(counters), counters.cb), "GetProcessMemoryInfo")
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    _check(_K32.GetProcessAffinityMask(_PROC, ctypes.byref(mask), ctypes.byref(system)),
           "GetProcessAffinityMask")
    if mask.value.bit_count() != 1:
        raise RuntimeError("Single-CPU affinity changed.")
    if int(counters.PeakWorkingSetSize) >= RSS_LIMIT_BYTES:
        raise MemoryError("Peak Windows resident set reached the 1 GiB limit.")
    return {
        **_STATE,
        "elapsed_seconds": time.perf_counter() - _START,
        "current_working_set_bytes": int(counters.WorkingSetSize),
        "peak_RSS_bytes_Windows_PeakWorkingSetSize": int(counters.PeakWorkingSetSize),
        "private_commit_bytes": int(counters.PrivateUsage),
        "peak_private_commit_bytes": int(counters.PeakPagefileUsage),
        "peak_RSS_below_1_GiB": True,
        "observed_affinity_mask_decimal": int(mask.value),
        "application_files_opened": sorted(_OPENED),
        "forbidden_array_or_checkpoint_open_attempts": list(_BLOCKED),
        "forbidden_array_or_checkpoint_open_attempt_count": len(_BLOCKED),
    }
