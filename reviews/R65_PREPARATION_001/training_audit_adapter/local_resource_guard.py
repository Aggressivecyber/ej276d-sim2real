"""Native Windows one-CPU/1-GiB working-set guard for local preparation only."""
import ctypes
import os
from pathlib import Path

LIMIT = 1024 ** 3


def apply(workspace):
    workspace = Path(workspace).resolve()
    cache = workspace / "runtime_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX"):
        os.environ[key] = str(cache)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    process = k.GetCurrentProcess()
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = ctypes.c_int
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(process, ctypes.byref(mask), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
    one = mask.value & -mask.value
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = ctypes.c_int
    if not k.SetProcessAffinityMask(process, one):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    k.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
    k.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    if not k.SetProcessWorkingSetSizeEx(process, 65536, LIMIT, 4):
        raise OSError(ctypes.get_last_error(), "SetProcessWorkingSetSizeEx")
    k.GetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint32)]
    k.GetProcessWorkingSetSizeEx.restype = ctypes.c_int
    lower, upper, flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_uint32()
    if not k.GetProcessWorkingSetSizeEx(process, ctypes.byref(lower), ctypes.byref(upper), ctypes.byref(flags)):
        raise OSError(ctypes.get_last_error(), "GetProcessWorkingSetSizeEx")
    if upper.value != LIMIT or not flags.value & 4:
        raise RuntimeError("native hard working-set maximum inactive")
    return {"PID": os.getpid(), "CPU_affinity_mask": one, "CPU_affinity_count": 1,
            "RSS_hard_max_bytes": upper.value, "RSS_limit_flags": flags.value,
            "cache_root": str(cache), "numeric_environment_threads": 1}


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    p = ctypes.WinDLL("psapi", use_last_error=True)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    value = Counters()
    value.cb = ctypes.sizeof(value)
    p.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    p.GetProcessMemoryInfo.restype = ctypes.c_int
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo")
    return {"peak_RSS_bytes": value.PeakWorkingSetSize, "RSS_bytes": value.WorkingSetSize,
            "peak_private_commit_bytes": value.PeakPagefileUsage}
