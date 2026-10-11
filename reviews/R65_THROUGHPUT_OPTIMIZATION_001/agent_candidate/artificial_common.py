"""Artificial-only loading, process guards, and bitwise comparisons."""
from __future__ import annotations
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_job_handle = None


def configure_local_cpu():
    """Pin Windows logical CPU 1 (mask 2); cap private commit at 1 GiB.

    A conservative hard private-commit limit is stronger than the requested
    RSS limit for ordinary private allocations. Peak resident bytes are also
    independently reported. Abort if the guard cannot be installed.
    """
    global _job_handle
    sys.dont_write_bytecode = True
    cache = HERE / "tmp"
    cache.mkdir(exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME"):
        os.environ[name] = str(cache)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Never initialize a local CUDA context in CPU checks.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.name != "nt":
        raise RuntimeError("Local CPU check is explicitly Windows / affinity mask 2")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), 2):
        raise OSError(ctypes.get_last_error(), "Cannot set affinity mask 2")
    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]
    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    k.CreateJobObjectW.restype = ctypes.c_void_p
    k.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    k.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    _job_handle = k.CreateJobObjectW(None, None)
    limit = Extended()
    limit.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    limit.ProcessMemoryLimit = 1024 ** 3
    if not _job_handle or not k.SetInformationJobObject(_job_handle, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
        raise OSError(ctypes.get_last_error(), "Cannot install 1 GiB private-commit cap")
    if not k.AssignProcessToJobObject(_job_handle, k.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "Cannot assign guarded job")


def memory_receipt():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    p = ctypes.WinDLL("psapi", use_last_error=True)
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    k.GetCurrentProcess.restype = ctypes.c_void_p
    p.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
    counts = Counters()
    counts.cb = ctypes.sizeof(counts)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(counts), counts.cb):
        raise OSError(ctypes.get_last_error(), "Cannot read RSS")
    if counts.PeakWorkingSetSize > 1024 ** 3:
        raise RuntimeError("Peak RSS exceeded 1 GiB")
    return {"peak_RSS_bytes": counts.PeakWorkingSetSize, "RSS_bytes": counts.WorkingSetSize,
            "peak_private_commit_bytes": counts.PeakPagefileUsage, "private_commit_cap_bytes": 1024 ** 3,
            "CPU_threads": 1, "logical_CPU_index": 1, "affinity_mask": 2}


def definition_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_helper(source_root):
    return definition_module("r65_original_network_update_helper", Path(source_root) / "network_update_check" / "network_update_check.py")


def native_definitions(source_root):
    source_root = Path(source_root).resolve(strict=True)
    sys.path.insert(0, str(source_root))
    network = definition_module("network", source_root / "network.py")
    flow = definition_module("r65_native_source_flow", source_root / "source_flow_core.py")
    return network, flow


def assert_bits(torch, left, right, label):
    assert left.dtype == right.dtype and left.shape == right.shape, label + ": metadata"
    a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(a, b), label + ": byte mismatch"


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synthetic_native(torch, source_root, device, batch):
    """Construct new artificial weights and inputs; never load saved state."""
    network, flow = native_definitions(source_root)
    torch.manual_seed(650103)
    model = network.ResponseNetwork()
    with torch.no_grad():
        # The original constructor zeroes the final head; undo that only for
        # synthetic checks so upstream VJPs are nonzero at every layer.
        model.final.weight.normal_(0.0, 0.02)
        model.final.bias.fill_(0.005)
    model = model.to(device).train()
    gen = torch.Generator(device="cpu").manual_seed(650104)
    t = torch.arange(480, dtype=torch.float64)
    pulse = torch.exp(-((t - 140.0) / 20.0).square())
    source = (0.015 + torch.arange(batch, dtype=torch.float64)[:, None] * 0.0002) * pulse
    source += torch.randn((batch, 480), generator=gen, dtype=torch.float64) * 0.0002
    if batch >= 4:
        source[::4, 180] = 0.58  # exercise active Box projection, not only identity
    a = torch.zeros(480, dtype=torch.float64)
    a[:100], a[100:400] = -3.0, 1.0
    mu = (a / 1200.0).repeat(8, 1)
    scale = torch.full((8, 480), 0.02, dtype=torch.float64)
    b = torch.arange(batch, dtype=torch.int64) % 8
    noise = torch.randn((batch, 480), generator=gen, dtype=torch.float64) * 0.001
    noise = noise - mu[b] * flow.q_torch(noise)[:, None]
    q = flow.q_torch(source)
    assert bool((q > 0).all()) and bool((q.abs() < 300).all())
    h = torch.linspace(-0.7, 0.7, batch, dtype=torch.float64)
    gy = torch.randn((batch, 480), generator=gen, dtype=torch.float64) * 0.01
    tensors = [v.to(device) for v in (source, noise, q, b, h, mu, scale)]
    return model, flow, tensors, gy.to(device)
