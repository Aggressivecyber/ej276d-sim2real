"""Local Windows resource owner and transport for root-reviewed R65 controls.

Importing this module performs no network, data, source-import, or filesystem
mutation. Runtime work begins only inside an entry point after root invokes it.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

STAGE = "R65_FULL_EMPIRICAL_D_RESPONSE_001"
DEFAULT_PROJECT = Path("E:/GAN/research/sim2real_two_papers_20261008_0305")
REMOTE_ROOT = "/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC"
REVIEW_SHA = "e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5"
HOST = "rylan@100.82.231.102"
ENTRIES = ("train_final.py", "numerical_check.py", "generate_all.py", "evaluate_saved.py",
           "replay_all.py", "summarize_and_plot.py")
POST_ENTRIES = ENTRIES[1:]
HERE = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Empty relative path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and all(
        part not in ("", ".", "..") and re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in path.parts), "Unsafe relative path: " + repr(name))
    return name


def local_path(root, name):
    path = root / safe_name(name)
    require(path.resolve().is_relative_to(root.resolve()), "Local path leaves intended directory")
    require(not path.is_symlink(), "Refusing symlink destination")
    return path


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_latest(path, value):
    # All durable per-operation records use write_new. Only this explicitly
    # identified display pointer is atomically replaced.
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_new(temp, value)
    os.replace(temp, path)


def parser(description, with_entry=False, allow_train=False):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    if with_entry:
        result.add_argument("entry", choices=ENTRIES if allow_train else POST_ENTRIES)
    return result


class ResourceOwner:
    """Actual CPU1, Windows Job commit1GiB, and RSS1GiB guard for controller."""
    def __init__(self, project, command):
        require(os.name == "nt" and sys.dont_write_bytecode, "Use Windows python -B")
        project = project.resolve(strict=True)
        require(project.drive.upper() in ("E:", "D:"), "Controller data must remain on E or D")
        self.stage = project / "work" / STAGE
        require(self.stage.is_dir(), "Frozen local stage directory not found")
        self.root = self.stage / "controller_runtime"
        self.root.mkdir(exist_ok=True)
        self.operation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
        self.command = command
        self.start = time.monotonic()
        self.limit = 1024 ** 3
        self.stop = threading.Event()
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.process = self.k32.GetCurrentProcess()
        ptr = ctypes.c_size_t
        self.k32.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ptr), ctypes.POINTER(ptr))
        self.k32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ptr)
        pmask, smask = ptr(), ptr()
        require(self.k32.GetProcessAffinityMask(self.process, ctypes.byref(pmask), ctypes.byref(smask)), "Read CPU affinity failed")
        self.cpu_mask = pmask.value & -pmask.value
        require(self.k32.SetProcessAffinityMask(self.process, self.cpu_mask), "Set CPU1 affinity failed")

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ptr), ("MaximumWorkingSetSize", ptr),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ptr), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ptr),
                        ("JobMemoryLimit", ptr), ("PeakProcessMemoryUsed", ptr), ("PeakJobMemoryUsed", ptr)]

        class MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ptr) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

        self.MEMORY_COUNTERS = MEMORY_COUNTERS
        self.psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(MEMORY_COUNTERS), wintypes.DWORD)
        self.k32.CreateJobObjectW.restype = wintypes.HANDLE
        self.k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self.k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        self.k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.job = self.k32.CreateJobObjectW(None, None)
        limit = EXTENDED_LIMIT()
        limit.BasicLimitInformation.LimitFlags = 0x100
        limit.ProcessMemoryLimit = self.limit
        require(self.job and self.k32.SetInformationJobObject(self.job, 9, ctypes.byref(limit), ctypes.sizeof(limit)), "Set Job commit limit failed")
        require(self.k32.AssignProcessToJobObject(self.job, self.process), "Assign controller to Job failed")
        self.temp = self.root / "tmp" / self.operation
        self.temp.mkdir(parents=True)
        for key in ("TEMP", "TMP", "TMPDIR"):
            os.environ[key] = str(self.temp)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        threading.Thread(target=self._watch, daemon=True).start()

    def memory(self):
        row = self.MEMORY_COUNTERS()
        row.cb = ctypes.sizeof(row)
        require(self.psapi.GetProcessMemoryInfo(self.process, ctypes.byref(row), row.cb), "Actual RSS probe failed")
        return {"RSS_bytes": int(row.WorkingSetSize), "peak_RSS_bytes": int(row.PeakWorkingSetSize),
                "private_bytes": int(row.PrivateUsage), "peak_commit_bytes": int(row.PeakPagefileUsage)}

    def _watch(self):
        while not self.stop.wait(.05):
            try:
                value = self.memory()
                if value["peak_RSS_bytes"] > self.limit:
                    write_new(self.root / (self.operation + ".RESOURCE_FAILURE.json"), value)
                    os._exit(91)
            except BaseException:
                os._exit(92)

    def finish(self, status, error=None):
        self.stop.set()
        value = self.memory()
        write_new(self.root / (self.operation + ".RESOURCE_RECEIPT.json"), {
            "UTC": utc(), "PID": os.getpid(), "command": self.command, "status": status, "error": error,
            "elapsed_seconds": time.monotonic() - self.start, "CPU_affinity_count": 1,
            "CPU_affinity_mask": self.cpu_mask, "process_commit_limit_bytes": self.limit,
            "RSS_watchdog_limit_bytes": self.limit, "RSS_poll_seconds": .05, "Python_B": True,
            "TEMP": str(self.temp), **value})


def verify_controller_bundle():
    seal = json.loads((HERE / "CONTROLLER_BINDING.json").read_text("utf8"))
    require(seal["implementation_review_SHA256"] == REVIEW_SHA and seal["frozen_file_count"] == 112,
            "Controller frozen review binding differs")
    for row in seal["files"]:
        path = local_path(HERE, row["name"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Controller source changed: " + row["name"])
    return seal


def verify_local_stage(owner):
    path = owner.stage / "ROOT_IMPLEMENTATION_REVIEW.json"
    require(sha(path) == REVIEW_SHA, "Local implementation review changed")
    review = json.loads(path.read_text("utf8"))
    rows = review["files"]
    require(len(rows) == 112 and len({row["name"] for row in rows}) == 112, "Local frozen list differs")
    # The running controller only needs to read config + metadata locally.
    # Remote frozen bytes are independently verified before every transition.
    cfg = json.loads((owner.stage / "config.json").read_text("utf8"))
    config_row = next(row for row in rows if row["name"] == "config.json")
    require(sha(owner.stage / "config.json") == config_row["sha256"] and cfg["remote_root"] == REMOTE_ROOT,
            "Local config binding differs")
    return {row["name"]: row for row in rows}


def remote_call(owner, mode, entry=None):
    verify_controller_bundle()
    verify_local_stage(owner)
    request = {"mode": mode, "entry": entry, "remote_root": REMOTE_ROOT, "review_SHA256": REVIEW_SHA}
    payload = (HERE / "r65_remote_gate.py").read_text("utf8")
    # The future import stays first. REQUEST is inserted immediately after it;
    # JSON is embedded as a Python string literal, never a shell expression.
    marker = "from __future__ import annotations\n"
    require(payload.count(marker) == 1, "Remote payload future-import marker differs")
    payload = payload.replace(marker, marker + "\nREQUEST = " + repr(request) + "\n", 1)
    operation = owner.operation + "_" + mode + "_" + uuid.uuid4().hex[:8]
    write_new(owner.root / (operation + ".REQUEST.json"), {"UTC": utc(), "request": request,
              "remote_payload_SHA256": hashlib.sha256(payload.encode()).hexdigest(), "completion_claim": False})
    command = ["ssh", "-T", "-o", "ForwardX11=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", HOST, "/usr/bin/python3 -B -"]
    try:
        completed = subprocess.run(command, input=payload, encoding="utf8", errors="strict",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False)
    except subprocess.TimeoutExpired as error:
        write_new(owner.root / (operation + ".UNKNOWN_TRANSPORT_OUTCOME.json"), {
            "UTC": utc(), "request": request, "error": repr(error), "remote_outcome": "unknown",
            "next_action": "Observe; never retry a launch from a timeout."})
        raise RuntimeError("SSH timed out; remote outcome is unknown. Poll before any manual action.") from error
    (owner.root / (operation + ".stdout.log")).write_text(completed.stdout, encoding="utf8")
    (owner.root / (operation + ".stderr.log")).write_text(completed.stderr, encoding="utf8")
    write_new(owner.root / (operation + ".SSH_NATIVE_RECEIPT.json"), {"UTC": utc(), "native_exit": completed.returncode,
              "request": request, "is_scientific_native_exit": False})
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, "Unexpected SSH output; outcome requires inspection")
    result = json.loads(lines[0])
    require(completed.returncode == 0 and result.get("ok") is True,
            "Remote gate refused: " + str(result.get("error", "transport/native failure")))
    value = result["result"]
    write_new(owner.root / (operation + ".RESULT.json"), value)
    write_latest(owner.root / "LATEST_REMOTE_OBSERVATION.json", {"operation": operation, "mode": mode, "result": value})
    return value


def launch_entry(owner, entry):
    require(entry in POST_ENTRIES, "This controller cannot launch training or recovery")
    before = remote_call(owner, "observe")
    require(before.get("state") == "READY" and before.get("ready_entry") == entry,
            "Requested post entry is not yet eligible; no launch intent was created")
    marker = owner.root / ("LOCAL_LAUNCH_" + entry + ".json")
    # An exclusive local intent record plus an exclusive remote intent record
    # prevent retry after uncertain transport, failure, or a second watcher.
    write_new(marker, {"UTC": utc(), "entry": entry, "operation": owner.operation,
                       "status": "INTENT_ONLY", "automatic_retry": False})
    result = remote_call(owner, "launch", entry)
    write_new(owner.root / ("LOCAL_LAUNCH_" + entry + ".SPAWN.json"), result)
    return result


def fetch_entry(owner, entry):
    require(entry in ENTRIES, "Invalid requested terminal entry")
    manifest = remote_call(owner, "manifest", entry)
    frozen = verify_local_stage(owner)
    protected = set(frozen) | {"ROOT_IMPLEMENTATION_REVIEW.json"}
    operation_root = owner.root / ("FETCH_" + owner.operation)
    operation_root.mkdir()
    write_new(operation_root / "REMOTE_MANIFEST.json", manifest)
    rows = manifest["files"]
    require(len({row["name"] for row in rows}) == len(rows), "Duplicate fetch filename")
    require(protected.issubset({row["name"] for row in rows}), "Remote manifest omits frozen files")
    actions = []
    # Preflight every destination before changing any local canonical output.
    for row in rows:
        name = safe_name(row["name"])
        require(isinstance(row["bytes"], int) and row["bytes"] >= 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "Malformed remote file record")
        dest = local_path(owner.stage, name)
        if name in protected and dest.exists():
            require(sha(dest) == row["sha256"], "Local protected frozen file differs: " + name)
        require(not name.startswith("controller_runtime/"), "Remote files collide with local controller evidence")
    for row in rows:
        name = row["name"]
        dest = local_path(owner.stage, name)
        if dest.is_file() and dest.stat().st_size == row["bytes"] and sha(dest) == row["sha256"]:
            actions.append({"name": name, "action": "existing_identical", "sha256": row["sha256"]})
            continue
        require(name not in protected or not dest.exists(), "Refusing protected-file replacement")
        staged = local_path(operation_root / "downloaded", name)
        staged.parent.mkdir(parents=True, exist_ok=True)
        source = HOST + ":" + REMOTE_ROOT + "/" + name
        # Manifest paths permit only safe POSIX segments. No shell=True or
        # wildcard command expansion is used. Each transfer has a finite timeout.
        command = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", source, str(staged)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=60, check=False)
        require(completed.returncode == 0, "SCP native failure; partial download retained: " + name)
        require(staged.stat().st_size == row["bytes"] and sha(staged) == row["sha256"], "Downloaded bytes differ: " + name)
        action = {"name": name, "action": "new_file", "sha256": row["sha256"], "bytes": row["bytes"]}
        if dest.exists():
            require(dest.is_file(), "Destination is not a regular file")
            previous_sha = sha(dest)
            archive = local_path(operation_root / "previous_local_versions", name)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, archive)
            require(sha(archive) == previous_sha, "Local previous-version archive failed verification")
            action.update(action="previous_version_preserved_then_replaced", previous_sha256=previous_sha,
                          previous_archive=str(archive))
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Target path was resolved and checked within owner.stage above.
        os.replace(staged, dest)
        require(sha(dest) == row["sha256"], "Canonical fetched bytes failed verification")
        actions.append(action)
        write_latest(operation_root / "FETCH_PROGRESS.json", {"UTC": utc(), "completed_files": len(actions), "total_files": len(rows)})
    # Recheck remote native identity and the full manifest after the transfer.
    after = remote_call(owner, "manifest", entry)
    require(after["files"] == rows and after["verified_terminal"]["active"] == manifest["verified_terminal"]["active"],
            "Remote stage changed during fetch; fetched evidence retained without a completion claim")
    result = {"UTC": utc(), "status": "COMPLETE", "terminal_entry": entry, "verified_files": len(rows),
              "actions": actions, "remote_manifest_unchanged": True, "previous_versions_preserved": True,
              "scientific_model_goal_complete": False, "operation_directory": str(operation_root)}
    write_new(operation_root / "FETCH_RECEIPT.json", result)
    return result


def run_owned(args, command, action):
    owner = ResourceOwner(args.project_root, command)
    try:
        result = action(owner)
        owner.finish("COMPLETE")
        return result
    except BaseException as error:
        owner.finish("FAILED_OR_UNKNOWN", repr(error))
        raise
