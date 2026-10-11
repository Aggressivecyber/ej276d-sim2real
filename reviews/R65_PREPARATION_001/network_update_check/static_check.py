"""Own the lightweight preparation guard; never import or run the network fixture."""
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
PROJECT = ROOT.parents[2]
R64 = PROJECT / "work" / "R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001"
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(TEMP)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
if not sys.dont_write_bytecode or os.name != "nt":
    raise RuntimeError("Windows python -B required for static preparation")
MODE = sys.argv[1] if len(sys.argv) > 1 else "read"
if MODE not in ("read", "syntax", "validate"):
    raise ValueError("Use read, syntax or validate")
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
    "R65_protocol": PROJECT / "tasks" / "R65_FULL_EMPIRICAL_D_RESPONSE_001.md",
    "R64_network_artificial": R64 / "network_replay_artificial.py",
    "R64_original_bindings": R64 / "distribution_adapter" / "original_bindings.py",
    "R64_portable_fixture": R64 / "portable_fixture.py",
    "R64_distribution_adapter": R64 / "distribution_adapter" / "distribution_gradient.py",
    "R64_noise": R64 / "noise_core.py",
    "R63_config": PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001" / "config.json",
    "R65_objective_adapter": ROOT.parent / "objective_adapter" / "objective_adapter.py",
    "R65_bound_manifest": ROOT.parent / "objective_adapter" / "BOUND_SOURCES.json",
    "own_helper": ROOT / "network_update_check.py",
    "own_tolerances": ROOT / "FROZEN_TOLERANCES.json",
    "own_fixture": ROOT / "FIXTURE_SPEC.json",
    "bound_original_loss": ROOT.parent / "objective_adapter" / "bound_r64" / "r63_original" / "distribution_loss.py",
    "bound_base_loss": ROOT.parent / "objective_adapter" / "bound_r64" / "r63_original" / "R61_distribution_loss.py",
    "bound_precision": ROOT.parent / "objective_adapter" / "bound_r64" / "r63_original" / "precision_order.py",
    "bound_core": ROOT.parent / "objective_adapter" / "bound_r64" / "r63_original" / "source_flow_core.py",
    "bound_network": ROOT.parent / "objective_adapter" / "bound_r64" / "r63_original" / "network.py",
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
def validate():
    helper = SPECS["own_helper"]
    text = helper.read_text("utf-8-sig")
    tree = ast.parse(text, filename=str(helper))
    compile(tree, str(helper), "exec")
    own_static = ROOT / "static_check.py"
    compile(ast.parse(own_static.read_text("utf-8-sig")), str(own_static), "exec")
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    expected = {
        "parameter_vjp_cpu_f64": (["output", "parameters", "gradient"], []),
        "scalar_parameter_gradient_cpu_f64": (["scalar", "parameters"], ["retain_graph"]),
        "install_cpu_f64_clip_step": (["parameters", "gradient_cpu_f64", "optimizer"], ["clip", "capture"]),
        "run_check": (["device"], ["objective_adapter_root"]),
    }
    signatures = {}
    for name, (positional, keyword) in expected.items():
        node = functions[name]
        assert [a.arg for a in node.args.args] == positional
        assert [a.arg for a in node.args.kwonlyargs] == keyword
        signatures[name] = ast.unparse(node.args)
    assert ast.literal_eval(functions["run_check"].args.defaults[0]) == "cpu"
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(v.name for v in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module)
        else:
            assert isinstance(node, (ast.Expr, ast.Assign, ast.FunctionDef, ast.ClassDef, ast.If))
    allowed_imports = {"__future__", "contextlib", "copy", "gc", "hashlib", "importlib.util", "json", "pathlib", "sys", "time", "traceback"}
    assert set(top_imports) <= allowed_imports
    forbidden = {"write_text", "write_bytes", "mkdir", "unlink", "rmdir", "rename", "replace", "save", "load", "loadtxt",
                 "set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms", "set_fastpath_enabled"}
    bad = [(n.lineno, n.func.attr) for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]
    assert not bad, bad
    setter_assignments = [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign))
                          for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
                          if isinstance(target, ast.Attribute) and ("backends" in ast.unparse(target) or "environ" in ast.unparse(target))]
    assert not setter_assignments
    step_nodes = [n for n in ast.walk(functions["install_cpu_f64_clip_step"]) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == "step" for n in step_nodes) == 1
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == "clip_grad_norm_" for n in step_nodes) == 1
    public_metadata = ("pre_clip_norm", "did_clip", "cast_calls", "parameter_tensor_cast_calls", "clip_calls", "optimizer_calls", "optimizer_step_calls")
    constants = {n.value for n in ast.walk(functions["install_cpu_f64_clip_step"]) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert set(public_metadata) <= constants
    source_text = SPECS["R64_network_artificial"].read_text("utf-8-sig")
    source_tree = ast.parse(source_text)
    original_tolerance = next(ast.literal_eval(n.value) for n in ast.walk(source_tree) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "tolerances" for t in n.targets))
    tolerance = json.loads(SPECS["own_tolerances"].read_text("utf8"))
    fixture = json.loads(SPECS["own_fixture"].read_text("utf8"))
    for own, old in (("relative_L2", "gradient_relative_L2"), ("max_absolute", "gradient_max_absolute"),
                     ("max_relative_to_reference_max", "gradient_max_relative_to_reference_max"),
                     ("reference_norm_strictly_above", "nonzero_reference_gradient_norm_above")):
        assert tolerance["gradient"][own] == original_tolerance[old]
    assert tolerance["optimizer"] == {
        "learning_rate": 1e-6, "betas": [.9,.999], "epsilon": 1e-8, "weight_decay": 0., "gradient_clip": 5.,
        "synthetic_steps_per_arm": 1,
        "rationale": "Original R63 Adam settings at update 1, original clip threshold, one artificial step per arm. No scientific model or optimizer is constructed."}
    first = json.loads((ROOT/"READ_04.BINDING.json").read_text("utf8"))
    first_map = {v["id"]:v["sha256"] for v in first["files"]}
    for name in ("own_tolerances", "own_fixture"):
        assert sha(SPECS[name]) == first_map[name], "Frozen input changed since first static review: "+name
    for device in ("cpu", "cuda"):
        case = fixture[device]
        assert len(case["source_counts"]) == len(case["REAL_counts"]) == 10
        assert all(x > 0 for x in case["source_counts"]+case["REAL_counts"])
        assert sum(case["source_counts"]) == case["SOURCE_N"]
        assert sum(case["REAL_counts"]) == case["REAL_N"]
        assert case["schedule"] == [[j,min(j+case["chunk"],case["SOURCE_N"])] for j in range(0,case["SOURCE_N"],case["chunk"])]
        assert case["SOURCE_N"]%case["chunk"] == 1
    assert fixture["C_batch_per_region"] == 32 and fixture["C_identity_batch"] == 64 and fixture["C_neural_rows"] == 384
    root = ROOT.parent / "objective_adapter"
    bound_path = root / "BOUND_SOURCES.json"
    bound = json.loads(bound_path.read_text("utf8"))
    objective_files = []
    for item in [{"file":"objective_adapter.py","sha256":"1b8f38324b96212ca3cc6d83f603c89d8b18f731ab7227998a8da718122dfaec"},
                 {"file":"BOUND_SOURCES.json","sha256":"6116059ca68f737e1a2b303fe941eda31b9c286c609c75ebc3b69df3e603f475"}] + bound["copies"]:
        rel = Path(item["file"])
        assert not rel.is_absolute() and ".." not in rel.parts and rel.suffix in (".py", ".json")
        p = (root/rel).resolve(strict=True)
        assert root.resolve(strict=True) in p.parents
        digest = sha(p)
        assert digest == item["sha256"], str(p)
        objective_files.append({"file":item["file"],"sha256":digest})
    reviewed = [{"id":key,"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
                for key,path in SPECS.items() if not key.startswith("own_")]
    binding = {
        "created_UTC": datetime.now(timezone.utc).isoformat(),
        "helper_file": helper.name, "helper_sha256": sha(helper),
        "own_inputs": {SPECS[k].name:sha(SPECS[k]) for k in ("own_tolerances", "own_fixture")},
        "objective_files": objective_files, "read_only_reviewed_sources": reviewed,
        "source_binding_scope": "Only source code and JSON configuration/metadata; no arrays or weights",
        "frozen_before_any_local_or_known_server_execution": True,
    }
    write_json("SOURCE_BINDING.json", binding)
    for record in reviewed:
        assert sha(Path(record["path"])) == record["sha256"]
    receipt = {
        "status":"STATIC_PREPARATION_PASSED_RUNTIME_UNEXECUTED",
        "helper_sha256":sha(helper), "source_binding_sha256":sha(ROOT/"SOURCE_BINDING.json"),
        "tolerances_sha256":sha(SPECS["own_tolerances"]), "fixture_spec_sha256":sha(SPECS["own_fixture"]),
        "static_checker_sha256":sha(own_static), "API":signatures, "public_update_metadata":list(public_metadata),
        "AST_parse_compile_pass":True,"static_checker_imports_helper":False,"top_level_imports":top_imports,
        "helper_file_writes_found":False,"helper_backend_or_thread_setters_found":False,
        "R64_network_gradient_tolerances_exact":True,"frozen_inputs_unchanged_since_first_static_review":True,
        "objective_definition_bindings_verified":len(objective_files),"reviewed_sources_unchanged":True,
        "CPU_runtime_status":"UNEXECUTED_REQUIRE_ROOT_SERVER","CUDA_runtime_status":"UNEXECUTED_REQUIRE_ROOT_SERVER",
        "artificial_network_executions":0,"synthetic_optimizer_steps_executed":0,
        "scientific_array_or_weight_files_read":0,"scientific_updates":0,
        "root_metadata_or_Git_writes":0,"resource":memory(),"CPU_affinity_count":1,
        "process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,
        "python_B":bool(sys.dont_write_bytecode),"TEMP":str(TEMP),
        "limitations":"AST checks cannot establish runtime correctness or resource fit; actual CPU384/CUDA checks remain for the root-owned supervised server",
    }
    write_json("STATIC_RECEIPT.json",receipt)
    own = [{"file":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in
           (helper,SPECS["own_tolerances"],SPECS["own_fixture"],ROOT/"SOURCE_BINDING.json",ROOT/"STATIC_RECEIPT.json",ROOT/"README.md",own_static)]
    preparation = {"status":receipt["status"],"files":own,"resource":memory(),
                   "CPU_affinity_count":1,"process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,
                   "actual_network_executions":0,"synthetic_optimizer_steps_executed":0,"scientific_inputs_read":0,
                   "CPU_status":receipt["CPU_runtime_status"],"CUDA_status":receipt["CUDA_runtime_status"],
                   "root_metadata_or_Git_writes":0,"external_definition_files_unchanged":True}
    write_json("PREPARATION_RECEIPT.json",preparation)
    for item in own:
        assert sha(ROOT/item["file"]) == item["sha256"]
    print(json.dumps({"status":receipt["status"],"helper_sha256":sha(helper),
                      "source_binding_sha256":sha(ROOT/"SOURCE_BINDING.json"),
                      "preparation_receipt_sha256":sha(ROOT/"PREPARATION_RECEIPT.json"),
                      "bindings_verified":len(objective_files),"runtime_executions":0,
                      "peak_RSS_bytes":memory()["peak_RSS_bytes"]}))

try:
    if MODE == "read":
        request = json.loads((ROOT/sys.argv[2]).read_text("utf8")) if len(sys.argv) > 2 else [{"id": key} for key in SPECS]
        result, inputs = {}, []
        for item in request:
            key = item["id"]
            path = SPECS[key]
            value = path.read_text("utf-8-sig")
            initial = sha(path)
            inputs.append({"id": key, "path": str(path), "bytes": path.stat().st_size, "sha256": initial})
            lines = value.splitlines()
            start, end = item.get("lines", [1, len(lines)])
            result[key] = ([{"name": x.name, "start": x.lineno, "stop": x.end_lineno} for x in ast.walk(ast.parse(value))
                            if isinstance(x, (ast.FunctionDef, ast.ClassDef))]
                           if item.get("definitions") else "\n".join(f"{i+1}: {v}" for i,v in enumerate(lines) if start <= i+1 <= end))
            if sha(path) != initial:
                raise RuntimeError("Source changed during read: "+key)
        name = Path(sys.argv[2]).stem if len(sys.argv) > 2 else "SOURCE_READ"
        write_json(name+".OUTPUT.json", result)
        write_json(name+".BINDING.json", {"files": inputs, "source_unchanged_during_read": True})
        print(json.dumps(result, ensure_ascii=False))
    elif MODE == "syntax":
        tree = ast.parse(SPECS["own_helper"].read_text("utf-8-sig"), filename=str(SPECS["own_helper"]))
        compile(tree, str(SPECS["own_helper"]), "exec")
        result = {"status": "STATIC_PARSE_COMPILE_ONLY_PASS", "helper_sha256": sha(SPECS["own_helper"]),
                  "definitions": [{"name": n.name, "line": n.lineno, "end": n.end_lineno} for n in ast.walk(tree)
                                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                  "runtime_imports_or_network_checks_executed": 0}
        write_json("SYNTAX_RECEIPT.json", result)
        print(json.dumps(result))
    else:
        validate()
    m = memory()
    if m["peak_RSS_bytes"] > LIMIT:
        raise RuntimeError("Resource limit exceeded")
    write_json(MODE+".RESOURCE_RECEIPT.json", {"status": "COMPLETE", "UTC": datetime.now(timezone.utc).isoformat(), "PID": os.getpid(),
                "elapsed_seconds": time.monotonic()-START, "CPU_count": 1, "CPU_affinity_mask": one_cpu,
                "process_commit_limit_bytes": LIMIT, "RSS_watchdog_limit_bytes": LIMIT, "RSS_poll_seconds": .05,
                "Python_B": bool(sys.dont_write_bytecode), "TEMP": str(TEMP), **m,
                "artificial_network_executions": 0, "synthetic_optimizer_steps_executed": 0,
                "scientific_file_reads": 0, "scientific_updates": 0, "root_metadata_or_Git_writes": 0})
finally:
    stop.set()
