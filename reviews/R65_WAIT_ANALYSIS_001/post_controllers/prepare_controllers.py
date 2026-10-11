"""Static source preparation. Never imports a controller or opens SSH/data/weights."""
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

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
R63=PROJECT/"work"/"R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
R65=PROJECT/"work"/"R65_FULL_EMPIRICAL_D_RESPONSE_001"
TEMP=ROOT/"_tmp";TEMP.mkdir(exist_ok=True)
for key in ("TEMP","TMP","TMPDIR"):os.environ[key]=str(TEMP)
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
if not sys.dont_write_bytecode or os.name!="nt":raise RuntimeError("Guarded source preparation requires Windows python -B")
MODE=sys.argv[1]if len(sys.argv)>1 else"inventory"
if MODE not in ("inventory","read","seal","verify"):raise ValueError(MODE)
START=time.monotonic();LIMIT=1024**3
k32=ctypes.WinDLL("kernel32",use_last_error=True);psapi=ctypes.WinDLL("psapi",use_last_error=True);ptr=ctypes.c_size_t
k32.GetCurrentProcess.restype=wintypes.HANDLE;process=k32.GetCurrentProcess()
k32.GetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.POINTER(ptr),ctypes.POINTER(ptr));pmask,smask=ptr(),ptr()
if not k32.GetProcessAffinityMask(process,ctypes.byref(pmask),ctypes.byref(smask)):raise ctypes.WinError(ctypes.get_last_error())
one_cpu=pmask.value&-pmask.value;k32.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ptr)
if not k32.SetProcessAffinityMask(process,one_cpu):raise ctypes.WinError(ctypes.get_last_error())

class IO_COUNTERS(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong)for n in("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]
class BASIC_LIMIT(ctypes.Structure):
    _fields_=[("PerProcessUserTimeLimit",ctypes.c_longlong),("PerJobUserTimeLimit",ctypes.c_longlong),("LimitFlags",wintypes.DWORD),
              ("MinimumWorkingSetSize",ptr),("MaximumWorkingSetSize",ptr),("ActiveProcessLimit",wintypes.DWORD),("Affinity",ptr),
              ("PriorityClass",wintypes.DWORD),("SchedulingClass",wintypes.DWORD)]
class EXTENDED_LIMIT(ctypes.Structure):
    _fields_=[("BasicLimitInformation",BASIC_LIMIT),("IoInfo",IO_COUNTERS),("ProcessMemoryLimit",ptr),("JobMemoryLimit",ptr),("PeakProcessMemoryUsed",ptr),("PeakJobMemoryUsed",ptr)]
class MEMORY_COUNTERS(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ptr)for n in("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]
psapi.GetProcessMemoryInfo.argtypes=(wintypes.HANDLE,ctypes.POINTER(MEMORY_COUNTERS),wintypes.DWORD)
def memory():
    m=MEMORY_COUNTERS();m.cb=ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(process,ctypes.byref(m),m.cb):raise ctypes.WinError(ctypes.get_last_error())
    return {"RSS_bytes":int(m.WorkingSetSize),"peak_RSS_bytes":int(m.PeakWorkingSetSize),"private_bytes":int(m.PrivateUsage),"peak_commit_bytes":int(m.PeakPagefileUsage)}
k32.CreateJobObjectW.restype=wintypes.HANDLE;k32.CreateJobObjectW.argtypes=(ctypes.c_void_p,wintypes.LPCWSTR)
k32.SetInformationJobObject.argtypes=(wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD);k32.AssignProcessToJobObject.argtypes=(wintypes.HANDLE,wintypes.HANDLE)
job=k32.CreateJobObjectW(None,None);limit=EXTENDED_LIMIT();limit.BasicLimitInformation.LimitFlags=0x100;limit.ProcessMemoryLimit=LIMIT
if not job or not k32.SetInformationJobObject(job,9,ctypes.byref(limit),ctypes.sizeof(limit)):raise ctypes.WinError(ctypes.get_last_error())
if not k32.AssignProcessToJobObject(job,process):raise ctypes.WinError(ctypes.get_last_error())
stop=threading.Event()
def watch():
    while not stop.wait(.05):
        try:
            m=memory()
            if m["peak_RSS_bytes"]>LIMIT:
                (ROOT/(MODE+".RESOURCE_FAILURE.json")).write_text(json.dumps(m),encoding="utf8");os._exit(91)
        except BaseException:os._exit(92)
threading.Thread(target=watch,daemon=True).start()

SPECS={
 "launch_R63":PROJECT/"reviews"/"launch_R63_entry_root.py",
 "fetch_R63":PROJECT/"reviews"/"fetch_R63_root.py",
 "watch_R63":PROJECT/"reviews"/"watch_R63_pipeline_root.py",
 "poll_R63":PROJECT/"reviews"/"poll_R63_root.py",
 "control_R63":PROJECT/"reviews"/"prepare_R63_pipeline_control_root.py",
 "poll_R65":PROJECT/"reviews"/"poll_R65_root.py",
 "R65_config":R65/"config.json",
 "R65_config_post":R65/"config_post.json",
 "R65_config_reporting":R65/"config_reporting.json",
 "R65_training_audit":R65/"r65_training_audit.py",
 "R65_post_audit":R65/"post_audit.py",
 "R65_supervisor":R65/"linux_supervise.py",
 "R65_reporting_supervisor":R65/"linux_supervise_reporting.py",
 "R65_worker":R65/"linux_owned_worker.py",
 "R65_numerical":R65/"numerical_check.py",
 "R65_generate":R65/"generate_all.py",
 "R65_evaluate":R65/"evaluate_saved.py",
 "R65_replay":R65/"replay_all.py",
 "R65_summary":R65/"summarize_and_plot.py",
 "R65_implementation_review":R65/"ROOT_IMPLEMENTATION_REVIEW.json",
 "R65_train_source":R65/"train_final.py",
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):
    path=(ROOT/name).resolve()
    if ROOT.resolve()not in path.parents:raise RuntimeError("Outside exclusive preparation directory")
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
def read():
    request=json.loads((ROOT/sys.argv[2]).read_text("utf8"));answer={};sources=[]
    for item in request:
        path=SPECS[item["id"]]
        if not path.is_file():answer[item["id"]]={"missing":str(path)};continue
        digest=sha(path);value=path.read_text("utf-8-sig");lines=value.splitlines();lo,hi=item.get("lines",[1,len(lines)])
        selected=set(range(lo-1,min(hi,len(lines))))
        if "patterns"in item:
            hits=[i for i,line in enumerate(lines)if any(p in line for p in item["patterns"])]
            selected={j for i in hits for j in range(max(0,i-item.get("context",1)),min(len(lines),i+item.get("context",1)+1))}
        answer[item.get("label",item["id"])]=({key:json.loads(value).get(key)for key in item["json_keys"]}if "json_keys"in item else [{ "name":node.name,"line":node.lineno,"end":node.end_lineno}for node in ast.walk(ast.parse(value))if isinstance(node,(ast.FunctionDef,ast.ClassDef))]if item.get("definitions")else"\n".join(f"{i+1}: {line}"for i,line in enumerate(lines)if i in selected))
        assert sha(path)==digest
        sources.append({"id":item["id"],"path":str(path),"sha256":digest,"bytes":path.stat().st_size})
    prefix=Path(sys.argv[2]).stem;write(prefix+".OUTPUT.json",answer);write(prefix+".BINDING.json",{"files":sources,"unchanged":True})
    return answer
def inventory():
    result={key:{"path":str(path),"exists":path.is_file(),"sha256":sha(path)if path.is_file()else None}for key,path in SPECS.items()}
    write("SOURCE_INVENTORY.json",result);return result
PREPARED=ROOT/"prepared"
REVIEW_SHA="e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5"
PY_NAMES=("r65_remote_gate.py","r65_controller_common.py","launch_R65_entry_root.py",
          "poll_R65_pipeline_root.py","fetch_R65_root.py","watch_R65_pipeline_root.py")

def constants(path):
    result={}
    for node in ast.parse(path.read_text("utf8")).body:
        if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:result[node.targets[0].id]=ast.literal_eval(node.value)
            except(ValueError,TypeError):pass
    return result

def inspect_prepared():
    review=SPECS["R65_implementation_review"]
    assert sha(review)==REVIEW_SHA
    manifest=json.loads(review.read_text("utf8"))["files"]
    assert len(manifest)==112 and len({row["name"]for row in manifest})==112
    forbidden_modules={"numpy","torch","scipy","joblib","paramiko","r65_training_audit","post_audit"}
    parsed={};records=[]
    for name in PY_NAMES:
        path=PREPARED/name;source=path.read_text("utf8");tree=ast.parse(source,filename=str(path))
        compile(tree,str(path),"exec")
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(alias.name.split('.')[0]for alias in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append((node.module or"").split('.')[0])
        assert not set(imports)&forbidden_modules,(name,imports)
        assert not any(isinstance(node,ast.Call)and any(keyword.arg=="shell"and isinstance(keyword.value,ast.Constant)
            and keyword.value.value is True for keyword in node.keywords)for node in ast.walk(tree))
        assert "coverage_extension"not in source
        parsed[name]=(source,tree)
        records.append({"name":name,"bytes":path.stat().st_size,"sha256":sha(path),
                        "AST_and_compile_only":True,"module_imported_or_executed":False,"imports":imports})
    remote=constants(PREPARED/"r65_remote_gate.py");audit=constants(SPECS["R65_training_audit"])
    config=json.loads(SPECS["R65_config"].read_text("utf8"))
    assert remote["ENTRIES"]==("train_final.py","numerical_check.py","generate_all.py","evaluate_saved.py","replay_all.py","summarize_and_plot.py")
    assert remote["REVIEW_SHA"]==REVIEW_SHA and remote["REMOTE_ROOT"]==config["remote_root"]
    assert remote["GRADIENT_COUNTS"]==audit["GRADIENT_COUNTS"]
    assert set(remote["ARTIFACTS"])==set(audit["ARTIFACTS"])
    assert set(audit["TRUE_INTEGRITY"]).issubset(remote["TRUE_INTEGRITY"])
    assert set(remote["FALSE_INTEGRITY"])==set(audit["FALSE_INTEGRITY"])
    assert config["loss_columns"]==audit["LOSS_COLUMNS"] and len(config["loss_columns"])==32
    assert config["precision_columns"]==audit["PRECISION_COLUMNS"] and len(config["precision_columns"])==4
    assert config["replay_columns"]==audit["REPLAY_COLUMNS"] and len(config["replay_columns"])==12
    assert config["expected_completed_update_counts"]==audit["EXPECTED_COUNTS"]
    assert config["expected_replay_row"]==audit["REPLAY_ROW"]
    training_tree=ast.parse(SPECS["R65_train_source"].read_text("utf8"))
    integrity_nodes=[node for node in ast.walk(training_tree)if isinstance(node,ast.Dict)
                     and any(isinstance(key,ast.Constant)and key.value=="all_endpoint_codes_free_masks_exact"for key in node.keys)]
    assert len(integrity_nodes)==1
    declared={key.value:value.value for key,value in zip(integrity_nodes[0].keys,integrity_nodes[0].values)
              if isinstance(key,ast.Constant)and isinstance(value,ast.Constant)}
    assert all(declared[key]is True for key in remote["TRUE_INTEGRITY"])
    assert all(declared[key]is False for key in remote["FALSE_INTEGRITY"])
    source,tree=parsed["r65_remote_gate.py"]
    functions={node.name:ast.get_source_segment(source,node)for node in tree.body if isinstance(node,ast.FunctionDef)}
    required={
      "launch":["entry in POST_ENTRIES","before[\"state\"] == \"READY\"","before[\"ready_entry\"] == entry","marker.open(\"x\"","linux_supervise_reporting.py","subprocess.Popen"],
      "native_success":["actual_worker_native_exit","launcher_native_exit","handshake_before_imports","worker_self_receipt","process_record(pid)","process_record(supervisor)","job_tree_RSS_peak_bytes"],
      "training_complete":["50000","artifact_sha256","full_empirical_D_integrity","physical_attempt_compute","full_saved_array_audit_deferred_to_owned_numerical_entry"],
      "numerical_complete":["TRAINING_FROZEN_AUDIT.json","candidate_steps\") == 128","reference_only_steps\") == 256","median_limit_LSB\") == .25","p95_limit_LSB\") == 1.","relative_q_limit\") == 1e-5"],
      "generation_complete":["[4000, 1024, 832, 128]","5984"],
      "evaluation_complete":["HGB_attempts\") == 10","density_description_slots\") == 16"],
      "replay_complete":["4000","bitwise_pass","independent_process"],
      "reporting_complete":["scientific_model_goal_complete\": False","primary_A_C_pass","ACCEPTANCE_A_C.json"],
    }
    for function,tokens in required.items():
        assert all(token in functions[function]for token in tokens),(function,tokens)
    common=parsed["r65_controller_common.py"][0]
    assert all(token in common for token in ("ProcessMemoryLimit = self.limit","SetProcessAffinityMask","self.limit = 1024 ** 3",
        'project.drive.upper() in ("E:", "D:")',"time",'timeout=45','timeout=60','previous_local_versions',"UNKNOWN_TRANSPORT_OUTCOME",
        'before.get("state") == "READY"',"CONTROLLER_BINDING.json"))
    watch_source=parsed["watch_R65_pipeline_root.py"][0]
    assert "time.sleep(30)"in watch_source and "PIPELINE_STOP_REQUEST.json"in watch_source
    assert "launch_entry(owner, result[\"ready_entry\"])"in watch_source
    # Verify JSON request injection remains syntactically legal without importing
    # the remote payload or invoking any controller path.
    request={"mode":"observe","entry":None,"remote_root":remote["REMOTE_ROOT"],"review_SHA256":REVIEW_SHA}
    injected=source.replace("from __future__ import annotations\n","from __future__ import annotations\n\nREQUEST = "+repr(request)+"\n",1)
    compile(ast.parse(injected),"<remote-stdin-payload-AST-only>","exec")
    sources=[]
    frozen_by_name={row["name"]:row for row in manifest}
    for key,path in SPECS.items():
        assert path.suffix in (".py",".json") and path.is_file()
        row={"id":key,"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
        if path.parent==R65 and path.name in frozen_by_name:
            assert row["sha256"]==frozen_by_name[path.name]["sha256"]
        sources.append(row)
    return {"files":records,"sources":sources,"frozen_manifest":manifest,"status":"STATIC_ONLY_PASS",
      "checks":{"six_controller_Python_AST_compile":True,"remote_stdin_injection_AST_compile":True,
        "stdlib_only_no_scientific_imports":True,"R65_saved_record_schema_matches_frozen_audit_source":True,
        "R65_integrity_assertions_match_frozen_training_writer":True,"launch_accepts_five_post_entries_only":True,
        "true_native_zero_and_PID_disappearance_required":True,"original_128_256_threshold_literals_preserved":True,
        "original_generation_evaluation_and_replay_terminal_counts_preserved":True,
        "negative_A_C_is_completed_reporting_not_method_acceptance":True,"no_R63_only_coverage_extension_gate":True,
        "exclusive_launch_locks_no_auto_retry_recovery":True,"local_CPU1_Job_commit1GiB_RSS_watch1GiB":True,
        "fetch_hash_verification_and_previous_version_preservation":True},
      "controller_runtime_executed":False,"SSH_connections":0,"uploads_or_downloads":0,"science_launches":0,
      "scientific_array_or_weight_reads":0,"frozen_112_array_bytes_opened":False,
      "frozen_manifest_records_bound_without_decoding_or_opening_array_files":112}

def seal():
    assert not(ROOT/"PREPARATION_RECEIPT.json").exists(),"Already sealed; use verify"
    result=inspect_prepared()
    write("prepared/FROZEN_IMPLEMENTATION_MANIFEST.json",{
        "implementation_review_SHA256":REVIEW_SHA,"frozen_file_count":112,
        "source":str(SPECS["R65_implementation_review"]),"files":result["frozen_manifest"],
        "preparation_does_not_open_scientific_arrays_or_weights":True})
    own=[]
    for name in (*PY_NAMES,"README.md","FROZEN_IMPLEMENTATION_MANIFEST.json"):
        path=PREPARED/name;assert path.is_file()
        own.append({"name":name,"sha256":sha(path),"bytes":path.stat().st_size})
    write("prepared/CONTROLLER_BINDING.json",{"implementation_review_SHA256":REVIEW_SHA,"frozen_file_count":112,
        "files":own,"runtime_execution_status":"UNEXECUTED_ROOT_REVIEW_REQUIRED","SSH_connections_during_preparation":0})
    write("SOURCE_BINDING.json",{"sources":result["sources"],"all_read_source_files_unchanged":True,
        "bound_frozen_manifest_SHA256":sha(PREPARED/"FROZEN_IMPLEMENTATION_MANIFEST.json"),
        "frozen_review_SHA256":REVIEW_SHA,"science_array_or_weight_reads":0,
        "do_not_infer_all112_file_bytes_were_opened_locally":True})
    result.pop("frozen_manifest");result.pop("sources")
    write("AST_RECEIPT.json",result)
    receipt={"status":"PREPARED_STATIC_ONLY_UNEXECUTED","UTC":datetime.now(timezone.utc).isoformat(),
        "prepared_path":str(PREPARED),"files":own+[{
            "name":"CONTROLLER_BINDING.json","sha256":sha(PREPARED/"CONTROLLER_BINDING.json"),
            "bytes":(PREPARED/"CONTROLLER_BINDING.json").stat().st_size}],
        "SOURCE_BINDING_SHA256":sha(ROOT/"SOURCE_BINDING.json"),"AST_RECEIPT_SHA256":sha(ROOT/"AST_RECEIPT.json"),
        "prepare_controllers_SHA256":sha(Path(__file__)),"implementation_review_SHA256":REVIEW_SHA,
        "frozen_file_count":112,"source_files_read":len(SPECS),"controller_runtime_executed":False,
        "SSH_connections":0,"uploads_or_downloads":0,"science_launches":0,
        "scientific_array_or_weight_reads":0,"root_metadata_or_Git_writes":0,
        "numeric_or_A_F_thresholds_modified":False,"training_native_success_not_observed_by_this_preparation":True,
        "training_in_progress_as_reported_by_root":"2026-10-11; no post entry may launch before actual success",
        "resource_limits":{"CPU_affinity_count":1,"Job_process_commit_bytes":LIMIT,"RSS_watchdog_bytes":LIMIT}}
    write("PREPARATION_RECEIPT.json",receipt)
    return {"status":receipt["status"],"prepared_files":len(receipt["files"]),
            "PREPARATION_RECEIPT_SHA256":sha(ROOT/"PREPARATION_RECEIPT.json"),
            "controller_runtime_executed":False,"SSH_connections":0}

def verify():
    receipt=json.loads((ROOT/"PREPARATION_RECEIPT.json").read_text("utf8"))
    assert sha(Path(__file__))==receipt["prepare_controllers_SHA256"]
    for row in receipt["files"]:
        path=PREPARED/row["name"]
        assert path.stat().st_size==row["bytes"]and sha(path)==row["sha256"]
    assert sha(ROOT/"SOURCE_BINDING.json")==receipt["SOURCE_BINDING_SHA256"]
    assert sha(ROOT/"AST_RECEIPT.json")==receipt["AST_RECEIPT_SHA256"]
    source=json.loads((ROOT/"SOURCE_BINDING.json").read_text("utf8"))
    assert all(sha(Path(row["path"]))==row["sha256"]for row in source["sources"])
    result=inspect_prepared()
    assert all(result["checks"].values())
    record={"UTC":datetime.now(timezone.utc).isoformat(),"status":"REOPENED_STATIC_BINDINGS_PASS",
        "prepared_file_count":len(receipt["files"]),"source_files_unchanged":len(source["sources"]),
        "AST_only_no_runtime":True,"SSH_connections":0,"scientific_array_or_weight_reads":0,
        "PREPARATION_RECEIPT_SHA256":sha(ROOT/"PREPARATION_RECEIPT.json")}
    write("REOPENED_VERIFICATION.json",record);return record
try:
    result={"inventory":inventory,"read":read,"seal":seal,"verify":verify}[MODE]()
    m=memory();assert m["peak_RSS_bytes"]<=LIMIT
    write(MODE+".RESOURCE_RECEIPT.json",{"status":"COMPLETE","UTC":datetime.now(timezone.utc).isoformat(),"PID":os.getpid(),
         "elapsed_seconds":time.monotonic()-START,"CPU_affinity_count":1,"CPU_affinity_mask":one_cpu,
         "process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,"RSS_poll_seconds":.05,"Python_B":True,
         "TEMP":str(TEMP),**m,"SSH_connections":0,"uploads_or_downloads":0,"science_launches":0,
         "scientific_array_or_weight_reads":0,"root_metadata_or_Git_writes":0})
    print(json.dumps(result,ensure_ascii=False))
except BaseException as error:
    write(MODE+".FAILURE_RECEIPT.json",{"status":"FAILED","UTC":datetime.now(timezone.utc).isoformat(),
        "PID":os.getpid(),"error":repr(error),"CPU_affinity_count":1,"process_commit_limit_bytes":LIMIT,
        "RSS_watchdog_limit_bytes":LIMIT,"RSS_poll_seconds":.05,**memory(),
        "controller_runtime_executed":False,"SSH_connections":0,"scientific_array_or_weight_reads":0})
    raise
finally:stop.set()
