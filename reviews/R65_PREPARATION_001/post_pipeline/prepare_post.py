"""Guarded source-text preparation only; never import post/scientific runners."""
from __future__ import annotations
import ast
import copy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
R63 = PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
R65 = PROJECT / "work" / "R65_FULL_EMPIRICAL_D_RESPONSE_001"
TEMP = ROOT / "_tmp"
TEMP.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(TEMP)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
if not sys.dont_write_bytecode or os.name != "nt":
    raise RuntimeError("Windows python -B required for static source preparation")
MODE = sys.argv[1] if len(sys.argv) > 1 else "inventory"
if MODE not in ("inventory", "read", "adapt", "verify"):
    raise ValueError("Unknown preparation mode")
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
        try:
            m = memory()
            if m["peak_RSS_bytes"] > LIMIT:
                (ROOT / (MODE+".RESOURCE_FAILURE.json")).write_text(json.dumps(m), encoding="utf8")
                os._exit(91)
        except BaseException:
            os._exit(92)
threading.Thread(target=watch, daemon=True).start()

SPECS = {
    "preparer": PROJECT/"reviews"/"prepare_R63_post_root.py",
    "R65_protocol": PROJECT/"tasks"/"R65_FULL_EMPIRICAL_D_RESPONSE_001.md",
    "R63_config_post": R63/"config_post.json",
    "R63_numerical": R63/"numerical_check.py",
    "R63_generate": R63/"generate_all.py",
    "R63_evaluate": R63/"evaluate_saved.py",
    "R63_replay": R63/"replay_all.py",
    "R63_inference": R63/"infer_saved.py",
    "R63_plot": R63/"render_R63_figures.py",
    "R63_report": R63/"summarize_and_plot.py",
    "R63_config": R63/"config.json",
    "R63_config_reporting": R63/"config_reporting.json",
    "R65_config": R65/"config.json",
    "audit_contract": ROOT.parent / "training_audit_adapter" / "TRAINING_RECEIPT_CONTRACT.json",
}
for name in ("canonical.py","flow_runtime.py","evaluation.py","diagnostic_core.py","assessment.py","common_math.py",
             "energy_width.py","data_support.py","model.py","legacy_model.py","flow_core.py","endpoint_common.py","flow_diagnostics.py"):
    SPECS["dependency_"+name.removesuffix(".py")]=R63/name

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(name, value):
    path=(ROOT/name).resolve()
    if ROOT.resolve() not in path.parents:
        raise RuntimeError("Attempted output outside exclusive directory")
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")

def source_read():
    request=json.loads((ROOT/sys.argv[2]).read_text("utf8"))
    answer={};binding=[]
    for item in request:
        path=SPECS[item["id"]]
        if not path.is_file():
            answer[item["id"]]={"missing":str(path)}
            continue
        digest=sha(path);lines=path.read_text("utf-8-sig").splitlines()
        lo,hi=item.get("lines",[1,len(lines)])
        selected=set(range(lo-1,min(hi,len(lines))))
        if "patterns" in item:
            hits=[i for i,line in enumerate(lines) if any(p in line for p in item["patterns"])]
            selected={j for i in hits for j in range(max(0,i-item.get("context",1)),min(len(lines),i+item.get("context",1)+1))}
        answer[item.get("label",item["id"])]="\n".join(f"{i+1}: {line}" for i,line in enumerate(lines) if i in selected)
        if sha(path)!=digest:
            raise RuntimeError("Read source changed")
        binding.append({"id":item["id"],"path":str(path),"sha256":digest,"bytes":path.stat().st_size})
    prefix=Path(sys.argv[2]).stem
    write_json(prefix+".OUTPUT.json",answer)
    write_json(prefix+".BINDING.json",{"files":binding,"source_unchanged":True})
    return answer

def inventory():
    inventory={"R63_work":str(R63),"R65_work":str(R65),
               "explicit_sources":{key:{"path":str(path),"exists":path.is_file()} for key,path in SPECS.items()},
               "R63_top_level_source_names":[p.name for p in sorted(R63.iterdir()) if p.is_file() and p.suffix in (".py",".json")],
               "preparer_text":SPECS["preparer"].read_text("utf-8-sig") if SPECS["preparer"].is_file() else None,
               "scientific_array_or_weight_reads":0}
    write_json("INVENTORY.json",inventory)
    return inventory

POST_FILES = ["canonical.py","flow_runtime.py","numerical_check.py","generate_all.py","evaluate_saved.py","replay_all.py",
              "evaluation.py","diagnostic_core.py","assessment.py","common_math.py","energy_width.py","data_support.py",
              "model.py","legacy_model.py","flow_core.py","endpoint_common.py","flow_diagnostics.py","infer_saved.py",
              "summarize_and_plot.py","render_R63_figures.py"]

def source_inventory_details():
    records=[]
    own_modules={Path(name).stem for name in POST_FILES}
    tags=("R63","R61","r61","COVERAGE","PRECISION_RESOLVED","TRAINING_FROZEN_AUDIT","HISTORICAL_","COVERAGE_BY_UPDATE")
    for name in POST_FILES:
        path=R63/name;digest=sha(path);source=path.read_text("utf-8-sig")
        tree=ast.parse(source,filename=str(path));compile(tree,str(path),"exec")
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imported.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom): imported.append(node.module)
        records.append({"source":str(path),"name":name,"sha256":digest,"bytes":path.stat().st_size,
                        "imports":sorted(set(imported)),"tag_lines":[{"line":i+1,"text":line}for i,line in enumerate(source.splitlines())if any(t in line for t in tags)]})
        assert sha(path)==digest
    result={"files":records,"source_count":len(records),"excluded_other_agent_files":["post_audit.py","r65_training_audit.py","coverage_audit.py"],
            "external_runtime_modules":["network.py","source_flow_core.py","noise_core.py","post_audit.py","r65_training_audit.py"],
            "all_AST_compiled_without_import":True,"scientific_execution":0}
    write_json("SOURCE_INVENTORY.json",result)
    return result

def adapt():
    inventory = source_inventory_details()
    sources = {v["name"]:v for v in inventory["files"]}
    extra = {key:{"path":str(SPECS[key]),"sha256":sha(SPECS[key])}for key in
             ("preparer","R63_config_post","R63_config_reporting","R65_config","R65_protocol","audit_contract")}
    old_cfg=json.loads(SPECS["R63_config_post"].read_text("utf8"))
    current=json.loads(SPECS["R65_config"].read_text("utf8"))
    contract=json.loads(SPECS["audit_contract"].read_text("utf8"))
    assert current["arm"]=="FULL_EMPIRICAL_RESPONSE"
    assert current["updates"]==50000 and current["generation_chunk"]==64 and current["solver_steps"]==128 and current["solver_check_steps"]==256
    assert current["loss_columns"]==contract["records"]["loss_columns"] and len(current["loss_columns"])==32
    assert current["precision_columns"]==contract["records"]["precision_columns"] and len(current["precision_columns"])==4
    labels={"COVERAGE_RESPONSE":"FULL_EMPIRICAL_RESPONSE","COVERAGE_REAL_CONTROL":"FULL_EMPIRICAL_REAL_CONTROL",
            "PRECISION_RESOLVED_RESPONSE":"COVERAGE_RESPONSE","PRECISION_RESOLVED_REAL_CONTROL":"COVERAGE_REAL_CONTROL",
            "R63":"R65","R61":"R63","r61":"r63"}
    pattern=re.compile("|".join(re.escape(k)for k in sorted(labels,key=len,reverse=True)))
    outputs={};records=[]

    def exact_replace(value,old,new,changes,label,count=1):
        actual=value.count(old)
        assert actual==count,(label,actual,count)
        changes.append({"kind":"explicit_text","label":label,"old":old,"new":new,"count":actual})
        return value.replace(old,new)

    class MathNormalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                return ast.copy_location(ast.Constant(value="<string-label-or-path>"),node)
            return node
        def visit_Name(self,node):
            name=node.id.replace("r61","prior").replace("r63","prior").replace("render_R63_figures","render_current_figures").replace("render_R65_figures","render_current_figures")
            return ast.copy_location(ast.Name(id=name,ctx=node.ctx),node)
        def visit_FunctionDef(self,node):
            node=self.generic_visit(node)
            node.name=node.name.replace("r61","prior").replace("r63","prior")
            return node
        def visit_alias(self,node):
            node.name=node.name.replace("render_R63_figures","render_current_figures").replace("render_R65_figures","render_current_figures")
            return node

    def structure(text):
        return ast.dump(MathNormalize().visit(ast.parse(text)),include_attributes=False)

    for name in POST_FILES:
        original=(R63/name).read_bytes();text=original.decode("utf8");changes=[]
        counts={key:len(re.findall(re.escape(key),text))for key in labels if key in text}
        converted=pattern.sub(lambda match:labels[match.group()],text)
        if counts:
            changes.append({"kind":"simultaneous_role_label_mapping","mapping":{k:labels[k]for k in counts},"counts":counts})
        if name=="evaluate_saved.py":
            converted=exact_replace(converted,"frozen_objective_extension_integrity","complete_empirical_D_original_C_integrity",changes,"training-integrity metadata label")
            old="R65 sole final50000 coverage extension; original seven losses, same R63 initialization, original six consumed streams and parents; new TRAIN temporal constants and independent seventh direction stream"
            new="R65 sole final50000 complete finite-parent D and original occurrence-noise C; original R63 initialization, seven C/direction streams and constants retained; independent eighth full-parent noise stream; D-C coupling and compute change"
            converted=exact_replace(converted,old,new,changes,"training scope disclosure")
        if name=="summarize_and_plot.py":
            old="R65 full-coordinate RMS and temporal joint distribution coverage; original full480 network, parent weights, seven losses and same-REAL operator"
            new="R65 complete finite-parent five-term D plus one original four-term C microbatch; original full480 network, source population and same-REAL operator; D-C noise coupling and compute change"
            converted=exact_replace(converted,old,new,changes,"method recipe disclosure")
            converted=exact_replace(converted,"close_fixed_R65_coverage_extension","close_fixed_R65_full_empirical_D_response",changes,"fixed candidate closing label",count=2)
        if name=="render_R63_figures.py":
            start="    fig,ax=plt.subplots(1,2,figsize=(12,4.6))\n"
            finish="    save(fig,'training_and_checks.png')\n"
            assert converted.count(start)==converted.count(finish)==1
            lo=converted.index(start);hi=converted.index(finish,lo)+len(finish)
            replacement='''    steps=np.arange(1,51)*1000
    fig,ax=plt.subplots(1,3,figsize=(16,4.8))
    D_names=['full_window','PSD','noise','coordinate_RMS','temporal_joint']
    C_names=['source_order','source_valley','REAL_identity','box_correction']
    for j,(keys,title) in enumerate([(D_names,'Five complete-parent D terms'),(C_names,'Four original C terms')]):
        for label in keys:
            block=loss[:,cols.index(label)].reshape(-1,1000).mean(1)
            ax[j].plot(steps,np.maximum(block,1e-12),label=label)
        ax[j].set(yscale='log',xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',title=title)
        ax[j].legend(fontsize=7)
    tr=M['training'][arm];audit=json.loads((R/'TRAINING_FROZEN_AUDIT.json').read_text())
    counter=tr['counter']
    txt='Committed effective exposures\\nComplete D SOURCE: '+format(counter['full_D_SOURCE_law_rows'],',')+'\\n'
    txt+='Complete D REAL: '+format(counter['full_D_REAL_law_rows'],',')+'\\n'
    txt+='Original C SOURCE: '+format(counter['SOURCE_training_rows'],',')+'\\n'
    txt+='Original C REAL identity: '+format(counter['REAL_identity_rows'],',')+'\\n'
    txt+='Discarded micro-D references: '+format(counter['REAL_reference_rows'],',')+'\\n\\n'
    txt+='Completed updates: '+str(tr['updates'])+'\\nTraining audit: '+('PASS' if audit['pass']else'FAIL')+'\\n'
    txt+='A / B / C: '+' / '.join('PASS' if acceptance[arm][k]['pass']else'FAIL' for k in ['A','B','C'])+'\\n\\n'
    txt+='D/C use different parent/noise laws.\\nRecovery compute is reported separately.\\nLosses do not replace response acceptance.'
    ax[2].axis('off');ax[2].text(.02,.97,txt,va='top',linespacing=1.5,fontsize=9)
    save(fig,'training_and_checks.png')
'''
            changes.append({"kind":"plot_record_semantics","label":"separate complete D and original C loss/exposure panels","old":converted[lo:hi],"new":replacement,"count":1})
            converted=converted[:lo]+replacement+converted[hi:]
            old="    frac=M['training_audit']['constraint_loss_trace_summary']['SOURCE_global_range_projection_fraction']['mean'] if 'training_audit' in M else audit['constraint_loss_trace_summary']['SOURCE_global_range_projection_fraction']['mean']"
            new="    trace=audit['constraint_loss_trace_summary']\n    frac=trace['SOURCE_global_range_projection_fraction']['mean']\n    full_frac=trace['full_D_SOURCE_range_projection_fraction']['mean']\n    identity_frac=trace['REAL_range_projection_fraction']['mean']"
            converted=exact_replace(converted,old,new,changes,"separate D/C projection summary keys")
            old="    info+=f'TRAIN update-mean range activation: {100*frac:.2f}%.\\nWeighted minibatch trace; not exhaustive\\nfrozen-TRAIN inference.'"
            new="    info+=f'TRAIN update-mean range activation:\\ncomplete D SOURCE: {100*full_frac:.2f}%.\\noriginal C SOURCE: {100*frac:.2f}%.\\noriginal C REAL identity: {100*identity_frac:.2f}%.\\nDifferent populations/noise laws; not pooled.'"
            converted=exact_replace(converted,old,new,changes,"D/C projection caption")
            start="    coverage=load(R/arm/'COVERAGE_BY_UPDATE_PRIVATE.npz');cv=coverage['values'];cn=coverage['columns'].tolist()"
            assert converted.count(start)==1
            lo=converted.index(start)
            assert converted[lo:].rstrip().endswith("save(fig,'coverage_loss_decomposition.png')")
            replacement='''    fig,ax=plt.subplots(2,2,figsize=(13.2,9.2))
    for key,label in [('D_total','Complete D'),('C_total','Original C'),('total_loss','Joint D+C')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[0,0].plot(steps,np.maximum(block,1e-12),label=label)
    ax[0,0].set(yscale='log',ylabel='Mean loss per 1,000 updates',title='Complete D and original C totals')
    for key,label in [('D_parameter_gradient_L2','Complete D'),('C_parameter_gradient_L2','Original C'),('gradient_norm','Joint D+C before clipping')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[0,1].plot(steps,np.maximum(block,1e-12),label=label)
    ax[0,1].set(yscale='log',ylabel='Mean parameter-gradient L2',title='Gradient magnitudes; no directional claim')
    for key,label in [('full_D_SOURCE_range_projection_fraction','Complete D SOURCE'),('SOURCE_global_range_projection_fraction','Original C SOURCE'),('REAL_range_projection_fraction','Original C REAL identity')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[1,0].plot(steps,100*block,label=label)
    ax[1,0].set(ylabel='Mean range-active fraction (%)',ylim=(0,None),title='D and C range activation kept separate')
    for key,label in [('full_D_SOURCE_projection_RMS_V','Complete D SOURCE'),('SOURCE_global_projection_RMS_V','Original C SOURCE'),('REAL_projection_RMS_V','Original C REAL identity')]:
        block=loss[:,cols.index(key)].reshape(-1,1000).mean(1)
        ax[1,1].plot(steps,1000*block,label=label)
    ax[1,1].set(ylabel='Mean projection RMS (mV)',ylim=(0,None),title='D and C correction magnitudes kept separate')
    for axis in ax.flat:
        axis.set(xlabel='Update (sole final 50,000)');axis.legend(fontsize=8)
    fig.suptitle('Committed training traces; complete D and original C have different parent/noise laws')
    save(fig,'full_D_original_C_training.png')
'''
            changes.append({"kind":"plot_record_semantics","label":"replace absent old coverage trace with committed 32-column D/C traces","old":converted[lo:],"new":replacement,"count":1})
            converted=converted[:lo]+replacement
        target=name.replace("render_R63_figures","render_R65_figures")
        candidate=converted.encode("utf8") if converted!=text else original
        ast_tree=ast.parse(converted,filename=target);compile(ast_tree,target,"exec")
        invariant = structure(text)==structure(converted)
        if name!="render_R63_figures.py":
            assert invariant,(name,"Non-label AST changed")
        assert "COVERAGE_BY_UPDATE_PRIVATE" not in converted
        if name=="evaluate_saved.py":
            assert "'historical_R63_root'" in converted
            assert "r63names={'R63_SOURCE':'COVERAGE_RESPONSE','R63_REAL_CONTROL':'COVERAGE_REAL_CONTROL'}" in converted
        if name=="summarize_and_plot.py":
            assert "paired_R65_minus_frozen_R63" in converted
        outputs[target]=candidate
        records.append({"source_name":name,"source_path":str(R63/name),"source_sha256":sources[name]["sha256"],
                        "destination":"prepared/"+target,"prepared_sha256":hashlib.sha256(candidate).hexdigest(),
                        "byte_identical":candidate==original,"AST_arithmetic_controlflow_unchanged_after_label_normalization":invariant,
                        "changes":changes})

    cfg=copy.deepcopy(old_cfg);cfg.update(current)
    obsolete=("expected_R61_initial_state_SHA","expected_R61_stream_hashes","fixed_endpoint_objective_extension","new_temporal_statistics_TRAIN_only")
    for key in obsolete:cfg.pop(key,None)
    arm=current["arm"];ctrl="FULL_EMPIRICAL_REAL_CONTROL"
    cfg.update(endpoints=[arm,ctrl],REAL_arms=["BASE_REAL_CONTROL",ctrl],
               endpoint_weight_paths={n:current["remote_root"]+"/"+arm+"/FINAL_FLOW.pt"for n in (arm,ctrl)},
               historical_training_root=current["remote_root"],
               parameter_initialization="same fresh original seed and initial tensor hash as frozen R63; R63 final model not loaded",
               objective=current["sole_scientific_change"],
               scientific_scope="R65 complete finite-parent D and original C; descriptive frozen R63 comparison; original A-F unchanged",
               training_source_global_measure="complete D uses every original equal-weight parent; original C retains regional importance correction",
               execution={"training":"one fresh original network, sole50000 committed updates, complete five-term D and one original four-term C microbatch; D-C coupling and compute differ from R63",
                          "inference":"same SOURCE/REAL operator, Heun128 cached displacement and original box-charge projection"},
               descriptive_previous_stage="R63",descriptive_previous_arms={"SOURCE":"COVERAGE_RESPONSE","REAL_CONTROL":"COVERAGE_REAL_CONTROL"},
               historical_solver_check_filename="HISTORICAL_R63_TRAIN_SOLVER_CHECK.json",
               postprocessing_registered_before_results=True)
    invariant_keys=("C2ST","train_each_domain","holdout_each_domain","bootstrap_repeats","bootstrap_seed","C2ST_views",
                    "HGB_max_fits","HGB_cap","HGB_plan","density_EM_max","width_slots","main_generation_rows","replay_rows",
                    "solver_steps","solver_check_steps","generation_chunk","preADC_charge_relative_error_limit",
                    "R32_root","R33_root","R39_root","R26_root","old_BASE_REAL_generated","old_BASE_REAL_metrics",
                    "frozen_eval_input_root","canonical_R51_root","local_input_paths","constants_path","noise_calibration_path","train_solver_input")
    for key in invariant_keys:assert cfg[key]==old_cfg[key],("Fixed evaluation setting changed",key)
    reporting=copy.deepcopy(current)
    reporting.update(allowed_scientific_entries=["summarize_and_plot.py"],GPU_scientific_entries=[],reporting_only=True,new_fit_or_inference_authorized=False)
    outputs["config_post.json"]=(json.dumps(cfg,indent=2,allow_nan=False)+"\n").encode()
    outputs["config_reporting.json"]=(json.dumps(reporting,indent=2,allow_nan=False)+"\n").encode()
    for source in records:assert sha(Path(source["source_path"]))==source["source_sha256"]
    for source in extra.values():assert sha(Path(source["path"]))==source["sha256"]
    destination=ROOT/"prepared"
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("Prepared source directory is nonempty; preserve prior binding before a new revision")
    destination.mkdir(exist_ok=True)
    for name,value in outputs.items():(destination/name).write_bytes(value)
    modules=set(Path(name).stem for name in outputs if name.endswith(".py"))
    external = ["network.py","source_flow_core.py","noise_core.py","post_audit.py","r65_training_audit.py"]
    dependencies={
        "root_owned_source_modules":external,
        "other_agent_modules_never_copied_or_modified":["post_audit.py","r65_training_audit.py"],
        "runtime_only_assets_not_opened_in_preparation":[
            {"target":"HISTORICAL_R63_TRAIN_SOLVER_CHECK.json","source":current["historical_R63_root"]+"/TRAIN_SOLVER_CHECK.json","purpose":"Frozen prior numerical gate; assert both_pass as before"},
            {"target":arm+"/FINAL_FLOW.pt","source":"Unique completed R65 update50000","purpose":"Same weight for SOURCE and REAL; root-owned training output"},
            {"target":"NOISE_CALIBRATION.npz","source":current["noise_calibration_path"],"purpose":"Original calibration; root copies unchanged"},
            {"target":"UNIT_CHARGE_CONSTANTS.npz","source":current["constants_path"],"purpose":"Portable inference package; root copies unchanged"}],
        "historical_evaluation_reference":{
            "root":current["historical_R63_root"],"metrics":"metrics.json","predictions":"results/C2ST_PRIVATE.npz",
            "SOURCE_arm":"COVERAGE_RESPONSE","REAL_control_arm":"COVERAGE_REAL_CONTROL","new_reference_fits":0},
        "old_BASE_R49_C2ST_references_unchanged":True,
        "audit_interface":{"imports":["constraint_description(d)","training_audit(root,cfg,receipt)"],
            "required_top_level_keys":["pass","checks","constraint_loss_trace_summary"],
            "constraint_summary_fields":["SOURCE_global_range_projection_fraction","REAL_range_projection_fraction","full_D_SOURCE_range_projection_fraction"],
            "summary_statistic":"mean, supplied by original ev.summary"},
        "training_trace_inputs":{"LOSS_BY_UPDATE_PRIVATE.npz":current["loss_columns"],
                                 "PRECISION_ORDER_BY_UPDATE_PRIVATE.npz":current["precision_columns"]},
        "source_projection_semantics":{"SOURCE_global_*":"original C microbatch using original masses",
                                       "REAL_*":"original C REAL identity microbatch",
                                       "full_D_SOURCE_*":"all original SOURCE parents in current full D law"},
        "no_COVERAGE_trace_required":True,"execution_and_controllers":"Root owns copying/deployment/supervisors/native receipts/launch/Git; none executed here",
        "config_reporting_supervisor":"Root may adapt its own R65 supervisor to config_reporting.json; no supervisor/controller source copied by this task"}
    write_json("RUNTIME_DEPENDENCIES.json",dependencies)
    config_diff={key:{"old":old_cfg.get(key),"new":cfg.get(key)}for key in sorted(set(old_cfg)|set(cfg))if old_cfg.get(key)!=cfg.get(key)}
    write_json("CONFIG_POST_DIFF.json",config_diff)
    binding={"created_UTC":datetime.now(timezone.utc).isoformat(),"owner":"/root/r64_endpoint_geometry_prep",
             "source_directory":str(R63),"destination_directory":str(destination),"source_records":records,"additional_source_bindings":extra,
             "all_prepared_files":[{"name":name,"sha256":hashlib.sha256(value).hexdigest(),"bytes":len(value)}for name,value in outputs.items()],
             "configuration_fixed_evaluation_keys_unchanged":list(invariant_keys),"removed_stale_R63_metadata_keys":list(obsolete),
             "prepared_source_count":len(records),"byte_identical_source_count":sum(v["byte_identical"]for v in records),
             "nonplot_AST_invariance_count":sum(v["AST_arithmetic_controlflow_unchanged_after_label_normalization"]for v in records),
             "plot_only_non_label_changes":"Separate complete D and original C loss/exposure/gradient/projection displays from32-column logs; no acceptance calculations",
             "root_confirmed_labels_and_history":True,"scientific_runner_imports":0,"scientific_array_or_weight_reads":0,
             "scientific_calculations_or_updates":0,"post_runtime_execution_status":"UNEXECUTED","root_metadata_or_Git_writes":0}
    write_json("SOURCE_BINDING.json",binding)
    return {"status":"PREPARED_SOURCE_ONLY_RUNTIME_UNEXECUTED","source_count":len(records),
            "byte_identical_sources":binding["byte_identical_source_count"],"nonplot_AST_invariance_count":binding["nonplot_AST_invariance_count"],
            "prepared_files":list(outputs),"source_binding_sha256":sha(ROOT/"SOURCE_BINDING.json"),
            "config_post_sha256":sha(destination/"config_post.json"),"render_sha256":sha(destination/"render_R65_figures.py")}

def verify():
    binding=json.loads((ROOT/"SOURCE_BINDING.json").read_text("utf8"))
    destination=ROOT/"prepared"
    expected={r["name"]:r for r in binding["all_prepared_files"]}
    assert sorted(p.name for p in destination.iterdir()if p.is_file())==sorted(expected)
    for name,row in expected.items():
        path=destination/name
        assert sha(path)==row["sha256"]
        if name.endswith(".py"):
            tree=ast.parse(path.read_text("utf8"),filename=str(path));compile(tree,str(path),"exec")
    own_modules={Path(name).stem for name in expected if name.endswith(".py")}
    external={"network","source_flow_core","noise_core","post_audit","r65_training_audit"}
    packages={"numpy","torch","scipy","sklearn","matplotlib"}
    imports=[]
    for name in expected:
        if not name.endswith(".py"):continue
        for node in ast.walk(ast.parse((destination/name).read_text("utf8"))):
            values=([x.name for x in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)else[])
            for value in values:
                base=value.split(".")[0]
                assert base in own_modules|external|packages|set(sys.stdlib_module_names),(name,value)
                imports.append({"file":name,"module":value,"resolution":"prepared"if base in own_modules else"root_owned"if base in external else"environment"})
    for row in binding["source_records"]:
        assert sha(Path(row["source_path"]))==row["source_sha256"]
        if row["byte_identical"]:
            assert (ROOT/row["destination"]).read_bytes()==Path(row["source_path"]).read_bytes()
    additional_current=[]
    for key,row in binding["additional_source_bindings"].items():
        current_sha=sha(Path(row["path"]))
        if key!="audit_contract":assert current_sha==row["sha256"],("Source configuration/protocol changed",key)
        additional_current.append({"id":key,"reviewed_sha256":row["sha256"],"current_sha256":current_sha,"unchanged":current_sha==row["sha256"]})
    old_cfg=json.loads(SPECS["R63_config_post"].read_text("utf8"))
    cfg=json.loads((destination/"config_post.json").read_text("utf8"))
    cfg_reporting=json.loads((destination/"config_reporting.json").read_text("utf8"))
    current=json.loads(SPECS["R65_config"].read_text("utf8"))
    contract=json.loads(SPECS["audit_contract"].read_text("utf8"))
    for key in binding["configuration_fixed_evaluation_keys_unchanged"]:assert cfg[key]==old_cfg[key]
    assert cfg["endpoints"]==["FULL_EMPIRICAL_RESPONSE","FULL_EMPIRICAL_REAL_CONTROL"]
    assert cfg["REAL_arms"]==["BASE_REAL_CONTROL","FULL_EMPIRICAL_REAL_CONTROL"]
    weight=current["remote_root"]+"/FULL_EMPIRICAL_RESPONSE/FINAL_FLOW.pt"
    assert set(cfg["endpoint_weight_paths"].values())=={weight}
    assert cfg["historical_training_root"]==current["remote_root"]
    assert cfg["historical_R63_root"]==current["historical_R63_root"]
    assert cfg_reporting["allowed_scientific_entries"]==["summarize_and_plot.py"]
    assert cfg_reporting["GPU_scientific_entries"]==[] and cfg_reporting["reporting_only"] and not cfg_reporting["new_fit_or_inference_authorized"]
    assert cfg["loss_columns"]==contract["records"]["loss_columns"]
    assert cfg["precision_columns"]==contract["records"]["precision_columns"]
    required_loss=["full_window","PSD","noise","coordinate_RMS","temporal_joint","source_order","source_valley","REAL_identity","box_correction",
                   "D_total","C_total","total_loss","D_parameter_gradient_L2","C_parameter_gradient_L2","gradient_norm",
                   "full_D_SOURCE_range_projection_fraction","SOURCE_global_range_projection_fraction","REAL_range_projection_fraction",
                   "full_D_SOURCE_projection_RMS_V","SOURCE_global_projection_RMS_V","REAL_projection_RMS_V"]
    assert set(required_loss)<=set(cfg["loss_columns"])
    plot=(destination/"render_R65_figures.py").read_text("utf8")
    assert "COVERAGE_BY_UPDATE_PRIVATE.npz" not in plot
    assert "PRECISION_ORDER_BY_UPDATE_PRIVATE.npz" in plot
    assert "full_D_original_C_training.png" in plot
    numerical=(destination/"numerical_check.py").read_text("utf8")
    assert "med<=.25*lsb and p95<=lsb and charge<=1e-5" in numerical
    assert numerical.index("assert audit['pass']")<numerical.index("net.load_state_dict")
    assert "HISTORICAL_R63_TRAIN_SOLVER_CHECK.json" in numerical
    evaluated=(destination/"evaluate_saved.py").read_text("utf8")
    assert "r63names={'R63_SOURCE':'COVERAGE_RESPONSE','R63_REAL_CONTROL':'COVERAGE_REAL_CONTROL'}" in evaluated
    assert "PAIRED_R65_MINUS_R63.json" in evaluated and "new_fit=False" in evaluated
    summary=(destination/"summarize_and_plot.py").read_text("utf8")
    assert "paired_R65_minus_frozen_R63" in summary and "paired_R65_minus_historical_R49_completion" in summary
    assert "import render_R65_figures" in summary
    assert not any((destination/name).exists()for name in ("post_audit.py","r65_training_audit.py","coverage_audit.py"))
    native=json.loads((ROOT/"ADAPT_NATIVE_EXIT_RECEIPT.json").read_text("utf-8-sig"))
    assert native["native_exit"]==0
    write_json("IMPORT_DEPENDENCIES.json",imports)
    checks={"prepared22_file_hashes_verified":len(expected)==22,"all20_python_AST_compile_pass":len(own_modules)==20,
            "fourteen_byte_identical_dependencies_and_replay":sum(r["byte_identical"]for r in binding["source_records"])==14,
            "nineteen_nonplot_numeric_controlflow_AST_checks":sum(r["AST_arithmetic_controlflow_unchanged_after_label_normalization"]for r in binding["source_records"])==19,
            "original_A_F_and_numerical_threshold_calculations_unchanged":True,"fixed_evaluation_config_values_unchanged":True,
            "R65_SOURCE_and_REAL_use_same_unique_weight":True,"R65_minus_R63_descriptive_only":True,
            "historical_BASE_R49_comparisons_unchanged":True,"D_and_C_projection_columns_separate":True,
            "new32_column_loss_and_four_column_precision_contract_match":True,"old_coverage_trace_not_read":True,
            "audit_before_unique_weight_load":True,"other_agent_audit_files_not_copied_or_modified":True,
            "no_scientific_source_import":True,"no_scientific_array_or_weight_read":True,"no_runtime_pipeline_execution":True}
    receipt={"status":"STATIC_POST_PREPARATION_VERIFIED_RUNTIME_UNEXECUTED","checks":checks,"passed":all(checks.values()),
             "binding_sha256":sha(ROOT/"SOURCE_BINDING.json"),"prepared_count":len(expected),
             "extra_source_current_checks":additional_current,"resource":memory(),"CPU_affinity_count":1,
             "process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,
             "scientific_runner_imports":0,"scientific_array_or_weight_reads":0,"scientific_calculations_or_updates":0,
             "post_pipeline_runtime_status":"UNEXECUTED_BY_THIS_PREPARATION","figures_rendered_or_opened":0,
             "root_metadata_or_Git_writes":0,"native_adaptation_exit":0,
             "runtime_dependency_caveat":"Root must supply its audited training modules, frozen historical numerical receipt, completed sole weights/constants and scientific results before supervised post execution"}
    write_json("STATIC_VERIFICATION.json",receipt)
    extras=["SOURCE_BINDING.json","STATIC_VERIFICATION.json","RUNTIME_DEPENDENCIES.json","CONFIG_POST_DIFF.json","IMPORT_DEPENDENCIES.json","README.md","prepare_post.py","ADAPT_NATIVE_EXIT_RECEIPT.json"]
    file_rows=[{"file":name,"sha256":sha(ROOT/name),"bytes":(ROOT/name).stat().st_size}for name in extras]
    preparation={"status":receipt["status"],"prepared_files":binding["all_prepared_files"],"metadata_files":file_rows,
                 "resource":memory(),"CPU_affinity_count":1,"process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,
                 "source_only":True,"scientific_runner_imports":0,"scientific_array_or_weight_reads":0,
                 "scientific_calculations_or_updates":0,"scientific_runtime_or_visual_QA_claim":False,"root_metadata_or_Git_writes":0}
    write_json("PREPARATION_RECEIPT.json",preparation)
    for row in file_rows:assert sha(ROOT/row["file"])==row["sha256"]
    return {"status":receipt["status"],"prepared_files":len(expected),"all_checks_passed":receipt["passed"],
            "preparation_receipt_sha256":sha(ROOT/"PREPARATION_RECEIPT.json"),"source_binding_sha256":sha(ROOT/"SOURCE_BINDING.json"),
            "static_verification_sha256":sha(ROOT/"STATIC_VERIFICATION.json"),"resource":memory(),
            "runtime_executions":0,"figures_rendered":0}

try:
    result={"inventory":inventory,"read":source_read,"adapt":adapt,"verify":verify}[MODE]()
    m=memory()
    if m["peak_RSS_bytes"]>LIMIT:
        raise RuntimeError("Resource limit exceeded")
    write_json(MODE+".RESOURCE_RECEIPT.json",{"status":"COMPLETE","UTC":datetime.now(timezone.utc).isoformat(),
               "PID":os.getpid(),"elapsed_seconds":time.monotonic()-START,"CPU_affinity_count":1,"CPU_affinity_mask":one_cpu,
               "process_commit_limit_bytes":LIMIT,"RSS_watchdog_limit_bytes":LIMIT,"RSS_poll_seconds":.05,"Python_B":True,
               "TEMP":str(TEMP),**m,"scientific_array_or_weight_reads":0,"scientific_runner_imports":0,
               "scientific_jobs_or_updates":0,"root_metadata_or_Git_writes":0})
    print(json.dumps(result,ensure_ascii=False))
finally:
    stop.set()
