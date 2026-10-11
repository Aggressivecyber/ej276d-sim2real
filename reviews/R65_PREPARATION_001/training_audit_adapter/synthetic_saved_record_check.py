"""Entirely synthetic saved-record integration; no scientific data or model execution."""
from pathlib import Path
from local_resource_guard import apply, memory
import sys
ROOT = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]).resolve()
RESOURCE = apply(RUN)
import ast
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import time

START = time.monotonic()
PROJECT = ROOT.parents[2]
FIXTURES = RUN / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=False)
VIOLATIONS = []
ARRAY_OPENS = []
def access_guard(event, args):
    if event in ("subprocess.Popen", "socket.connect"):
        VIOLATIONS.append({"event":event})
        raise RuntimeError("synthetic saved-record check cannot spawn or connect")
    if event == "open" and not isinstance(args[0], int):
        path = Path(args[0]).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(x in mode for x in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and not path.is_relative_to(RUN):
            VIOLATIONS.append({"event":"write_outside_attempt","path":str(path)})
            raise RuntimeError("synthetic writes must remain inside its own attempt")
        if path.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".pkl"}:
            if not path.is_relative_to(FIXTURES):
                VIOLATIONS.append({"event":"non_synthetic_array_or_weight_open","path":str(path)})
                raise RuntimeError("real arrays/weights are prohibited: " + str(path))
            ARRAY_OPENS.append(str(path))
sys.addaudithook(access_guard)
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n","utf8")

def npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)

def read(path):
    return json.loads(Path(path).read_text("utf8"))

seal = read(ROOT / "CODE_SEAL_BEFORE_SYNTHETIC.json")
for row in seal["files"]:
    assert sha(ROOT / row["name"]) == row["sha256"], row["name"]
evaluation_path = PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001" / "evaluation.py"
evaluation_text = evaluation_path.read_text("utf8")
tree = ast.parse(evaluation_text)
summary_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "summary")
summary_source = ast.get_source_segment(evaluation_text, summary_node)
imports_dir = RUN / "import_sources"
imports_dir.mkdir()
summary_file = imports_dir / "evaluation.py"
summary_file.write_text("import numpy as np\n\n"+summary_source+"\n","utf8")
copied_summary = next(n for n in ast.parse(summary_file.read_text("utf8")).body
                      if isinstance(n,ast.FunctionDef))
assert ast.dump(copied_summary,include_attributes=False) == ast.dump(summary_node,include_attributes=False)
sys.path.insert(0, str(imports_dir))
sys.path.insert(0, str(ROOT))
post = importlib.import_module("post_audit")
audit_module = importlib.import_module("r65_training_audit")

schema = read(ROOT.parent / "ROOT_TRAINING_SCHEMA.json")
template = read(PROJECT / "work" / "R65_FULL_EMPIRICAL_D_RESPONSE_001" / "config.json")
U, NS, NR = 50000, 6593, 21047
assert schema["updates"] == U and schema["SOURCE_N"] == NS and schema["REAL_N"] == NR
arm = "FULL_EMPIRICAL_RESPONSE"
expected = schema["expected_completed_update_counts"]
base = FIXTURES / "base"
history = FIXTURES / "synthetic_R63_history"
inputs_dir = FIXTURES / "synthetic_inputs"
p = base / arm
p.mkdir(parents=True)
history.mkdir()
inputs_dir.mkdir()
source_ids = np.arange(NS,dtype=np.uint16)
source_bin = (np.arange(NS)%10-1).astype(np.int16)
real_bin = (np.arange(NR)%10-1).astype(np.int16)
source_psd = .4 + np.arange(NS,dtype=np.float64)*1e-6

def distribute(total, count):
    values = np.full(count,total//count,dtype=np.int64)
    values[:total%count] += 1
    return values

def stratified(bins):
    values = np.empty(len(bins),dtype=np.int64)
    for b in range(-1,9):
        where = np.flatnonzero(bins == b)
        values[where] = distribute(1600000,len(where))
    return values

count_s, count_r = stratified(source_bin), stratified(real_bin)
count_i = distribute(3200000,NR)
common_parents = dict(source_counts=count_s, REAL_reference_counts=count_r,
    REAL_identity_counts=count_i,source_bin0=source_bin,REAL_bin0=real_bin,source_ids=source_ids)
npz(history/"PARENT_EXPOSURES_PRIVATE.npz",**common_parents,source_original_psd=source_psd)
npz(p/"PARENT_EXPOSURES_PRIVATE.npz",**common_parents,
    full_D_source_counts=np.full(NS,U,dtype=np.int64),
    full_D_REAL_counts=np.full(NR,U,dtype=np.int64))
npz(history/"SOURCE_PSD_PRECISION_PRIVATE.npz",source_ids=source_ids,
    source_original_psd=source_psd,PSD_interval_low=source_psd-1e-7,PSD_interval_high=source_psd+1e-7)
write(history/"SOURCE_ORDER_PRECISION_DEFINITION.json", {
    "synthetic_fixture":True,"interval_rule":template["source_order_rule"],
    "all_original_denominator_pairs_retained":True,"new_threshold_fit":False})
for name in ("SOURCE_PSD_PRECISION_PRIVATE.npz","SOURCE_ORDER_PRECISION_DEFINITION.json"):
    shutil.copyfile(history/name,base/name)
for filename in ("TRAIN_PRIVATE.npz","UNIT_CHARGE_CONSTANTS.npz","NOISE_CALIBRATION.npz","TRAIN_SOLVER_INPUT.npz"):
    npz(inputs_dir/filename,synthetic_placeholder=np.array([1.],dtype=np.float64))
write(inputs_dir/"VALLEY_DEFINITION.json",{"synthetic_fixture":True,"unused_saved_record_placeholder":True})
real_counts = np.bincount(real_bin.astype(np.int64)+1,minlength=10)
npz(inputs_dir/"TRAIN_DISTRIBUTION_CONSTANTS.npz",REAL_counts=real_counts)
center,scale = np.zeros((10,31),dtype=np.float64),np.ones((10,31),dtype=np.float64)
npz(inputs_dir/"TRAIN_TEMPORAL_CONSTANTS.npz",temporal_center=center,
    temporal_scale=scale,temporal_REAL_counts=real_counts)
write(history/"TRAIN_TEMPORAL_CONSTANTS.json",{
    "synthetic_fixture":True,"center":center.tolist(),"scale":scale.tolist(),
    "fitted_REAL_TRAIN_rows":NR,"new_statistics_SHA256":sha(inputs_dir/"TRAIN_TEMPORAL_CONSTANTS.npz"),
    "old_statistics_SHA256":sha(inputs_dir/"TRAIN_DISTRIBUTION_CONSTANTS.npz"),
    "fitted_before_first_update":True,"scientific_response_fit":False,"teacher_fit":False})
shutil.copyfile(history/"TRAIN_TEMPORAL_CONSTANTS.json",base/"TRAIN_TEMPORAL_CONSTANTS.json")
for filename in ("UNIT_CHARGE_CONSTANTS.npz","NOISE_CALIBRATION.npz",
                 "TRAIN_DISTRIBUTION_CONSTANTS.npz","TRAIN_TEMPORAL_CONSTANTS.npz"):
    shutil.copyfile(inputs_dir/filename,base/filename)
streams = schema["original_streams"]
seeds = template["PCG64_seeds"]
initial_states = {name:np.random.PCG64(seed).state for name,seed in seeds.items()}
final_states = {}
for name,seed in seeds.items():
    generator = np.random.PCG64(seed)
    generator.advance(1)
    final_states[name] = generator.state
chain = {name:hashlib.sha256(("SYNTHETIC_ONLY_NOT_CONSUMED_STREAM:"+name).encode()).hexdigest()
         for name in seeds}
old_receipt = {
    "synthetic_fixture":True,"initial_state_SHA":audit_module.INITIAL_SHA,
    "PCG64_seeds":{name:seeds[name] for name in streams},
    "stream_hashes":{name:chain[name] for name in streams},
    "stream_hash_algorithm":"Synthetic saved-record metadata; not a real stream-consumption claim",
}
write(history/"TRAINING_RECEIPT.json",old_receipt)
write(history/"FINAL_RANDOM_STATES.json",{name:final_states[name] for name in streams})
cfg = copy.deepcopy(template)
cfg.update(synthetic_fixture=True,server_python="unused_synthetic_fixture",
    remote_root=str(base),historical_R63_root=str(history),
    expected_R63_stream_hashes=old_receipt["stream_hashes"],
    expected_R33_environment={"python":"synthetic_saved_record",
      "numpy":"synthetic_declared_environment","torch":"not_executed",
      "scipy":"not_executed","device":"synthetic_no_neural_device","GPU":"no_GPU_execution"})
cfg["input_paths"] = {
    "TRAIN_path":str(inputs_dir/"TRAIN_PRIVATE.npz"),
    "constants_path":str(inputs_dir/"UNIT_CHARGE_CONSTANTS.npz"),
    "noise_calibration_path":str(inputs_dir/"NOISE_CALIBRATION.npz"),
    "train_solver_input":str(inputs_dir/"TRAIN_SOLVER_INPUT.npz"),
    "valley_definition":str(inputs_dir/"VALLEY_DEFINITION.json"),
    "distribution_statistics":str(inputs_dir/"TRAIN_DISTRIBUTION_CONSTANTS.npz"),
    "temporal_statistics":str(inputs_dir/"TRAIN_TEMPORAL_CONSTANTS.npz"),
    "R63_training_receipt":str(history/"TRAINING_RECEIPT.json"),
    "R63_parent_exposures":str(history/"PARENT_EXPOSURES_PRIVATE.npz"),
    "R63_final_random_states":str(history/"FINAL_RANDOM_STATES.json"),
    "R63_config":str(history/"config.json"),
}
for key in ("TRAIN_path","constants_path","noise_calibration_path","train_solver_input","valley_definition"):
    cfg[key] = cfg["input_paths"][key]
cfg["source_local_TRAIN"] = cfg["input_paths"]["TRAIN_path"]
old_cfg = copy.deepcopy(cfg)
old_cfg["PCG64_seeds"] = old_receipt["PCG64_seeds"]
old_cfg["stage"] = "SYNTHETIC_R63_METADATA_ONLY"
write(history/"config.json",old_cfg)
write(base/"config.json",cfg)
binding = {"synthetic_fixture":True,"inputs":{
    name:{"path":path,"bytes":Path(path).stat().st_size,"sha256":sha(path)}
    for name,path in cfg["input_paths"].items()}}
write(base/"INPUT_BINDING.json",binding)
(base/"PROTOCOL.md").write_text("Purely synthetic saved-record fixture. No training or scientific evidence.\n","utf8")
for filename in ("post_audit.py","r65_training_audit.py"):
    shutil.copyfile(ROOT/filename,base/filename)
write(base/"ROOT_IMPLEMENTATION_REVIEW.json",{
    "synthetic_fixture":True,"files":[{"name":name,"sha256":sha(base/name)}
    for name in ("post_audit.py","r65_training_audit.py","config.json","PROTOCOL.md")]})
write(base/"ARTIFICIAL_SUITE_RECEIPT.json",{"synthetic_fixture":True,
    "placeholder_for_binding_only":True,"real_training_permission":False})
write(p/"INITIALIZATION.json",{
    "synthetic_fixture":True,"seed":20261010590,"state_tensor_SHA":audit_module.INITIAL_SHA,
    "loaded_previous_trained_response":False,
    "note":"Required contract identifier, not a tensor constructed or loaded in this check"})
write(p/"INITIAL_RANDOM_STATES.json",initial_states)
write(p/"FINAL_RANDOM_STATES.json",final_states)
(p/"FINAL_FLOW.pt").write_bytes(b"SYNTHETIC PLACEHOLDER BYTES. NOT A MODEL. DO NOT LOAD.\n")
columns = schema["loss_columns"]
row = {name:0. for name in columns}
row.update(full_window=.2,coordinate_RMS=.1,PSD=.5,noise=.3,temporal_joint=.4,
           source_order=.05,source_valley=.02,REAL_identity=.03,box_correction=.01,
           gradient_norm=.11,D_parameter_gradient_L2=.1,C_parameter_gradient_L2=.03,
           overall_PSD_W1=.005,Q1_Q7_mean_PSD_W1=.005,
           SOURCE_global_range_projection_fraction=.1,REAL_range_projection_fraction=.2,
           full_D_SOURCE_range_projection_fraction=.15,
           SOURCE_global_projection_RMS_V=.001,REAL_projection_RMS_V=.002,
           full_D_SOURCE_projection_RMS_V=.0015,
           full_D_max_relative_q_f64=1e-16,C_max_relative_q_f64=5e-16,max_relative_q_f64=5e-16,
           full_D_max_relative_q_f32=2e-8,C_max_relative_q_f32=1e-8,max_relative_q_f32=2e-8,
           Q3_h_one_soft_valley=.1,Q4_h_one_soft_valley=.1)
row["D_total"] = sum(row[name] for name in schema["full_D_scalar_order"])
row["C_total"] = sum(row[name] for name in schema["C_scalar_order"])
row["total_loss"] = row["D_total"]+row["C_total"]
values = np.tile(np.array([row[name] for name in columns],dtype=np.float64),(U,1))
for step in range(1,U+1):
    if step <= cfg["warmup_updates"]:
        rate = cfg["lr_initial"]+(cfg["lr"]-cfg["lr_initial"])*(step-1)/(cfg["warmup_updates"]-1)
    else:
        fraction = (step-cfg["warmup_updates"])/(U-cfg["warmup_updates"])
        rate = cfg["lr_final"]+.5*(cfg["lr"]-cfg["lr_final"])*(1+math.cos(math.pi*fraction))
    values[step-1,columns.index("learning_rate")] = rate
npz(p/"LOSS_BY_UPDATE_PRIVATE.npz",columns=np.asarray(columns),values=values)
npz(p/"PRECISION_ORDER_BY_UPDATE_PRIVATE.npz",columns=np.asarray(schema["precision_columns"]),
    values=np.tile(np.array([.07,.02,.5,.25],dtype=np.float64),(U,1)))
npz(p/"REPLAY_BY_UPDATE_PRIVATE.npz",columns=np.asarray(schema["replay_columns"]),
    values=np.tile(np.asarray(schema["expected_replay_row"],dtype=np.int64),(U,1)))
discarded_columns = schema["full_D_scalar_order"]+["original_nine_total","discarded_D_total"]
discarded_values = np.array([.21,.11,.51,.31,.41],dtype=np.float64)
discarded_total = float(discarded_values.sum())
discarded_row = np.r_[discarded_values,discarded_total+row["C_total"],discarded_total]
npz(p/"DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz",columns=np.asarray(discarded_columns),
    values=np.tile(discarded_row,(U,1)))
counter = dict(expected)
counter.update({k:0 for k in audit_module.ZERO_COUNTERS})
environment = dict(cfg["expected_R33_environment"],model_training_mode=True,
    MHA_fastpath=False,TF32=False,deterministic_algorithms=True,CPU_threads=1)
receipt = {
    "synthetic_fixture":True,"actual_scientific_training_runs":0,
    "arm":arm,"updates":U,"selection":"sole final50000","new_response_training_runs":1,
    "counter":counter,"model_seed":20261010590,"initial_state_SHA":audit_module.INITIAL_SHA,
    "final_state_SHA":hashlib.sha256(b"SYNTHETIC_NO_TENSORS").hexdigest(),
    "actual_unique_SOURCE_REAL_reference_REAL_identity":[NS,NR,NR],
    "PCG64_seeds":seeds,"stream_hashes":chain,
    "stream_hash_algorithm":old_receipt["stream_hash_algorithm"],
    "projection_direction_calls":U,"temporal_direction_calls":U,"full_SOURCE_epsilon_calls":U,
    "source_order_rule":cfg["source_order_rule"],"precision_order_trace_rows":U,
    "precision_order_columns":schema["precision_columns"],
    "teacher_waveforms_constructed":0,"R58_law_or_atoms_training_target":False,
    "h_PDG_network_inputs":False,"source_val_target_val_decoded":False,
    "DEV_read":0,"EVAL_read":0,"CAL_decode":0,
    "max_output_relative_q_error_f64":5e-16,"max_output_relative_q_error_f32":2e-8,
    "GPU_allocated_peak_bytes":0,"GPU_reserved_peak_bytes":0,"CPU_threads":1,
    "compute_budget_enforced":False,"model_goal_complete":False,
    "full_empirical_D_integrity":dict(
        {name:True for name in audit_module.TRUE_INTEGRITY},
        **{name:False for name in audit_module.FALSE_INTEGRITY}),
    "gradient_execution_counts":dict(audit_module.GRADIENT_COUNTS),
    "discarded_D_columns":discarded_columns,"discarded_D_trace_rows":U,
    "artifact_sha256":{name:sha(p/name) for name in audit_module.ARTIFACTS},
    "frozen_binding":{
        "protocol_SHA":sha(base/"PROTOCOL.md"),"config_SHA":sha(base/"config.json"),
        "input_binding_SHA":sha(base/"INPUT_BINDING.json"),
        "implementation_review_SHA":sha(base/"ROOT_IMPLEMENTATION_REVIEW.json"),
        "environment":environment},
    "artificial_suite_SHA256":sha(base/"ARTIFICIAL_SUITE_RECEIPT.json"),
    "physical_attempt_compute":[{
        "attempt_id":"synthetic_uninterrupted","resumed_from_update":0,
        "last_completed_update":U,"counter":dict(expected),
        "known_uncommitted_optimizer_steps":0,"counters_complete":True,"unknown_intervals":[]}],
    "recovery_history":[],
}
write(p/"TRAINING_RECEIPT.json",receipt)
write(RUN/"FIXTURE_PROVENANCE.json",{
    "scope":"ALL NPZ/JSON/PT test records under fixtures are artificial and carry no scientific evidence.",
    "all_array_values_generated_by_this_script":True,"weight_is_unloadable_placeholder_bytes":True,
    "50000_is_only_the_contract_record_length":True,
    "declared_training_counters_are_synthetic_schema_values_not_observed_compute":True,
    "actual_neural_rows":0,"actual_neural_forward_calls":0,"actual_optimizer_steps":0,
    "real_scientific_array_or_weight_reads":0,
    "source_summary_function_AST_preserved":True,
    "evaluation_dependency_scope":"Original summary function copied verbatim; unrelated plotting/classifier imports are omitted in this synthetic-only dependency.",
    "source_summary_file_SHA256":sha(evaluation_path),
    "copied_summary_function_SHA256":hashlib.sha256(summary_source.encode()).hexdigest(),
    "candidate_source_seal_SHA256":sha(ROOT/"CODE_SEAL_BEFORE_SYNTHETIC.json")})

matrix = []
def run_case(name, path, expected_pass, expected_failed):
    candidate = read(path/arm/"TRAINING_RECEIPT.json")
    result = post.training_audit(path, {
        "arm":arm,"historical_training_root":str(path),
        "endpoint_weight_paths":{arm:str(path/arm/"FINAL_FLOW.pt")}},candidate)
    failed = sorted(k for k,v in result["checks"].items() if not v)
    good = result["pass"] is expected_pass and set(failed) == set(expected_failed)
    row_result = {"case":name,"expected_pass":expected_pass,"actual_pass":result["pass"],
        "expected_failed_checks":sorted(expected_failed),"actual_failed_checks":failed,
        "test_pass":good,"audit_check_count":len(result["checks"]),
        "record_root":str(path),"audit_error":result.get("error")}
    if "physical_compute" in result:
        row_result["logical_updates"] = result["physical_compute"]["logical_completed_counter"]["new_response_updates"]
        row_result["physical_optimizer_calls"] = result["physical_compute"]["known_physical_counter_lower_bounds"]["new_response_updates"]
        row_result["known_uncommitted_optimizer_steps"] = result["physical_compute"]["known_uncommitted_optimizer_steps"]
    write(RUN/(name+"_AUDIT.json"),result)
    matrix.append(row_result)
    write(RUN/"TEST_MATRIX.json",matrix)
    if not good:
        raise AssertionError(json.dumps(row_result))

def variant(name):
    path = FIXTURES/name
    shutil.copytree(base,path)
    return path

def mutate_table(path,filename,change):
    with np.load(path/arm/filename,allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    change(data["values"])
    npz(path/arm/filename,**data)
    rec = read(path/arm/"TRAINING_RECEIPT.json")
    rec["artifact_sha256"][filename] = sha(path/arm/filename)
    write(path/arm/"TRAINING_RECEIPT.json",rec)

try:
    run_case("valid_contract",base,True,[])
    bad_replay = variant("bad_replay")
    mutate_table(bad_replay,"REPLAY_BY_UPDATE_PRIVATE.npz",
                 lambda a:a.__setitem__((0,3),103))
    run_case("bad_replay",bad_replay,False,
             ["all104_chunk_replays_and_single_reverse_counts_every_update"])
    bad_total = variant("bad_D_plus_C_total")
    mutate_table(bad_total,"LOSS_BY_UPDATE_PRIVATE.npz",
                 lambda a:a.__setitem__((0,columns.index("total_loss")),a[0,0]+.125))
    run_case("bad_D_plus_C_total",bad_total,False,
             ["total_loss_ordered_component_sum","total_is_D_plus_C"])
    bad_stream = variant("bad_original_seven_stream")
    rec = read(bad_stream/arm/"TRAINING_RECEIPT.json")
    rec["stream_hashes"]["source_rows"] = "0"*64
    write(bad_stream/arm/"TRAINING_RECEIPT.json",rec)
    run_case("bad_original_seven_stream",bad_stream,False,
             ["original_seven_seed_chain_and_final_states_equal_bound_R63"])
    recovery = variant("valid_partial_extra_Adam_recovery")
    rec = read(recovery/arm/"TRAINING_RECEIPT.json")
    def physical_for(steps):
        return {k:(v//U)*steps for k,v in expected.items()}
    rec["physical_attempt_compute"] = [
        {"attempt_id":"synthetic_failed_after_Adam","resumed_from_update":0,
         "last_completed_update":500,"counter":physical_for(501),
         "known_uncommitted_optimizer_steps":1,"counters_complete":True,"unknown_intervals":[]},
        {"attempt_id":"synthetic_resumed","resumed_from_update":500,
         "last_completed_update":U,"counter":physical_for(U-500),
         "known_uncommitted_optimizer_steps":0,"counters_complete":True,"unknown_intervals":[]}]
    rec["recovery_history"] = [{
        "from_attempt_id":"synthetic_failed_after_Adam","to_attempt_id":"synthetic_resumed",
        "checkpoint_update":500,"checkpoint_sha256":hashlib.sha256(b"SYNTHETIC_CHECKPOINT").hexdigest(),
        "bindings_verified":True,"previous_PID_confirmed_dead":True}]
    write(recovery/arm/"TRAINING_RECEIPT.json",rec)
    run_case("valid_partial_extra_Adam_recovery",recovery,True,[])
    assert matrix[-1]["logical_updates"] == 50000
    assert matrix[-1]["physical_optimizer_calls"] == 50001
    assert matrix[-1]["known_uncommitted_optimizer_steps"] == 1
    assert not VIOLATIONS
    for record in seal["files"]:
        assert sha(ROOT/record["name"]) == record["sha256"],record["name"]
    status = "pass"
except BaseException as exc:
    status = "fail"
    error = repr(exc)
finally:
    RESOURCE.update(memory())
    result = {"status":status,"matrix":matrix,"resource":RESOURCE,
        "elapsed_s":time.monotonic()-START,"access_guard_violations":VIOLATIONS,
        "synthetic_array_weight_file_open_events":len(ARRAY_OPENS),
        "real_scientific_array_or_weight_reads":0,"scientific_forward_or_fit_calls":0,
        "actual_optimizer_steps":0,"scientific_model_or_weights_loaded":False,
        "candidate_source_seal_preserved":all(sha(ROOT/r["name"]) == r["sha256"] for r in seal["files"]),
        "runtime":{"python":sys.version,"numpy":np.__version__},
        "source_summary_function_AST_preserved":True,
        "production_evaluation_module_imported":False,
        "error":locals().get("error")}
    write(RUN/"RESULT.json",result)
    print(json.dumps(result))
if status != "pass":
    raise SystemExit(1)
