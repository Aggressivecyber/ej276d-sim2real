"""Text/AST only: never imports candidate modules or opens scientific arrays."""
from pathlib import Path
from local_resource_guard import apply, memory
ROOT = Path(__file__).resolve().parent
RESOURCE = apply(ROOT / "static_validation")
import ast
import hashlib
import json
import sys
import time

START = time.monotonic()
PROJECT = ROOT.parents[2]
OLD = PROJECT / "work" / "R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001"
NEW = PROJECT / "work" / "R65_FULL_EMPIRICAL_D_RESPONSE_001"
schema_path = ROOT.parent / "ROOT_TRAINING_SCHEMA.json"
files = [
    ROOT / "post_audit.py", ROOT / "r65_training_audit.py",
    ROOT / "TRAINING_RECEIPT_CONTRACT.json", ROOT / "local_resource_guard.py",
    ROOT / "validate_static.py", schema_path,
    OLD / "post_audit.py", OLD / "coverage_audit.py", OLD / "numerical_check.py",
    OLD / "generate_all.py", OLD / "summarize_and_plot.py", OLD / "evaluation.py",
    OLD / "common_math.py", NEW / "train_final.py", NEW / "config.json",
    PROJECT / "tasks" / "R65_FULL_EMPIRICAL_D_RESPONSE_001.md",
]
opened = []
def guard(event, args):
    if event == "open" and not isinstance(args[0], int):
        path = Path(args[0]).resolve()
        if path.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".pkl"}:
            raise RuntimeError("static validator refuses scientific/binary record: " + str(path))
        opened.append(str(path))
sys.addaudithook(guard)

texts = {str(path): path.read_text("utf8") for path in files}
trees = {str(path): ast.parse(texts[str(path)], filename=str(path))
         for path in files if path.suffix == ".py"}
schema = json.loads(texts[str(schema_path)])
contract = json.loads(texts[str(ROOT / "TRAINING_RECEIPT_CONTRACT.json")])
helper = trees[str(ROOT / "r65_training_audit.py")]
post = trees[str(ROOT / "post_audit.py")]
old_post = trees[str(OLD / "post_audit.py")]
trainer = trees[str(NEW / "train_final.py")]
num = trees[str(OLD / "numerical_check.py")]
def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
def literal(name):
    node = next(n for n in helper.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)
def dict_keys(node):
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
checks = {}
checks["all_python_sources_parse"] = True
checks["constraint_description_AST_identical_to_R63"] = (
    ast.dump(function(post, "constraint_description"), include_attributes=False)
    == ast.dump(function(old_post, "constraint_description"), include_attributes=False))
checks["preserved_public_function_signatures"] = all(
    [a.arg for a in function(post, name).args.args] == args
    for name, args in [("constraint_description", ["d"]),
                       ("training_audit", ["root", "cfg", "receipt"])])
checks["adapter_delegates_three_original_arguments"] = (
    isinstance(function(post, "training_audit").body[0], ast.Return)
    and ast.unparse(function(post, "training_audit").body[0].value)
    == "audit_saved_records(root, cfg, receipt)")
checks["frozen_schema_constants_equal"] = all(
    literal(name) == schema[key] for name, key in [
        ("LOSS_COLUMNS", "loss_columns"), ("PRECISION_COLUMNS", "precision_columns"),
        ("REPLAY_COLUMNS", "replay_columns"), ("REPLAY_ROW", "expected_replay_row"),
        ("EXPECTED_COUNTS", "expected_completed_update_counts"),
        ("D_TERMS", "full_D_scalar_order"), ("C_TERMS", "C_scalar_order"),
        ("OLD_STREAMS", "original_streams")])
checks["gradient_count_contract_equal"] = (
    literal("GRADIENT_COUNTS") == contract["required_receipt_fields"]["gradient_execution_counts"])
receipt_assign = next(n for n in ast.walk(trainer) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "receipt" for t in n.targets)
                      and isinstance(n.value, ast.Dict))
receipt_values = {k.value: v for k, v in zip(receipt_assign.value.keys, receipt_assign.value.values)
                  if isinstance(k, ast.Constant)}
missing = sorted(set(contract["required_receipt_fields"]) - set(receipt_values))
checks["trainer_all_required_top_receipt_fields_present"] = not missing
integrity_keys = dict_keys(receipt_values["full_empirical_D_integrity"])
checks["trainer_all_integrity_flags_present"] = set(literal("TRUE_INTEGRITY") + literal("FALSE_INTEGRITY")) <= integrity_keys
artifact_comp = receipt_values["artifact_sha256"]
checks["trainer_all_saved_artifact_bindings_present"] = (
    isinstance(artifact_comp, ast.DictComp)
    and set(literal("ARTIFACTS")) <= set(ast.literal_eval(artifact_comp.generators[0].iter)))
physical_fn = function(trainer, "physical_schema_counter")
physical_keys = dict_keys(next(n.value for n in physical_fn.body if isinstance(n, ast.Return)))
checks["physical_counter_schema_keys_exact"] = physical_keys == set(literal("EXPECTED_COUNTS"))
top_reading_calls = []
for n in helper.body:
    if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
        continue
    if any(isinstance(x, ast.Call) for x in ast.walk(n)):
        top_reading_calls.append(getattr(n, "lineno", None))
checks["helper_no_top_level_file_or_scientific_execution"] = not top_reading_calls
imports = [n.module for n in ast.walk(helper) if isinstance(n, ast.ImportFrom)]
imports += [a.name for n in ast.walk(helper) if isinstance(n, ast.Import) for a in n.names]
checks["audit_has_no_torch_or_network_import"] = not any(
    name and ("torch" in name or "network" in name or "distribution_loss" in name) for name in imports)
checks["old_coverage_extension_not_called"] = not any(
    isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    and n.func.id == "coverage_extension_audit" for n in ast.walk(helper))
audit_line = min(n.lineno for n in ast.walk(num) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "training_audit")
load_line = min(n.lineno for n in ast.walk(num) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "torch" and n.func.attr == "load")
assert_lines = [n.lineno for n in ast.walk(num) if isinstance(n, ast.Assert)
                and "audit['pass']" in ast.unparse(n.test)]
checks["inherited_numerical_audit_assert_precedes_model_load"] = any(
    audit_line < line < load_line for line in assert_lines)
checks["inherited_numerical_source_hash_unchanged"] = hashlib.sha256(
    (OLD / "numerical_check.py").read_bytes()).hexdigest() == "00695d5998dfb4056977f89036403d1948f0d9b2ce7a184e3679eb80e39ad764"
tolerances = []
for n in ast.walk(helper):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("allclose", "isclose"):
        tolerances.append({k.arg: ast.literal_eval(k.value) for k in n.keywords if k.arg in ("rtol", "atol")})
checks["original_record_numeric_tolerances_only"] = (
    bool(tolerances) and all(t in [{"rtol":1e-12,"atol":1e-12},{"rtol":1e-10,"atol":1e-10}] for t in tolerances))
checks["physical_explicit_uncommitted_Adam_difference_present"] = (
    'attempt["known_uncommitted_optimizer_steps"] == physical_steps - (end - start)'
    in texts[str(ROOT / "r65_training_audit.py")])
bindings = [{"path":str(path),"bytes":path.stat().st_size,
             "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
RESOURCE.update(memory())
result = {
    "status":"pass" if all(checks.values()) else "fail", "checks":checks,
    "missing_trainer_top_receipt_fields":missing, "source_bindings":bindings,
    "source_AST_checks_only":True, "candidate_modules_imported":0,
    "scientific_arrays_or_weights_opened":0, "scientific_forward_fit_or_loss_calls":0,
    "numeric_record_tolerances_found":tolerances,
    "resource":RESOURCE,"elapsed_s":time.monotonic()-START,
    "limitations":["AST and text checks do not establish runtime or scientific correctness.",
                   "Trainer source bindings reflect this read, not later root edits."],
}
(ROOT / "STATIC_VALIDATION.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n","utf8")
print(json.dumps({"status":result["status"],"checks":checks,"resource":RESOURCE,"elapsed_s":result["elapsed_s"]}))
raise SystemExit(0 if result["status"] == "pass" else 1)
