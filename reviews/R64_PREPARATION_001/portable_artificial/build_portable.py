"""Static source port only; never executes a fixture, torch or scientific data."""
from __future__ import annotations
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "distribution_gradient"
GEOMETRY = ROOT.parent / "endpoint_geometry"
source_path = ORIGINAL / "artificial_fixture.py"
source_bytes = source_path.read_bytes()
SOURCE_SHA = "5273878b94ef509e466a0ed5250b4f6c3b7b87ac4e87cee6c10b48fa06126c6e"
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA
tree = ast.parse(source_bytes.decode("utf-8-sig"))
original_receipt = json.loads((ORIGINAL / "ARTIFICIAL_RECEIPT.json").read_text("utf-8-sig"))
assert len(original_receipt["tests"]) == 69
adapter_names = ["original_bindings.py", "weighted_empirical_w1.py", "distribution_gradient.py",
                 "SOURCE_BINDINGS.json", "FROZEN_TOLERANCES.json"]
adapter_names += ["r63_original/" + name for name in
                  ("network.py", "noise_core.py", "source_flow_core.py", "precision_order.py",
                   "R61_distribution_loss.py", "distribution_loss.py", "config.json")]
adapter_hashes = {name: hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() for name in adapter_names}
expected_tol = "24c3c78dad18846fd56180fccfff19a1e0f835908f649ab2eecae50fe43c266c"
assert adapter_hashes["FROZEN_TOLERANCES.json"] == expected_tol
outer = next(node for node in tree.body if isinstance(node, ast.Try) and isinstance(node.body[0], ast.Import))

retained = []
omitted = []
for node in outer.body:
    code = ast.unparse(node)
    skip = (isinstance(node, (ast.Import, ast.ImportFrom)) and
            any(term in code for term in ("distribution_gradient", "original_bindings", "weighted_empirical_w1")))
    skip = skip or code.startswith(("torch.set_num_threads(", "torch.set_num_interop_threads(",
                                   "torch.use_deterministic_algorithms(", "torch.backends.mha.set_fastpath_enabled("))
    skip = skip or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                    (ast.unparse(node.value.func).endswith("write_text") or ast.unparse(node.value.func) == "print"))
    skip = skip or code == "stage('receipt_written')"
    if skip:
        omitted.append({"line": node.lineno, "statement": code.splitlines()[0]})
    else:
        retained.append(copy.deepcopy(node))

edits = []


class DevicePort(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        func = ast.unparse(node.func)
        if func == "torch.device" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cpu":
            edits.append({"line": node.lineno, "kind": "original_R63_loss_device"})
            return ast.copy_location(ast.Name(id="device", ctx=ast.Load()), node)
        if func in {"torch.tensor", "torch.as_tensor", "torch.zeros", "torch.ones", "torch.full", "torch.arange", "torch.empty"}:
            if not any(k.arg == "device" for k in node.keywords):
                node.keywords.append(ast.keyword(arg="device", value=ast.Name(id="device", ctx=ast.Load())))
                edits.append({"line": node.lineno, "kind": "selected_device_tensor_constructor", "call": func})
        if func in {"dg.FrozenDistributionObjective", "WrongMass"}:
            if not any(k.arg == "device" for k in node.keywords):
                node.keywords.append(ast.keyword(arg="device", value=ast.Name(id="device", ctx=ast.Load())))
                edits.append({"line": node.lineno, "kind": "selected_device_distribution_objective"})
        if func == "torch.from_numpy":
            edits.append({"line": node.lineno, "kind": "CPU_array_to_selected_device"})
            return ast.copy_location(ast.Call(func=ast.Attribute(value=node, attr="to", ctx=ast.Load()),
                                              args=[ast.Name(id="device", ctx=ast.Load())], keywords=[]), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "numpy":
            value = node.func.value
            detach = ast.Call(func=ast.Attribute(value=value, attr="detach", ctx=ast.Load()), args=[], keywords=[])
            cpu = ast.Call(func=ast.Attribute(value=detach, attr="cpu", ctx=ast.Load()), args=[], keywords=[])
            node.func.value = cpu
            edits.append({"line": node.lineno, "kind": "explicit_detach_CPU_numpy"})
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "TESTS":
            node.value = ast.Subscript(value=ast.Name(id="state", ctx=ast.Load()), slice=ast.Constant(value="tests"), ctx=ast.Load())
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "receipt":
            assert isinstance(node.value, ast.Dict)
            keep = []
            for key, value in zip(node.value.keys, node.value.values):
                if key.value in {"CPU_affinity", "RSS_limit_bytes", "peak_RSS_bytes"}:
                    edits.append({"line": node.lineno, "kind": "supervisor_owned_resource_field", "field": key.value})
                    continue
                if key.value == "implementation_files":
                    value = ast.Name(id="manifest", ctx=ast.Load())
                if key.value == "limitations":
                    value.elts[-1] = ast.Constant(value="Selected-device artificial fixture only; first-order gradients only; no scientific arrays, weights or training are used.")
                keep.append((key, value))
            node.value.keys, node.value.values = [p[0] for p in keep], [p[1] for p in keep]
        return self.generic_visit(node)


before = ast.Module(body=copy.deepcopy(retained), type_ignores=[])
ported = DevicePort().visit(ast.Module(body=retained, type_ignores=[]))
assert sum(isinstance(n, ast.Assert) for n in ast.walk(before)) == sum(isinstance(n, ast.Assert) for n in ast.walk(ported))


def comparison_definitions(module):
    return [{"test_name_expression": ast.unparse(n.args[0]), "frozen_rule": ast.unparse(n.args[3])}
            for n in ast.walk(module) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "compare"]


assert comparison_definitions(before) == comparison_definitions(ported)
prefix = ast.parse('''
START = time.monotonic()
def stage(name):
    state['stages'].append({'stage': name, 'elapsed_s': round(time.monotonic()-START, 3)})
''').body
function = ast.FunctionDef(name="_run_fixture",
    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
                      ("device", "ROOT", "dg", "R63", "FrozenMass", "TOL", "TOL_SHA", "manifest", "state")],
                       vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
    body=prefix + ported.body + [ast.Return(value=ast.Name(id="receipt", ctx=ast.Load()))], decorator_list=[])
generated = ast.unparse(ast.fix_missing_locations(function)) + "\n"
destination = ROOT / "portable_fixture.py"
head = destination.read_text("utf8").split("# BEGIN GENERATED FROZEN FIXTURE BODY", 1)[0]
head = head.replace("__ADAPTER_HASH_MAP__", repr(adapter_hashes))
head = head.replace("__ORIGINAL_TEST_NAMES__", repr(tuple(original_receipt["tests"])))
result = head + "# BEGIN GENERATED FROZEN FIXTURE BODY\n" + generated
ast.parse(result)
compile(result, str(destination), "exec")
destination.write_text(result, encoding="utf8")

protected = []
protected_sets = (
    (GEOMETRY, ("endpoint_geometry.py", "artificial_check.py", "README.md", "PROTOCOL_DRAFT.json", "ARTIFICIAL_CHECK.json", "SOURCE_BINDING.json", "PREPARATION_RECEIPT.json")),
    (ORIGINAL, tuple(adapter_names) + ("artificial_fixture.py", "ARTIFICIAL_RECEIPT.json", "PREPARATION_RECEIPT.json", "NATIVE_EXIT_RECEIPT.json")),
)
for base, names in protected_sets:
    for name in names:
        path = base / name
        protected.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
report = {
    "UTC": datetime.now(timezone.utc).isoformat(),
    "status": "static_port_generated_no_fixture_execution",
    "original_fixture_path": str(source_path), "original_fixture_sha256": SOURCE_SHA,
    "original_receipt_sha256": hashlib.sha256((ORIGINAL / "ARTIFICIAL_RECEIPT.json").read_bytes()).hexdigest(),
    "original_tolerances_sha256": expected_tol,
    "portable_fixture_sha256_before_local_execution": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "expected_test_count": 69, "expected_test_names": list(original_receipt["tests"]),
    "retained_assert_count": sum(isinstance(n, ast.Assert) for n in ast.walk(before)),
    "comparison_name_and_tolerance_expressions_unchanged": True,
    "comparison_definitions": comparison_definitions(before),
    "device_only_AST_edits_and_resource_metadata_removal": edits,
    "omitted_setup_or_IO_statements": omitted,
    "adapter_hashes": adapter_hashes,
    "protected_existing_files": protected,
    "scientific_array_or_weight_reads": 0,
    "note": "Only source/config/previous artificial receipt text was read. No fixture body executed in this static port step. No original bytes were changed."
}
(ROOT / "PORT_SOURCE_BINDING.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
print(json.dumps({"status": report["status"], "expected_tests": 69, "asserts_preserved": report["retained_assert_count"],
                  "device_edits": len(edits), "portable_sha256": report["portable_fixture_sha256_before_local_execution"]}))
