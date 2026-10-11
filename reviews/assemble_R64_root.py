"""Assemble byte-bound artificial-tested R64 helpers; no scientific imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shutil

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
P = R/'reviews/R64_PREPARATION_001'
assert not (S/'ASSEMBLY_RECEIPT.json').exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
records = []


def copy(source, target, expected=None):
    if expected is not None:
        assert sha(source) == expected, str(source)
    assert not target.exists(), str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    records.append(dict(source=str(source), target=str(target.relative_to(S)), bytes=target.stat().st_size, sha256=sha(target)))


base = P/'distribution_gradient'
receipt = json.loads((base/'PREPARATION_RECEIPT.json').read_text('utf8'))
assert receipt['native_exit_code'] == 0 and receipt['scientific_array_reads'] == 0
for name in ['distribution_gradient.py', 'original_bindings.py', 'weighted_empirical_w1.py',
             'SOURCE_BINDINGS.json', 'FROZEN_TOLERANCES.json']:
    expected = next(row['sha256'] for row in receipt['artifact_files'] if row['file'] == name)
    copy(base/name, S/'distribution_adapter'/name, expected)
bindings = json.loads((base/'SOURCE_BINDINGS.json').read_text('utf8'))
for name in ['network.py', 'noise_core.py', 'source_flow_core.py', 'precision_order.py',
             'R61_distribution_loss.py', 'distribution_loss.py', 'config.json']:
    expected = next(row['sha256'] for row in bindings if row['file'] == name)
    copy(base/'r63_original'/name, S/'distribution_adapter/r63_original'/name, expected)
helpers = [
    ('endpoint_geometry/endpoint_geometry.py', 'endpoint_geometry.py', '275425853ed2681ed8bbd88d7b1207bddeb5dd59fd8a8296069db5c3d474e9b3'),
    ('gradient_statistics/gradient_statistics.py', 'gradient_statistics.py', '99478d4e69b9c42d07d783b1cccdc6a85536e1992a9db2c2abf29d33fd5d5bdf'),
    ('portable_artificial/portable_fixture.py', 'portable_fixture.py', 'c19a1ffadcbfd41c10e76aab1bd3c300b9fb102c5a7a3b5b1000a6dfe006c753'),
    ('network_replay_artificial.py', 'network_replay_artificial.py', None),
]
for source, target, expected in helpers:
    copy(P/source, S/target, expected)
copy(R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/residual_math.py', S/'residual_math.py')
for path in S.rglob('*.py'):
    ast.parse(path.read_text('utf8'), filename=str(path))
value = dict(UTC=datetime.now(timezone.utc).isoformat(), owner='/root', status='artificial_tested_helpers_assembled',
             helpers=records, python_AST_pass=True, scientific_arrays_or_weights_opened=0,
             all_sources_and_tolerances_byte_identical=True, scientific_execution_started=False,
             model_goal_complete=False)
(S/'ASSEMBLY_RECEIPT.json').write_bytes((json.dumps(value, indent=2)+'\n').encode())
print(json.dumps(dict(status=value['status'], copied_helpers=len(records), python_AST_pass=True)))
