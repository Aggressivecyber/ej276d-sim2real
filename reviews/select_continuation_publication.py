"""Select explicit source/summary paths; retain event data and weights locally."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument('--stage')
p.add_argument('--receipt', required=True)
p.add_argument('--root', nargs='*', default=[])
a = p.parse_args()
paths = [ROOT / s for s in a.root]
if a.stage:
    stage = ROOT / 'work' / a.stage
    names = {
        'config.json', 'PROTOCOL.md', 'REPORT.md', 'METRICS_COMPACT.json',
        'START_RECEIPT.json', 'RUN_HISTORY.json', 'FINAL_MANIFEST.json',
        'FINAL_RECEIPT.json', 'NUMERIC_RECEIPT.json', 'READINESS.json',
        'INPUTS.json', 'SOURCE_BINDINGS.json', 'PREPARATION_SOURCE_READS.json',
        'PRE_NUMERIC_NONFINITE_GUARD.json', 'PAIRED_TRAINING_RECEIPT.json',
        'REPLAY_RECEIPT.json', 'VISUAL_QA.json', 'LAYOUT_REPAIR_RECEIPT.json',
        'RESOURCE_RECEIPT.json', 'FIGURE_QA.json', 'LAYOUT_QA_RECEIPT.json',
        'DELIVERY_NATIVE_LINKS.json',
        'FROZEN_BINDING_CHECK.json', 'EVAL_EXTRACTION_RECEIPT.json',
    }
    paths += [v for v in stage.iterdir() if v.is_file() and
              (v.suffix == '.py' or v.name in names)]
    paths += list(stage.glob('numeric_*/NATIVE_RECEIPT.json'))
    paths += list(stage.glob('results/figures/*.png'))
    for name in ('TRAINING_RECEIPT.json', 'TRAINING_TRACE.json', 'GENERATION_RECEIPT.json'):
        paths += list(stage.glob('arms/*/' + name))

paths = sorted(set(paths), key=lambda v: v.as_posix())
entries = []
issues = []
credentials = re.compile(rb'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{20,}')

def inspect(v, where):
    if isinstance(v, dict):
        for k, value in v.items():
            label = str(k).lower()
            if label in {'parent_ids', 'event_ids', 'sample_ids', 'waveforms', 'white_noise', 'latent_values'} and isinstance(value, list):
                issues.append({'kind': 'possible_event_payload', 'location': where + '/' + str(k)})
            inspect(value, where + '/' + str(k))
    elif isinstance(v, list):
        if len(v) >= 1000 and all(not isinstance(x, (dict, list)) for x in v):
            issues.append({'kind': 'long_primitive_array', 'location': where, 'length': len(v)})
        for i, value in enumerate(v):
            if isinstance(value, (dict, list)):
                inspect(value, where + '/' + str(i))

for path in paths:
    assert path.resolve().is_relative_to(ROOT), path
    assert path.is_file(), path
    relative = path.relative_to(ROOT).as_posix()
    assert path.suffix.lower() not in {'.pt', '.pth', '.npz', '.npy', '.csv', '.log', '.bin', '.zip'}
    payload = path.read_bytes()
    if credentials.search(payload):
        issues.append({'kind': 'credential_pattern', 'path': relative})
    if path.suffix.lower() == '.json':
        inspect(json.loads(payload.decode('utf-8-sig')), relative)
    entries.append({'path': relative, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})

result = {'files': entries, 'file_count': len(entries), 'total_bytes': sum(v['bytes'] for v in entries),
          'issues': issues, 'scientific_binary_payloads_selected': False,
          'status': 'ready_for_explicit_staging' if not issues else 'requires_local_inspection'}
target = ROOT / a.receipt
assert target.resolve().is_relative_to(ROOT)
with target.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'receipt': a.receipt, 'file_count': result['file_count'],
                  'total_bytes': result['total_bytes'], 'issues': issues}, ensure_ascii=False))
if issues:
    raise SystemExit(2)
