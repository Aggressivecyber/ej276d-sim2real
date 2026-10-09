"""Final byte manifest and control-only QA; no scientific arrays are decoded."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEADLINE = dt.datetime.fromisoformat('2026-10-09T18:52:01.024475+00:00')
EXCLUDED = {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}

def utc():
    return dt.datetime.now(dt.timezone.utc)

def main():
    assert utc() < DEADLINE, 'seal deadline passed'
    assert not any((ROOT / p).exists() for p in EXCLUDED), 'stage was already sealed'
    required = ['REPORT.md', 'REPRODUCTION.md', 'METRICS_COMPACT.json',
                'RESOURCE_RECEIPT.json', 'NATIVE_CHAIN.json', 'FIGURE_QA.json',
                'REPLAY_RECEIPT.json', 'GATE_METRICS.json', 'SOURCE_SEMANTICS.json']
    assert all((ROOT / p).is_file() for p in required), 'missing delivery'
    resource = json.loads((ROOT / 'RESOURCE_RECEIPT.json').read_text(encoding='utf-8'))
    assert resource['science_elapsed_including_failures_s'] <= resource['science_limit_s']
    assert resource['actual_worker_RSS_peak_bytes'] <= resource['RSS_limit_bytes']
    assert resource['GPU_allocated_peak_bytes'] <= resource['GPU_allocated_limit_bytes']
    counters = resource['evaluation_counters']
    assert counters['HGB_attempts'] == 40 and counters['density_description_slots'] == 32
    assert counters['density_EM_attempts'] == 29 and counters['main_generation_rows'] == 8000
    assert resource['replay_rows'] == 8000
    figures = list((ROOT / 'results' / 'figures').glob('*.png'))
    assert len(figures) == 16
    entries = []
    returned = 0
    for p in sorted(ROOT.rglob('*')):
        if p.is_symlink():
            raise RuntimeError('symlink in stage: ' + str(p))
        if not p.is_file():
            continue
        name = p.relative_to(ROOT).as_posix()
        if name in EXCLUDED:
            continue
        assert utc() < DEADLINE, 'deadline during byte manifest'
        before = p.stat()
        returned += before.st_size
        assert returned + 1024 * 1024 <= 512 * 1024 * 1024, 'output cap'
        h = hashlib.sha256()
        n = 0
        with p.open('rb') as stream:
            while True:
                b = stream.read(1024 * 1024)
                if not b:
                    break
                n += len(b)
                h.update(b)
        after = p.stat()
        assert n == before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        entries.append({'path': name, 'bytes': n, 'sha256': h.hexdigest()})
    manifest = {
        'stage': 'R32_CONDITIONAL_MEAN_TRANSLATION_001',
        'sealed_UTC': utc().isoformat(), 'seal_child_PID': os.getpid(),
        'member_count': len(entries), 'member_bytes': returned,
        'byte_hash_read_bytes': returned, 'scientific_array_decodes_during_seal': 0,
        'excluded_self_references': sorted(EXCLUDED),
        'exclusion_reason': 'Manifest cannot hash itself; final receipt is written after native child exit.',
        'required_artifacts_present': True, 'final_figures_count': len(figures),
        'members': entries,
    }
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    assert utc() < DEADLINE
    (ROOT / 'FINAL_MANIFEST.json').write_bytes(raw)
    print(json.dumps({'PID': os.getpid(), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'manifest_bytes': len(raw), 'member_count': len(entries),
                      'member_bytes': returned, 'byte_hash_read_bytes': returned,
                      'control_only_QA': True, 'science_recomputed': False}))

if __name__ == '__main__':
    main()
