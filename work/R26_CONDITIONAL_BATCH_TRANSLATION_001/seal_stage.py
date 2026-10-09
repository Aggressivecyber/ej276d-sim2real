"""Pure control finalization. No scientific arrays or models are decoded."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEADLINE = datetime.fromisoformat('2026-10-09T14:00:38.792322+00:00')
EXCLUDED = {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}


def utc():
    return datetime.now(timezone.utc)


def check_clock():
    if utc() >= DEADLINE:
        raise RuntimeError('R26 seal deadline reached')


def read_json(name):
    return json.loads((ROOT / name).read_bytes())


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_json(name, value):
    check_clock()
    (ROOT / name).write_bytes(encode(value))


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    started = utc()
    timer = time.perf_counter()
    check_clock()
    if any((ROOT / name).exists() for name in EXCLUDED):
        raise RuntimeError('Existing final seal must not be overwritten')
    resource = read_json('RESOURCE_RECEIPT.json')
    compact = read_json('METRICS_COMPACT.json')
    history = read_json('RUN_HISTORY.json')
    replay = read_json('REPLAY_RECEIPT.json')
    qa = read_json('FIGURE_QA.json')
    assert len(history) == 6
    assert all('stop_UTC' in r and 'actual_worker_native_exit' in r for r in history)
    assert [r['entry'] for r in history if r['actual_worker_native_exit'] != 0] == ['evaluate_endpoints.py']
    assert abs(sum(r['elapsed_s'] for r in history) - resource['science_s']) < 1e-8
    assert resource['science_s'] <= 1800 and resource['training_s'] <= 900
    assert resource['actual_worker_RSS_peak_bytes'] <= 4 * 1024**3
    assert resource['GPU_peak_allocated_bytes'] <= 4 * 1024**3
    assert resource['memory_probe_failures'] == 0
    assert replay['pass'] is True and replay['rows'] == 6000
    assert qa['all_figures_readable'] is True and len(qa['reviewed']) == 10
    counts = resource['counts']
    assert counts['G_updates'] == counts['D_updates'] == counts['reader_updates'] == 2950
    assert counts['HGB_attempts'] == 21 and counts['density_EM_attempts'] == 44
    assert resource['EM_original_consumed'] == resource['EM_recovery_consumed'] == 22
    assert counts['main_generation_rows'] == 6000
    assert counts['repeat_noise_rows'] == 3072 and counts['source_probe_rows'] == 2496
    assert counts['EVAL_read'] == counts['CAL_waveform_decode'] == 0
    assert compact['private_payloads_published'] is False

    # A copied evaluator counter describes evaluation only, not the training phase.
    correction = {
        'field': 'response_updates',
        'inherited_evaluation_only_value': counts['response_updates'],
        'actual_response_training_updates': counts['G_updates'],
        'scope': 'whole R26 stage',
        'original_native_and_scientific_results_unchanged': True,
    }
    counts['response_updates'] = counts['G_updates']
    compact['counters']['response_updates'] = counts['G_updates']
    compact['counters']['D_updates'] = counts['D_updates']
    compact['counters']['reader_updates'] = counts['reader_updates']
    compact['counter_scope_correction'] = correction
    resource['counter_scope_correction'] = correction
    write_json('METRICS_COMPACT.json', compact)

    # Remove doubled Windows CRs so Markdown table rows stay adjacent.
    normalized = []
    for name in ('REPORT.md', 'REPRODUCTION.md'):
        p = ROOT / name
        before = p.read_bytes()
        after = before.decode('utf-8').replace('\r\n', '\n').replace('\r', '').encode('utf-8')
        if after != before:
            check_clock()
            p.write_bytes(after)
            normalized.append(name)

    resource['finalization'] = {
        'PID': os.getpid(),
        'start_UTC': started.isoformat(),
        'scientific_fit_generation_updates': 0,
        'normalization_only_files': normalized,
        'run_history_native_failures_preserved': True,
        'output_bytes_exact_in_FINAL_RECEIPT': True,
        'final_byte_hash_is_not_scientific_decode': True,
    }
    write_json('RESOURCE_RECEIPT.json', resource)

    files = []
    for p in sorted(ROOT.rglob('*')):
        check_clock()
        if p.is_symlink():
            raise RuntimeError('Unexpected symlink in stage')
        if not p.is_file() or p.relative_to(ROOT).as_posix() in EXCLUDED:
            continue
        files.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                      'sha256': sha_file(p)})
    member_bytes = sum(f['bytes'] for f in files)
    assert member_bytes < resource['new_output_cap_bytes']
    manifest = {
        'stage': resource['stage'], 'UTC': utc().isoformat(), 'seal_PID': os.getpid(),
        'files': files, 'file_count': len(files), 'total_member_bytes': member_bytes,
        'exclusions': sorted(EXCLUDED),
        'exclusion_reason': 'Manifest and final receipt are mutually self-referential control files; receipt binds manifest SHA.',
        'full_stage_output_cap_bytes': resource['new_output_cap_bytes'],
        'science_array_decode_during_seal': 0,
    }
    write_json('FINAL_MANIFEST.json', manifest)
    manifest_path = ROOT / 'FINAL_MANIFEST.json'
    manifest_hash = sha_file(manifest_path)
    key_hashes = {name: next(f['sha256'] for f in files if f['path'] == name)
                  for name in ('REPORT.md', 'METRICS_COMPACT.json', 'RESOURCE_RECEIPT.json', 'REPLAY_RECEIPT.json')}
    receipt = {
        'stage': resource['stage'], 'status': 'sealed', 'PID': os.getpid(),
        'start_UTC': started.isoformat(), 'control_timestamp_before_final_write_UTC': utc().isoformat(),
        'elapsed_s_before_final_receipt_write': time.perf_counter() - timer,
        'deadline_UTC': DEADLINE.isoformat(), 'manifest_sha256': manifest_hash,
        'manifest_member_count': len(files), 'stage_file_count': len(files) + 2,
        'output_total_bytes': 0, 'key_sha256': key_hashes,
        'science_s': resource['science_s'], 'training_s': resource['training_s'],
        'counts': counts, 'all_actual_scientific_jobs_terminal': True,
        'science_fit_generation_during_seal': 0, 'no_further_stage_writes': True,
        'process_native_exit_evidence': 'The caller exec_command response records the actual native exit; this file is written before process exit.',
        'last_write_evidence': 'Caller stdout reports FINAL_RECEIPT filesystem mtime after the last write.',
        'scope_correction': correction,
        'self_referential_exclusions': sorted(EXCLUDED),
    }
    # Determine exact file bytes without re-writing the self-referential receipt.
    for _ in range(10):
        data = encode(receipt)
        total = member_bytes + manifest_path.stat().st_size + len(data)
        if total == receipt['output_total_bytes']:
            break
        receipt['output_total_bytes'] = total
    else:
        raise RuntimeError('Receipt size did not stabilize')
    assert total <= resource['new_output_cap_bytes']
    check_clock()
    target = ROOT / 'FINAL_RECEIPT.json'
    target.write_bytes(data)
    # No file mutations after this point.
    print(json.dumps({
        'PID': os.getpid(), 'status': 'sealed',
        'last_stage_write_UTC': datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(),
        'actual_stop_UTC': utc().isoformat(), 'manifest_sha256': manifest_hash,
        'final_receipt_sha256': hashlib.sha256(data).hexdigest(),
        'manifest_member_count': len(files), 'stage_file_count': len(files) + 2,
        'output_total_bytes': total, 'key_sha256': key_hashes,
        'science_s': resource['science_s'], 'no_further_stage_writes': True,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
