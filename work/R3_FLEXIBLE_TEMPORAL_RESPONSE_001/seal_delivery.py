"""One stdlib-only delivery pass; no scientific arrays or recomputation."""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def utc():
    return dt.datetime.now(dt.timezone.utc)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


start = read('START.json')
deadline = dt.datetime.fromisoformat(start['stage_deadline_UTC'])
assert utc() < deadline
assert not (ROOT / 'FINAL_RECEIPT.json').exists(), 'delivery already fixed'


def write(name, obj):
    assert utc() < deadline, 'stage deadline'
    data = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    assert len(data) <= 65536
    (ROOT / name).write_bytes(data)


prep = read('PREPARATION_RECEIPT.json')
for name, expected in prep['source'].items():
    assert digest(ROOT / name) == expected, f'executed source changed: {name}'
    ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name)
assert digest(ROOT / 'config.json')['sha256'] == prep['config_sha256']
m = read('metrics.json')
history = read('RUN_HISTORY.json')
replay = read('REPLAY_CHECK.json')
assert len(history) == 2 and all(x['worker_native_exit'] == 0 for x in history)
assert replay['bitwise_equal'] and replay['float32_values_compared'] == 960000
assert replay['parent_ID_order_equal'] and replay['max_absolute_difference_V'] == 0
assert digest(ROOT / 'model.pt')['sha256'] == replay['model_SHA256']
assert digest(ROOT / 'generation_inputs.npz')['sha256'] == replay['generation_inputs_SHA256']
assert digest(ROOT / 'GENERATION_RANDOM_STATE.pt')['sha256'] == replay['generation_RNG_SHA256']
assert m['current_model_fit_DEV_events'] == 0 and m['EVAL_read'] == 0
assert m['training']['optimizer_updates'] == 2000

figures = sorted(x.name for x in (ROOT / 'results/figures').glob('*.png'))
assert len(figures) == 9
write('DELIVERY_QA.json', {
    'UTC': utc().isoformat(), 'PID': os.getpid(),
    'executed_source_and_config_bindings_unchanged': True,
    'source_AST_parse': True,
    'plots_actually_viewed_before_delivery': figures,
    'plot_layout': 'Readable axes, legends and complete distributions; no clipping found.',
    'current_model_fit_DEV_events': m['current_model_fit_DEV_events'],
    'EVAL_read': 0, 'SSH': 0, 'new_G4': 0,
    'R3_QC_valid': m['domains']['R3']['QC_valid'],
    'REAL_QC_valid': m['domains']['REAL']['QC_valid'],
    'ADC_clip_count': m['generation']['ADC_clip_count'],
    'padding_count': m['generation']['padding_count'],
    'fresh_saved_weight_replay': replay,
    'R3_seed_before_model_constructor': 202610083,
    'R2_original_hidden_initialization': 'unknown; exact external checkpoint is bound',
    'whole_response_similarity_pass': False,
    'remaining_findings': [
        'Near-perfect domain classification remains.',
        'q3/q4 conditional distances and q4 coverage worsen relative to R2.',
        'Some retained single-event responses contain leading shoulders or local double peaks.',
        'Artificial groups are from one acquisition, not independent physical runs.'
    ],
    'preserved_config_explanation_discrepancy':
        'Composition-pair exact ties use stable source-array order in executed code; config SHA-tie text is not implemented. No pair was changed.',
    'science_recomputed_for_delivery': False
})

write('METRICS_COMPACT.json', {
    'status': 'fixed_DEV_diagnostic_completed; full_response_similarity_failed',
    'training': m['training'],
    'domain_vs_REAL': m['domain_vs_REAL'],
    'R3_C2ST': m['C2ST']['R3_vs_REAL'],
    'R3_region_QC_counts': {k: v['PSD']['counts'][0]
                           for k, v in m['domain_vs_REAL']['R3'].items()},
    'CAL_reference_valid': m['CAL_reference_valid'],
    'q4_no_fallback': m['q4_no_fallback'],
    'generation': m['generation'],
    'parent_association_R3': m['parent_association_R3'],
    'fixed_latent_condition_examples': m['fixed_latent_condition_examples'],
    'replay': replay,
    'complete_metrics': 'metrics.json',
    'not_independent_external_test': True,
    'EVAL_read': 0
})

numeric_sum = sum(x['wall_s'] for x in history)
reserved_numeric_sum = history[-1]['numeric_cumulative_wall_s']
bytes_before_final_inventory = sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
assert reserved_numeric_sum <= 1200
assert max(x['peak_RSS_observed'] for x in history) <= 4 * 1024 ** 3
assert bytes_before_final_inventory + 65536 <= 512 * 1024 ** 2
write('RESOURCE_RECEIPT.json', {
    'UTC': utc().isoformat(), 'PID': os.getpid(),
    'start': start, 'one_fixed_recipe': True,
    'response_training_jobs': 1, 'saved_weight_replay_jobs': 1,
    'concurrent_numeric_jobs': 1,
    'optimizer_updates': 2000, 'CPU_threads': 4,
    'numeric_wall_s_sum': numeric_sum,
    'supervisor_cumulative_numeric_wall_s': reserved_numeric_sum,
    'numeric_limit_s': 1200,
    'peak_child_RSS_observed_B': max(x['peak_RSS_observed'] for x in history),
    'peak_child_RSS_observed_limit_B': 4 * 1024 ** 3,
    'RSS_probe_failures': sum(x['memory_probe_failures'] for x in history),
    'RSS_is_sampled_observed_peak': True,
    'output_before_final_inventory_B': bytes_before_final_inventory,
    'remaining_delivery_write_upper_B': 65536,
    'new_stage_output_limit_B': 512 * 1024 ** 2,
    'native_runs': history,
    'native_tool_evidence': [
        {'kind': 'training_generation_evaluation', 'start_tool': 'da15c8',
         'completed_tool': '3eb502', 'session_ID': 21991, 'native_exit': 0},
        {'kind': 'fresh_saved_weight_replay', 'completed_tool': 'e8b443', 'native_exit': 0}
    ],
    'source_preparation_tool': {'tool': 'a8ed2c', 'PID': 62160, 'native_exit': 0},
    'failure_or_retry_count': 0,
    'warning': 'Tensor-to-scalar warning preserved in run_001/stderr.log; no scientific rerun.',
    'EVAL_read': 0, 'SSH': 0, 'new_G4': 0,
    'write_locations': [str(ROOT).replace('\\', '/')],
    'cache_locations': ['E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp',
                        str(ROOT / 'runtime_cache').replace('\\', '/')],
    'delivery_hashes_are_byte_verification_only': True,
    'old_stages_modified': False
})

members = []
for path in sorted(ROOT.rglob('*')):
    if path.is_file() and path.name not in {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}:
        members.append({'path': path.relative_to(ROOT).as_posix(), **digest(path)})
write('FINAL_MANIFEST.json', {
    'UTC': utc().isoformat(), 'PID': os.getpid(),
    'member_count': len(members), 'member_bytes': sum(x['bytes'] for x in members),
    'self_reference_excluded': ['FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'],
    'hash_operation': 'Pure byte SHA256; no scientific array decoding.',
    'files': members
})

anchors = {name: digest(ROOT / name) for name in [
    'REPORT.md', 'metrics.json', 'METRICS_COMPACT.json', 'model.pt', 'generated.npz',
    'generation_inputs.npz', 'GENERATION_RANDOM_STATE.pt', 'REPLAY_CHECK.json',
    'RESOURCE_RECEIPT.json', 'FINAL_MANIFEST.json'
]}
write('FINAL_RECEIPT.json', {
    'seal_UTC': utc().isoformat(), 'seal_PID': os.getpid(),
    'status': 'completed; no further stage writes',
    'stage_deadline_UTC': start['stage_deadline_UTC'],
    'stage_write_stop_after_this_file': True,
    'manifest_member_count': len(members), 'anchors': anchors,
    'final_native_exit': 'The calling native-tool receipt is authoritative; not self-asserted here.',
    'science_result': 'PSD and rise marginal improvement; full joint response still clearly distinguishable.',
    'replay_pass': True, 'EVAL_read': 0, 'new_fit_during_delivery': 0
})
files = [x for x in ROOT.rglob('*') if x.is_file()]
total = sum(x.stat().st_size for x in files)
assert total <= 512 * 1024 ** 2
last = max(x.stat().st_mtime_ns for x in files)
print(json.dumps({
    'seal_PID': os.getpid(), 'completed_UTC': utc().isoformat(),
    'last_file_write_UTC': dt.datetime.fromtimestamp(last / 1e9, dt.timezone.utc).isoformat(),
    'file_count': len(files), 'total_bytes': total,
    'manifest': digest(ROOT / 'FINAL_MANIFEST.json'),
    'receipt': digest(ROOT / 'FINAL_RECEIPT.json'), 'anchors': anchors,
    'numeric_cumulative_s': reserved_numeric_sum,
    'source_and_config_unchanged': True, 'science_recomputed': False
}, ensure_ascii=False))
