from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
B = R / 'reviews/R63_WAIT_PARALLEL_001/literature/c1_preparation'
receipt_path = B / 'CODE_PREPARATION_RECEIPT.json'
receipt_bytes = receipt_path.read_bytes()
receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
assert receipt_sha == '26682374b35a7f7cf4312c43cf763002aa86ac1610fe51ca1bfea13f1d6a8a0a'
receipt = json.loads(receipt_bytes)
results = json.loads((B / 'attempt_001/ARTIFICIAL_TEST_RESULTS.json').read_text('utf8'))
assert receipt['status'] == results['status'] == 'passed'
assert receipt['test_groups'] == receipt['passed_groups'] == results['test_groups'] == results['passed_groups'] == 9
assert receipt['immutable_reports_byte_identical'] is True
for key in ['scientific_arrays_read', 'scientific_model_checkpoints_loaded',
            'scientific_forwards_or_gradients_or_fits', 'network_training_updates']:
    assert receipt[key] == 0
assert len(receipt['attempts']) == 1 and receipt['attempts'][0]['exit_code'] == 0
bindings = receipt['input_bindings_before'] + receipt['source_and_fixture_bindings'] + receipt['output_bindings']
checked = []
for item in bindings:
    path = Path(item['path'])
    assert path.drive.upper() in ['E:', 'D:']
    content = path.read_bytes()
    current = hashlib.sha256(content).hexdigest()
    assert len(content) == item['bytes'] and current == item['sha256'], str(path)
    checked.append({'path': str(path.relative_to(R)).replace('\\', '/'),
                    'bytes': len(content), 'SHA256': current, 'match': True})
assert len(checked) == len({item['path'] for item in checked}) == 12
out = {
    'UTC': datetime.now(timezone.utc).isoformat(),
    'owner': '/root',
    'status': 'code_preparation_accepted_for_future_registered_integration',
    'pass': True,
    'receipt': str(receipt_path.relative_to(R)).replace('\\', '/'),
    'receipt_SHA256': receipt_sha,
    'source_review': 'Root reviewed the weighted CDF partition, fixed tie branch, analytic and independent Fraction oracle tests, duplicate-parent VJP accumulation and separate fixed-REAL versus shared-parameter REAL roles before this audit.',
    'binding_checks': checked,
    'native_exit_code': receipt['attempts'][0]['exit_code'],
    'test_group_count': 9,
    'passed_groups': 9,
    'groups': [item['name'] for item in results['results']],
    'original_runtime': receipt['runtime'],
    'original_peak_RSS_bytes': receipt['attempts'][0]['resource_peak_bytes'],
    'original_elapsed_seconds': receipt['attempts'][0]['elapsed_seconds'],
    'tests_rerun_by_this_audit': False,
    'original_reports_unchanged': True,
    'scientific_data_or_weights_loaded': 0,
    'scientific_forwards_gradients_or_updates': 0,
    'limitations': [
        'These are artificial scalar/vector objectives, not the integrated R63 five-term distribution objective.',
        'No original network, scientific endpoint, CUDA replay or scientific gradient was exercised.',
        'Stable tie-selected gradients are conditional algorithmic choices, not unique gradients at ties.',
        'Parent-fixed-noise finite reference and fixed directions differ from the original stochastic training law.',
        'Exact integer mass partition does not make floating-point distances, objectives or VJPs exact arithmetic.',
        'Future scientific use requires a distinct registered protocol and objective-level artificial compatibility checks.'
    ]
}
target = R / 'reviews/R63_WAIT_PARALLEL_001/root_analysis/C1_ROOT_CODE_REVIEW.json'
assert not target.exists()
target.write_bytes((json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
print(json.dumps({'pass': True, 'bindings': 12, 'artificial_groups': 9,
                  'scientific_computations': 0, 'review': str(target)}, ensure_ascii=False))
