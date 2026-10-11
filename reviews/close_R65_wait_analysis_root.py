"""Close text-only waiting preparation; no scientific imports or operations."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[1]
W = R / 'reviews/R65_WAIT_ANALYSIS_001'
S = R / 'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
import local_resource_guard as guard
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 8)
resource = guard.apply(W / 'root_closure')
read = lambda p: json.loads(p.read_text('utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = []

def bind(p, expected=None, size=None):
    value = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
    assert expected is None or value['sha256'] == expected, str(p)
    assert size is None or value['bytes'] == size, str(p)
    records.append(value)
    return value

lit = W / 'finite_empirical_literature'
seal = read(lit / 'BINDING_RECEIPT.json')
bind(lit / 'BINDING_RECEIPT.json', '47ec7ec0a04d6f5db714cfb26c3f1d5aefb9fc09f5cad82696ee694ca96fe181')
for name, row in seal['artifacts'].items():
    bind(lit / name, row['sha256'], row['bytes'])
inputs = read(lit / 'INPUT_BINDING.json')['files']
assert len(inputs) == 15
for row in inputs:
    bind(Path(row['path']), row['sha256'], row['bytes'])
assert read(lit / 'NATIVE_EXIT_RECEIPT.json')['native_exit_code'] == 0
hyp = read(lit / 'HYPOTHESES.json')
assert hyp['status'] == 'PROPOSED_ONLY_NOT_REGISTERED_NOT_EXECUTED'
assert not hyp['applies_to_active_R65'] and not hyp['original_A_F_changed']
op = read(W / 'ROOT_OPERATIONAL_REVIEW.json')
assert op['post_controller_offline_checks_passed'] == 23
assert op['recovery_saved_record_cases_passed'] == 28
assert not op['scientific_entry_may_launch_now']
bind(W / 'ROOT_OPERATIONAL_REVIEW.json')
for row in op['files']:
    bind(R / row['path'], row['sha256'], row['bytes'])
for sub in ('root_synthesis', 'final_validation'):
    for p in sorted((W / sub).glob('*')):
        if p.is_file():
            bind(p)
result = {
    'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'ROOT_WAIT_PREPARATION_REVIEW_COMPLETE',
    'literature_primary_sources_verified': [
        {'url': 'https://proceedings.mlr.press/v80/rainforth18a/rainforth18a.pdf', 'locator': 'section 3, Theorem 1'},
        {'url': 'https://arxiv.org/pdf/1312.2128v1', 'locator': 'section 1, Theorem 1'},
        {'url': 'https://arxiv.org/pdf/1412.6980v9', 'locator': 'Algorithm 1'},
        {'url': 'https://arxiv.org/html/2101.01792v1', 'locator': 'Theorems 29-30'},
        {'url': 'https://arxiv.org/pdf/2001.09061v1', 'locator': 'Propositions 2.1-2.2'}
    ],
    'scientific_findings_are_theory_or_previous_metadata_only': True,
    'Jensen_scope': 'fixed theta, directions, feature maps, scales, reference and region masses; forward five D values only',
    'no_claim_of_unbiased_STE_gradient_or_A_F_improvement': True,
    'coprime_archive_counts_do_not_prove_PSD_threshold_unattainable': True,
    'H1_H2_status': hyp['status'],
    'D_E_preparation_not_execution': True,
    'GAP_Q6_empty_is_conditional_on_actual_charge_and_edge_binding': True,
    'post_controller_offline_cases': 23,
    'recovery_offline_cases': 28,
    'existing_scientific_R65_implementation_unchanged': True,
    'actual_training_native_exit_not_yet_observed': True,
    'watcher_eligible_after_agents_idle': True,
    'scientific_post_entry_eligible_now': False,
    'new_scientific_jobs': 0,
    'scientific_arrays_or_weights_decoded': 0,
    'original_A_F_changed': False,
    'model_goal_complete': False,
    'resource': dict(resource, **guard.memory()),
    'files': list({row['path']: row for row in records}.values())
}
dest = W / 'ROOT_CLOSURE_REVIEW.json'
assert not dest.exists(), 'Inspect an existing closure instead of overwriting it.'
dest.write_bytes((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
assert read(dest) == result
print(json.dumps({'status': result['status'], 'bound_files': len(result['files']),
                  'path': str(dest), 'sha256': sha(dest), 'resource': result['resource']}, indent=2))
