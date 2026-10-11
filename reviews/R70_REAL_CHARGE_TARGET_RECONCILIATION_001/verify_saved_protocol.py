"""Read back saved target semantics, inherited thresholds, lineage and references."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
P = Path(__file__).resolve().parent
R = P.parents[1]
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'verification_runtime')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


receipt = read(P / 'ROOT_RECONCILIATION_RECEIPT.json')
old = read(R / 'PROJECT_ACCEPTANCE.json')
new = read(R / 'PROJECT_ACCEPTANCE_REAL_CHARGE_V5.json')
pointer = read(R / 'CURRENT_ACCEPTANCE.json')
assert pointer['version'] == new['version'] == 5
assert sha(R / pointer['registry']) == pointer['registry_sha256']
assert sha(R / pointer['document']) == pointer['document_sha256'] == new['document_sha256']
for name, digest in {**receipt['original_acceptance_SHA256'], **receipt['historical_frozen_SHA256']}.items():
    assert sha(R / name) == digest, name
checks = []
for key, value in old['final_response'].items():
    if key == 'noise_and_source_and_boundary_and_replay':
        continue
    assert new['final_response'][key] == value, key
    checks.append(key)
assert new['utility'] == old['utility']
for key in ('median_LSB_max', 'p95_LSB_max', 'independent_main_replay_rows_per_model'):
    assert new['numerical_precision'][key] == old['numerical_precision'][key], key
assert new['numerical_precision']['preADC_relative_Q_target_error_max'] == old['numerical_precision']['preADC_relative_q_max']
assert new['numerical_precision']['old_SOURCE_Q_conservation_gate_replaced']
assert new['numerical_precision']['Q_predicted_independent_of_output_integral']
assert not new['charge_target']['SOURCE_original_Q_hard_preservation']
assert new['charge_target']['SOURCE_parent_identity_composition_order_and_weight_preserved']
assert not new['charge_target']['REAL_identity_bypass_or_domain_flag_allowed']
assert new['scope']['retain_all_legal_SOURCE_TRAIN_including_nominal_gap']
assert new['scope']['hide_GAP_REAL_TRAIN_and_all_derived_statistics']
assert not new['scope']['unobserved_mask_means_zero_probability']
assert not new['scope']['GAP_may_borrow_FULL_weights_or_gap_REAL_statistics']
assert not new['historical_R1_R69_results_changed']
assert new['source_and_boundary']['original_source_parent_count'] == 2000
assert new['source_and_boundary']['original_outside_support_count_retained'] == 417
assert read(R / 'jobs.json')['active_scientific_jobs'] == 0
assert sha(R / 'jobs.json') == sha(P / 'before_target_correction/jobs.json')
for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
    data = read(R / name)
    assert data['acceptance_registry'] == pointer['registry']
    assert data['acceptance_document'] == pointer['document']
    assert data['acceptance_version'] == 5
    assert data['application_goal_status'] == read(P / 'APP_GOAL_OBSERVATION.json')['result']['goal']['status'] == 'paused'
    assert data['active_scientific_jobs'] == 0 and data['current_job'] is None
    assert not data['model_goal_complete'] and not data['original_A_F_complete']
    assert not data['numeric_execution_released']
    assert not data['R65_cost_hold']['automatic_resume']
for name in ('AGENTS.md', 'STATUS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'LONG_TERM_GOAL_20261010.md'):
    current = (R / name).read_bytes()
    original = (P / 'before_target_correction' / name).read_bytes()
    assert current.endswith(original), name
    assert pointer['document'] in current[:3000].decode('utf8'), name
stage = next(x for x in read(R / 'experiments.json')['experiments'] if x['id'] == P.name)
assert not stage['scientific_execution'] and stage['model_fits'] == stage['model_inferences'] == stage['science_array_reads'] == 0
audited = []
for manifest, key in [('acceptance_audit/INPUT_SHA256.json', 'inputs'), ('method_audit/SHA256_MANIFEST.json', 'scientific_sources')]:
    for row in read(P / manifest)[key]:
        assert sha(R / row['path']) == row['sha256'], row['path']
        audited.append(row['path'])
report_hashes = {str(Path(x).as_posix()): sha(P / x) for x in ['acceptance_audit/REPORT.md', 'method_audit/REPORT.md']}
result = {
    'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_STATIC_PROTOCOL_READBACK',
    'native_exit': 0, 'reviewed_document_SHA256': new['document_sha256'],
    'reviewed_registry_SHA256': pointer['registry_sha256'],
    'inherited_response_numeric_fields_checked': checks,
    'all_utility_fields_equal': True, 'numerical_tolerance_values_equal_explicit_new_Q_target': True,
    'original_acceptance_and_historical_frozen_hashes_unchanged': True,
    'five_original_markdown_bodies_preserved_byte_for_byte': True,
    'root_metadata_and_current_pointer_consistent': True,
    'audited_source_files_rehashed': sorted(set(audited)), 'audit_report_SHA256': report_hashes,
    'root_semantic_decision': 'Accept direct human REAL-Q priority; keep original source parents and weights; replace absolute source-Q invariance explicitly; use output-Q conditional evaluation and REAL-only heldout gap.',
    'same_treated_REAL_control_retained': True, 'gap_mask_is_missing_not_zero_mass': True,
    'new_scientific_execution': False, 'scientific_gate_results_inferred_from_static_checks': False,
    'model_goal_complete': False, 'app_goal_changed': False, 'resource': {**guard, **memory()},
}
(P / 'ROOT_STATIC_VERIFICATION.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
print(json.dumps({'status': result['status'], 'verified_static_sources': len(set(audited)),
                  'historical_hashes_unchanged': True, 'model_goal_complete': False}, ensure_ascii=False))
