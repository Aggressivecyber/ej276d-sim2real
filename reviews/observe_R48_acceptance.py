"""Record the registered numerical decision from existing R48 metadata only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
REL = 'work/' + STAGE + '/'
OUT = ROOT / 'reviews/R48_ACCEPTANCE_OBSERVATION_001.json'


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


assert not OUT.exists()
check = read(REL + 'TRAIN_SOLVER_CHECK.json')
training = read(REL + 'TRAINING_COMPACT.json')['training']
arms = ['GUIDED_FIXED_NOISE_CONTROL', 'GUIDED_NOISE_COMPLETION']
assert set(training) == set(arms) == set(check['models'])
assert check['both_pass'] is False and check['DEV_read_or_generated'] == 0
assert all(training[a]['updates'] == 50000 for a in arms)
assert training[arms[0]]['initial_state_SHA'] == training[arms[1]]['initial_state_SHA']
assert training[arms[0]]['stream_hashes'] == training[arms[1]]['stream_hashes']
assert len(training[arms[1]]['paired_stream_equality']) == 7
assert all(training[arms[1]]['paired_stream_equality'].values())
rows = {}
for arm in arms:
    r = check['models'][arm]
    assert r['median_limit_LSB'] == .25 and r['p95_limit_LSB'] == 1.
    assert r['relative_q_limit'] == 1e-5 and r['candidate_steps'] == 64 and r['reference_only_steps'] == 128
    assert r['RMS_median_LSB'] <= .25 and r['RMS_p95_LSB'] > 1.
    assert r['preADC_relative_q_max'] <= 1e-5 and r['pass'] is False
    rows[arm] = {k: r[k] for k in ['RMS_median_LSB', 'RMS_p95_LSB', 'preADC_relative_q_max',
                                  'median_limit_LSB', 'p95_limit_LSB', 'relative_q_limit', 'pass']}
bindings = []
for name in ['tasks/' + STAGE + '.md', REL + 'TRAIN_SOLVER_CHECK.json', REL + 'TRAINING_COMPACT.json']:
    raw = (ROOT / name).read_bytes()
    bindings.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
result = {'stage': STAGE, 'observed_UTC': datetime.now(timezone.utc).isoformat(),
    'actual_control_PID': os.getpid(), 'metadata_only_no_scientific_recomputation': True,
    'decision': 'close_fixed_R48_recipe_at_registered_numerical_gate_before_DEV',
    'primary': arms[1], 'fixed_control': arms[0], 'numerical_results': rows,
    'same_initial_state_and_all_seven_streams': True, 'new_response_updates_per_arm': 50000,
    'new_response_training_runs_completed': 2,
    'A_pass': None, 'A_status': 'not_evaluated_no_DEV_outputs',
    'B_pass': None, 'B_status': 'not_evaluated_no_DEV_outputs',
    'C_pass': False, 'C_status': 'registered_64_128_p95_numerical_requirement_failed',
    'remaining_C_requirements': 'not_evaluated_in_R48',
    'D_E_status': 'not_executed_no_A_C_accepted_candidate',
    'F_status': 'stage_evidence_delivery_pending; no_accepted_final_model_or_main_replay',
    'DEV_generated_rows': 0, 'HGB_attempts': 0, 'density_EM_attempts': 0, 'main_replay_rows': 0,
    'counter_basis': 'Registered stop branch and numerical receipt; reconcile against sealed resource receipt before publication.',
    'original_scientific_acceptance_A_F_changed': False,
    'higher_step_search_performed': False, 'project_model_goal_complete': False,
    'interpretation': 'Both fixed 64-step endpoints fail the prespecified p95 integration comparison. This does not establish that guided noise completion is ineffective or identify physical noise.',
    'loss_interpretation': 'The arms have different identity velocity targets; lower identity training loss alone is not an equal-task output-quality comparison.',
    'method_budget': {'registered_plans': 3, 'maximum_plans': 3, 'automatic_extension_allowed': False},
    'next_action': 'Finish registered negative-result figures, seal, private Git publication and D backup; start no new scientific method or step search.',
    'bindings': bindings}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'path': str(OUT), 'decision': result['decision'], 'actual_control_PID': os.getpid(),
                  'A': None, 'B': None, 'C': False, 'model_goal_complete': False}))
