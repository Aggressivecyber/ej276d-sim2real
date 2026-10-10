"""Reconcile executor-owned receipts, without submitting or polling remote science."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_FROZEN_HEUN128_RESPONSE_001'
WORK = ROOT / 'work' / STAGE
NOW = datetime.now(timezone.utc).isoformat()
OBSERVATION = 'reviews/R49_LIVE_OBSERVATION.json'


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


assert not (WORK / 'FINAL_RECEIPT.json').exists(), 'Final receipt requires final reconciliation.'
native = []
for path in sorted(WORK.glob('*.NATIVE_RECEIPT.json')):
    data = read(path)
    if 'actual_worker_native_exit' in data and 'elapsed_s' in data:
        native.append(dict(data, receipt=path.relative_to(ROOT).as_posix()))
closed = {row['entry'] for row in native}
started, running, unknown = [], [], []
for path in sorted(WORK.glob('*.LOCAL_ACTIVE_JOB.json')):
    data = read(path)
    handshake = data.get('actual_remote_handshake')
    if not handshake:
        continue
    item = {'entry': data['entry'], 'actual_worker_PID': handshake['actual_worker_PID'],
        'start_UTC': handshake['start_UTC'], 'handshake_before_imports': handshake['handshake_before_imports'],
        'limit_s': handshake['limit_s'], 'local_status_receipt': path.relative_to(ROOT).as_posix()}
    assert item['limit_s'] is None and not data['entry'].startswith(('train_', 'calibrate'))
    started.append(item)
    if data['entry'] not in closed:
        if 'SSH_native_exit' in data:
            unknown.append(dict(item, status='awaiting_native_receipt_remote_state_not_inferred_from_SSH_exit',
                                SSH_native_exit=data['SSH_native_exit']))
        else:
            running.append(item)
assert len(running) <= 1
science = sum(float(row['elapsed_s']) for row in native)
obs = {'stage': STAGE, 'observed_UTC': NOW, 'actual_root_control_PID': os.getpid(),
    'completed_native_science_s': science, 'started_native_jobs': started,
    'last_observed_running_jobs': running, 'awaiting_native_receipt_jobs': unknown,
    'native_completed_jobs': native, 'response_training_runs_started': 0,
    'inflight_elapsed_not_added_until_native_completion': True,
    'remote_fresh_process_query_by_root': False,
    'evidence_basis': 'Executor-owned native receipts and actual worker handshake; no duplicate remote polling.'}
numeric_path = WORK / 'TRAIN_SOLVER_CHECK.json'
if numeric_path.is_file():
    result = read(numeric_path)
    obs['numerical_check'] = {'both_pass': result['both_pass'], 'models': {
        name: {k: row[k] for k in ['RMS_median_LSB', 'RMS_p95_LSB', 'preADC_relative_q_max', 'pass',
                                  'candidate_steps', 'reference_only_steps']}
        for name, row in result['models'].items()}}
save(ROOT / OBSERVATION, obs)
budget = read(ROOT / 'RESOURCE_BUDGET.json')
block = budget['improvement_block']
assert budget['compute_budget_enforced'] is False and block['total_science_seconds_max'] is None
previous = [x for x in block['actual_stage_receipts'] if x['stage'] != STAGE]
assert abs(sum(x['science_s'] for x in previous) - 6380.315270759122) < 1e-8
block['actual_science_seconds_recorded'] = sum(x['science_s'] for x in previous) + science
block['response_training_runs_started'] = sum(x.get('response_training_runs_started', 0) for x in previous)
block['actual_stage_receipts'] = previous + [{'stage': STAGE, 'receipt': OBSERVATION,
    'science_s': science, 'response_training_runs_started': 0, 'new_response_updates': 0,
    'status': 'in_progress_completed_native_jobs_only', 'stage_science_complete': False,
    'inflight_science_not_yet_in_total': bool(running or unknown)}]
next(x for x in block['registered_stages'] if x['stage'] == STAGE).update(
    status='scientific_execution_in_progress', actual_completed_native_science_s=science,
    response_training_runs_started=0)
budget['updated_UTC'] = NOW
save(ROOT / 'RESOURCE_BUDGET.json', budget)
active = None if unknown or not running else len(running)
current = running[0] if running else unknown[0] if unknown else {
    'stage': STAGE, 'status': 'native_stage_transition_or_handshake_not_yet_returned',
    'remote_process_count': 'unknown_until_next_actual_handshake_or_final_receipt'}
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    data = read(ROOT / name)
    assert data['current_stage'] == STAGE
    data.update(status='R49_in_progress_model_goal_unmet', current_stage_progress='scientific_execution_in_progress',
        current_activity=current.get('entry', 'native_stage_transition_observed'),
        active_scientific_jobs=active, last_native_observation=OBSERVATION, current_job=current,
        current_stage_response_training_runs_started=0, updated_UTC=NOW, updated_at_utc=NOW)
    save(ROOT / name, data)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(ROOT / name)
    row = next(x for x in data[key] if x.get(identity) == STAGE)
    row.update(status='scientific_execution_in_progress', last_observed_UTC=NOW,
        actual_native_jobs=native, last_observed_running_jobs=running,
        awaiting_native_receipt_jobs=unknown, active_scientific_jobs=active,
        actual_completed_native_science_s=science, actual_science_seconds=science,
        response_training_runs_started=0)
    save(ROOT / name, data)
print(json.dumps({'UTC': NOW, 'PID': os.getpid(), 'completed_science_s': science,
    'block_science_s': block['actual_science_seconds_recorded'], 'running': running,
    'awaiting_native_receipt_jobs': unknown, 'numerical_check': obs.get('numerical_check')}))
