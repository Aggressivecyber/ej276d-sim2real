"""Read executor-owned local receipts; preserve unresolved remote work separately."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
WORK = ROOT / 'work' / STAGE
OBS = 'reviews/R49_LOCAL_LIVE_OBSERVATION.json'
NOW = datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


assert not (WORK / 'FINAL_RECEIPT.json').exists()
history = read(WORK / 'RUN_HISTORY.json') if (WORK / 'RUN_HISTORY.json').is_file() else []
active = read(WORK / 'ACTIVE_JOB.json') if (WORK / 'ACTIVE_JOB.json').is_file() else None
running = active if active and active.get('actual_worker_PID') and 'launcher_native_exit' not in active else None
science = sum(float(row['elapsed_s']) for row in history)
obs = {'stage': STAGE, 'observed_UTC': NOW, 'root_control_PID': os.getpid(),
    'actual_completed_local_native_science_s': science, 'actual_completed_local_jobs': history,
    'last_observed_local_running_job': running, 'local_phase_transition_unresolved': not bool(running),
    'remote_previous_attempt_terminal': 'unknown', 'active_remote_scientific_jobs': None,
    'global_single_physical_job_confirmed': False,
    'confirmed_remote_completed_science_s': 2107.3375221949973,
    'unfinished_remote_replay_full_elapsed_s': None,
    'unknown_remote_replay_cost_not_zeroed_or_invented': True,
    'new_response_training_runs': 0, 'new_response_updates': 0, 'new_calibration_fits': 0,
    'new_scientific_jobs_submitted_by_root': False, 'compute_budget_enforced': False}
numeric = WORK / 'TRAIN_SOLVER_CHECK.json'
if numeric.is_file():
    n = read(numeric)
    obs['numerical_check'] = {'both_pass': n['both_pass'], 'models': {
        name: {key: row[key] for key in ['RMS_median_LSB', 'RMS_p95_LSB', 'preADC_relative_q_max', 'pass',
                                       'candidate_steps', 'reference_only_steps']}
        for name, row in n['models'].items()}}
write(ROOT / OBS, obs)
budget = read(ROOT / 'RESOURCE_BUDGET.json')
block = budget['improvement_block']
assert not budget['compute_budget_enforced'] and block['method_plans_max'] is None
previous = [row for row in block['actual_stage_receipts'] if row['stage'] != STAGE]
assert abs(sum(row['science_s'] for row in previous) - 8487.652792954119) < 1e-8
block['actual_stage_receipts'] = previous + [{'stage': STAGE, 'parent_stage': 'R49_FROZEN_HEUN128_RESPONSE_001',
    'receipt': OBS, 'science_s': science, 'response_training_runs_started': 0, 'new_response_updates': 0,
    'stage_science_complete': False, 'inflight_science_not_yet_in_total': bool(running),
    'status': 'local_recovery_in_progress_completed_native_jobs_only'}]
block['actual_science_seconds_recorded'] = sum(row['science_s'] for row in block['actual_stage_receipts'])
next(row for row in block['registered_stages'] if row['stage'] == STAGE).update(
    status='local_scientific_execution_in_progress', actual_completed_native_science_s=science,
    response_training_runs_started=0)
budget['updated_UTC'] = NOW
write(ROOT / 'RESOURCE_BUDGET.json', budget)
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    data = read(ROOT / name)
    assert data['current_stage'] == STAGE
    data.update(status='R49_local_recovery_in_progress_model_goal_unmet', updated_UTC=NOW, updated_at_utc=NOW,
        current_stage_progress='local_scientific_execution_in_progress',
        current_activity=running['entry'] if running else 'local_phase_transition_waiting_for_actual_handshake',
        current_job=running or {'stage': STAGE, 'status': 'phase_transition_or_admission_pending'},
        last_native_observation=OBS, active_local_scientific_jobs=1 if running else None,
        active_scientific_jobs=None, active_remote_scientific_jobs=None,
        remote_job_state='unknown_user_reports_maintenance', current_stage_response_training_runs_started=0)
    write(ROOT / name, data)
for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
    data = read(ROOT / name)
    next(row for row in data[key] if row.get(identity) == STAGE).update(
        status='local_scientific_execution_in_progress', last_observed_UTC=NOW, last_native_observation=OBS,
        actual_native_jobs=history, last_observed_local_running_job=running,
        active_local_scientific_jobs=1 if running else None,
        actual_completed_native_science_s=science, actual_science_seconds=science,
        response_training_runs_started=0)
    write(ROOT / name, data)
print(json.dumps({'UTC': NOW, 'PID': os.getpid(), 'local_completed_science_s': science,
    'block_confirmed_completed_science_s': block['actual_science_seconds_recorded'],
    'local_running': None if not running else {'entry': running['entry'], 'actual_worker_PID': running['actual_worker_PID']},
    'remote_terminal': 'unknown', 'numerical_check': obs.get('numerical_check')}))
