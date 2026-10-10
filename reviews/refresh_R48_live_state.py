"""Reconcile local native receipts into root ledgers without scientific execution."""
from datetime import datetime, timezone
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
stage = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
work = root / 'work' / stage
now = datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text('utf-8-sig'))

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', 'utf8')

native = []
for path in sorted(work.glob('*.NATIVE_RECEIPT.json')):
    data = read(path)
    if 'actual_worker_native_exit' in data and 'elapsed_s' in data:
        native.append(dict(data, receipt=path.relative_to(root).as_posix()))
closed = {row['entry'] for row in native}
started, running = [], []
for path in sorted(work.glob('*.LOCAL_ACTIVE_JOB.json')):
    data = read(path)
    handshake = data.get('actual_remote_handshake')
    if not handshake:
        continue
    item = {'entry': data['entry'], 'actual_worker_PID': handshake['actual_worker_PID'],
            'start_UTC': handshake['start_UTC'], 'handshake_before_imports': handshake['handshake_before_imports'],
            'local_status_receipt': path.relative_to(root).as_posix()}
    started.append(item)
    if data['entry'] not in closed and 'SSH_native_exit' not in data:
        running.append(item)
progress = []
for path in sorted(work.glob('train_*.SSH.stdout.log')):
    latest = None
    for line in path.read_text('utf8', errors='replace').splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'actual_training_progress' in obj:
            latest = obj['actual_training_progress']
    if latest:
        progress.append(latest)
compact_path = work / 'TRAINING_COMPACT.json'
if compact_path.is_file():
    completed_training = read(compact_path).get('training', {})
    by_arm = {row['arm']: row for row in progress}
    for arm, receipt in completed_training.items():
        final_trace = receipt.get('final_trace')
        if receipt.get('updates') == 50000 and final_trace and final_trace['global_step'] == 50000:
            by_arm[arm] = dict(final_trace, actual_completed=True,
                               final_training_evidence=compact_path.relative_to(root).as_posix())
    progress = [by_arm[arm] for arm in sorted(by_arm)]
completed_s = sum(float(row['elapsed_s']) for row in native)
training_started = sum(item['entry'].startswith('train_') for item in started)
assert len(running) <= 1
obs = {'stage': stage, 'observed_UTC': now, 'completed_native_science_s': completed_s,
       'inflight_elapsed_not_added_until_native_completion': True,
       'started_native_jobs': started, 'last_observed_running_jobs': running,
       'response_training_runs_started': training_started, 'latest_training_progress': progress,
       'native_completed_jobs': native, 'remote_fresh_process_query_by_root': False,
       'evidence_basis': 'Executor-owned native receipts and streaming progress, no duplicate SSH polling.'}
save(root / 'reviews/R48_LIVE_OBSERVATION.json', obs)
budget = read(root / 'RESOURCE_BUDGET.json')
block = budget['improvement_block']
previous = [x for x in block['actual_stage_receipts'] if x['stage'] != stage]
block['actual_science_seconds_recorded'] = sum(x['science_s'] for x in previous) + completed_s
block['response_training_runs_started'] = sum(x.get('response_training_runs_started', 0) for x in previous) + training_started
block['actual_stage_receipts'] = previous + [{'stage': stage, 'receipt': 'reviews/R48_LIVE_OBSERVATION.json',
    'science_s': completed_s, 'response_training_runs_started': training_started,
    'status': 'in_progress_completed_native_jobs_only', 'stage_science_complete': False,
    'inflight_science_not_yet_in_total': bool(running)}]
registered = next(x for x in block['registered_stages'] if x['stage'] == stage)
registered.update(status='scientific_execution_in_progress', actual_completed_native_science_s=completed_s,
                  response_training_runs_started=training_started)
budget['updated_UTC'] = now
save(root / 'RESOURCE_BUDGET.json', budget)
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    data = read(root / name)
    assert data['current_stage'] == stage
    data.update(status='R48_in_progress_model_goal_unmet', current_stage_progress='scientific_execution_in_progress',
        current_activity=running[0]['entry'] if running else 'native_stage_transition_observed',
        active_scientific_jobs=len(running), last_native_observation='reviews/R48_LIVE_OBSERVATION.json',
        current_job=running[0] if running else {'stage': stage, 'status': 'no_inflight_native_job_in_local_receipts'},
        latest_training_progress=progress, updated_UTC=now)
    save(root / name, data)
jobs = read(root / 'jobs.json')
row = next(x for x in jobs['jobs'] if x.get('stage') == stage)
row.update(status='scientific_execution_in_progress', last_observed_UTC=now,
           actual_native_jobs=native, last_observed_running_jobs=running)
save(root / 'jobs.json', jobs)
experiments = read(root / 'experiments.json')
row = next(x for x in experiments['experiments'] if x['id'] == stage)
row.update(status='scientific_execution_in_progress', actual_completed_native_science_s=completed_s,
           response_training_runs_started=training_started, last_observed_UTC=now)
save(root / 'experiments.json', experiments)
print(json.dumps({'observed_UTC': now, 'stage_completed_science_s': completed_s,
    'block_recorded_science_s': block['actual_science_seconds_recorded'],
    'block_training_runs_started': block['response_training_runs_started'],
    'last_observed_running': running, 'progress': progress}))
