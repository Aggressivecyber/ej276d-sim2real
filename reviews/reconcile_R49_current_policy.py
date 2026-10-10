"""Align current dashboard metadata with the already authorized R49 protocol."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_FROZEN_HEUN128_RESPONSE_001'
DEST = ROOT / 'reviews/R49_CURRENT_POLICY_RECONCILIATION_001'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')


assert not DEST.exists()
policy = read('RESOURCE_BUDGET.json')
assert policy['resource_version'] == 3 and policy['compute_budget_enforced'] is False
assert policy['improvement_block']['total_science_seconds_max'] is None
assert policy['improvement_block']['all_response_training_runs_max'] is None
ack = read('work/' + STAGE + '/EXECUTOR_ACK.json')
assert ack['stage'] == STAGE and ack['planned_new_response_training_runs'] == 0
DEST.mkdir()
changes = []
for name in ['PROGRAM.json', 'PROJECT_PROGRESS.json']:
    obj = read(name)
    assert obj['current_stage'] == STAGE
    shutil.copy2(ROOT / name, DEST / name)
    previous = {key: obj.get(key) for key in [
        'improvement_block_science_seconds_max', 'improvement_block_response_training_runs_max',
        'execution_location', 'executor_start_observation', 'last_native_observation']}
    obj.update(improvement_block_science_seconds_max=None,
        improvement_block_response_training_runs_max=None, compute_budget_enforced=False,
        deadlines_enforced=False, execution_policy='registered_scientific_design_continuous_execution_without_compute_quota',
        current_stage_response_training_runs_started=0,
        latest_training_progress_stage='R48_GUIDED_NOISE_COMPLETION_FLOW_001',
        latest_training_progress_role='historical_frozen_weights_no_R49_training',
        executor_start_observation='work/' + STAGE + '/EXECUTOR_ACK.json',
        last_native_observation=None,
        current_policy_reconciliation='reviews/R49_CURRENT_POLICY_RECONCILIATION_001/RECEIPT.json',
        updated_UTC=NOW, updated_at_utc=NOW)
    obj['execution_location'].update(mode='server_CPU12_frozen_numerical_generation_evaluation_replay_local_auxiliary',
        current_stage_response_training=False, current_stage_GPU_used=False,
        GPU_limits_apply_only_if_GPU_execution_is_separately_registered=True)
    save(ROOT / name, obj)
    changes.append({'file': name, 'previous_values': previous})

save(DEST / 'RECEIPT.json', {'UTC': NOW, 'actual_control_PID': os.getpid(),
    'action': 'correct_current_dashboard_metadata_to_existing_R49_registration',
    'no_new_scientific_authorization_or_threshold_change': True, 'science_started_by_root': False,
    'historical_stage_files_modified': False, 'before_snapshots': changes,
    'actual_science_accounting_unchanged': policy['improvement_block']['actual_science_seconds_recorded'],
    'authority': 'reviews/NO_COMPUTE_BUDGET_20261010_001/RECEIPT.json',
    'stage_registration': 'reviews/R49_REGISTRATION_001/RECEIPT.json'})
print(json.dumps({'UTC': NOW, 'PID': os.getpid(), 'status': 'current_R49_policy_metadata_aligned',
    'receipt': 'reviews/R49_CURRENT_POLICY_RECONCILIATION_001/RECEIPT.json'}))
