"""Select sealed local R49 evidence and the preserved incomplete remote attempt."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
REMOTE = 'R49_FROZEN_HEUN128_RESPONSE_001'
WORK = ROOT / 'work' / STAGE
OLD = ROOT / 'work' / REMOTE
SELECTION = 'reviews/R49_PUBLICATION_SELECTION_001.json'
assert not (ROOT / SELECTION).exists()
assert json.loads((WORK / 'FINAL_RECEIPT.json').read_text('utf8'))['status'] == 'sealed'
names = [
    'AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
    'RESOURCE_BUDGET.json', 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
    'SERVER_RESOURCE_POLICY.md', 'jobs.json', 'experiments.json', 'R49_REPORT.md', 'R49_DELIVERY_INDEX.json',
    'tasks/' + STAGE + '.md', 'tasks/' + REMOTE + '.md',
    'reviews/ROOT_R49_IMPLEMENTATION_REVIEW_001.json', 'reviews/ROOT_R49_LOCAL_IMPLEMENTATION_REVIEW_001.json',
    'reviews/ROOT_R49_LOCAL_KEY_RESULT_REVIEW_001.json',
    'reviews/R49_LOCAL_ACCEPTANCE_OBSERVATION_001.json', 'reviews/R49_LOCAL_SEAL_NATIVE_RECEIPT.json',
    'reviews/R49_LOCAL_RESOURCE_RECONCILIATION_001.json', 'reviews/R49_LOCAL_ROOT_FIGURE_QA_001.json',
    'reviews/R49_LOCAL_ROOT_DIRECTION_001.json', 'reviews/R49_LOCAL_ROOT_RECONCILIATION_NATIVE_RECEIPT.json',
    'reviews/R49_LOCAL_LIVE_OBSERVATION.json', 'reviews/R49_LOCAL_REGISTRATION_NATIVE_RECEIPT.json',
    'reviews/R49_LOCAL_REGISTRATION_PUBLICATION_RECEIPT.json', 'reviews/R49_REMOTE_INTERRUPTION_001.json',
    'reviews/R49_ACCEPTANCE_OBSERVATION_001.json', 'reviews/R49_ROOT_DIRECTION_001.json',
    'reviews/R49_LIVE_OBSERVATION.json', 'reviews/R49_REGISTRATION_NATIVE_RECEIPT.json',
    'reviews/R49_REGISTRATION_PUBLICATION_RECEIPT.json',
    'reviews/prepare_R49_local_delivery_tools.py', 'reviews/finalize_R49_local_scientific_review.py',
    'reviews/finalize_R49_scientific_review.py', 'reviews/publish_R49.py',
    'reviews/backup_R49_incremental.py', 'reviews/finalize_R49_delivery_completion.py',
    'reviews/prepare_R49_local_publication.py', 'reviews/publish_R49_local.py',
    'reviews/backup_R49_local_incremental.py', 'reviews/finalize_R49_local_delivery_completion.py',
    'reviews/inspect_R49_local_implementation.py', 'reviews/refresh_R49_local_live_state.py',
    'reviews/observe_R49_local_acceptance.py', 'reviews/register_R49_local_recovery.py',
    'reviews/publish_R49_local_registration.py', 'reviews/select_continuation_publication.py',
    'reviews/reconcile_R49_current_policy.py', 'reviews/inspect_R49_implementation.py',
    'reviews/observe_R49_acceptance.py', 'reviews/refresh_R49_live_state.py',
    'reviews/read_R49_local_status.ps1',
    'reviews/record_R49_local_direction.py',
    'reviews/observe_R49_adc_metadata.py', 'reviews/ROOT_R49_ADC_SOURCE_OBSERVATION_001.json',
]
names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'reviews/R49_CURRENT_POLICY_RECONCILIATION_001').glob('*.json')))
names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'reviews/R49_LOCAL_REGISTRATION_001').glob('*')) if p.is_file())
local_required = [
    'config.json', 'config_cpu.json', 'PROTOCOL.md', 'REPORT.md', 'REPRODUCTION.md', 'EXECUTOR_ACK.json', 'SOURCE_BINDINGS.json',
    'LOCAL_ENVIRONMENT.json', 'LOCAL_INPUT_INVENTORY.json', 'NOISE_CALIBRATION.json',
    'TRAIN_SOLVER_CHECK.json', 'GENERATION_NUMERIC_RECEIPT.json', 'GENERATION_RECEIPT.json',
    'INITIAL_NOISE_DESCRIPTIONS.json', 'REAL_IDENTITY_DESCRIPTIONS.json', 'CHARGE_CONSERVATION_SUMMARY.json',
    'EVALUATION_NUMERIC_RECEIPT.json', 'REPLAY_RECEIPT.json', 'SUMMARY_PLOT_NUMERIC_RECEIPT.json',
    'TRAIN_PLOT_NUMERIC_RECEIPT.json', 'ACCEPTANCE_A_C.json', 'GATE_METRICS.json', 'METRICS_COMPACT.json',
    'FIGURE_QA.json', 'FIGURE_QA_SUMMARY.json', 'RESOURCE_RECEIPT.json', 'NATIVE_CHAIN.json',
    'LOCAL_PIPELINE_STARTED.json', 'LOCAL_PIPELINE_TERMINAL.json', 'RUN_HISTORY.json',
    'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json', 'ENVIRONMENT_IMPORT_FAILURE_001.json', 'PREPARATION_FAILURE_001.json',
]
names.extend('work/' + STAGE + '/' + name for name in local_required)
for directory in [WORK, OLD]:
    names.extend(p.relative_to(ROOT).as_posix() for p in sorted(directory.glob('*.py')))
    for pattern in ['*.NATIVE_RECEIPT.json', '*.WORKER_RECEIPT.json', '*.SCIENCE_PID.json', '*.RSS_ACK.json', '*.RESOURCE_ADMISSION.json']:
        names.extend(p.relative_to(ROOT).as_posix() for p in sorted(directory.glob(pattern)))
    for name in ['PREPARATION_ACK.json', 'SOURCE_BINDINGS.json', 'config.json', 'config_cpu.json', 'PROTOCOL.md',
                 'EXECUTOR_ACK.json', 'RESOURCE_CHECK_NATIVE.json', 'SERVER_PREFLIGHT.json', 'DEPLOYMENT_RECEIPT.json',
                 'NOISE_CALIBRATION.json', 'TRAIN_SOLVER_CHECK.json', 'GENERATION_NUMERIC_RECEIPT.json',
                 'GENERATION_RECEIPT.json', 'EVALUATION_NUMERIC_RECEIPT.json', 'INITIAL_NOISE_DESCRIPTIONS.json',
                 'REAL_IDENTITY_DESCRIPTIONS.json', 'CHARGE_CONSERVATION_SUMMARY.json', 'TRANSFER_HISTORY.json',
                 'OUTPUT_RETURN_RECEIPT.json', 'DISPLAY_CORRECTION_RECEIPT.json', 'POST_GUARD_BINDING.json',
                 'DOWNSTREAM_PREEXEC_REPAIR.json', 'REPLAY_NEGATIVE_SEAL_PREEXEC_CORRECTION.json',
                 'SUMMARY_SCHEMA_PREEXEC_CORRECTION.json', 'OUTAGE_STATUS_QUERY_001.json',
                 'LOCAL_SOURCE_FREEZE.json', 'LOCAL_COMPLETION_HELPERS.json', 'LOCAL_RESOURCE_WAIT.json',
                 'LOCAL_EVALUATION_MEMORY_EVIDENCE.json', 'PROTOCOL.json',
                 'EVALUATION_LOG_LABEL_CORRECTION_001.json',
                 'CONTROL_READ_FAILURE_001.json', 'CONTROL_OPTIONAL_STATUS_READ_001.json',
                 'CONTROL_REPORT_PATCH_FAILURE_001.json',
                 'BOUNDARY_LAYOUT_CORRECTION_001.json',
                 'display_initial/boundary_differences.png',
                 'display_initial/render_figures.boundary_initial.py',
                 'display_initial/supervise_local.boundary_initial.py',
                 'control_source_history/evaluate_saved.log_label_before.py']:
        if (directory / name).is_file():
            names.append((directory / name).relative_to(ROOT).as_posix())
    for arm in ['GUIDED_FIXED_NOISE_CONTROL', 'GUIDED_NOISE_COMPLETION']:
        names.append((directory / arm / 'TRAIN_SOLVER_CHECK.json').relative_to(ROOT).as_posix())
names.extend(p.relative_to(ROOT).as_posix() for p in sorted((WORK / 'results/figures').glob('*.png')))
names = sorted(set(names))
assert all((ROOT / name).is_file() for name in names), [name for name in names if not (ROOT / name).is_file()]
assert not any(Path(name).name in ['TRAINING_RECEIPT.json', 'TRAINING_TRACE.json', 'metrics.json'] for name in names)
selector = ROOT / 'reviews/select_continuation_publication.py'
# Keep the explicit list in process to avoid Windows CreateProcess argument-length limits.
sys.argv = [str(selector), '--receipt', SELECTION, '--root', *names]
runpy.run_path(str(selector), run_name='__main__')
