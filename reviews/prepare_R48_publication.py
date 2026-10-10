"""Build the explicit R48 source/summary/figure whitelist; no raw arrays or weights."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R48_GUIDED_NOISE_COMPLETION_FLOW_001'
PREFIX = 'work/' + STAGE + '/'
assert json.loads((ROOT / PREFIX / 'FINAL_RECEIPT.json').read_text('utf8'))['status'] == 'sealed'
assert not (ROOT / 'reviews/R48_PUBLICATION_SELECTION_001.json').exists()
names = [
    'AGENTS.md', 'PLAN.md', 'PROJECT_OBJECTIVE.md', 'PROGRAM.json', 'PROJECT_PROGRESS.json',
    'RESOURCE_BUDGET.json', 'PROJECT_ACCEPTANCE.md', 'PROJECT_ACCEPTANCE.json',
    'SERVER_RESOURCE_POLICY.md', 'jobs.json', 'experiments.json',
    'R48_REPORT.md', 'R48_DELIVERY_INDEX.json', 'tasks/' + STAGE + '.md',
    'reviews/ROOT_R48_PRETRAIN_REVIEW_001.json', 'reviews/ROOT_R48_KEY_RESULT_REVIEW_001.json',
    'reviews/R48_ACCEPTANCE_OBSERVATION_001.json', 'reviews/R48_SEAL_NATIVE_RECEIPT.json',
    'reviews/R48_RESOURCE_RECONCILIATION_001.json', 'reviews/R48_ROOT_RECONCILIATION_NATIVE_RECEIPT.json',
    'reviews/R48_LIVE_OBSERVATION.json', 'reviews/R48_EXECUTOR_START_OBSERVATION.json',
    'reviews/R48_REGISTRATION_PUBLICATION_RECEIPT.json',
    'reviews/observe_R48_acceptance.py', 'reviews/finalize_R48_scientific_review.py',
    'reviews/prepare_R48_publication.py', 'reviews/publish_R48.py',
    'reviews/backup_R48_incremental.py', 'reviews/finalize_R48_delivery_completion.py',
    'reviews/refresh_R48_live_state.py', 'reviews/select_continuation_publication.py',
]
names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / PREFIX).glob('*.py')))
names.extend(PREFIX + n for n in [
    'config.json', 'config_cpu.json', 'PROTOCOL.json', 'REPORT.md', 'REPRODUCTION.md',
    'EXECUTOR_ACK.json', 'SERVER_PREFLIGHT.json', 'DEPLOYMENT_RECEIPT.json', 'SERVER_START_RESOURCE.json',
    'PRETRAIN_FREEZE.json', 'PRETRAIN_RESOURCE_CORRECTION.json', 'FORMULA_CHECK.json',
    'NOISE_CALIBRATION.json', 'TRAINING_COMPACT.json', 'TRAIN_SOLVER_CHECK.json',
    'ACCEPTANCE_A_C.json', 'GATE_METRICS.json', 'METRICS_COMPACT.json',
    'FIGURE_QA.json', 'DISPLAY_CORRECTION_RECEIPT.json', 'TRAIN_PLOT_NUMERIC_RECEIPT.json',
    'RESOURCE_RECEIPT.json', 'NATIVE_CHAIN.json', 'RESOURCE_CHECK_NATIVE.json',
    'TRANSFER_HISTORY.json', 'OUTPUT_RETURN_RECEIPT.json', 'POST_DEPLOY_TRAIN_only.json',
    'POST_ENTRY_AST.json', 'POST_GUARD_BINDING.json', 'POST_NUMERIC_SOURCE_BINDINGS.json',
    'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json',
])
for entry in ['calibrate.py', 'train_control.py', 'train_completion.py', 'check_models.py', 'render_train.py', 'render_noise_display.py']:
    names.extend(PREFIX + entry + suffix for suffix in ['.NATIVE_RECEIPT.json', '.WORKER_RECEIPT.json', '.SCIENCE_PID.json', '.RSS_ACK.json'])
for arm in ['GUIDED_FIXED_NOISE_CONTROL', 'GUIDED_NOISE_COMPLETION']:
    names.extend(PREFIX + arm + '/' + n for n in ['ENVIRONMENT.json', 'INITIALIZATION.json', 'TRAIN_SOLVER_CHECK.json'])
names.extend(PREFIX + 'results/figures/' + n for n in [
    'training_source_identity.png', 'TRAIN_numerical_check.png', 'TRAIN_noise_operator.png', 'TRAIN_initial_noise_rule.png'])
names = sorted(set(names))
assert all((ROOT / name).is_file() for name in names)
assert not any(Path(n).name in ['TRAINING_RECEIPT.json', 'TRAINING_TRACE.json'] for n in names)
result = subprocess.run([sys.executable, '-B', str(ROOT / 'reviews/select_continuation_publication.py'),
    '--receipt', 'reviews/R48_PUBLICATION_SELECTION_001.json', '--root', *names], cwd=ROOT)
raise SystemExit(result.returncode)
