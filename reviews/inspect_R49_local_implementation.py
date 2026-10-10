"""Read-only inspection of local recovery source, configuration and native receipts."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_LOCAL_HEUN128_RECOVERY_001'
WORK = ROOT / 'work' / STAGE
REMOTE = ROOT / 'work/R49_FROZEN_HEUN128_RESPONSE_001'
OUT = ROOT / 'reviews/ROOT_R49_LOCAL_IMPLEMENTATION_REVIEW_001.json'
assert not OUT.exists()
cfg = json.loads((WORK / 'config.json').read_text('utf8'))
assert cfg['solver_steps'] == 128 and cfg['solver_check_steps'] == 256
assert cfg['generation_chunk'] == 64 and cfg['CPU_threads'] == cfg['threads'] == 4
assert cfg['new_response_training_runs'] == cfg['new_response_updates'] == cfg['updates'] == 0
null_fields = ['science_budget', 'science_budget_s', 'wall_budget_s', 'deadline_UTC',
    'absolute_deadline_UTC', 'training_wall_s', 'training_total_limit_s', 'numeric_wall_limit_s',
    'output_limit_bytes', 'disk_free_reserve_bytes']
assert all(cfg[k] is None for k in null_fields)
assert cfg['compute_budget_enforced'] is cfg['deadlines_enforced'] is False
assert cfg['RSS_limit_bytes'] == 4 * 1024**3 and cfg['GPU_count'] == 0
assert cfg['remote_root'] is None and cfg['remote_original_replay_terminal'] == 'unknown'
names = ['network.py', 'noise_core.py', 'source_flow_core.py', 'flow_runtime.py', 'check_models.py',
    'generate_all.py', 'flow_diagnostics.py', 'evaluate_saved.py', 'replay_all.py', 'evaluation.py',
    'diagnostic_core.py', 'assessment.py', 'energy_width.py', 'data_support.py', 'common_math.py',
    'model.py', 'legacy_model.py', 'endpoint_common.py', 'temporal_common.py', 'mean_description.py',
    'flow_core.py', 'summarize_and_plot.py', 'render_figures.py', 'render_train.py', 'make_report.py']
bindings = []
for name in names:
    original = (REMOTE / name).read_text('utf8').replace('\r', '')
    expected = original.replace('torch.set_num_threads(12)', 'torch.set_num_threads(4)')
    expected = expected.replace('torch.set_num_interop_threads(12)', 'torch.set_num_interop_threads(4)')
    expected = expected.replace("os.environ[key]='12'", "os.environ[key]='4'").replace("'CPU_threads':12", "'CPU_threads':4")
    if name == 'make_report.py':
        expected = expected.replace("read('REMOTE_RUN_HISTORY.json')", "read('RUN_HISTORY.json')")
        expected = expected.replace('服务器同一CPU后端、12线程', '本机同一CPU后端、4线程')
    if name == 'render_figures.py':
        expected = expected.replace('import render_train\n', '')
        expected = expected.replace("    render_train.run(root,cfg,json.loads((root/'TRAIN_SOLVER_CHECK.json').read_text('utf8')))\n", '')
    actual = (WORK / name).read_text('utf8').replace('\r', '')
    if name != 'make_report.py':
        assert actual == expected, name
    else:
        assert "read('RUN_HISTORY.json')" in actual
        assert '本次是用户授权的本机恢复' in actual and '终态unknown' in actual
    ast.parse(actual, filename=name)
    raw = (WORK / name).read_bytes()
    bindings.append({'path': 'work/' + STAGE + '/' + name, 'bytes': len(raw),
                     'sha256': hashlib.sha256(raw).hexdigest()})
guard = (WORK / 'supervise_local.py').read_text('utf8')
worker = (WORK / 'owned_worker.py').read_text('utf8')
assert "if (ROOT/(entry+'.used')).exists():raise RuntimeError" in guard
assert "if max(peak,tree)>CFG['RSS_limit_bytes']" in guard
assert "if time.monotonic()-start>20 and handle is None" in guard
assert "'science_budget_s':None" in guard and "'wall_budget_s':None" in guard
assert "'CPU_not_gated_on_GPU':True" in guard
assert "'before_scientific_imports':True" in worker and 'runpy.run_path' in worker
native = json.loads((WORK / 'check_models.py.NATIVE_RECEIPT.json').read_text('utf8'))
numeric = json.loads((WORK / 'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
assert native['actual_worker_native_exit'] == native['launcher_native_exit'] == 0
assert native['handshake_before_imports'] and numeric['both_pass'] and numeric['DEV_read_or_generated'] == 0
inventory = json.loads((WORK / 'LOCAL_INPUT_INVENTORY.json').read_text('utf8'))
assert not inventory['missing'] and inventory['scientific_arrays_decoded'] == 0
for name in ['config.json', 'supervise_local.py', 'owned_worker.py', 'run_local_pipeline.py',
             'LOCAL_INPUT_INVENTORY.json', 'LOCAL_ENVIRONMENT.json']:
    raw = (WORK / name).read_bytes()
    bindings.append({'path': 'work/' + STAGE + '/' + name, 'bytes': len(raw),
                     'sha256': hashlib.sha256(raw).hexdigest()})
review = {'stage': STAGE, 'reviewed_UTC': datetime.now(timezone.utc).isoformat(),
    'actual_root_control_PID': os.getpid(), 'review_type': 'source_and_native_metadata_only',
    'null_budget_fields': null_fields, 'candidate_reference_steps': [128, 256], 'chunk': 64,
    'CPU_threads': 4, 'RSS_safety_bytes': 4 * 1024**3, 'new_training_updates': 0,
    'science_source_compared_to_preserved_remote_attempt': [name for name in names if name != 'make_report.py'],
    'local_science_source_differences': ['12 to 4 CPU thread settings', 'local RUN_HISTORY reference and report text'],
    'report_generator_read_tool': 'd6a787',
    'report_generator_separately_adapted_for_local_provenance': True,
    'render_train_duplication_removed': {'read_tool': 'd57db1',
        'change': 'TRAIN plot rendering runs once in its registered pipeline job instead of also inside render_figures.'},
    'initial_inspection_control_exit': {'tool': '9b86c2', 'native_exit': 1,
        'cause': 'Report generator includes local provenance paragraphs; scientific source equality assumptions do not apply to prose.'},
    'second_inspection_control_exit': {'tool': '448134', 'native_exit': 1,
        'cause': 'Separate TRAIN plotting entry makes the additional render_figures import/call redundant; source diff inspected.'},
    'science_arrays_compared_between_platforms': False, 'migration_content_tests': False,
    'both_original_128_TRAIN_rows_checked_before_DEV': True,
    'first_local_native_receipt': native, 'local_numerical_pass': numeric['both_pass'],
    'root_scientific_computation': False, 'full_model_scientific_acceptance_inferred_from_source': False,
    'remote_previous_replay_terminal': 'unknown', 'global_single_physical_job_confirmed': False,
    'bindings': bindings}
OUT.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'PID': os.getpid(), 'review': OUT.relative_to(ROOT).as_posix(),
    'science_source_files_inspected': len(names), 'local_numerical_pass': True,
    'full_model_scientific_acceptance_known': False}, ensure_ascii=False))
