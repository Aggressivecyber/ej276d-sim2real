"""Record source/metadata review of the frozen 128/256 implementation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'R49_FROZEN_HEUN128_RESPONSE_001'
WORK = ROOT / 'work' / STAGE
OLD = ROOT / 'work/R48_GUIDED_NOISE_COMPLETION_FLOW_001'
OUT = ROOT / 'reviews/ROOT_R49_IMPLEMENTATION_REVIEW_001.json'
assert not OUT.exists()
cfg = json.loads((WORK / 'config.json').read_text('utf8'))
assert cfg['solver_steps'] == 128 and cfg['solver_check_steps'] == 256
assert cfg['generation_chunk'] == 64 and cfg['CPU_threads'] == 12
assert cfg['new_response_training_runs'] == cfg['new_response_updates'] == cfg['updates'] == 0
null_fields = ['science_budget', 'science_budget_s', 'wall_budget_s', 'deadline_UTC',
    'absolute_deadline_UTC', 'training_wall_s', 'training_total_limit_s', 'numeric_wall_limit_s',
    'output_limit_bytes', 'disk_free_reserve_bytes']
assert all(cfg[k] is None for k in null_fields)
assert cfg['compute_budget_enforced'] is cfg['deadlines_enforced'] is False
unchanged = ['network.py', 'noise_core.py']
assert all((WORK / n).read_bytes() == (OLD / n).read_bytes() for n in unchanged)
old_flow = (OLD / 'source_flow_core.py').read_text('utf8').replace('\r', '')
flow = (WORK / 'source_flow_core.py').read_text('utf8').replace('\r', '')
assert flow == old_flow.replace('completion,steps=64,chunk=64', 'completion,steps=128,chunk=64')
guard = (WORK / 'linux_supervise_post.py').read_text('utf8')
assert "'limit_s':None" in guard and "'GPU_admission_applied':False" in guard
assert "if total>cfg['RSS_limit_bytes']" in guard
assert "if time.monotonic()-start>20 and row['actual_worker_PID']is None" in guard
assert "'cause':'unknown'" in guard
assert "if(ROOT/(entry+'.used')).exists():raise RuntimeError" in guard
active = json.loads((WORK / 'check_models.py.LOCAL_ACTIVE_JOB.json').read_text('utf8'))
hs = active['actual_remote_handshake']
assert hs['actual_worker_PID'] == 208979 and hs['handshake_before_imports']
assert hs['limit_s'] is None and hs['stage_output_limit_bytes'] is None
names = ['config.json', 'check_models.py', 'flow_runtime.py', 'source_flow_core.py',
         'linux_supervise_post.py', *unchanged]
bindings = []
for name in names:
    raw = (WORK / name).read_bytes()
    bindings.append({'path': 'work/' + STAGE + '/' + name, 'bytes': len(raw),
                     'sha256': hashlib.sha256(raw).hexdigest()})
review = {'stage': STAGE, 'reviewed_UTC': datetime.now(timezone.utc).isoformat(),
    'actual_root_control_PID': os.getpid(), 'review_type': 'key_source_and_native_metadata_read_only',
    'read_tools': ['356440', '3f81a7', '133ece'], 'null_budget_fields': null_fields,
    'candidate_reference_steps': [128, 256], 'chunk': 64, 'CPU_threads': 12,
    'network_and_noise_source_unchanged': unchanged, 'only_solver_default_changed_in_source_flow_core': True,
    'raw_white_and_original_TRAIN_ordinal_reused': True,
    'frozen_R48_weight_paths_reused_no_response_fit': cfg['endpoint_weight_paths'],
    'both_original_128_TRAIN_rows_checked_before_DEV': True,
    'scientific_timing_and_stage_byte_kill_removed': True,
    'identity_handshake_and_RSS_safety_retained': True, 'CPU_GPU_admission_unnecessary': True,
    'first_actual_worker_PID': hs['actual_worker_PID'], 'first_actual_start_UTC': hs['start_UTC'],
    'actual_scientific_native_exit_at_review': None,
    'root_scientific_computation': False, 'migration_consistency_test': False,
    'reproduction_or_scientific_pass_inferred_from_source': False, 'bindings': bindings}
OUT.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'PID': os.getpid(), 'review': OUT.relative_to(ROOT).as_posix(),
                  'scientific_pass_known': False, 'worker_PID': hs['actual_worker_PID']}))
