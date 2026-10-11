"""Root fixes the R65 scientific design; no scientific arrays are opened."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import ctypes

R = Path(__file__).resolve().parents[1]
P = R/'reviews/R65_PREPARATION_001'
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
h = k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h, 8)
assert k.SetProcessWorkingSetSizeEx(h, 16*1024**2, 1024**3, 6)
read = lambda p: json.loads(p.read_text('utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = R/'tasks/R65_FULL_EMPIRICAL_D_RESPONSE_001.md'
review = read(P/'protocol_review/REVIEW.json')
assert sha(P/'PROTOCOL_BEFORE_INTEGRATION_DETAILS.md') == review['reviewed_protocol_sha256']
assert review['verdict']['preserves_original_acceptance_at_protocol_level']
assert [b['id'] for b in review['blockers']] == ['B1']
assert read(R/'R64_DELIVERY_COMPLETION_RECEIPT.json')['status'] == 'delivered'
assert read(R/'reviews/R64_ROOT_DIRECTION_001.json')['original_A_F_changed'] is False
schema = read(P/'ROOT_TRAINING_SCHEMA.json')
assert len(schema['loss_columns']) == 32 and len(schema['precision_columns']) == 4
assert len(schema['replay_columns']) == len(schema['expected_replay_row']) == 12
assert schema['updates'] == 50000
assert schema['new_stream']['draw_per_update'] == 'standard_normal((6593,480)).astype(<f4)'
source = (R/'reviews/register_R65_root.py').read_text('utf8')
ast.parse(source)
result = {
    'UTC': datetime.now(timezone.utc).isoformat(), 'root_PID': os.getpid(),
    'stage': schema['stage'], 'ready_for_registration': True,
    'ready_for_scientific_launch': False, 'original_A_F_changed': False,
    'protocol_sha256': sha(protocol), 'schema_sha256': sha(P/'ROOT_TRAINING_SCHEMA.json'),
    'prior_review_sha256': sha(P/'protocol_review/REVIEW.json'),
    'prior_reviewed_draft_sha256': review['reviewed_protocol_sha256'],
    'root_direction_sha256': sha(R/'reviews/R64_ROOT_DIRECTION_001.json'),
    'registration_source_AST_pass': True,
    'implementation_actions_before_launch': [
        'B1: R65-specific training audit for distinct D/C/discarded-D records, preserving numerical/evaluation gates.',
        'I1: Original seven streams matched to R63; once-created eighth stream exact normal API, state and chain; each-step direction binding.',
        'I2: One joint D GY, all original64 chunks, f64 CPU sum plus one original C, one cast/clip/Adam.',
        'I3: Atomic post-update checkpoint, exact binding and RNG resume, distinct physical and effective counters.',
        'I4: Full-D and C diagnostics/exposures separated; no fabricated legacy sole-change field.',
        'Frozen CPU/CUDA artificial objective and actual-network proofs and implementation source bindings.'
    ],
    'interpretation': 'Complete finite-parent, one fresh noise per parent objective; not a noise-integrated population objective. D-C coupling, exposures, compute and rounding paths change. R64 B128 negative and R60 markers remain closed.',
    'training_gate_order': 'artificial checks -> TRAIN-only training and audit -> original128/256 numeric gate -> original complete A-C -> independent4000-row replay -> D/E only on original pass',
    'scientific_arrays_opened': 0, 'trained_weights_loaded': False,
    'new_training_runs_started': 0, 'model_goal_complete': False,
    'resources': {'CPU_affinity': [3], 'native_RSS_hard_max_bytes': 1024**3},
}
target = P/'ROOT_PROTOCOL_REVIEW.json'
assert not target.exists()
target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode())
assert read(target) == result
print(json.dumps({'ready_for_registration': True, 'ready_for_scientific_launch': False,
                  'protocol_SHA': result['protocol_sha256']}))
