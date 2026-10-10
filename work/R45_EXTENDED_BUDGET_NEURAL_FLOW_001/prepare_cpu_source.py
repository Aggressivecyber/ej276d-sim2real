from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'R43_SCALAR_TIME_NEURAL_FLOW_001'
names = [
    'generate_all.py', 'replay_all.py', 'flow_runtime.py', 'flow_diagnostics.py',
    'evaluation.py', 'diagnostic_core.py', 'assessment.py', 'energy_width.py',
    'data_support.py', 'common_math.py', 'model.py', 'legacy_model.py',
    'endpoint_common.py', 'temporal_common.py', 'mean_description.py',
    'covariance_description.py', 'random_diagnostics.py',
]
records = []
for name in names:
    source = (OLD / name).read_bytes()
    code = source.decode('utf8')
    if name in ['generate_all.py', 'replay_all.py']:
        code = code.replace('SCALAR_TIME_FLOW_FINAL', 'EXTENDED_FLOW_FINAL')
        code = code.replace('SCALAR_TIME_FLOW_REAL_CONTROL', 'EXTENDED_FLOW_REAL_CONTROL')
        code = code.replace('torch.set_num_threads(4)', 'torch.set_num_threads(12)')
        code = code.replace('torch.set_num_interop_threads(4)', 'torch.set_num_interop_threads(12)')
        code = code.replace("(ROOT/'config.json')", "(ROOT/'config_cpu.json')")
    ast.parse(code)
    target = code.encode('utf8')
    (ROOT / name).write_bytes(target)
    records.append({'name': name, 'source': str(OLD/name),
                    'source_sha256': hashlib.sha256(source).hexdigest(),
                    'new_sha256': hashlib.sha256(target).hexdigest(), 'bytes': len(target)})
record = {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
          'source_only': True, 'scientific_arrays_read': 0,
          'response_forward_update_or_fit': 0, 'copies': records,
          'CPU_generation_and_replay': 'same server Python/NumPy/Torch CPU12, chunk64',
          'no_upload_or_execution_during_active_training': True,
          'TRAIN_solver_gate_precedes_any_DEV_payload_read': True}
(ROOT/'CPU_SOURCE_PREPARATION.json').write_text(json.dumps(record, indent=2)+'\n', 'utf8')
print(json.dumps(record))
