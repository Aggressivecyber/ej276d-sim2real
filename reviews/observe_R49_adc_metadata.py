"""Record existing remote aggregate metadata for a future, separate ADC diagnosis."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / 'work/R49_FROZEN_HEUN128_RESPONSE_001'
LOCAL = ROOT / 'work/R49_LOCAL_HEUN128_RECOVERY_001'
OUT = ROOT / 'reviews/ROOT_R49_ADC_SOURCE_OBSERVATION_001.json'
assert not OUT.exists()
metrics_path = REMOTE / 'metrics.json'
metrics = json.loads(metrics_path.read_text('utf8'))
identity = json.loads((REMOTE / 'REAL_IDENTITY_DESCRIPTIONS.json').read_text('utf8'))
names = ['REAL', 'TRAIN_REFERENCE', 'BASE_REAL_CONTROL',
         'GUIDED_FIXED_NOISE_REAL_CONTROL', 'GUIDED_COMPLETION_REAL_CONTROL']
rows = {}
for name in names:
    feats = metrics['local'][name]['regions']['overall']['features']
    rows[name] = {key: feats[key] for key in ['ADC_lattice_RMS_steps', 'baseline_mean_V', 'baseline_RMS_V']}
legacy = LOCAL / 'legacy_model.py'
legacy_source = legacy.read_text('utf8')
assert 'v=np.asarray(x,dtype=np.float32)' in legacy_source
assert 'out=(-.5+np.rint((np.clip(v,-.5,.5)+.5)*16383)/16383).astype(np.float32)' in legacy_source
record = {
    'UTC': datetime.now(timezone.utc).isoformat(), 'actual_control_PID': os.getpid(),
    'scope': 'Read existing remote aggregates and preserved ADC source only; current local outputs not yet evaluated.',
    'source_metrics': str(metrics_path.relative_to(ROOT)).replace('\\', '/'),
    'source_metrics_SHA256': hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    'ADC_source': str(legacy.relative_to(ROOT)).replace('\\', '/'),
    'ADC_source_SHA256': hashlib.sha256(legacy.read_bytes()).hexdigest(),
    'actual_read_tools': ['0a03f4', 'af3e87', 'a04db8'],
    'remote_existing_descriptive_features': rows,
    'remote_existing_same_REAL_identity_descriptions': identity,
    'observation': 'The recorded ADC lattice residual is near zero in all domains, but its floating-point-scale distribution differs between RAW REAL and generated REAL. The legacy ADC implementation performs rounding reconstruction after a float32 cast. These facts motivate a separately registered representation/quantization diagnosis.',
    'limitations': ['No counterfactual ADC or classifier was run in this observation.',
        'No causal attribution of the full-voltage BA is established.',
        'The lattice residual differences are far below one ADC LSB; this is not evidence of large physical voltage distortion.',
        'Baseline distribution and functional identity summaries do not exclude conditional voltage or other waveform differences.',
        'Remote aggregates must not be represented as metrics of the local recovery arrays.'],
    'current_R49_implementation_or_threshold_changes': False, 'original_A_F_unchanged': True,
    'new_scientific_array_decodes': 0, 'new_scientific_model_runs': 0,
    'new_classifier_or_density_fits': 0, 'new_scientific_stage_registered': False,
    'next_diagnostic_proposal': 'After local R49 delivery, a separate saved-output protocol can compare RAW REAL identity through the existing ADC, before/after ADC voltage representations, and baseline-subtracted voltage using the same saved groups. Keep original R49 acceptance immutable and do not treat this proposal as an accepted method.'
}
OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf8')
print(json.dumps({'receipt': OUT.relative_to(ROOT).as_posix(), 'PID': os.getpid(),
    'existing_remote_aggregates_only': True, 'new_scientific_computation': False}, ensure_ascii=False))
