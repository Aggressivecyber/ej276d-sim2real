"""Render frozen R78 results only; preserve original failed summary and all metrics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time

P = Path(__file__).resolve().parent
R = P.parents[2]
S = R / 'work/R78_FULL_RESPONSE_EVALUATION_001'
sys.path.insert(0, str(R / 'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply, memory
guard = apply(P / 'runtime')
sys.path.insert(0, str(S))
sys.path.insert(0, str(P))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return json.loads((S / name).read_text('utf-8'))

if __name__ == '__main__':
    start = time.monotonic()
    names = ['metrics.json', 'GATE_METRICS.json', 'ACCEPTANCE_A_C.json',
             'ROOT_AGGREGATE_EVIDENCE.json', 'config_post.json',
             'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz', 'EVALUATION_NUMERIC_RECEIPT.json',
             'SUMMARY_FAILURE.json', 'summarize_and_plot.py.NATIVE_RECEIPT.json']
    before = {name: sha(S / name) for name in names}
    m, gate, acceptance, evidence, config = [read(name) for name in names[:5]]
    assert m['status'] == 'complete'
    assert gate['primary_A_C_pass'] is False
    from render_full_response_repaired import run
    figures = run(S, config, m, gate, acceptance, evidence)
    assert len(figures) == 16
    assert before == {name: sha(S / name) for name in names}
    receipt = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'FROZEN_RESULT_PLOTS_REPAIRED', 'elapsed_s': time.monotonic() - start,
        'reason': 'Historical REAL density rows contain saved fit parameters but no histogram_counts.',
        'change': 'Draw saved REAL mixture curves with dashed line and explicit labels; retain original SOURCE/BASE histograms.',
        'original_summary_failure_and_native_exit1_preserved': True,
        'source_event_arrays_read': False, 'new_model_forward_rows': 0,
        'new_HGB_fits': 0, 'new_EM_fits': 0,
        'Q_ADC_CDF_identity_migrations_recomputed': False,
        'only_existing_derived_moment_packet_read': True,
        'scientific_inputs_unchanged_SHA256': before,
        'renderer_SHA256': sha(P / 'render_full_response_repaired.py'),
        'primary_A_C_pass': gate['primary_A_C_pass'],
        'figure_files': figures, 'guard': dict(guard, **memory()),
        'root_visual_QA_pending': True,
    }
    with (P / 'PLOT_REPAIR_RECEIPT.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['figure_files', 'scientific_inputs_unchanged_SHA256']}, allow_nan=False))
