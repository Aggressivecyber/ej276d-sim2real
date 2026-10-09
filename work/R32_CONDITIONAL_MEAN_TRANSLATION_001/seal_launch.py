"""Capture the real native seal child termination before the last stage write."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEADLINE = dt.datetime.fromisoformat('2026-10-09T18:52:01.024475+00:00')

def main():
    start = dt.datetime.now(dt.timezone.utc)
    remaining = (DEADLINE - start).total_seconds()
    assert remaining > 1, 'no remaining seal time'
    assert not (ROOT / 'FINAL_RECEIPT.json').exists()
    t = time.monotonic()
    child = subprocess.Popen([sys.executable, '-B', str(ROOT / 'seal_stage.py')], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timeout = False
    try:
        out, err = child.communicate(timeout=min(45, remaining - 1))
    except subprocess.TimeoutExpired:
        timeout = True
        child.kill()
        out, err = child.communicate()
    native = child.returncode
    summary = json.loads(out.decode('utf-8')) if native == 0 else None
    receipt = {
        'stage': 'R32_CONDITIONAL_MEAN_TRANSLATION_001',
        'status': 'sealed' if native == 0 else 'seal_failed',
        'closure_control_PID': os.getpid(), 'closure_parent_PID': os.getppid(),
        'seal_child_PID': child.pid, 'seal_child_native_exit': native,
        'seal_child_timeout': timeout, 'start_UTC': start.isoformat(),
        'child_completed_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
        'elapsed_control_s': time.monotonic() - t,
        'native_child_stdout': out.decode('utf-8', errors='replace'),
        'native_child_stderr': err.decode('utf-8', errors='replace'),
        'manifest_binding': summary,
        'no_further_stage_writes': True,
        'science_recomputed_during_finalization': False,
        'science_elapsed_s': 345.523252972899, 'science_limit_s': 2400,
        'scientific_gate_passed': False,
        'all_final_figures_actual_viewed': 16,
        'next_scientific_task_started': False,
        'executor_seal_deadline_UTC': DEADLINE.isoformat(),
        'last_write_started_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if summary:
        receipt['stage_regular_file_count'] = summary['member_count'] + 2
        receipt['stage_regular_file_bytes'] = 0
    def encode():
        return (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    for _ in range(8):
        raw = encode()
        if not summary:
            break
        exact = summary['member_bytes'] + summary['manifest_bytes'] + len(raw)
        if exact == receipt['stage_regular_file_bytes']:
            break
        receipt['stage_regular_file_bytes'] = exact
    raw = encode()
    assert dt.datetime.now(dt.timezone.utc) < DEADLINE, 'deadline before final receipt'
    assert len(raw) < 64 * 1024
    if summary:
        assert summary['member_bytes'] + summary['manifest_bytes'] + len(raw) <= 512 * 1024 * 1024
    final_path = ROOT / 'FINAL_RECEIPT.json'
    final_path.write_bytes(raw)
    actual_last_write = dt.datetime.fromtimestamp(final_path.stat().st_mtime, dt.timezone.utc)
    print(json.dumps({'closure_control_PID': os.getpid(), 'seal_child_PID': child.pid,
                      'seal_child_native_exit': native, 'status': receipt['status'],
                      'actual_last_write_UTC': actual_last_write.isoformat(),
                      'stage_regular_file_count': receipt.get('stage_regular_file_count'),
                      'stage_regular_file_bytes': receipt.get('stage_regular_file_bytes'),
                      'FINAL_MANIFEST_sha256': summary['manifest_sha256'] if summary else None,
                      'FINAL_RECEIPT_sha256': hashlib.sha256(raw).hexdigest(),
                      'no_further_stage_writes': True}))
    return native if native is not None else 1

if __name__ == '__main__':
    raise SystemExit(main())
