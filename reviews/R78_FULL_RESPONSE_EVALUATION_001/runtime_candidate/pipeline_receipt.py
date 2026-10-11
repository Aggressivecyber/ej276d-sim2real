"""Record sequential stage orchestration; scientific native exits remain primary."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
ENTRIES = ['generate_full_response.py','replay_full_response.py',
           'evaluate_saved.py','summarize_and_plot.py']


def read(path):
    return json.loads(path.read_text('utf8'))


def write(path, value, exclusive=False):
    with path.open('x'if exclusive else'w',encoding='utf8')as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


mode = sys.argv[1]
cfg = read(ROOT/'config.json')
assert cfg['allowed_scientific_entries'] == ENTRIES
if mode == 'started':
    assert not (ROOT/'PIPELINE_COMPLETION.json').exists()
    write(ROOT/'PIPELINE_IDENTITY.json',dict(UTC=utc(),shell_PID=os.getppid(),
        config_SHA256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
        entries=ENTRIES,scientific_workers_sequential=True),exclusive=True)
elif mode == 'progress':
    entry = sys.argv[2]
    assert entry in ENTRIES and read(ROOT/'PIPELINE_IDENTITY.json')['shell_PID'] == os.getppid()
    write(ROOT/'PIPELINE_PROGRESS.json',dict(UTC=utc(),current_entry=entry,
        entry_index=ENTRIES.index(entry),total_entries=len(ENTRIES),shell_PID=os.getppid()))
elif mode in ['completed','failed']:
    identity = read(ROOT/'PIPELINE_IDENTITY.json')
    assert identity['shell_PID'] == os.getppid()
    receipts = {}
    for entry in ENTRIES:
        path = ROOT/(entry+'.NATIVE_RECEIPT.json')
        if path.exists():
            row = read(path)
            receipts[entry] = dict(actual_worker_PID=row['actual_worker_PID'],
                actual_worker_native_exit=row['actual_worker_native_exit'],
                termination_reason=row['termination_reason'],
                elapsed_s=row['elapsed_s'],receipt_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    if mode == 'completed':
        assert list(receipts) == ENTRIES
        assert all(type(row['actual_worker_native_exit'])is int and
            row['actual_worker_native_exit'] == 0 and row['termination_reason'] is None
            for row in receipts.values())
        gate = read(ROOT/'GATE_METRICS.json')
        assert (ROOT/'SUMMARY_PLOT_NUMERIC_RECEIPT.json').exists()
        status = 'ALL_FOUR_OWNED_WORKERS_EXITED_ZERO'
        passed = gate['primary_A_C_pass']
        failure_entry,supervisor_code = None,None
    else:
        status = 'SEQUENTIAL_EXECUTION_STOPPED_ON_ENTRY_FAILURE'
        failure_entry,supervisor_code = sys.argv[2],int(sys.argv[3])
        assert failure_entry in ENTRIES and supervisor_code != 0
        passed = None
    write(ROOT/'PIPELINE_COMPLETION.json',dict(UTC=utc(),status=status,shell_PID=os.getppid(),
        scientific_native_receipts=receipts,primary_A_C_pass=passed,
        failed_entry=failure_entry,failed_supervisor_return_code=supervisor_code,
        pipeline_shell_native_exit='not_observed_by_child; root must separately confirm process gone',
        model_A_F_complete=False,D_E_executed=False),exclusive=True)
else:
    raise ValueError(mode)
