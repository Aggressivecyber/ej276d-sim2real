"""Read completed small summaries while the normal private payload copy runs."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import json
import subprocess

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
D = R/'reviews/R64_RESULT_ANALYSIS_001/completed_summary_copy'
D.mkdir(exist_ok=True, parents=True)
remote = json.loads((S/'config.json').read_text('utf8'))['remote_root']
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << 3)
names = ['RESULT.json', 'PARAMETER_GRADIENT_SUMMARY.json', 'ENDPOINT_GRADIENT_SUMMARY.json',
         'MACRO_LOSS_AND_GRADIENT_RECORDS.json', 'diagnose_train.py.NATIVE_RECEIPT.json', 'FIGURE_MANIFEST.json']
code = 'from pathlib import Path\nimport json\np=Path('+repr(remote)+')\n'+'''
n=json.loads((p/'diagnose_train.py.NATIVE_RECEIPT.json').read_text())
assert n['actual_worker_native_exit']==n['worker_self_receipt']['native_exit']==0
assert n['termination_reason']is None and not Path('/proc',str(n['actual_worker_PID'])).exists()
assert json.loads((p/'DIAGNOSTIC_PROGRESS.json').read_text())['phase']=='registered_diagnostic_endpoint_complete'
assert not(p/'DIAGNOSTIC_FAILURE.json').exists()
'''+ 'names='+repr(names)+'\nprint(json.dumps({name:(p/name).read_text() for name in names}))\n'
args = ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'rylan@100.82.231.102', '/usr/bin/python3 -B -']
p = subprocess.run(args, input=code, text=True, capture_output=True, timeout=60)
assert p.returncode == 0, dict(native_exit=p.returncode, stderr=p.stderr)
values = json.loads(p.stdout)
for name, value in values.items():
    assert not (D/name).exists()
    json.loads(value)
    (D/name).write_text(value, 'utf8')
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), source=remote, files=names,
    remote_native_exit=0, actual_PID_absent=True, all_32_macro_groups_complete=True,
    metadata_read_native_exit=p.returncode, full_private_payload_transfer_is_separate=True,
    migration_content_consistency_test=False)
(D/'COPY_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', 'utf8')
print(json.dumps(receipt, indent=2))
