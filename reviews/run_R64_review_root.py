"""Record native output for the single saved-artifact audit invocation."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
entry = R/'reviews/review_R64_results_root.py'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
mask, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system))
allowed = [i for i in range(64) if mask.value & (1 << i)]
chosen = allowed[min(3, len(allowed)-1)]
assert kernel.SetProcessAffinityMask(handle, 1 << chosen)
started = time.monotonic()
prefix = S/'ROOT_ARTIFACT_REVIEW'
assert not prefix.with_suffix('.NATIVE_RECEIPT.json').exists(), 'Preserve and resolve an earlier audit attempt before repeating.'
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('TEMP', 'TMP', 'TMPDIR'):
    env[key] = str(S/'cache/root_saved_artifact_review')
Path(env['TEMP']).mkdir(exist_ok=True, parents=True)
with prefix.with_suffix('.stdout.log').open('w', encoding='utf8') as out, prefix.with_suffix('.stderr.log').open('w', encoding='utf8') as err:
    p = subprocess.Popen([sys.executable, '-B', str(entry)], cwd=R, env=env, stdout=out, stderr=err)
    print(json.dumps(dict(actual_audit_PID=p.pid, controller_PID=os.getpid(), inherited_CPU_affinity=[chosen])), flush=True)
    code = p.wait()
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), actual_audit_PID=p.pid, controller_PID=os.getpid(),
    native_exit=code, elapsed_s=time.monotonic()-started, entry=str(entry), source_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
    stdout_path=str(prefix.with_suffix('.stdout.log')), stderr_path=str(prefix.with_suffix('.stderr.log')))
prefix.with_suffix('.NATIVE_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', 'utf8')
print(json.dumps(receipt, indent=2))
if code:
    print(prefix.with_suffix('.stderr.log').read_text('utf8'))
    raise SystemExit(code)
