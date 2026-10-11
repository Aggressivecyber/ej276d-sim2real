"""Run one named, root-reviewed operational control; preserve native output."""
from pathlib import Path
import ctypes
import json
import subprocess
import sys
P=Path(__file__).resolve().parent
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
name=sys.argv[1]
assert name in ['repair_artificial_output_collision.py']
remote=json.loads((P/'REMOTE_CONTROL_LOCATION.json').read_text('utf8'))['remote_directory']
source=P/name
upload=subprocess.run(['scp','-q','-o','BatchMode=yes',str(source),
 'rylan@100.82.231.102:'+remote+'/'+name],text=True,capture_output=True,timeout=30)
assert upload.returncode==0,upload.stderr
code='import runpy\nrunpy.run_path('+repr(remote+'/'+name)+',run_name='+repr('__main__')+')\n'
run=subprocess.run(['ssh','-T','-o','BatchMode=yes','rylan@100.82.231.102','/usr/bin/python3 -B -'],
 input=code,text=True,capture_output=True,timeout=60)
(P/(source.stem+'.NATIVE.stdout.log')).write_text(run.stdout,encoding='utf8')
(P/(source.stem+'.NATIVE.stderr.log')).write_text(run.stderr,encoding='utf8')
print(run.stdout)
print(run.stderr)
raise SystemExit(run.returncode)
