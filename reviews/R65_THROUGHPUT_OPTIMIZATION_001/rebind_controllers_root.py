"""Preserve the stopped controller, then change only its reviewed source binding."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import ctypes
import hashlib
import json
import re
import shutil
import sys

P=Path(__file__).resolve().parent
R=P.parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
old=R/'reviews/R65_WAIT_ANALYSIS_001/post_controllers/prepared'
new=P/'new_controller_package'
OLD='e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5'
NEW='5d26170c74bfd1294511d2684455942bd29581fca7ac6f4414e95aaa6134339c'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2);f.write('\n')
def make_launcher():
    launcher=(R/'reviews/start_R65_watch_root.ps1').read_text('utf8')
    launcher=launcher.replace('reviews\\R65_WAIT_ANALYSIS_001\\post_controllers\\prepared', 'reviews\\R65_THROUGHPUT_OPTIMIZATION_001\\new_controller_package')
    launcher=launcher.replace('reviews\\R65_WAIT_ANALYSIS_001\\ROOT_CLOSURE_REVIEW.json', 'reviews\\R65_THROUGHPUT_OPTIMIZATION_001\\ROOT_CONTROLLER_REBIND_REVIEW.json')
    launcher=launcher.replace('ROOT_WAIT_PREPARATION_REVIEW_COMPLETE','ROOT_CONTROLLER_REBIND_REVIEW_COMPLETE')
    assert str(new.name) in launcher and 'ROOT_CONTROLLER_REBIND_REVIEW.json' in launcher
    target=P/'start_rebound_watch_root.ps1'
    assert not target.exists()
    target.write_text(launcher,encoding='utf8')
if '--launcher-only' in sys.argv:
    prior=read(P/'ROOT_CONTROLLER_REBIND_REVIEW.json')
    assert prior['status']=='ROOT_CONTROLLER_REBIND_REVIEW_COMPLETE'
    assert sha(new/'CONTROLLER_BINDING.json')==prior['new_binding_SHA256']
    make_launcher()
    print(json.dumps({'launcher_ready':True,'existing_review_reverified':True}))
    raise SystemExit(0)
assert not new.exists() and not (P/'ROOT_CONTROLLER_REBIND_REVIEW.json').exists()
review=read(S/'ROOT_IMPLEMENTATION_REVIEW.json')
assert sha(S/'ROOT_IMPLEMENTATION_REVIEW.json')==NEW and len(review['files'])==115
for row in review['files']:assert sha(S/row['name'])==row['sha256']
binding=read(old/'CONTROLLER_BINDING.json')
assert binding['implementation_review_SHA256']==OLD and binding['frozen_file_count']==112
for row in binding['files']:assert sha(old/row['name'])==row['sha256']
runtime=S/'controller_runtime'
launch=read(runtime/'ROOT_WATCHER_LAUNCH.json')
stopped=read(runtime/'PIPELINE_WATCH_STOPPED.json')
assert launch['PID']==53832 and stopped['state']=='LOCAL_WATCH_STOPPED'
assert stopped['remote_worker_killed'] is False
pids=(ctypes.c_ulong*32768)();needed=ctypes.c_ulong()
assert ctypes.WinDLL('psapi',use_last_error=True).EnumProcesses(pids,ctypes.sizeof(pids),ctypes.byref(needed))
assert launch['PID'] not in pids[:needed.value//ctypes.sizeof(ctypes.c_ulong)]
archived=P/'old_controller_runtime_20261011_045543_stopped'
assert not archived.exists()
assert runtime.resolve().is_relative_to(R.resolve()) and archived.resolve().is_relative_to(R.resolve())
runtime.rename(archived)
new.mkdir()
for file in old.iterdir():
    assert file.is_file()
    shutil.copy2(file,new/file.name)
delta=[]
for name in ['r65_controller_common.py','r65_remote_gate.py']:
    before=(old/name).read_text('utf8')
    after=re.sub(r'\b112\b','115',before.replace(OLD,NEW))
    assert after!=before and OLD not in after
    restored=re.sub(r'\b115\b','112',after.replace(NEW,OLD))
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(before),include_attributes=False)
    (new/name).write_text(after,encoding='utf8')
    delta.append({'name':name,'before_SHA256':sha(old/name),'after_SHA256':sha(new/name),
      'AST_identical_after_restoring_only_review_hash_and_frozen_count':True})
manifest=read(new/'FROZEN_IMPLEMENTATION_MANIFEST.json')
manifest.update(implementation_review_SHA256=NEW,frozen_file_count=115,files=review['files'])
(new/'FROZEN_IMPLEMENTATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
readme=(old/'README.md').read_text('utf8')
readme=re.sub(r'\b112\b','115',readme.replace(OLD,NEW))
readme='This package is rebound to the root-reviewed, numerically equivalent gradient-pack upgrade. Post-training eligibility and transition behavior are unchanged. See ../ROOT_CONTROLLER_REBIND_REVIEW.json.\n\n'+readme
(new/'README.md').write_text(readme,encoding='utf8')
binding.update(implementation_review_SHA256=NEW,frozen_file_count=115,
 runtime_execution_status='ROOT_STATIC_REBIND_REVIEW_COMPLETE_NATIVE_OBSERVATION_PENDING')
binding['offline_metadata_tests']['scope']='Prior original-binding tests; not rerun. Rebinding alters only hash and frozen file count in executable AST.'
for row in binding['files']:row.update(sha256=sha(new/row['name']),bytes=(new/row['name']).stat().st_size)
(new/'CONTROLLER_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf8')
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'status':'ROOT_CONTROLLER_REBIND_REVIEW_COMPLETE',
 'prior_controller_PID_absent':True,'prior_runtime_preserved_at':str(archived),
 'prior_binding_SHA256':sha(old/'CONTROLLER_BINDING.json'),'new_binding_SHA256':sha(new/'CONTROLLER_BINDING.json'),
 'old_implementation_review_SHA256':OLD,'new_implementation_review_SHA256':NEW,
 'source_changes':delta,'all_other_executable_controller_sources_byte_identical':all(
     sha(f)==sha(old/f.name) for f in new.glob('*.py') if f.name not in {r['name'] for r in delta}),
 'science_or_acceptance_policy_changed':False,'new_scientific_computation':False,
 'watcher_launched':False,'native_observation_pending':True}
assert receipt['all_other_executable_controller_sources_byte_identical']
write(P/'ROOT_CONTROLLER_REBIND_REVIEW.json',receipt)
make_launcher()
print(json.dumps(receipt))
