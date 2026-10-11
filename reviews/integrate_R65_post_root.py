"""Integrate verified R65 audit and frozen evaluation adapters, without running."""
from pathlib import Path
import ast
import ctypes
import hashlib
import json
import shutil

R=Path(__file__).resolve().parents[1]
S=R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
P=R/'reviews/R65_PREPARATION_001'
H=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h=k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text('utf8'))
post=P/'post_pipeline'
assert sha(post/'SOURCE_BINDING.json')=='e76e474effb66673600bd2c23f758eec93beee72e7e7181b4432a776be851d1b'
assert read(post/'STATIC_VERIFICATION.json')['passed']
records=read(post/'SOURCE_BINDING.json')['all_prepared_files']
assert len(records)==22
copied={}
for record in records:
    name=record['name']
    src=post/'prepared'/name
    assert src.is_file() and sha(src)==record['sha256']
    assert not (S/name).exists(),name
    shutil.copy2(src,S/name)
    copied[name]=record['sha256']
audit=P/'training_audit_adapter'
for name,expected in {
    'post_audit.py':'8a92aef6c15892ccbf97409c8f121b02b1b08bfe85b1d6f6286948fc9807e4e2',
    'r65_training_audit.py':'88c26b435621e8454254447ec03e0c6a2ae391f1463c601ea13c7e06861d88ad',
}.items():
    assert sha(audit/name)==expected
    assert not (S/name).exists()
    shutil.copy2(audit/name,S/name)
    copied[name]=expected
assert read(H/'TRAIN_SOLVER_CHECK.json')['both_pass']
for src,name in ((H/'TRAIN_SOLVER_CHECK.json','HISTORICAL_R63_TRAIN_SOLVER_CHECK.json'),
                 (H/'linux_supervise_reporting.py','linux_supervise_reporting.py')):
    assert not (S/name).exists()
    shutil.copy2(src,S/name)
    copied[name]=sha(src)
for name in copied:
    if name.endswith('.py'):
        ast.parse((S/name).read_text('utf-8-sig'))
record={'bound_SHA256':copied,'scientific_execution_started':False,
    'post_pipeline_source_binding_SHA':sha(post/'SOURCE_BINDING.json'),
    'original_A_F_and_numerical_math_unchanged':True,
    'actual_pipeline_execution_and_figures_QA':'not_yet_run',
    'historical_R63_numerical_gate_both_pass':True}
(S/'POSTPROCESS_FILE_BINDING.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps({'integrated_files':len(copied),'AST_pass':True,'scientific_execution_started':False}))
