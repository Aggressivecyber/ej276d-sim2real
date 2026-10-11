"""Copy only sealed definition/proof bundles into the registered R65 stage."""
from pathlib import Path
import ast
import ctypes
import hashlib
import json
import shutil

R = Path(__file__).resolve().parents[1]
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
P = R/'reviews/R65_PREPARATION_001'
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
k.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
h = k.GetCurrentProcess()
assert k.SetProcessAffinityMask(h,8) and k.SetProcessWorkingSetSizeEx(h,16*1024**2,1024**3,6)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text('utf8'))
src = P/'objective_adapter'
manifest = read(src/'PREEXECUTION_BINDING.json')
names = [r['file'] for r in manifest['files']]
assert len(names) == len(set(names))
for record in manifest['files']:
    assert sha(src/record['file']) == record['sha256'],record['file']
assert sha(src/'objective_adapter.py') == '1b8f38324b96212ca3cc6d83f603c89d8b18f731ab7227998a8da718122dfaec'
dst = S/'objective_adapter'
assert not dst.exists()
dst.mkdir()
copied = []
for name in names+['PREEXECUTION_BINDING.json','README.md','DELIVERY_RECEIPT.json']:
    target = dst/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src/name,target)
    copied.append({'name':str(target.relative_to(S)).replace('\\','/'),'sha256':sha(src/name)})
src = P/'network_update_check'
assert sha(src/'network_update_check.py') == 'db8976450aec139e61b7a39b5d6f2c8234028b676fb578eea1819cda8ee77945'
assert sha(src/'SOURCE_BINDING.json') == '7e3a60f76968da95d96fd20953d0cd34459a8591406bad99aaa52ef137587afb'
dst = S/'network_update_check'
assert not dst.exists()
dst.mkdir()
for name in ('network_update_check.py','SOURCE_BINDING.json','FROZEN_TOLERANCES.json',
             'FIXTURE_SPEC.json','README.md','PREPARATION_RECEIPT.json','STATIC_RECEIPT.json'):
    shutil.copy2(src/name,dst/name)
    copied.append({'name':str((dst/name).relative_to(S)).replace('\\','/'),'sha256':sha(src/name)})
for p in (S/'objective_adapter').rglob('*.py'):
    ast.parse(p.read_text('utf-8-sig'))
ast.parse((dst/'network_update_check.py').read_text('utf8'))
for name in ('SOURCE_PSD_PRECISION_PRIVATE.npz','SOURCE_ORDER_PRECISION_DEFINITION.json','TRAIN_TEMPORAL_CONSTANTS.json'):
    src = R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'/name
    assert sha(S/name) == sha(src)
    copied.append({'name':name,'sha256':sha(src),'frozen_source':str(src)})
receipt = {'files':copied,'scientific_arrays_decoded':False,'tests_executed':False,
           'description':'Copies of frozen definitions and artificial fixtures, before native scientific launch.'}
(S/'HELPER_INTEGRATION.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
print(json.dumps({'copied_files':len(copied),'AST_pass':True,'scientific_execution_started':False}))
