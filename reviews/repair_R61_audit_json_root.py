"""Repair one JSON scalar type; verify the complete audit on saved TRAIN records."""
from pathlib import Path
import json,hashlib,shutil,os,sys,ctypes
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
assert S.drive.upper()in ['E:','D:']
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMP=str(S/'cache/tmp'),TEMP=str(S/'cache/tmp'),TMPDIR=str(S/'cache/tmp'),TORCH_HOME=str(S/'cache/torch'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
sys.dont_write_bytecode=True
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
import numpy as np
sys.path.insert(0,str(S))
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
assert read(S/'LOCAL_PIPELINE_CONTROL.json')['status']=='scientific_entry_failed_no_retry'
assert not read(S/'LIVE_OBSERVATION.json')['actual_process']['exists']
failed=read(S/'numerical_check.py.NATIVE_RECEIPT.json')
assert failed['actual_worker_native_exit']==1 and 'bool_' in failed['worker_self_receipt']['error']
assert read(S/'train_final.py.NATIVE_RECEIPT.json')['actual_worker_native_exit']==0
assert 'numerical_check.py", line 20' in (S/'numerical_check.py.stderr.log').read_text('utf8')
assert not (S/'TRAIN_SOLVER_CHECK.json').exists() and not (S/'GENERATION_RECEIPT.json').exists()
A=S/'repairs/AUDIT_JSON_001';A.mkdir(parents=True,exist_ok=True)
assert not (A/'REPAIR_REVIEW.json').exists()
names=['post_audit.py','POSTPROCESS_FILE_BINDING.json','LOCAL_PIPELINE_CONTROL.json','LOCAL_PIPELINE_CONTROL.started.json','LOCAL_LAUNCH_numerical_check.py.json','REMOTE_ACTIVE_JOB.json','REMOTE_RUN_HISTORY.json']
names+=sorted(p.name for p in S.glob('numerical_check.py.*')if p.is_file())
names+=sorted(p.name for p in S.glob('LAUNCH_numerical_check.py*')if p.is_file())
for n in sorted(set(names)):
    if (A/n).exists():assert (A/n).read_bytes()==(S/n).read_bytes(),n
    else:shutil.copy2(S/n,A/n)
old=(S/'post_audit.py').read_text('utf8')
before="'loss_records_complete_finite':losses.shape==(50000,20) and np.isfinite(losses).all(),"
after="'loss_records_complete_finite':bool(losses.shape==(50000,20) and np.isfinite(losses).all()),"
assert old.count(before)==1
new=old.replace(before,after);assert new.replace(after,before)==old
P=read(S/'config_post.json');C=read(S/'config.json');arm=C['arm'];train=read(S/arm/'TRAINING_RECEIPT.json')
H=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
local_cfg=dict(P,historical_R59_root=str(H));local_cfg['input_paths']=dict(P['input_paths'],distribution_statistics=str(H/'TRAIN_DISTRIBUTION_CONSTANTS.npz'))
old_ns={};new_ns={};exec(compile(old,'post_audit_before.py','exec'),old_ns);exec(compile(new,'post_audit_after.py','exec'),new_ns)
old_audit=old_ns['training_audit'](S,local_cfg,train)
assert isinstance(old_audit['checks']['loss_records_complete_finite'],np.bool_)
try:json.dumps(old_audit,allow_nan=False)
except TypeError as e:assert 'not JSON serializable' in str(e)
else:raise AssertionError('Original saved-record failure not reproduced')
audit=new_ns['training_audit'](S,local_cfg,train)
assert audit==old_audit and audit['pass'] and all(type(v)is bool for v in audit['checks'].values())
assert json.loads(json.dumps(audit,allow_nan=False))==audit
for vals in [np.zeros((2,2)),np.array([[0.,np.nan],[0.,0.]]),np.array([[0.,np.inf],[0.,0.]])]:
    original=vals.shape==(2,2) and np.isfinite(vals).all();converted=bool(original)
    assert converted==original and json.loads(json.dumps({'finite':converted}))['finite']is converted
protected=read(S/'POSTPROCESS_PREPARATION.json')['training_files_and_protocol_acceptance_untouched']
assert all(sha(Path(path))==digest for path,digest in protected.items())
binding=read(S/'POSTPROCESS_FILE_BINDING.json');assert all(sha(S/n)==v for n,v in binding['bound_SHA256'].items())
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
mem=PMC();mem.cb=ctypes.sizeof(mem);assert psapi.GetProcessMemoryInfo(handle,ctypes.byref(mem),mem.cb) and mem.PeakWorkingSetSize<=4*1024**3
(S/'post_audit.py').write_bytes((A/'post_audit.py').read_bytes().replace(before.encode(),after.encode()))
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'pass':True,'scope':'Wrap loss_records_complete_finite in native bool for JSON serialization only','original_failure_reproduced_on_saved_TRAIN':True,'all_audit_values_equal_before_after':True,'all_28_checks_pass':len(audit['checks'])==28 and audit['pass'],'check_count':len(audit['checks']),'JSON_roundtrip_pass':True,'finite_and_nonfinite_boolean_semantics_retained':True,'training_protocol_acceptance_protected_hashes_unchanged':True,'old_post_audit_SHA256':sha(A/'post_audit.py'),'new_post_audit_SHA256':sha(S/'post_audit.py'),'frozen_weight_SHA256':sha(S/arm/'FINAL_FLOW.pt'),'failed_attempt':failed,'new_response_updates':0,'new_neural_forward_rows':0,'DEV_EVAL_CAL_decoded':0,'CPU_affinity':cpus,'worker_peak_RSS_bytes':mem.PeakWorkingSetSize,'audit':audit}
write(A/'REPAIR_REVIEW.json',receipt)
binding['bound_SHA256']['post_audit.py']=sha(S/'post_audit.py');binding.update(binding_revision=2,serialization_only_repair='repairs/AUDIT_JSON_001/REPAIR_REVIEW.json',previous_binding='repairs/AUDIT_JSON_001/POSTPROCESS_FILE_BINDING.json')
write(S/'POSTPROCESS_FILE_BINDING.json',binding)
print(json.dumps({k:v for k,v in receipt.items()if k not in ['audit','failed_attempt']},indent=2))
