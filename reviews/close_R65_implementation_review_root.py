"""Seal the registered R65 implementation; stdlib/static preparation only."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import ctypes
import hashlib
import json
import os
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
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (S/'ROOT_IMPLEMENTATION_REVIEW.json').exists()
driver=read(P/'driver_review/REVIEW.json')
closed=read(P/'driver_review/BINDING_RECEIPT.json')
assert driver['unresolved_blockers']==[] and closed['native_review_process_exit_code']==0
assert closed['source_bindings_unchanged_on_reopen'] and closed['all_output_text_JSON_AST_reopened']
assert sha(S/'train_final.py')==driver['reviewed_driver_SHA256']==closed['reviewed_driver_SHA256']
assert sha(S/'r65_training_audit.py')==driver['reviewed_audit_SHA256']
for name,rec in closed['artifacts'].items():
    assert sha(P/'driver_review'/name)==rec['sha256'],name
cfg,old=read(S/'config.json'),read(H/'config.json')
for key in driver['frozen_config_keys_equal_R63']:
    assert cfg[key]==old[key],key
assert cfg['expected_completed_update_counts']==driver['expected_completed_counts_recomputed_from_registered_population_chunk_batch_sizes']
assert sha(S/'PROTOCOL.md')==sha(R/cfg['protocol'])
for rec in read(S/'HELPER_INTEGRATION.json')['files']:
    assert sha(S/rec['name'])==rec['sha256'],rec['name']
for name,digest in read(S/'POSTPROCESS_FILE_BINDING.json')['bound_SHA256'].items():
    assert sha(S/name)==digest,name
post=read(P/'post_pipeline/STATIC_VERIFICATION.json')
assert post['passed'] and all(post['checks'].values())
assert sha(P/'post_pipeline/SOURCE_BINDING.json')==post['binding_sha256']
audit=read(P/'training_audit_adapter/STATIC_VALIDATION.json')
assert audit['status']=='pass' and all(audit['checks'].values())
synthetic=read(P/'training_audit_adapter/synthetic_saved_record_check/attempt_001/RESULT.json')
native=read(P/'training_audit_adapter/synthetic_saved_record_check/attempt_001/NATIVE_EXIT.json')
assert synthetic['status']=='pass' and len(synthetic['matrix'])==5 and all(x['test_pass'] for x in synthetic['matrix'])
assert native['native_exit_code']==0 and synthetic['real_scientific_array_or_weight_reads']==0
for name in ('post_audit.py','r65_training_audit.py'):
    assert sha(S/name)==sha(P/'training_audit_adapter'/name)
for name in ('network.py','noise_core.py','source_flow_core.py','distribution_loss.py',
        'R61_distribution_loss.py','precision_order.py','SOURCE_PSD_PRECISION_PRIVATE.npz',
        'TRAIN_DISTRIBUTION_CONSTANTS.npz','TRAIN_TEMPORAL_CONSTANTS.npz',
        'UNIT_CHARGE_CONSTANTS.npz','NOISE_CALIBRATION.npz','linux_supervise.py','linux_owned_worker.py'):
    assert sha(S/name)==sha(H/name),name
evidence_names=(
    'driver_review/REVIEW.json','driver_review/BINDING_RECEIPT.json',
    'driver_review/SOURCE_BINDINGS.json','driver_review/NATIVE_EXIT_RECEIPT.json',
    'training_audit_adapter/RECEIPT.json','training_audit_adapter/SOURCE_BINDINGS.json',
    'training_audit_adapter/STATIC_VALIDATION.json','training_audit_adapter/TRAINING_RECEIPT_CONTRACT.json',
    'training_audit_adapter/synthetic_saved_record_check/attempt_001/RESULT.json',
    'training_audit_adapter/synthetic_saved_record_check/attempt_001/NATIVE_EXIT.json',
    'post_pipeline/SOURCE_BINDING.json','post_pipeline/STATIC_VERIFICATION.json',
    'post_pipeline/PREPARATION_RECEIPT.json','post_pipeline/VERIFY_NATIVE_EXIT_RECEIPT.json',
)
for name in evidence_names:
    target=S/'preparation_receipts'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    shutil.copy2(P/name,target)
    assert sha(target)==sha(P/name)
files=[]
AST=[]
JSON=[]
for path in sorted(S.rglob('*')):
    if not path.is_file():
        continue
    name=path.relative_to(S).as_posix()
    assert not path.is_symlink() and path.suffix in ('.py','.json','.md','.npz'),name
    assert not any(p in ('cache','__pycache__','artificial_results','native_attempts') for p in path.parts),name
    if path.suffix=='.py':
        ast.parse(path.read_text('utf8'),filename=name)
        AST.append(name)
    if path.suffix=='.json':
        read(path)
        JSON.append(name)
    files.append({'name':name,'sha256':sha(path),'bytes':path.stat().st_size})
for name in ('deploy_R65_root.py','poll_R65_root.py'):
    ast.parse((R/'reviews'/name).read_text('utf8'))
review={
    'UTC':datetime.now(timezone.utc).isoformat(),'root_PID':os.getpid(),
    'stage':cfg['stage'],'ready_for_launch':True,'original_A_F_changed':False,
    'meaning':'Approved to launch the registered owned entry. Eight frozen CPU/CUDA artificial checks must pass inside that entry before scientific TRAIN decode. This is not a runtime or scientific acceptance pass.',
    'protocol_SHA256':sha(S/'PROTOCOL.md'),'config_SHA256':sha(S/'config.json'),
    'driver_SHA256':sha(S/'train_final.py'),'source_audit_SHA256':sha(S/'r65_training_audit.py'),
    'original_acceptance_SHA256':{name:sha(R/name) for name in ('PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md')},
    'files':files,'static_AST_pass_files':AST,'JSON_reopened_files':JSON,
    'independent_static_review_unresolved_blockers':0,'independent_fixed_static_findings':6,
    'training_audit_synthetic_cases_passed':5,'training_audit_static_checks_passed':17,
    'postprocessing_original_threshold_and_calculation_checks_passed':True,
    'frozen_config_keys_equal_R63':driver['frozen_config_keys_equal_R63'],
    'original_network_constraints_constants_and_supervisor_byte_identical':True,
    'artificial_server_suite_still_required':True,'scientific_arrays_decoded':0,
    'scientific_weights_loaded':False,'new_response_updates':0,'model_goal_complete':False,
    'recovery_validation':'Static code review and synthetic saved-record audit only; no native checkpoint/fault-injection evidence claimed.',
    'CPU_affinity':[3],'native_RSS_hard_max_bytes':1024**3,
    'limits':{'server_CPU':12,'server_RSS_bytes':48*1024**3,'server_GPU_bytes':10*1024**3,
              'single_scientific_job':True,'compute_budget':None,'local_new_files':'E/D only'},
}
out=S/'ROOT_IMPLEMENTATION_REVIEW.json'
out.write_bytes((json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
assert read(out)==review
print(json.dumps({'review_SHA256':sha(out),'files':len(files),'AST_pass':len(AST),
                  'JSON_reopened':len(JSON),'ready_for_launch':True,'scientific_updates':0},indent=2))
