"""Root closure of already-executed offline control/recovery checks.

This does not rerun tests, connect to the server, or start scientific work.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import ctypes
import hashlib
import json
import os

R = Path(__file__).resolve().parents[1]
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
W = R/'reviews/R65_WAIT_ANALYSIS_001'
P = W/'post_controllers'
B = P/'prepared'
Q = W/'recovery_procedure'
K = ctypes.WinDLL('kernel32',use_last_error=True)
K.GetCurrentProcess.restype = ctypes.c_void_p
K.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
K.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_ulong]
H = K.GetCurrentProcess()
assert K.SetProcessAffinityMask(H,8)
assert K.SetProcessWorkingSetSizeEx(H,16*1024**2,1024**3,6)
read = lambda path:json.loads(path.read_text('utf8'))
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
records = []

def bind(path, wanted=None, size=None):
    assert path.resolve().is_relative_to(R.resolve())
    digest = sha(path)
    if wanted is not None:
        assert digest == wanted,str(path)
    if size is not None:
        assert path.stat().st_size == size,str(path)
    if path.suffix == '.py':
        ast.parse(path.read_text('utf8'),filename=str(path))
    elif path.suffix == '.json':
        read(path)
    records.append({'path':path.relative_to(R).as_posix(),'bytes':path.stat().st_size,'sha256':digest})
    return digest

review_hash = bind(S/'ROOT_IMPLEMENTATION_REVIEW.json',
 'e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5')
frozen = read(S/'ROOT_IMPLEMENTATION_REVIEW.json')['files']
assert len(frozen) == 112 and len({row['name'] for row in frozen}) == 112
for row in frozen:
    bind(S/row['name'],row['sha256'],row['bytes'])

post = read(P/'DELIVERY_RECEIPT.json')
bind(P/'PREPARATION_RECEIPT.json',post['PREPARATION_RECEIPT_SHA256'])
bind(B/'CONTROLLER_BINDING.json',post['CONTROLLER_BINDING_SHA256'])
package = read(B/'CONTROLLER_BINDING.json')
assert package['implementation_review_SHA256'] == review_hash and package['frozen_file_count'] == 112
for row in package['files']:
    bind(B/row['name'],row['sha256'],row['bytes'])
for name,digest in post['source_SHA256'].items():
    assert sha(B/name) == digest
for row in read(P/'SOURCE_BINDING.json')['sources']:
    bind(Path(row['path']),row['sha256'],row['bytes'])
tests = read(P/'synthetic_gate_checks/attempt_001/RESULT.json')
bind(P/'synthetic_gate_checks/attempt_001/RESULT.json',post['synthetic_RESULT_SHA256'])
bind(P/'SYNTHETIC_ATTEMPT_001.NATIVE_EXIT_RECEIPT.json',post['synthetic_native_receipt_SHA256'])
bind(P/'synthetic_gate_checks/attempt_001/RESOURCE_RECEIPT.json',post['synthetic_resource_receipt_SHA256'])
assert tests['status'] == 'PASS' and tests['passed'] == tests['total'] == len(tests['checks']) == 23
assert all(row['pass'] is True for row in tests['checks'])
assert tests['subprocess_Popen_calls'] == tests['SSH_connections'] == tests['science_launches'] == 0
for name,digest in tests['production_controller_sources'].items():
    assert sha(B/name) == digest
assert read(P/'SYNTHETIC_ATTEMPT_001.NATIVE_EXIT_RECEIPT.json')['native_exit_code'] == 0
assert read(P/'VERIFY.NATIVE_EXIT_RECEIPT.json')['native_exit_code'] == 0
assert read(P/'DELIVERY.NATIVE_EXIT_RECEIPT.json')['native_exit_code'] == 0
resource = read(P/'synthetic_gate_checks/attempt_001/RESOURCE_RECEIPT.json')
assert resource['CPU_affinity_count'] == 1 and resource['peak_RSS_bytes'] < 1024**3
assert resource['process_commit_limit_bytes'] == resource['RSS_watchdog_limit_bytes'] == 1024**3

recovery = read(Q/'PREPARATION_RECEIPT.json')
for row in recovery['key_artifacts']:
    bind(Q/row['name'],row['sha256'],row['bytes'])
for row in read(Q/'SOURCE_BINDINGS.json')['inputs']:
    bind(Path(row['path']),row['sha256'],row['bytes'])
selfcheck = read(Q/'selfcheck_saved_records/attempt_001/SELF_CHECK.json')
assert selfcheck['pass'] is True and selfcheck['case_count'] == len(selfcheck['cases']) == 28
assert selfcheck['positive_cases'] == 8 and selfcheck['negative_cases'] == 20
assert all(row['pass'] is True for row in selfcheck['cases'])
assert all(selfcheck['static_checks'].values()) and not selfcheck['guard_violations']
assert read(Q/'SELFCHECK_NATIVE_attempt_001.json')['native_exit'] == 0
assert selfcheck['actual_native_recovery_validated'] is False
assert selfcheck['checkpoint_file_reads'] == selfcheck['scientific_array_reads'] == selfcheck['native_launches'] == 0
assert selfcheck['resources']['CPU_affinity_count'] == 1 and selfcheck['memory']['peak_RSS_bytes'] < 1024**3
for p in (P/'SOURCE_BINDING.json',P/'AST_RECEIPT.json',P/'DELIVERY_RECEIPT.json',
 P/'VERIFY.NATIVE_EXIT_RECEIPT.json',P/'REOPENED_VERIFICATION.json',
 Q/'PREPARATION_RECEIPT.json',Q/'DELIVERY_BINDING.json',Path(__file__)):
    bind(p)
unique = {record['path']:record for record in records}
result = {'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),
 'status':'ROOT_REVIEWED_OFFLINE_CONTROL_AND_RECOVERY_PREPARATION',
 'frozen_R65_file_count':112,'frozen_implementation_review_SHA256':review_hash,
 'post_controller_package_root':str(B),'post_controller_offline_checks_passed':23,
 'post_controller_offline_native_exit':0,'post_controller_peak_RSS_bytes':resource['peak_RSS_bytes'],
 'recovery_saved_record_cases_passed':28,'recovery_native_exit':0,
 'recovery_peak_RSS_bytes':selfcheck['memory']['peak_RSS_bytes'],
 'root_review':{
   'post_native_nonce_PID_and_prior_process_death_required':True,
   'post_final50000_and_complete_saved_artifact_metadata_required':True,
   'unchanged_owned_numerical_audit_then_generation_evaluation_replay_report':True,
   'negative_A_C_is_valid_completed_report_not_goal_completion':True,
   'exclusive_launch_intents_and_unknown_transport_no_automatic_retry':True,
   'no_train_recovery_or_D_E_F_entry_in_post_launcher':True,
   'actual_train_exit_not_yet_observed_by_this_review':True,
   'recovery_requires_current_PID_death_native_evidence_and_closed_archive':True,
   'recovery_requires_existing_published_checkpoint_digest_no_tmp_promotion':True,
   'recovery_preserves_creator_vs_previous_attempt_and_unknown_physical_work':True,
   'stale_worker_receipt_clearance_required_only_after_actual_archive':True,
   'before_first500_no_supported_same_run_checkpoint':True},
 'operational_read_only_poll_ready':True,'watcher_only_after_preparation_agents_idle':True,
 'scientific_entry_may_launch_now':False,'actual_native_recovery_tested':False,
 'scientific_implementation_changed':False,'original_A_F_modified':False,
 'model_goal_complete':False,'new_scientific_jobs':0,'scientific_arrays_decoded':0,
 'CPU_affinity_mask':8,'RSS_cap_bytes':1024**3,'files':list(unique.values())}
destination = W/'ROOT_OPERATIONAL_REVIEW.json'
assert not destination.exists()
destination.write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
assert read(destination) == result
print(json.dumps({'review':str(destination),'bound_files':len(unique),'post_cases':23,'recovery_cases':28,
 'read_only_poll_ready':True,'post_science_launch_now':False,'model_goal_complete':False},indent=2))
