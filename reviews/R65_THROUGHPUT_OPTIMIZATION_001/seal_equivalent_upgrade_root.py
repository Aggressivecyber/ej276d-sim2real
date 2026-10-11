"""Seal reviewed current-source bindings, preserving prior source and checkpoint lineage."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import ctypes
import hashlib
import json
import os
import shutil
import tarfile

P = Path(__file__).resolve().parent
R = P.parents[1]
S = R/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
D = P/'prepared_upgrade'
B = P/'local_before_upgrade'
k = ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
assert not B.exists()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')

prep = json.loads((D/'PREPARATION_RECEIPT.json').read_text('utf8'))
request_path = P/'recovery_actual/reviewed_request/RECOVERY_REQUEST.DRAFT.json'
request = json.loads(request_path.read_text('utf8'))
validation = json.loads(request_path.with_name('EVIDENCE_VALIDATION.json').read_text('utf8'))
stop = json.loads((P/'recovery_actual/STOP_AND_ARCHIVE_RECEIPT.json').read_text('utf8'))
assert validation['evidence_kind'] == 'root_saved_evidence'
assert request['committed_update'] == stop['checkpoint_update'] == 500
assert stop['native_exit'] == -15 and stop['original_worker_PID_absent'] and stop['original_supervisor_PID_absent']
assert request['checkpoint_SHA256'] == stop['checkpoint_SHA256'] == '63d958bfffbb75b57362005361f4adf3ef828de201bd6b2ebd5933df34fa3b5e'
assert request['preserved_physical_attempt_compute'][0]['last_completed_update'] == 787
old_review_path = S/'ROOT_IMPLEMENTATION_REVIEW.json'
old_review_sha = sha(old_review_path)
assert old_review_sha == request['frozen_binding']['implementation_review_SHA']
old = json.loads(old_review_path.read_text('utf8'))
for row in old['files']:
    assert sha(S/row['name']) == row['sha256'],row['name']
names = ['train_final.py','network_update_check/network_update_check.py','network_update_check/SOURCE_BINDING.json']
assert {row['name'] for row in prep['changed_source_files']} == set(names)
for row in prep['changed_source_files']:
    assert sha(S/row['name']) == row['before_SHA256']
    assert sha(D/row['name']) == row['after_SHA256']
    if row['name'].endswith('.py'):
        ast.parse((D/row['name']).read_text('utf8'))
B.mkdir()
for name in names+['ROOT_IMPLEMENTATION_REVIEW.json','ARTIFICIAL_SUITE_RECEIPT.json']:
    target = B/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(S/name,target)
    assert sha(target) == sha(S/name)
changes = []
for row in prep['changed_source_files']:
    changes.append(dict(row,before_archive_relative_path='native_attempts/'+request['previous_native_attempt_id']+'/stage/'+row['name']))
evidence = {}
for kind,origin in [('CPU',P/'agent_candidate/CPU_ARTIFICIAL_RECEIPT.json'),('CUDA',P/'attempt_001/CUDA_ARTIFICIAL_RECEIPT.json')]:
    target = D/'implementation_upgrade'/(kind+'_EQUIVALENCE.json')
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(origin,target)
    assert json.loads(target.read_text('utf8'))['status'] == 'passed'
    evidence[kind] = {'path':target.relative_to(D).as_posix(),'SHA256':sha(target)}
upgrade = {'schema_version':1,'kind':'equivalent_gradient_pack_only','UTC':datetime.now(timezone.utc).isoformat(),
           'owner':'/root','logical_run_id':request['logical_run_id'],'scientific_parameters_changed':False,
           'prior_frozen_binding':request['frozen_binding'],'checkpoint_SHA256':request['checkpoint_SHA256'],
           'checkpoint_committed_update':request['committed_update'],'checkpoint_bytes_modified':False,
           'changed_source_files':changes,'equivalence_evidence':evidence,
           'original_scientific_training_loop_AST_unchanged':True,'CPU_float64_chunk_add_order_unchanged':True,
           'original_source_population_random_streams_and_50000_endpoint_unchanged':True,
           'old_native_attempt_archive':'native_attempts/'+request['previous_native_attempt_id'],
           'intentional_interruption_native_exit':-15,'previous_logical_completed_update_lower_bound':787,
           'saved_evidence_bundle_SHA256':stop['evidence_bundle_SHA256'],
           'recovery_request_SHA256':sha(request_path),
           'root_reviewed_integration_report_SHA256':sha(P/'agent_candidate/INTEGRATION_REVIEW.md'),
           'new_artificial_suite_required_before_old_checkpoint_deserialization':True}
write(D/'IMPLEMENTATION_UPGRADE.json',upgrade)
new_names = names+['IMPLEMENTATION_UPGRADE.json',evidence['CPU']['path'],evidence['CUDA']['path']]
rows = []
for row in old['files']:
    p = D/row['name'] if row['name'] in names else S/row['name']
    rows.append({'name':row['name'],'sha256':sha(p),'bytes':p.stat().st_size})
for name in new_names[3:]:
    p = D/name
    rows.append({'name':name,'sha256':sha(p),'bytes':p.stat().st_size})
assert len(rows) == len({r['name'] for r in rows}) == 115
new = {'UTC':datetime.now(timezone.utc).isoformat(),'root_PID':os.getpid(),'stage':old['stage'],
       'ready_for_launch':True,'original_A_F_changed':False,
       'meaning':'Resume the same run from its byte-preserved checkpoint with a verified equivalent gradient-pack implementation; original eight artificial checks must pass again. Not scientific acceptance.',
       'prior_implementation_review_SHA256':old_review_sha,
       'prior_implementation_review_archive':'native_attempts/'+request['previous_native_attempt_id']+'/stage/ROOT_IMPLEMENTATION_REVIEW.json',
       'implementation_upgrade':'IMPLEMENTATION_UPGRADE.json',
       'protocol_SHA256':old['protocol_SHA256'],'config_SHA256':old['config_SHA256'],
       'driver_SHA256':sha(D/'train_final.py'),'original_acceptance_SHA256':old['original_acceptance_SHA256'],
       'files':rows,'original_frozen_files_unchanged':109,'changed_source_files':names,'added_evidence_files':new_names[3:],
       'scientific_update_loop_AST_unchanged':True,'VJP_C_reverse_clip_and_Adam_callers_AST_unchanged':True,
       'candidate_CPU_and_CUDA_byte_checks_passed':True,'native_same_run_restore_yet_verified':False,
       'artificial_server_suite_still_required':True,'model_goal_complete':False,'limits':old['limits']}
write(D/'ROOT_IMPLEMENTATION_REVIEW.json',new)
new_names.append('ROOT_IMPLEMENTATION_REVIEW.json')
for name in new_names:
    target=S/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(D/name,target)
shutil.copyfile(request_path,D/'RECOVERY_REQUEST.json')
new_names.append('RECOVERY_REQUEST.json')
for row in rows:
    assert sha(S/row['name']) == row['sha256']
assert {name:sha(R/name) for name in old['original_acceptance_SHA256']} == old['original_acceptance_SHA256']
manifest = {'UTC':datetime.now(timezone.utc).isoformat(),'root_reviewed':True,'previous_review_SHA256':old_review_sha,
            'new_review_SHA256':sha(D/'ROOT_IMPLEMENTATION_REVIEW.json'),'files':[
                {'name':name,'sha256':sha(D/name),'bytes':(D/name).stat().st_size} for name in new_names],
            'old_attempt_id':request['previous_native_attempt_id'],'checkpoint_SHA256':request['checkpoint_SHA256'],
            'checkpoint_update':500,'source_changes':changes}
write(D/'UPGRADE_INSTALL_MANIFEST.json',manifest)
with tarfile.open(D/'UPGRADE_BUNDLE.tar','x') as tar:
    for name in new_names+['UPGRADE_INSTALL_MANIFEST.json']:
        tar.add(D/name,arcname=name)
write(P/'ROOT_UPGRADE_SEAL_RECEIPT.json',dict(manifest,local_source_upgrade_installed=True,
      remote_source_upgrade_installed=False,old_native_attempt_archived_and_stopped=True))
print(json.dumps({'local_upgrade_sealed':True,'frozen_file_count':115,'new_review_SHA256':manifest['new_review_SHA256'],
                  'checkpoint_update':500,'new_remote_training_launched':False}))
