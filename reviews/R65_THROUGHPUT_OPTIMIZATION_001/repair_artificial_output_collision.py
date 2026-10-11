"""Archive a proven pre-training startup failure and prepare the same checkpoint again."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import shutil

S=Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC')
O=Path(__file__).resolve().parent
old='37c43e76bb064c44937ebe888d58d0da'
failed='c9306986025c4474b87e5e800b300a43'
os.sched_setaffinity(0,{11})
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(n):return json.loads((S/n).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
assert not (O/'STARTUP_COLLISION_REPAIR_RECEIPT.json').exists()
assert not any((Path('/proc')/str(p)).exists() for p in [328790,328795])
native=read('train_final.py.NATIVE_RECEIPT.json')
hs=read('train_final.py.SCIENCE_PID.json')
failure=read('TRAINING_FAILURE.json')
assert hs['nonce']==failure['native_attempt_id']==failed and hs['PID']==native['actual_worker_PID']==328795
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==1
assert failure['phase']=='startup_before_artificial_checks' and failure['updates']==failure['current_update']==0
assert failure['error']=="FileExistsError(17, 'File exists')" and 'target.mkdir(exist_ok=False)' in failure['traceback']
assert all(v==0 for v in failure['known_completed_physical_operations_this_attempt'].values())
assert all(v==0 for v in failure['physical_schema_counter_lower_bound'].values())
assert not (S/'SAME_RUN_RESTORE_RECEIPT.json').exists()
current=read('ROOT_IMPLEMENTATION_REVIEW.json')
for row in current['files']:assert sha(S/row['name'])==row['sha256'],row['name']
request=read('RECOVERY_REQUEST.json')
assert request['previous_native_attempt_id']==old and len(request['preserved_physical_attempt_compute'])==1
assert not request['prior_recovery_history'] and request['committed_update']==500
assert sha(S/'cache/RECOVERY_SAME_RUN.pt')==request['checkpoint_SHA256']
assert sha(S/request['checkpoint_publication_receipt_path'])==request['checkpoint_publication_receipt_SHA256']
archive=S/'native_attempts'/failed/'stage'
assert not archive.exists()
archive.mkdir(parents=True)
records={}
for p in sorted(S.rglob('*')):
    if p.is_relative_to(S/'native_attempts') or not p.is_file():continue
    assert not p.is_symlink()
    rel=p.relative_to(S).as_posix();target=archive/rel
    assert target.resolve().is_relative_to(archive.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    h=sha(p);shutil.copyfile(p,target)
    assert sha(target)==h==sha(p) and target.stat().st_ino!=p.stat().st_ino
    records[rel]={'sha256':h,'bytes':p.stat().st_size,'archive_relative_path':target.relative_to(S).as_posix()}
write(S/'native_attempts'/failed/'ARCHIVE_INVENTORY.json',{'UTC':datetime.now(timezone.utc).isoformat(),
 'attempt_id':failed,'all_stage_files_except_prior_attempts_preserved':True,'files':records})
physical={'attempt_id':failed,'PID':328795,'resumed_from_update':500,'last_completed_update':500,
 'completed_operations':failure['known_completed_physical_operations_this_attempt'],
 'counter':failure['physical_schema_counter_lower_bound'],'counters_complete':True,'unknown_intervals':[],
 'known_uncommitted_optimizer_steps':0,'exact_completed_operation_counts':True,'unknown_after_crash_fields':[],
 'prior_native_attempts_separately_preserved':True,'evidence_source_kind':'native_startup_failure_before_checkpoint_deserialization',
 'evidence_SHA256':sha(S/'TRAINING_FAILURE.json'),'logical_baseline_from_incoming_request_before_restore':True,
 'restoration_completed':False,'scientific_TRAIN_decoded':False,
 'zero_science_reason':'Caught FileExistsError before run_all first check; source reaches no tensor load, model construction or training loop before this exception. Native failure and wrapper receipt both exit1.'}
history={'from_attempt_id':old,'to_attempt_id':failed,'checkpoint_creator_attempt_id':old,
 'checkpoint_update':500,'checkpoint_sha256':request['checkpoint_SHA256'],
 'checkpoint_publication_receipt_path':request['checkpoint_publication_receipt_path'],
 'checkpoint_publication_receipt_SHA256':request['checkpoint_publication_receipt_SHA256'],
 'bindings_verified':True,'previous_PID_confirmed_dead':True,'restoration_completed':False,
 'meaning':'Bound recovery requested and root verified original checkpoint bytes; this startup attempt failed before tensor deserialization.',
 'implementation_upgrade_declaration_SHA256':sha(S/'IMPLEMENTATION_UPGRADE.json')}
revised=copy.deepcopy(request)
revised['preserved_physical_attempt_compute'].append(physical)
revised['prior_recovery_history'].append(history)
revised['previous_native_attempt_id']=failed
assert len(revised['prior_recovery_history'])==len(revised['preserved_physical_attempt_compute'])-1
removed=S/'native_attempts'/failed/'removed_live_names'
removed.mkdir()
names=['train_final.py.WORKER_RECEIPT.json','train_final.py.SCIENCE_PID.json','train_final.py.RSS_ACK.json',
 'train_final.py.NATIVE_RECEIPT.json','train_final.py.START_RESOURCE.json','train_final.py.stdout.log',
 'train_final.py.stderr.log','REMOTE_ACTIVE_JOB.json','PHYSICAL_PROGRESS.json','TRAINING_FAILURE.json',
 'TRAIN_PROGRESS.json','REPLAY_MISMATCH.json','RECOVERY_REQUEST.json','ARTIFICIAL_SUITE_PROGRESS.json']
for name in names:
    p=S/name
    if p.exists():
        assert sha(p)==records[name]['sha256']
        q=removed/name;assert q.resolve().is_relative_to(removed.resolve()) and not q.exists()
        q.parent.mkdir(parents=True,exist_ok=True);p.rename(q)
# Preserve the original artificial outputs rather than weakening exclusive creation.
art=S/'artificial_results'
assert art.is_dir() and art.resolve().is_relative_to(S) and not art.is_symlink()
assert all(sha(p)==records[p.relative_to(S).as_posix()]['sha256'] for p in art.rglob('*') if p.is_file())
art.rename(removed/'artificial_results')
write(S/'RECOVERY_REQUEST.json',revised)
assert sha(S/'train_final.py.used')==records['train_final.py.used']['sha256']
(S/'train_final.py.used').rename(removed/'train_final.py.used')
for row in current['files']:assert sha(S/row['name'])==row['sha256']
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'status':'startup_artificial_directory_collision_repaired',
 'native_failed_attempt':failed,'actual_native_exit':1,'actual_worker_and_supervisor_absent':True,
 'scientific_updates_in_failed_attempt':0,'checkpoint_deserialized_in_failed_attempt':False,
 'checkpoint_and_publication_unchanged':True,'checkpoint_update':500,
 'new_recovery_request_SHA256':sha(S/'RECOVERY_REQUEST.json'),'archived_files':len(records),
 'current_frozen_115_files_unchanged':True,'new_training_started':False}
write(O/'STARTUP_COLLISION_REPAIR_RECEIPT.json',receipt)
print(json.dumps(receipt),flush=True)
