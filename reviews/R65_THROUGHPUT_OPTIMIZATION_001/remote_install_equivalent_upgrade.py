"""Install a root-reviewed equivalent-source revision after verified native death/archive."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tarfile

O = Path(__file__).resolve().parent
S = Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC')
nonce = '37c43e76bb064c44937ebe888d58d0da'
os.sched_setaffinity(0,{11})

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

assert not (O/'INSTALL_RECEIPT.json').exists()
assert not any((Path('/proc')/str(pid)).exists() for pid in [291399,291394])
stop = read(O/'STOP_AND_ARCHIVE_RECEIPT.json')
assert stop['status'] == 'old_native_attempt_stopped_and_preserved' and stop['native_exit'] == -15
dest = O/'upgrade_payload'
dest.mkdir(exist_ok=False)
with tarfile.open(O/'UPGRADE_BUNDLE.tar') as tar:
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile() for m in tar.getmembers())
    tar.extractall(dest,filter='data')
manifest = read(dest/'UPGRADE_INSTALL_MANIFEST.json')
assert manifest['root_reviewed'] and manifest['old_attempt_id'] == nonce
assert manifest['previous_review_SHA256'] == sha(S/'ROOT_IMPLEMENTATION_REVIEW.json')
for row in manifest['files']:
    assert sha(dest/row['name']) == row['sha256'],row['name']
request = read(dest/'RECOVERY_REQUEST.json')
assert request['checkpoint_SHA256'] == manifest['checkpoint_SHA256'] == sha(S/'cache/RECOVERY_SAME_RUN.pt')
publication = S/request['checkpoint_publication_receipt_path']
assert sha(publication) == request['checkpoint_publication_receipt_SHA256']
old_review = read(S/'ROOT_IMPLEMENTATION_REVIEW.json')
for row in old_review['files']:
    assert sha(S/row['name']) == row['sha256'],row['name']
for row in manifest['source_changes']:
    assert sha(S/row['name']) == row['before_SHA256']
    assert sha(S/row['before_archive_relative_path']) == row['before_SHA256']
archive = S/'native_attempts'/nonce/'stage'
move_names = ['train_final.py.WORKER_RECEIPT.json','train_final.py.SCIENCE_PID.json',
 'train_final.py.RSS_ACK.json','train_final.py.NATIVE_RECEIPT.json','train_final.py.START_RESOURCE.json',
 'train_final.py.stdout.log','train_final.py.stderr.log','REMOTE_ACTIVE_JOB.json','PHYSICAL_PROGRESS.json',
 'TRAINING_FAILURE.json','TRAIN_PROGRESS.json','REPLAY_MISMATCH.json','RECOVERY_REQUEST.json',
 'RECOVERY_CHECKPOINT.json.tmp','RECOVERY_CHECKPOINT_PENDING.json.tmp','cache/RECOVERY_SAME_RUN.tmp.pt']
removed = S/'native_attempts'/nonce/'removed_live_names'
removed.mkdir()
for name in move_names:
    source = S/name
    if source.exists():
        assert source.resolve().is_relative_to(S) and not source.is_symlink()
        assert sha(source) == sha(archive/name),name
        target = removed/name
        assert target.resolve().is_relative_to(removed.resolve()) and not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
for row in manifest['files']:
    name = row['name']
    source = dest/name
    target = S/name
    assert target.resolve().is_relative_to(S)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_name(target.name+'.upgrade_tmp')
    assert not temp.exists()
    shutil.copyfile(source,temp)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temp,target)
    assert sha(target) == row['sha256']
current_review = read(S/'ROOT_IMPLEMENTATION_REVIEW.json')
assert sha(S/'ROOT_IMPLEMENTATION_REVIEW.json') == manifest['new_review_SHA256']
assert len(current_review['files']) == 115
for row in current_review['files']:
    assert sha(S/row['name']) == row['sha256'],row['name']
assert sha(S/'cache/RECOVERY_SAME_RUN.pt') == request['checkpoint_SHA256']
assert sha(publication) == request['checkpoint_publication_receipt_SHA256']
assert (S/'train_final.py.used').is_file() and sha(S/'train_final.py.used') == sha(archive/'train_final.py.used')
(S/'train_final.py.used').rename(removed/'train_final.py.used')
fd=os.open(S,os.O_RDONLY|os.O_DIRECTORY)
os.fsync(fd)
os.close(fd)
record={'UTC':datetime.now(timezone.utc).isoformat(),'status':'equivalent_upgrade_installed_ready_for_same_run_resume',
 'old_worker_and_supervisor_absent':True,'source_changes':manifest['source_changes'],
 'new_review_SHA256':manifest['new_review_SHA256'],'new_frozen_file_count':115,
 'checkpoint_update':request['committed_update'],'checkpoint_SHA256':request['checkpoint_SHA256'],
 'recovery_request_SHA256':sha(S/'RECOVERY_REQUEST.json'),'checkpoint_and_publication_bytes_unchanged':True,
 'old_active_names_preserved':True,'new_training_started':False}
with (O/'INSTALL_RECEIPT.json').open('x') as f:
    json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record),flush=True)
