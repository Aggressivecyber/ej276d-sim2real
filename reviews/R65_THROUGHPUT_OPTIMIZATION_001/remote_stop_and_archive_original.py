"""Root-authorized one-time stop/archive for a tested equivalent implementation upgrade."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time

S = Path('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R65_ROOT_20261011_030107_UTC')
O = Path(__file__).resolve().parent
E = O/'saved_evidence'
NONCE = '37c43e76bb064c44937ebe888d58d0da'
RUN = 'R65_FULL_EMPIRICAL_D_RESPONSE_001@'+str(S)
ENTRY = 'train_final.py'
ARM = 'FULL_EMPIRICAL_RESPONSE'
os.sched_setaffinity(0, {11})

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def read(name):
    return json.loads((S/name).read_text())

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def proc(pid):
    p = Path('/proc')/str(pid)
    return {'PID':pid, 'cwd':str((p/'cwd').resolve()),
            'command':(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
            'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19])}

assert O != S and not O.is_relative_to(S)
assert not E.exists() and not (S/'native_attempts'/NONCE).exists()
assert not (S/'RECOVERY_REQUEST.json').exists(), 'Expected original first attempt'
cfg = read('config.json')
review = read('ROOT_IMPLEMENTATION_REVIEW.json')
assert sha(S/'ROOT_IMPLEMENTATION_REVIEW.json') == 'e0a4b101886f5cb10b205364fe8414ca2db1fa211cbb058de7d4113f85ba24f5'
for row in review['files']:
    assert sha(S/row['name']) == row['sha256'], row['name']
hs = read(ENTRY+'.SCIENCE_PID.json')
active = read('REMOTE_ACTIVE_JOB.json')
assert hs['nonce'] == NONCE and hs['PID'] == active['actual_worker_PID'] == 291399
assert hs['parent_PID'] == active['supervisor_PID'] == 291394
assert active.get('actual_worker_native_exit') is None and active['entry'] == ENTRY
before = proc(hs['PID'])
assert before['cwd'] == str(S) and 'linux_owned_worker.py train_final.py' in before['command']
assert read('RECOVERY_CHECKPOINT.json')['committed_update'] >= 500
assert not any(p.is_symlink() for p in S.rglob('*')), 'Refuse stage symlinks before archive'
write(O/'STOP_INTENT.json', {'UTC':utc(), 'owner':'/root', 'reason':'Apply byte-equivalent gradient-pack optimization after CPU and CUDA verification',
                           'worker':before, 'supervisor_PID':hs['parent_PID'], 'checkpoint_bytes_to_preserve':True,
                           'scientific_endpoint_updates':50000, 'scientific_parameters_changed':False})
os.kill(hs['PID'], signal.SIGTERM)
deadline = time.monotonic()+30
while (Path('/proc')/str(hs['PID'])).exists() or (Path('/proc')/str(hs['parent_PID'])).exists():
    assert time.monotonic() < deadline, 'Wait for actual worker and supervisor termination; do not resubmit'
    time.sleep(.1)
native = read(ENTRY+'.NATIVE_RECEIPT.json')
assert native['actual_worker_native_exit'] == -15 and native['actual_worker_PID'] == hs['PID']
assert native['supervisor_PID'] == hs['parent_PID']
assert 'worker_self_receipt' not in native, 'Unexpected stale worker receipt requires review'
owned = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        command = (p/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace')
        cwd = str((p/'cwd').resolve())
        if 'python' in command and (cwd == str(S) or str(S/'linux_owned_worker.py') in command):
            owned.append({'PID':int(p.name),'command':command,'cwd':cwd})
    except (OSError, PermissionError):
        pass
assert not owned, owned
E.mkdir()
prefix = 'attempt/'+NONCE+'/'
refs = {}
def put(role, name, *, value=None, original=None):
    p = E/name
    p.parent.mkdir(parents=True,exist_ok=True)
    if original is not None:
        p.write_bytes(original.read_bytes())
    else:
        write(p,value)
    refs[role] = {'path':name,'sha256':sha(p)}

for role, name in [('science_pid',ENTRY+'.SCIENCE_PID.json'),('rss_ack',ENTRY+'.RSS_ACK.json'),
                   ('native_receipt',ENTRY+'.NATIVE_RECEIPT.json'),('physical_snapshot','PHYSICAL_PROGRESS.json')]:
    put(prefix+role, role+'.json', original=S/name)
death = {'UTC':utc(),'logical_run_id':RUN,'attempt_id':NONCE,'actual_worker_PID':hs['PID'],
         'supervisor_PID':hs['parent_PID'],'same_host_and_process_namespace_verified':True,
         'saved_handshake_and_native_identity_verified':True,'actual_worker_PID_absent':True,
         'supervisor_PID_absent':True,'no_matching_attempt_or_owned_descendants_alive':True,
         'native_receipt_SHA256':sha(S/(ENTRY+'.NATIVE_RECEIPT.json')),
         'collector':'root remote_stop_and_archive_original.py /proc and supervisor wait receipt',
         'stop_reason':'Intentional engineering replacement after artificial equivalence verification; native -15 is retained.'}
put(prefix+'death_confirmation','death_confirmation.json',value=death)

archive = S/'native_attempts'/NONCE/'stage'
assert archive.resolve().is_relative_to((S/'native_attempts').resolve())
archive.mkdir(parents=True)
paths = sorted(p for p in S.rglob('*') if p.is_file() and not p.is_relative_to(S/'native_attempts'))
records = {}
for p in paths:
    assert not p.is_symlink() and p.resolve().is_relative_to(S)
    rel = p.relative_to(S).as_posix()
    target = archive/rel
    assert target.resolve().is_relative_to(archive.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    before_sha, size = sha(p), p.stat().st_size
    shutil.copyfile(p,target)
    assert sha(target) == before_sha == sha(p) and target.stat().st_size == size == p.stat().st_size
    assert target.stat().st_ino != p.stat().st_ino
    records[rel] = {'exists':True,'archive_relative_path':target.relative_to(S).as_posix(),
                    'source_sha256':before_sha,'archive_sha256':before_sha,
                    'source_bytes':size,'archive_bytes':size,'regular_file':True,'source_symlink':False}
finals = ['LOSS_BY_UPDATE_PRIVATE.npz','PRECISION_ORDER_BY_UPDATE_PRIVATE.npz','REPLAY_BY_UPDATE_PRIVATE.npz',
          'DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz','PARENT_EXPOSURES_PRIVATE.npz','FINAL_FLOW.pt',
          'FINAL_TORCH_RNG.pt','FINAL_RANDOM_STATES.json','TRAINING_RECEIPT.json']
mutable = [ENTRY+s for s in ['.used','.stdout.log','.stderr.log','.SCIENCE_PID.json','.RSS_ACK.json',
           '.WORKER_RECEIPT.json','.NATIVE_RECEIPT.json','.START_RESOURCE.json']]+[
           'REMOTE_ACTIVE_JOB.json','REMOTE_RUN_HISTORY.json','LAUNCH_RECEIPT.json','RECOVERY_REQUEST.json',
           'RECOVERY_CHECKPOINT.json','RECOVERY_CHECKPOINT_PENDING.json','cache/RECOVERY_SAME_RUN.pt',
           'cache/RECOVERY_SAME_RUN.tmp.pt','PHYSICAL_PROGRESS.json','TRAINING_FAILURE.json','REPLAY_MISMATCH.json',
           'TRAIN_STARTED.json','TRAIN_PROGRESS.json','ARTIFICIAL_SUITE_RECEIPT.json',ARM+'/ENVIRONMENT.json',
           ARM+'/INITIALIZATION.json',ARM+'/INITIAL_RANDOM_STATES.json',ARM+'/TRAINING_TRACE.json']+[
           ARM+'/'+n for n in finals]
for name in mutable:
    if name not in records:
        assert not (S/name).exists()
        records[name] = {'exists':False}
inventory = {'UTC':utc(),'logical_run_id':RUN,'attempt_id':NONCE,'archive_root':'native_attempts/'+NONCE,
             'all_stage_files_except_native_attempts_inventory_closed':True,
             'all_existing_files_copied_and_rehashed':True,'prior_attempt_archives_preserved':True,
             'independent_byte_copies_no_hardlinks_or_symlinks':True,'files':records,
             'existing_files':len(paths),'copied_bytes':sum(r['source_bytes'] for r in records.values() if r['exists'])}
put(prefix+'archive_inventory','archive_inventory.json',value=inventory)
publication = read('RECOVERY_CHECKPOINT.json')
binding = publication['frozen_binding']
for row in review['files']:
    assert sha(S/row['name']) == row['sha256']
inputs = read('INPUT_BINDING.json')['inputs']
for key,path in cfg['input_paths'].items():
    assert sha(Path(path)) == inputs[key]['sha256'], key
env_code = '''import os,sys,json,numpy,scipy,torch
os.sched_setaffinity(0,set(range(12)))
torch.set_num_threads(12);torch.set_num_interop_threads(12)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.mha.set_fastpath_enabled(False)
print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'torch':torch.__version__,
 'scipy':scipy.__version__,'device':'cuda:0','GPU':torch.cuda.get_device_name(0),
 'CPU_threads':torch.get_num_threads(),'CPU_affinity':sorted(os.sched_getaffinity(0)),
 'model_training_mode':True,'MHA_fastpath':torch.backends.mha.get_fastpath_enabled(),
 'TF32':torch.backends.cuda.matmul.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}))
'''
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(O),TMP=str(O),TEMP=str(O),
           CUDA_CACHE_PATH=str(O/'cuda_cache'),TORCH_HOME=str(O/'torch_cache'))
env_run = subprocess.run([cfg['server_python'],'-B','-c',env_code],env=env,text=True,capture_output=True,timeout=30)
assert env_run.returncode == 0, env_run.stderr
observed_environment = json.loads(env_run.stdout)
assert observed_environment == binding['environment'], (observed_environment,binding['environment'])
verification = {'UTC':utc(),'logical_run_id':RUN,'frozen_binding':binding,
                'trainer_SHA256':sha(S/'train_final.py'),'all_reviewed_source_hashes_verified':True,
                'all_scientific_input_hashes_verified':True,'environment_exactly_matches_bound_environment':True,
                'observed_environment':observed_environment,'no_scientific_parameters_changed':True,
                'verification_scope':'saved_root_observation_no_training_advance',
                'environment_native_exit':env_run.returncode}
put('frozen_verification','frozen_verification.json',value=verification)
put('artificial_suite','artificial_suite.json',original=S/'ARTIFICIAL_SUITE_RECEIPT.json')
for role,name in [('publication_committed','RECOVERY_CHECKPOINT.json'),('publication_pending','RECOVERY_CHECKPOINT_PENDING.json')]:
    if (S/name).exists():
        put(role,role+'.json',original=S/name)
cp = S/'cache/RECOVERY_SAME_RUN.pt'
observation = {'UTC':utc(),'logical_run_id':RUN,'path':'cache/RECOVERY_SAME_RUN.pt','exists':cp.exists(),
               'regular_file':cp.is_file(),'observed_after_all_prior_processes_dead':True,
               'hash_observed_without_checkpoint_deserialization':True,'no_tmp_promotion_performed':True,
               'symlink':cp.is_symlink(),'sha256':sha(cp),'bytes':cp.stat().st_size}
assert observation['sha256'] == publication['checkpoint_SHA256']
put('checkpoint_observation','checkpoint_observation.json',value=observation)
manifest = {'schema_version':1,'evidence_kind':'root_saved_evidence','logical_run_id':RUN,'frozen_binding':binding,
            'attempts':[{'attempt_id':NONCE,'resumed_from_update':0,'physical_source_kind':'PHYSICAL_PROGRESS'}],
            'references':refs}
write(E/'EVIDENCE_BUNDLE.json',manifest)
receipt = {'UTC':utc(),'status':'old_native_attempt_stopped_and_preserved','native_exit':native['actual_worker_native_exit'],
           'original_worker_PID_absent':True,'original_supervisor_PID_absent':True,
           'archived_files':len(paths),'archived_bytes':inventory['copied_bytes'],
           'checkpoint_update':publication['committed_update'],'checkpoint_SHA256':observation['sha256'],
           'last_completed_update_lower_bound':read('PHYSICAL_PROGRESS.json')['last_completed_update'],
           'new_training_launched':False,'frozen_files_unchanged':len(review['files']),
           'saved_evidence_path':str(E),'evidence_bundle_SHA256':sha(E/'EVIDENCE_BUNDLE.json')}
write(O/'STOP_AND_ARCHIVE_RECEIPT.json',receipt)
print(json.dumps(receipt),flush=True)
