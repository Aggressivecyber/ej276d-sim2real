"""Resolve the verified prelaunch Windows/POSIX manifest failure; one launch."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
C = json.loads((S/'config.json').read_text('utf8'))
remote, host = C['remote_root'], 'rylan@100.82.231.102'
Q = R/'reviews/R64_DEPLOYMENT_REPAIR_001'
Q.mkdir(exist_ok=False)
utc = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


for name in ['SOURCE_PACKAGE.tar', 'ROOT_IMPLEMENTATION_REVIEW.json', 'DEPLOYMENT_RECEIPT.json']:
    shutil.copy2(S/name, Q/name)
old = json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))
old_package_sha = sha(S/'SOURCE_PACKAGE.tar')
for record in old['files']:
    assert sha(S/record['name']) == record['sha256'], record['name']
    record['name'] = record['name'].replace('\\', '/')
assert all('\\' not in row['name'] for row in old['files'])
old.update(UTC=utc(), prelaunch_manifest_portability_repair='Normalize local Windows relative paths to POSIX package/remote paths; scientific worker/source/config unchanged')
write(S/'ROOT_IMPLEMENTATION_REVIEW.json', old)
files = [row['name'] for row in old['files']] + ['ROOT_IMPLEMENTATION_REVIEW.json']
with tarfile.open(S/'SOURCE_PACKAGE.tar', 'w') as tar:
    for name in files:
        tar.add(S/name, arcname=name, recursive=False)
with tarfile.open(S/'SOURCE_PACKAGE.tar') as tar:
    assert tar.getnames() == files
manifest = [dict(name=name, sha256=sha(S/name)) for name in files]
events = []


def invoke(argv, code=None):
    p = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=60)
    events.append(dict(argv=argv, native_exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
    write(Q/'RECEIPT.json', dict(UTC=utc(), repair='Windows/POSIX relative manifest representation',
                              scientific_source_config_changed=False, original_deployment_failure_preserved=True,
                              original_package_SHA256=old_package_sha, package_SHA256=sha(S/'SOURCE_PACKAGE.tar'), events=events))
    assert p.returncode == 0, events[-1]
    return p.stdout


# Reuse the root-owned metadata updater definition only, never execute the old dispatcher.
tree = ast.parse((R/'reviews/deploy_R64_root.py').read_text('utf8'))
updater = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'update_stage')
exec(compile(ast.Module(body=[updater], type_ignores=[]), 'bound_update_stage', 'exec'))
ssh = ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, '/usr/bin/python3 -B -']
prefix = 'from pathlib import Path\nimport json,subprocess,time,os,hashlib,tarfile\np=Path('+repr(remote)+')\n'
resolve = prefix+'expected_old='+repr(old_package_sha)+'\n'+'''
assert p.is_dir()
assert not(p/'LAUNCH_ONCE.json').exists()and not(p/'diagnose_train.py.used').exists()
assert not(p/'REMOTE_ACTIVE_JOB.json').exists()
assert [v.name for v in p.iterdir()]==['SOURCE_PACKAGE.tar']
assert hashlib.sha256((p/'SOURCE_PACKAGE.tar').read_bytes()).hexdigest()==expected_old
(p/'SOURCE_PACKAGE.tar').rename(p/'SOURCE_PACKAGE__PRELAUNCH_WINDOWS_PATH_FAILURE.tar')
print(json.dumps({'only_code_package_was_present':True,'launch_marker_existed':False,'old_package_preserved':True,'scientific_job_started':False}))
'''
invoke(ssh, resolve)
invoke(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', str(S/'SOURCE_PACKAGE.tar'), host+':'+remote+'/SOURCE_PACKAGE.tar'])
extract = prefix+'manifest='+repr(manifest)+'\nexpected='+repr(sha(S/'SOURCE_PACKAGE.tar'))+'\n'+'''
assert hashlib.sha256((p/'SOURCE_PACKAGE.tar').read_bytes()).hexdigest()==expected
with tarfile.open(p/'SOURCE_PACKAGE.tar')as tar:
    assert tar.getnames()==[v['name']for v in manifest]
    assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in tar.getmembers())
    tar.extractall(p,filter='data')
assert all(hashlib.sha256((p/v['name']).read_bytes()).hexdigest()==v['sha256']for v in manifest)
print(json.dumps({'files_verified':len(manifest),'all_relative_paths_POSIX':True,'no_scientific_arrays_decoded':True}))
'''
invoke(ssh, extract)
job = dict(stage=C['stage'], status='launch_requested', remote_root=remote, active_scientific_jobs=1, UTC=utc())
update_stage('launch_requested', job)
launch = prefix+'python='+repr(C['server_python'])+'\n'+'''
marker=p/'LAUNCH_ONCE.json'
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'diagnose_train.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(2)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())if(p/'REMOTE_ACTIVE_JOB.json').exists()else None
record['immediate_exit']=child.poll()
print(json.dumps(record))
'''
record = json.loads(invoke(ssh, launch))
job.update(status='diagnostic_running' if record['immediate_exit'] is None else 'launch_returned_check_native_receipts',
           launch=record, active_scientific_jobs=int(record['immediate_exit'] is None), UTC=utc())
update_stage(job['status'], job)
write(S/'LAUNCH_RECEIPT.json', job)
print(json.dumps(dict(status=job['status'], supervisor_PID=record['supervisor_PID'],
                     actual_worker_PID=(record['actual_job'] or {}).get('actual_worker_PID'), remote_root=remote)))
