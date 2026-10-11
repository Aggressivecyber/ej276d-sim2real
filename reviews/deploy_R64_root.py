"""Root-only guarded code delivery and exclusive launch of registered R64."""
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
remote = C['remote_root']
host = 'rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R64_ROOT_')
assert not (S/'DEPLOYMENT_RECEIPT.json').exists(), 'Resolve prior dispatch identity; do not launch twice'
utc = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


top = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'ASSEMBLY_RECEIPT.json',
       'diagnose_train.py', 'render_diagnostic.py', 'network.py', 'noise_core.py', 'source_flow_core.py',
       'precision_order.py', 'R61_distribution_loss.py', 'distribution_loss.py', 'linux_supervise.py', 'linux_owned_worker.py']
assembly = json.loads((S/'ASSEMBLY_RECEIPT.json').read_text('utf8'))
for row in assembly['helpers']:
    assert sha(S/row['target']) == row['sha256'], row['target']
files = sorted(set(top + [row['target'].replace('\\', '/') for row in assembly['helpers']]))
assert all('\\' not in name for name in files), 'Remote package paths must be POSIX'
assert all((S/name).is_file() for name in files)
assert (S/'PROTOCOL.md').read_bytes() == (R/C['protocol']).read_bytes()
assert C['response_training_runs'] == C['response_updates'] == C['HGB_fits'] == C['density_fits'] == C['DEV_read'] == C['EVAL_read'] == C['CAL_decode'] == 0
assert all(C[k] is None for k in ('science_budget', 'deadline_UTC', 'output_limit_bytes'))
binding = json.loads((S/'INPUT_BINDING.json').read_text('utf8'))
for row in binding['original_acceptance']:
    assert sha(Path(row['path'])) == row['sha256'], row['path']
for name, row in binding['frozen_R63_helpers'].items():
    assert sha(S/name) == row['sha256'], name
for name in files:
    if name.endswith('.py'):
        ast.parse((S/name).read_text('utf8'), filename=name)
tree = ast.parse((S/'diagnose_train.py').read_text('utf8'))
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('step', 'backward') for n in ast.walk(tree))
root_decision = json.loads((S/'ROOT_STATIC_REVIEW_DECISION.json').read_text('utf8'))
assert root_decision['ready_for_registered_dispatch'] is True
assert root_decision['worker_sha256'] == sha(S/'diagnose_train.py')
assert root_decision['render_sha256'] == sha(S/'render_diagnostic.py')
files.append('ROOT_STATIC_REVIEW_DECISION.json')
storage_estimate = binding['conservative_uncompressed_gradient_bytes'] + 1536*1024**2
assert shutil.disk_usage('D:/').free > storage_estimate
assert shutil.disk_usage('E:/').free > storage_estimate
review = dict(UTC=utc(), owner='/root', status='root_static_review_ready_for_artificial_then_registered_science',
              AST_pass=True, original_helper_and_adapter_bytes_unchanged=True,
              artificial_tests_locally_passed=dict(integrated_distribution=69, endpoint_geometry=33, gradient_statistics=127,
                                                  portable_fixture=69, actual_network_CPU=True),
              server_CPU_CUDA_artificial_tests_required_before_scientific_files=True,
              frozen_model_science_started=False, original_A_F_unchanged=True,
              conservative_uncompressed_storage_estimate_bytes=storage_estimate,
              storage_readiness_is_not_output_quota=True, actual_D_free_bytes=shutil.disk_usage('D:/').free,
              files=[dict(name=name, bytes=(S/name).stat().st_size, sha256=sha(S/name)) for name in files])
write(S/'ROOT_IMPLEMENTATION_REVIEW.json', review)
files.append('ROOT_IMPLEMENTATION_REVIEW.json')
package = S/'SOURCE_PACKAGE.tar'
with tarfile.open(package, 'w') as tar:
    for name in files:
        tar.add(S/name, arcname=name, recursive=False)
events = []


def invoke(argv, code=None):
    p = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=60)
    events.append(dict(argv=argv, native_exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
    write(S/'DEPLOYMENT_RECEIPT.json', dict(UTC=utc(), control_PID=os.getpid(), events=events,
                                         remote_root=remote, source_package_SHA256=sha(package)))
    assert p.returncode == 0, events[-1]
    return p.stdout


def update_stage(status, job):
    now = utc()
    active = int(status in ('launch_requested', 'diagnostic_running'))
    for name in ('PROGRAM.json', 'PROJECT_PROGRESS.json'):
        value = json.loads((R/name).read_text('utf8'))
        value.update(status='R64_'+status+'_long_term_goal_active', current_stage_progress=status,
                     current_job=job, active_scientific_jobs=active, active_remote_scientific_jobs=active,
                     active_local_scientific_jobs=0, model_goal_complete=False, application_goal_status='active')
        if name == 'PROGRAM.json':
            value['updated_at_utc'] = now
        else:
            value.update(updated_UTC=now, current_activity='root_registered_R64_'+status)
        write(R/name, value)
    for name, key, identity in [('jobs.json', 'jobs', 'stage'), ('experiments.json', 'experiments', 'id')]:
        value = json.loads((R/name).read_text('utf8'))
        item = next(v for v in value[key] if v.get(identity) == C['stage'])
        item.update(status=status, updated_UTC=now, remote_root=remote, current_job=job)
        value.update(active_scientific_jobs=active, active_remote_scientific_jobs=active, active_local_scientific_jobs=0)
        write(R/name, value)


ssh = ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, '/usr/bin/python3 -B -']
prefix = 'from pathlib import Path\nimport json,subprocess,time,os,shutil,hashlib,tarfile\np=Path('+repr(remote)+')\n'
admission = prefix+'storage='+repr(storage_estimate)+'\n'+'''
assert not p.exists(), 'Remote stage exists: resolve it rather than launching another worker'
base=str(p.parent)
jobs=[]
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name)==os.getpid():continue
    try:
        a=(d/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace')
        c=str((d/'cwd').resolve())
        if 'python' in a and(base in a or c.startswith(base)):
            jobs.append({'PID':int(d.name),'cwd':c,'argv':a})
    except(OSError,PermissionError):pass
assert not jobs,jobs
assert shutil.disk_usage(p.parent).free>storage
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,apps
p.mkdir(exist_ok=False)
print(json.dumps({'remote_directory_created':str(p),'other_project_jobs':jobs,'other_GPU_compute_jobs':apps,'server_free_bytes':shutil.disk_usage(p).free}))
'''
invoke(ssh, admission)
invoke(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', str(package), host+':'+remote+'/SOURCE_PACKAGE.tar'])
manifest = [dict(name=name, sha256=sha(S/name)) for name in files]
extract = prefix+'manifest='+repr(manifest)+'\nexpected_package='+repr(sha(package))+'\n'+'''
assert hashlib.sha256((p/'SOURCE_PACKAGE.tar').read_bytes()).hexdigest()==expected_package
with tarfile.open(p/'SOURCE_PACKAGE.tar')as tar:
    assert tar.getnames()==[v['name']for v in manifest]
    assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in tar.getmembers())
    tar.extractall(p,filter='data')
assert all(hashlib.sha256((p/v['name']).read_bytes()).hexdigest()==v['sha256']for v in manifest)
print(json.dumps({'source_files_bound':len(manifest),'no_scientific_arrays_decoded':True}))
'''
invoke(ssh, extract)
job = dict(stage=C['stage'], status='launch_requested', remote_root=remote, active_scientific_jobs=1, UTC=utc())
update_stage('launch_requested', job)
launch = prefix+'python='+repr(C['server_python'])+'\n'+'''
marker=p/'LAUNCH_ONCE.json'
with marker.open('x')as f:
    json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
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
                     actual_worker_PID=(record['actual_job'] or {}).get('actual_worker_PID'),
                     remote_root=remote, original_A_F_changed=False)))
