from __future__ import annotations
import ast, datetime, hashlib, json, os, pathlib, shutil
root = pathlib.Path(__file__).resolve().parent
failed = root / 'failed_001'
failed.mkdir(exist_ok=False)
names = ['numeric_worker.py.stdout.log', 'numeric_worker.py.stderr.log',
         'LOCAL_SCIENCE_TOKEN.json', 'ACTIVE_JOB.json', 'RUN_HISTORY.json',
         'LOCAL_NATIVE_RECEIPT.json', 'READINESS.json', 'numeric_worker.py',
         'local_supervise.py', 'config.json', 'PROTOCOL.md']
manifest = []
for name in names:
    p = root / name
    b = p.read_bytes()
    shutil.copyfile(p, failed / name)
    manifest.append({'name': name, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
for p in root.glob('FAILURE_*.json'):
    b = p.read_bytes()
    shutil.copyfile(p, failed / p.name)
    manifest.append({'name': p.name, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
prior = json.loads((failed / 'RUN_HISTORY.json').read_text('utf8'))
assert len(prior) == 1 and prior[0]['exit_code'] == 1
worker = (root / 'numeric_worker.py').read_text('utf8')
old = "tables=frozen['tables'];lambdas=np.asarray(frozen['lambdas'])"
new = "tables=frozen['bins'];assert [r['bin_id_1based'] for r in tables]==list(range(1,9));lambdas=np.asarray([r['lambda_value'] for r in tables],dtype='f8')"
assert worker.count(old) == 1
worker = worker.replace(old, new)
ast.parse(worker)
(root / 'numeric_worker.py').write_text(worker, 'utf8')
supervisor = (root / 'local_supervise.py').read_text('utf8')
assert supervisor.count('history=[]') == 1
supervisor = supervisor.replace('history=[]', "history=json.loads((ROOT/'failed_001/RUN_HISTORY.json').read_text('utf8'))")
ast.parse(supervisor)
(root / 'local_supervise.py').write_text(supervisor, 'utf8')
# The native receipt proved this specific child is terminal. Its one-use token
# remains in failed_001; remove only the now-closed top-level admission token.
(root / 'LOCAL_SCIENCE_TOKEN.json').unlink()
receipt = {'PID': os.getpid(), 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'failure_preserved': manifest, 'failure_numeric_seconds': prior[0]['elapsed_s'],
           'G_updates_in_failed_run': 0, 'D_updates_in_failed_run': 0,
           'repair': 'Read unchanged bins/lambda_value schema; carry failed native wall into cumulative budget.',
           'frozen_numbers_or_scientific_recipe_changed': False,
           'remote_job_or_SSH': False}
(root / 'ENGINEERING_REPAIR_001.json').write_text(json.dumps(receipt, indent=2) + '\n', 'utf8')
print(json.dumps({'PID': os.getpid(), 'UTC': receipt['UTC'], 'repair': receipt['repair'], 'failure_numeric_seconds': receipt['failure_numeric_seconds']}))
