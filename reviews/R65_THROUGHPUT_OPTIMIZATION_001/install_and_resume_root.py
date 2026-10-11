"""Install the reviewed payload, then start exactly one same-run resume supervisor."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import json
import subprocess
import sys

P = Path(__file__).resolve().parent
S = P.parents[1]/'work/R65_FULL_EMPIRICAL_D_RESPONSE_001'
remote_control=json.loads((P/'REMOTE_CONTROL_LOCATION.json').read_text('utf8'))['remote_directory']
cfg=json.loads((S/'config.json').read_text('utf8'))
remote=cfg['remote_root']
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),8)
assert not (P/'RESUME_LAUNCH_RECEIPT.json').exists()
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
     'rylan@100.82.231.102','/usr/bin/python3 -B -']
if '--launch-only' not in sys.argv:
    for source in [P/'remote_install_equivalent_upgrade.py',P/'prepared_upgrade/UPGRADE_BUNDLE.tar']:
        run=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(source),
                            'rylan@100.82.231.102:'+remote_control+'/'+source.name],text=True,capture_output=True,timeout=30)
        assert run.returncode==0,run.stderr
    code='import runpy\nrunpy.run_path('+repr(remote_control+'/remote_install_equivalent_upgrade.py')+',run_name="__main__")\n'
    installed=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=60)
    (P/'INSTALL_NATIVE.stdout.log').write_text(installed.stdout,encoding='utf8')
    (P/'INSTALL_NATIVE.stderr.log').write_text(installed.stderr,encoding='utf8')
    assert installed.returncode==0,(installed.stdout,installed.stderr)
    installation=json.loads(installed.stdout)
    (P/'REMOTE_INSTALL_RECEIPT.json').write_text(json.dumps(installation,indent=2)+'\n',encoding='utf8')
else:
    installation=json.loads((P/'REMOTE_INSTALL_RECEIPT.json').read_text('utf8'))
    check='from pathlib import Path\nimport json,hashlib\np=Path('+repr(remote)+')\n'
    check+='assert hashlib.sha256((p/"ROOT_IMPLEMENTATION_REVIEW.json").read_bytes()).hexdigest() == '+repr(installation['new_review_SHA256'])+'\n'
    check+='assert not (p/"LAUNCH_train_final.py_resume_001.json").exists() and not (p/"train_final.py.used").exists()\n'
    verified=subprocess.run(ssh,input=check,text=True,capture_output=True,timeout=20)
    assert verified.returncode==0,verified.stderr
    for suffix in ['stdout.log','stderr.log']:
        failed=P/('RESUME_LAUNCH_NATIVE.'+suffix)
        saved=P/('RESUME_LAUNCH_NATIVE.attempt_001_syntax.'+suffix)
        if failed.exists():
            assert not saved.exists()
            saved.write_bytes(failed.read_bytes())
code='from pathlib import Path\nimport os,json,subprocess,time\np=Path('+repr(remote)+')\npython='+repr(cfg['server_python'])+'\n'+'''
assert not any((Path('/proc')/str(pid)).exists() for pid in [291399,291394])
assert not (p/'train_final.py.used').exists() and (p/'RECOVERY_REQUEST.json').is_file()
marker=p/'LAUNCH_train_final.py_resume_001.json'
with marker.open('x') as f:
    json.dump({'launch_intent_unix':time.time(),'root_authorized_equivalent_resume':True},f)
out=(p/'train_final.py.resume_001.supervisor.stdout.log').open('xb')
err=(p/'train_final.py.resume_001.supervisor.stderr.log').open('xb')
cpus=list(range(12))
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'train_final.py'],cwd=p,
 stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
 preexec_fn=lambda:os.sched_setaffinity(0,cpus))
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'remote_root':str(p),
 'same_logical_run':True,'resume_checkpoint':500,'entry':'train_final.py','CPU_affinity':cpus}
time.sleep(2)
record['immediate_supervisor_exit']=child.poll()
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text()) if (p/'REMOTE_ACTIVE_JOB.json').exists() else None
marker.write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
launch=subprocess.run(ssh,input=code,text=True,capture_output=True,timeout=25)
(P/'RESUME_LAUNCH_NATIVE.stdout.log').write_text(launch.stdout,encoding='utf8')
(P/'RESUME_LAUNCH_NATIVE.stderr.log').write_text(launch.stderr,encoding='utf8')
assert launch.returncode==0,(launch.stdout,launch.stderr)
record=json.loads(launch.stdout)
record['local_observed_UTC']=datetime.now(timezone.utc).isoformat()
(P/'RESUME_LAUNCH_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
assert record['immediate_supervisor_exit'] is None and record['actual_job'] is not None,record
print(json.dumps({'same_run_resume_launched':True,'supervisor_PID':record['supervisor_PID'],
 'actual_worker_PID':record['actual_job']['actual_worker_PID'],'native_exit':record['actual_job'].get('actual_worker_native_exit'),
 'checkpoint':500,'new_review_SHA256':installation['new_review_SHA256']}))
