from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os,sys
R=Path(__file__).resolve().parents[1];S=R/'work/R54_CONTINUOUS_CHARGE_FLOW_001'
C=json.loads((S/'config.json').read_text());entry=sys.argv[1]
assert entry in ['numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_figures_only.py','render_final_figures.py','render_legend_only.py']
prefix='from pathlib import Path\nimport json,subprocess,time,os\np=Path('+repr(C['remote_root'])+')\nentry='+repr(entry)+'\npython='+repr(C['server_python'])+'\n'
code=prefix+'''
assert (p/entry).exists()
active=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
assert 'actual_worker_native_exit' in active and active['termination_reason'] is None,active
if entry=='render_figures_only.py':
 assert active['entry']=='summarize_and_plot.py' and active['actual_worker_native_exit']==1 and active['worker_self_receipt']['error']=="KeyError('histogram_counts')",active
elif entry=='render_final_figures.py':
 assert active['entry']=='render_figures_only.py' and active['actual_worker_native_exit']==1 and 'TypeError' in active['worker_self_receipt']['error'],active
else:assert active['actual_worker_native_exit']==0,active
assert not Path('/proc',str(active['actual_worker_PID'])).exists(),active
train=json.loads((p/'CONTINUOUS_Q_RESPONSE/TRAINING_RECEIPT.json').read_text())
assert train['updates']==50000 and all(train['same_R48_source_identity_and_random_streams'].values())
assert all(train['same_R53_all_original_and_teacher_streams'].values()) and train['continuous_Q_nonzero_gradient_steps']>0
if entry!='numerical_check.py':assert json.loads((p/'TRAIN_SOLVER_CHECK.json').read_text())['both_pass']
if entry in ['evaluate_saved.py','replay_all.py']:assert json.loads((p/'GENERATION_NUMERIC_RECEIPT.json').read_text())['status']=='complete'
if entry in ['summarize_and_plot.py','render_figures_only.py','render_final_figures.py']:
 assert json.loads((p/'EVALUATION_NUMERIC_RECEIPT.json').read_text())['status']=='complete'
 assert json.loads((p/'REPLAY_RECEIPT.json').read_text())['rows']==4000
marker=p/('LAUNCH_'+entry+'.json')
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/(entry+'.supervisor.stdout.log')).open('wb');err=(p/(entry+'.supervisor.stderr.log')).open('wb')
supervisor='linux_supervise_reporting.py' if entry=='summarize_and_plot.py' else 'linux_supervise.py'
if entry=='render_figures_only.py':supervisor='linux_supervise_reporting_recovery.py'
if entry=='render_final_figures.py':supervisor='linux_supervise_reporting_recovery2.py'
if entry=='render_legend_only.py':supervisor='linux_supervise_legend.py'
child=subprocess.Popen([python,'-B',str(p/supervisor),entry],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'entry':entry,'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())
record['immediate_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
p=subprocess.run(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=30)
record={'UTC':datetime.now(timezone.utc).isoformat(),'entry':entry,'control_PID':os.getpid(),'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(S/('LOCAL_LAUNCH_'+entry+'.json')).write_text(json.dumps(record,indent=2)+'\n','utf8')
assert p.returncode==0,record
print(p.stdout)
