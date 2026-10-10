from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,time,subprocess,traceback
R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text('utf8'))
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
if(R/'LOCAL_PIPELINE_STARTED.json').exists():raise RuntimeError('local pipeline already started; inspect state, do not duplicate')
assert json.loads((R/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']
write('LOCAL_PIPELINE_STARTED.json',{'PID':os.getpid(),'UTC':utc(),'new_training':0,'remote_replay_terminal':'unknown'})
steps=[]
try:
    for entry in ['generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_train.py']:
        while True:
            p=subprocess.run([sys.executable,'-B',str(R/'supervise_local.py'),entry],cwd=R)
            if p.returncode!=2:break
            write('LOCAL_RESOURCE_WAIT.json',{'PID':os.getpid(),'UTC':utc(),'entry':entry,
                'status':'waiting for actual available physical memory; no science worker started','user_apps_untouched':True})
            time.sleep(60)
        steps.append({'entry':entry,'supervisor_native_exit':p.returncode,'UTC':utc()})
        write('LOCAL_PIPELINE_PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'steps':steps})
        if p.returncode!=0:raise RuntimeError((entry,'actual guarded execution failed',p.returncode))
    write('LOCAL_PIPELINE_TERMINAL.json',{'PID':os.getpid(),'UTC':utc(),'status':'complete','steps':steps})
except BaseException as e:
    write('LOCAL_PIPELINE_TERMINAL.json',{'PID':os.getpid(),'UTC':utc(),'status':'engineering_failure','steps':steps,
        'error':repr(e),'traceback':traceback.format_exc(),'no_implicit_resubmission':True})
    raise
