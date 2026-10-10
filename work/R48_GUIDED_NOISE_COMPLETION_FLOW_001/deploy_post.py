from pathlib import Path
from datetime import datetime,timezone
import ast,json,subprocess,os,sys,time
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));mode=sys.argv[1]
names=['linux_supervise_post.py','render_train.py']if mode=='TRAIN_only'else[
 'config_cpu.json','flow_runtime.py','flow_diagnostics.py','generate_all.py','evaluate_saved.py','replay_all.py',
 'evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','common_math.py','model.py','legacy_model.py',
 'endpoint_common.py','temporal_common.py','mean_description.py','flow_core.py','linux_supervise_post.py','render_train.py','render_figures.py','summarize_and_plot.py']
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'mode':mode,'normal_transfer_no_equivalence_test':True,'operations':[]}
start=time.monotonic()
for n in names:
    d=(R/n).read_bytes()
    if n.endswith('.py'):ast.parse(d.decode('utf8'))
    p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(R/n),'rylan@100.82.231.102:'+C['remote_root']+'/'+n],capture_output=True,text=True)
    rec['operations'].append({'file':n,'native_exit':p.returncode,'size':len(d),'stderr':p.stderr[-2000:]})
    (R/('POST_DEPLOY_'+mode+'.json')).write_bytes((json.dumps(rec,indent=2)+'\n').encode())
    if p.returncode:raise RuntimeError(rec['operations'][-1])
rec['elapsed_s']=time.monotonic()-start;(R/('POST_DEPLOY_'+mode+'.json')).write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps(rec),flush=True)
