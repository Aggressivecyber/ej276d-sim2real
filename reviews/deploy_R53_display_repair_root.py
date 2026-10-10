from pathlib import Path
from datetime import datetime,timezone
import json,ast,subprocess
R=Path(__file__).resolve().parents[1];S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
C=json.loads((S/'config_reporting.json').read_text());C['allowed_scientific_entries']=['render_figures_only.py']
(S/'config_reporting_recovery.json').write_text(json.dumps(C,indent=2)+'\n','utf8')
sup=(S/'linux_supervise_reporting.py').read_text().replace("ROOT/'config_reporting.json'","ROOT/'config_reporting_recovery.json'")
(S/'linux_supervise_reporting_recovery.py').write_text(sup,'utf8')
names=['render_R53_figures.py','render_figures_only.py','config_reporting_recovery.json','linux_supervise_reporting_recovery.py','DISPLAY_REPAIR_AUTHORITY.json']
for n in names:
 if n.endswith('.py'):ast.parse((S/n).read_text(),filename=n)
host='rylan@100.82.231.102';remote=C['remote_root'];events=[]
def call(argv,code=None):
 p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=30)
 events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 (S/'DISPLAY_REPAIR_DEPLOYMENT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'events':events},indent=2)+'\n','utf8')
 assert p.returncode==0,events[-1]
 return p.stdout
code='from pathlib import Path\nimport json,shutil\np=Path('+repr(remote)+')\n'+'''
a=json.loads((p/'summarize_and_plot.py.NATIVE_RECEIPT.json').read_text())
assert a['actual_worker_native_exit']==1 and a['worker_self_receipt']['error']=="KeyError('histogram_counts')"
assert not Path('/proc',str(a['actual_worker_PID'])).exists()
assert not (p/'render_figures_only.py.used').exists()
d=p/'display_originals';d.mkdir(exist_ok=True)
dest=d/'render_R53_figures.before_histogram_fix.py';assert not dest.exists();shutil.copy2(p/'render_R53_figures.py',dest)
print(json.dumps({'original_error_preserved':True,'original_plot_source_preserved':True,'new_fit_or_inference':0}))
'''
print(call(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],code))
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n)for n in names],host+':'+remote+'/'])
print('Display-only repair deployed')
