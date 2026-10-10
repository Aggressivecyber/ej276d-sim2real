from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,shutil,ast
R=Path(__file__).resolve().parents[1];S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
C=json.loads((S/'config.json').read_text());host='rylan@100.82.231.102'
for name in ['UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']:
    assert not (S/name).exists();shutil.copy2(R/'work/R43_SCALAR_TIME_NEURAL_FLOW_001'/name,S/name)
names=['summarize_and_plot.py','render_R53_figures.py','config_reporting.json','linux_supervise_reporting.py',
       'infer_saved.py','replay_all.py','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']
for name in names:
    if name.endswith('.py'):ast.parse((S/name).read_text('utf8'))
events=[]
def call(argv,code=None):
    p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    (S/'REPORTING_DEPLOYMENT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'events':events},indent=2)+'\n','utf8')
    assert p.returncode==0,events[-1]
    return p.stdout
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
prefix='from pathlib import Path\nimport json,ast\np=Path('+repr(C['remote_root'])+')\n'
code=prefix+'assert not (p/"replay_all.py.used").exists()\nassert not (p/"summarize_and_plot.py.used").exists()\nnames='+repr([n for n in names if n!='replay_all.py'])+'\nassert not any((p/n).exists()for n in names)\nprint("unexecuted replay updated to portable loader; reporting performs no new fits")\n'
call(ssh,code)
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n)for n in names],host+':'+C['remote_root']+'/'])
print(call(ssh,prefix+'names='+repr(names)+'\nfor n in names:\n if n.endswith(".py"):ast.parse((p/n).read_text(),filename=n)\nprint(json.dumps({"files":len(names),"AST":"pass","new_scientific_execution":0}))\n'))
