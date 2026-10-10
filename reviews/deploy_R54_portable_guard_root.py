from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
R=Path(__file__).resolve().parents[1];S=R/'work/R54_CONTINUOUS_CHARGE_FLOW_001';C=json.loads((S/'config.json').read_text());host='rylan@100.82.231.102'
code='from pathlib import Path\nimport shutil\np=Path('+repr(C['remote_root'])+')\n'+'''assert not (p/'replay_all.py.used').exists()
d=p/'code_pre_execution';d.mkdir(exist_ok=True)
assert not (d/'infer_saved.before_Windows_guard.py').exists()
shutil.copy2(p/'infer_saved.py',d/'infer_saved.before_Windows_guard.py')
print('Unexecuted portable loader preserved before Windows cache guard')
'''
events=[]
for args,src in [(['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -'],code),(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(S/'infer_saved.py'),host+':'+C['remote_root']+'/'],None)]:
 p=subprocess.run(args,input=src,text=True,capture_output=True,timeout=30);events.append({'argv':args,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0,events[-1]
(S/'PORTABLE_STORAGE_GUARD_DEPLOYMENT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'events':events,'scientific_formula_changed':False,'before_first_replay':True},indent=2)+'\n','utf8')
print(json.dumps({'deployed':True,'new_inference':0,'training_code_changed':False}))
