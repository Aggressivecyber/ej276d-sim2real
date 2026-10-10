from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path(__file__).resolve().parents[1];S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
C=json.loads((S/'config_post.json').read_text());host='rylan@100.82.231.102'
names=['config_post.json','canonical.py','flow_runtime.py','numerical_check.py','generate_all.py',
       'evaluate_saved.py','replay_all.py','evaluation.py','diagnostic_core.py','assessment.py',
       'common_math.py','energy_width.py','data_support.py','model.py','legacy_model.py',
       'flow_core.py','endpoint_common.py','flow_diagnostics.py','HISTORICAL_TRAIN_SOLVER_CHECK.json',
       'POSTPROCESS_PREPARATION.json']
events=[]
def call(argv,code=None):
    p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
    events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    (S/'POSTPROCESS_DEPLOYMENT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),'events':events},indent=2)+'\n','utf8')
    assert p.returncode==0,events[-1]
    return p.stdout
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
prefix='from pathlib import Path\nimport json,ast\np=Path('+repr(C['remote_root'])+')\n'
call(ssh,prefix+'names='+repr(names)+'\nassert p.is_dir()\nassert not any((p/n).exists() for n in names)\nprint("new postprocessing files only; training files retained")\n')
call(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n)for n in names],host+':'+C['remote_root']+'/'])
code=prefix+'''
c=json.loads((p/'config_post.json').read_text())
for name in '''+repr(names)+''':
 if name.endswith('.py'):ast.parse((p/name).read_text(),filename=name)
inputs=[Path(c['canonical_R51_root'])/'CANONICAL_C2ST.json',Path(c['canonical_R51_root'])/'CANONICAL_PREDICTIONS_PRIVATE.npz',
 Path(c['train_solver_input']),Path(c['R39_root'])/'metrics.json',Path(c['R26_root'])/'results/C2ST_PRIVATE.npz',
 Path(c['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz',Path(c['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',
 Path(c['frozen_eval_input_root'])/'PROBE_INPUTS_CORRECTED.npz',Path(c['frozen_eval_input_root'])/'JOINT8_INPUTS_PRIVATE.npz',
 Path(c['local_input_paths']['width_definition']),Path(c['old_BASE_REAL_generated'])]
missing=[str(q)for q in inputs if not q.exists()];assert not missing,missing
print(json.dumps({'postprocessing_files':len('''+repr(names)+'''),'reference_paths_present':len(inputs),'AST':'pass','new_scientific_computation':0}))
'''
print(call(ssh,code))
