from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'R43_SCALAR_TIME_NEURAL_FLOW_001'
raw=(OLD/'evaluate_saved.py').read_bytes()
code=raw.decode('utf8')
code=code.replace('SCALAR_TIME_FLOW_FINAL','EXTENDED_FLOW_FINAL').replace('SCALAR_TIME_FLOW_REAL_CONTROL','EXTENDED_FLOW_REAL_CONTROL')
code=code.replace("(ROOT/'config.json')","(ROOT/'config_cpu.json')")
code=code.replace("os.environ[key]='4'","os.environ[key]='12'")
code=code.replace('torch.set_num_threads(4)','torch.set_num_threads(12)').replace('torch.set_num_interop_threads(4)','torch.set_num_interop_threads(12)')
code=code.replace("r7=ROOT.parent/'R7_RESPONSE_RESIDUAL_DIAGNOSIS_001'","r7=pathlib.Path(CFG['R7_root'])")
needle="    write('metrics.json',metrics)"
assert code.count(needle)==1
code=code.replace(needle,"    from budget_comparison import describe as budget_compare\n    metrics['short_budget_R43_comparison']=budget_compare(metrics,CFG,ROOT,SOURCE_ARM)\n"+needle)
code=code.replace('R43 saved evaluation complete','R45 saved evaluation complete')
ast.parse(code);(ROOT/'evaluate_saved.py').write_text(code,'utf8')
for name in ['budget_comparison.py','check_train_solver.py']:
    ast.parse((ROOT/name).read_text('utf8'))
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_only':True,'arrays_read':0,
     'original':str(OLD/'evaluate_saved.py'),'original_sha256':hashlib.sha256(raw).hexdigest(),
     'new_sha256':hashlib.sha256(code.encode()).hexdigest(),'new_HGB_plan':20,'new_density_slots':16,
     'additional_R43_comparison':'saved predictions and eight aggregate distances only; no refits',
     'remote_execution_not_started':True}
(ROOT/'EVALUATION_SOURCE_PREPARATION.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec))
