from pathlib import Path
import json,ast
R=Path(__file__).resolve().parents[1];S=R/'work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001'
p=(R/'reviews/deploy_R57_root.py').read_text('utf8').replace('R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001','R58_JOINT_DISTRIBUTION_FEASIBILITY_001').replace('R57_ROOT_','R58_ROOT_')
a=p.index('files = [');b=p.index('\nfor name in files:',a)
p=p[:a]+"files = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'numeric_worker.py', 'joint_core.py', 'verify_joint.py', 'render_joint.py', 'noise_core.py', 'linux_supervise_post.py', 'linux_owned_worker.py']"+p[b:]
p=p.replace("assert C['response_training_runs'] == C['HGB_fits'] == C['ODE_generation'] == C['DEV_read'] == C['EVAL_read'] == 0","assert C['LP_attempts'] == 1 and C['response_training_runs'] == C['teacher_waveforms_constructed'] == C['HGB_fits'] == C['ODE_generation'] == C['DEV_read'] == C['EVAL_read'] == 0")
a=p.index("    'reviewed_definitions': [");b=p.index("    'scientific_execution_started': False",a)
p=p[:a]+"    'reviewed_definitions': ['all original SOURCE equal weights and all REAL integer contributions', 'fixed512 quantile-mean atoms and64 SOURCE midrank segments', 'row and column constrained finite joint probability family', 'fixed-marginal covariance is linear and proportional to Spearman', 'exact union-support CDF integral for each original REAL conditional W1', 'frozen Q3/Q4 h==1 valley constraints without relabeling', 'one LP only, saved-solution independent verification', 'separate ADC short/long integral necessary bounds', 'no teacher waveforms, neural training or new DEV evaluation'],\n"+p[b:]
assert 'teacher_core' not in p and 'render_teacher' not in p
(R/'reviews/deploy_R58_root.py').write_text(p,'utf8')
p=(R/'reviews/poll_R57_root.py').read_text('utf8').replace('R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001','R58_JOINT_DISTRIBUTION_FEASIBILITY_001')
p=p.replace("names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','COUPLING_RECEIPT.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']","names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','LP_CONSTRUCTION.json','LP_SOLVER_RECEIPT.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']")
p=p.replace("'coupling': out.get('COUPLING_RECEIPT.json'), 'numeric': out.get('NUMERIC_RECEIPT.json')","'solver': out.get('LP_SOLVER_RECEIPT.json'), 'numeric': out.get('NUMERIC_RECEIPT.json')")
(R/'reviews/poll_R58_root.py').write_text(p,'utf8')
p=(R/'reviews/fetch_R57_root.py').read_text('utf8').replace('R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001','R58_JOINT_DISTRIBUTION_FEASIBILITY_001')
(R/'reviews/fetch_R58_root.py').write_text(p,'utf8')
files=list(S.glob('*.py'))+list((R/'reviews').glob('*R58*root*.py'))
for path in files:ast.parse(path.read_text('utf8'),filename=str(path))
print(json.dumps({'AST':'passed','files':len(files),'scientific_execution_started':False}))
