from pathlib import Path
import json, ast
R=Path(__file__).resolve().parents[1]
S=R/'work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001'
p=(R/'reviews/deploy_R55_root.py').read_text('utf8')
p=p.replace('R55_PSD_SOURCE_COMPATIBILITY_001','R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001').replace('R55_ROOT_','R56_ROOT_')
a=p.index("files = [");b=p.index('\nfor name in files:',a)
p=p[:a]+"files = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'diagnose_train.py', 'render_diagnostic.py', 'network.py', 'noise_core.py', 'source_flow_core.py', 'teacher_core.py', 'canonical.py', 'linux_supervise.py', 'linux_owned_worker.py']"+p[b:]
p=p.replace("assert C['LP_attempts'] == 1 and C['response_training_runs'] == C['HGB_fits'] == C['ODE_generation'] == 0","assert C['response_training_runs'] == C['HGB_fits'] == C['DEV_read'] == C['EVAL_read'] == 0")
a=p.index("    'reviewed_definitions': [");b=p.index("    'scientific_execution_started': False",a)
p=p[:a]+"    'reviewed_definitions': ['original source/REAL coverage and frozen exposure counts', 'same full/partial coupling and original R54 teacher', 'same PCG64 first-cycle path with no updates', 'PSD derivative and tangent residual energy in original coordinate metric', 'all6593 original source parents one noise draw Heun128', 'all invalid/outside rows retained and exposure repeats disclosed', 'predefined six-bin direction markers are not A-F acceptance'],\n"+p[b:]
p=p.replace("'linux_supervise_post.py'),'numeric_worker.py'","'linux_supervise.py'),'diagnose_train.py'")
assert 'numeric_worker' not in p and 'LP_attempts' not in p
(R/'reviews/deploy_R56_root.py').write_text(p,'utf8')
p=(R/'reviews/poll_R55_root.py').read_text('utf8').replace('R55_PSD_SOURCE_COMPATIBILITY_001','R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001')
p=p.replace("names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','LP_CONSTRUCTION.json','LP_SOLVER_RECEIPT.json','LOWER_BOUND_CERTIFICATE.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']","names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','COVERAGE.json','NUMERIC_RECEIPT.json','FAILURE.json','diagnose_train.py.NATIVE_RECEIPT.json']")
p=p.replace("'solver': out.get('LP_SOLVER_RECEIPT.json'), 'certificate': out.get('LOWER_BOUND_CERTIFICATE.json')","'coverage': out.get('COVERAGE.json'), 'numeric': out.get('NUMERIC_RECEIPT.json')")
(R/'reviews/poll_R56_root.py').write_text(p,'utf8')
p=(R/'reviews/fetch_R55_root.py').read_text('utf8').replace('R55_PSD_SOURCE_COMPATIBILITY_001','R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001')
(R/'reviews/fetch_R56_root.py').write_text(p,'utf8')
for path in list(S.glob('*.py'))+[R/'reviews'/n for n in ['register_R56_root.py','prepare_R56_control_root.py','deploy_R56_root.py','poll_R56_root.py','fetch_R56_root.py']]:ast.parse(path.read_text('utf8'),filename=str(path))
print(json.dumps({'AST':'passed','prepared':['deploy_R56_root.py','poll_R56_root.py','fetch_R56_root.py'],'scientific_execution_started':False}))
