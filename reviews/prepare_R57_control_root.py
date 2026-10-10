from pathlib import Path
import json, ast
R=Path(__file__).resolve().parents[1]
S=R/'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001'
p=(R/'reviews/deploy_R55_root.py').read_text('utf8')
p=p.replace('R55_PSD_SOURCE_COMPATIBILITY_001','R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001').replace('R55_ROOT_','R57_ROOT_')
a=p.index('files = [');b=p.index('\nfor name in files:',a)
p=p[:a]+"files = ['config.json', 'INPUT_BINDING.json', 'PROTOCOL.md', 'REPRODUCTION.md', 'numeric_worker.py', 'render_teacher.py', 'coupling_core.py', 'weighted_metrics.py', 'noise_core.py', 'teacher_core.py', 'canonical.py', 'linux_supervise_post.py', 'linux_owned_worker.py']"+p[b:]
p=p.replace("assert C['LP_attempts'] == 1 and C['response_training_runs'] == C['HGB_fits'] == C['ODE_generation'] == 0","assert C['response_training_runs'] == C['HGB_fits'] == C['ODE_generation'] == C['DEV_read'] == C['EVAL_read'] == 0")
a=p.index("    'reviewed_definitions': [");b=p.index("    'scientific_execution_started': False",a)
p=p[:a]+"    'reviewed_definitions': ['ten original-Q regions with all6593 SOURCE and21047 REAL TRAIN parents', 'exact integer split masses and fixed16 Q groups', 'within-group monotone signed-PSD coupling with original index ties', 'global SOURCE equal margins and conditional REAL equal margins', 'unchanged frozen pulse-only charge correction and canonical ADC', 'all failed and overrange mass retained', 'source_h used only for frozen TRAIN valley diagnostic', 'all480 waveform moments and fixed first edges', 'fixed TRAIN conditions do not replace A-F'],\n"+p[b:]
assert 'LP_attempts' not in p
(R/'reviews/deploy_R57_root.py').write_text(p,'utf8')
p=(R/'reviews/poll_R55_root.py').read_text('utf8').replace('R55_PSD_SOURCE_COMPATIBILITY_001','R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001')
p=p.replace("names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','LP_CONSTRUCTION.json','LP_SOLVER_RECEIPT.json','LOWER_BOUND_CERTIFICATE.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']","names=['REMOTE_ACTIVE_JOB.json','PROGRESS.json','COUPLING_RECEIPT.json','NUMERIC_RECEIPT.json','FAILURE.json','numeric_worker.py.NATIVE_RECEIPT.json']")
p=p.replace("'solver': out.get('LP_SOLVER_RECEIPT.json'), 'certificate': out.get('LOWER_BOUND_CERTIFICATE.json')","'coupling': out.get('COUPLING_RECEIPT.json'), 'numeric': out.get('NUMERIC_RECEIPT.json')")
(R/'reviews/poll_R57_root.py').write_text(p,'utf8')
p=(R/'reviews/fetch_R55_root.py').read_text('utf8').replace('R55_PSD_SOURCE_COMPATIBILITY_001','R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001')
(R/'reviews/fetch_R57_root.py').write_text(p,'utf8')
for path in list(S.glob('*.py'))+[R/'reviews'/n for n in ['register_R57_root.py','prepare_R57_control_root.py','deploy_R57_root.py','poll_R57_root.py','fetch_R57_root.py']]:ast.parse(path.read_text('utf8'),filename=str(path))
print(json.dumps({'AST':'passed','prepared':['deploy_R57_root.py','poll_R57_root.py','fetch_R57_root.py'],'scientific_execution_started':False}))
