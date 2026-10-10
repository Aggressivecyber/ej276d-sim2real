from pathlib import Path
from datetime import datetime,timezone
import ast,json,shutil
R=Path(__file__).resolve().parents[1];S=R/'work/R54_CONTINUOUS_CHARGE_FLOW_001';O=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
C=json.loads((S/'config.json').read_text());P=json.loads((O/'config_post.json').read_text());P.update(C)
P.update(endpoints=['CONTINUOUS_Q_RESPONSE','CONTINUOUS_Q_REAL_CONTROL'],REAL_arms=['BASE_REAL_CONTROL','CONTINUOUS_Q_REAL_CONTROL'],endpoint_weight_paths={n:C['remote_root']+'/CONTINUOUS_Q_RESPONSE/FINAL_FLOW.pt' for n in ['CONTINUOUS_Q_RESPONSE','CONTINUOUS_Q_REAL_CONTROL']},historical_training_root=C['remote_root'],scientific_scope='R54 sole continuous-logQ conditioning ablation; R53 and R51 frozen comparisons reused',historical_R53_controls_new_GPU_candidate_disclosed=True)
(S/'config_post.json').write_text(json.dumps(P,indent=2)+'\n','utf8')
files=['canonical.py','flow_runtime.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py','data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py','infer_saved.py','summarize_and_plot.py','render_R53_figures.py']
for name in files:
 dest=S/name.replace('render_R53','render_R54');assert not dest.exists(),name
 value=(O/name).read_text('utf8').replace('JOINT_PULSE_RESPONSE','CONTINUOUS_Q_RESPONSE').replace('JOINT_PULSE_REAL_CONTROL','CONTINUOUS_Q_REAL_CONTROL').replace('R53','R54')
 dest.write_text(value,'utf8')
for name in ['HISTORICAL_TRAIN_SOLVER_CHECK.json','UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json']:
 assert not (S/name).exists();shutil.copy2(O/name,S/name)
Q=dict(C);Q.update(allowed_scientific_entries=['summarize_and_plot.py'],GPU_scientific_entries=[],reporting_only=True,new_fit_or_inference_authorized=False)
(S/'config_reporting.json').write_text(json.dumps(Q,indent=2)+'\n','utf8')
sup=(S/'linux_supervise.py').read_text().replace("ROOT/'config.json'","ROOT/'config_reporting.json'")
(S/'linux_supervise_reporting.py').write_text(sup,'utf8')
for p in S.glob('*.py'):ast.parse(p.read_text('utf8'),filename=str(p))
(S/'POSTPROCESS_PREPARATION.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','training_files_untouched':True,'new_generation_rows':5984,'replay_rows':4000,'new_HGB_fits':10,'density_slots':16,'frozen_R53_paired_comparison_to_add_before_execution':True,'same_raw_SOURCE_definitions':True,'all_asts_parsed':True,'new_scientific_execution':0},indent=2)+'\n','utf8')
print(json.dumps({'files_prepared':len(files)+5,'training_code_untouched':True,'AST':'pass'}))
