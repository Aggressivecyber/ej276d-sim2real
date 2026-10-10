from pathlib import Path
from datetime import datetime,timezone
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
text=(R/'work/R49_FROZEN_HEUN128_RESPONSE_001/summarize_and_plot.py').read_text('utf8')
def replace(old,new):
 global text
 assert old in text,old
 text=text.replace(old,new)
replace("R/'config_cpu.json'","R/'config_post.json'")
replace("S,L=C['arms'];A=[S,L];RC=['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']",
        "L=C['arms'][0];A=[L];RC=['JOINT_PULSE_REAL_CONTROL']")
replace("'historical_R48_updates_per_arm':50000,'new_training_updates_per_arm':0", "'new_R53_updates_per_arm':50000,'new_training_updates_per_arm':50000")
replace("'independent_CPU_full_chain_replay_rows'","'independent_GPU_full_chain_replay_rows'")
replace("g6['historical_R48_paired_training_pass']=training[arm]['updates']==50000 and all(training[L]['paired_stream_equality'].values()) and training[S]['initial_state_SHA']==training[L]['initial_state_SHA']",
        "g6['frozen_final_training_stream_pass']=training[arm]['updates']==50000 and all(training[arm]['same_R48_source_identity_and_random_streams'].values()) and training[arm]['same_R48_target_total_occurrence_counts'] and json.loads((R/'TEACHER_VALIDATION.json').read_text())['passed']")
replace("g6['historical_R48_paired_training_pass']","g6['frozen_final_training_stream_pass']")
replace("'recipe':'fixed full-guide and REAL-identity flow; conservative projected colored noise completion'",
        "'recipe':'R53 fixed joint charge-PSD occurrence coupling and pulse-only teacher; full guide and REAL identity'" )
replace("'close_fixed_R49_Heun128_recipe'","'close_fixed_R53_joint_pulse_teacher_recipe'")
replace("'paired_primary_minus_control_C2ST'","'paired_R53_minus_historical_R49_completion'")
replace("'paired_completion_minus_fixed_same_REAL_C2ST'","'paired_R53_minus_historical_R49_completion'")
replace("'historical_R48_training':tc,'new_training_runs':0", "'R53_training':tc,'new_training_runs':1")
replace("'source_loss_final','identity_loss_final'", "'source_loss_final','identity_loss_final','same_R48_source_identity_and_random_streams','same_R48_target_total_occurrence_counts'")
# Remove a duplicate dictionary key inherited from the two-arm report.
replace(" 'paired_R53_minus_historical_R49_completion':M['paired_R53_minus_historical_R49_completion'],\n 'paired_R53_minus_historical_R49_completion':M['paired_R53_minus_historical_R49_completion'],",
        " 'paired_R53_minus_historical_R49_completion':M['paired_R53_minus_historical_R49_completion'],")
replace('import render_figures\nrender_figures.run(R,C,M,gate,acceptance)',
        'import render_R53_figures\nrender_R53_figures.run(R,C,M,gate,acceptance)')
ast.parse(text);(S/'summarize_and_plot.py').write_text(text,'utf8')
C=json.loads((S/'config.json').read_text());C['allowed_scientific_entries']=['summarize_and_plot.py'];C['GPU_scientific_entries']=[]
C['reporting_only']=True;C['new_fit_or_inference_authorized']=False
(S/'config_reporting.json').write_text(json.dumps(C,indent=2)+'\n','utf8')
sup=(S/'linux_supervise.py').read_text();assert "ROOT/'config.json'" in sup
sup=sup.replace("ROOT/'config.json'","ROOT/'config_reporting.json'")
(S/'linux_supervise_reporting.py').write_text(sup,'utf8')
print(json.dumps({'summary':'prepared','new_fits_or_inference':0,'gates':'original_A_C_and_six_stage_gates','UTC':datetime.now(timezone.utc).isoformat()}))
