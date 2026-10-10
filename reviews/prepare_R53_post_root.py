from pathlib import Path
from datetime import datetime, timezone
import ast, json, shutil

R = Path(__file__).resolve().parents[1]
S = R / 'work/R53_PULSE_ONLY_TEACHER_FLOW_001'
OLD = R / 'work/R49_FROZEN_HEUN128_RESPONSE_001'
C = json.loads((S / 'config.json').read_text('utf8'))
P = json.loads((OLD / 'config.json').read_text('utf8'))
P.update(C)
P.update(endpoints=['JOINT_PULSE_RESPONSE', 'JOINT_PULSE_REAL_CONTROL'],
    REAL_arms=['BASE_REAL_CONTROL', 'JOINT_PULSE_REAL_CONTROL'],
    endpoint_weight_paths={n:C['remote_root']+'/JOINT_PULSE_RESPONSE/FINAL_FLOW.pt'
                           for n in ['JOINT_PULSE_RESPONSE', 'JOINT_PULSE_REAL_CONTROL']},
    historical_training_root=C['remote_root'],
    canonical_R51_root=json.loads((R/'work/R51_CANONICAL_SOURCE_TEACHER_DIAGNOSTIC_001/config.json').read_text())['remote_root'],
    C2ST_views=['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local'],
    HGB_max_fits=10, planned_new_classifier_fits=10, planned_new_density_slots=16,
    inference_device='cuda:0', postprocessing_registered_before_results=True,
    canonical_ADC_features=True, raw_SOURCE_input_quantized_for_features=False,
    historical_CPU_controls_new_GPU_candidate_disclosed=True,
    scientific_scope='R53 single frozen50000 endpoint; R51 canonical comparisons reused')
(S/'config_post.json').write_text(json.dumps(P,indent=2)+'\n','utf8')
files=['evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py',
       'data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py',
       'flow_diagnostics.py']
for name in files:
    assert not (S/name).exists(), name
    shutil.copy2(OLD/name,S/name)
shutil.copy2(R/'work/R49_LOCAL_HEUN128_RECOVERY_001/TRAIN_SOLVER_CHECK.json',
             S/'HISTORICAL_TRAIN_SOLVER_CHECK.json')

def change(text,old,new):
    assert old in text, old
    return text.replace(old,new)

runtime=(OLD/'flow_runtime.py').read_text('utf8')
runtime=change(runtime,'self.network.eval();self.network.requires_grad_(False)',
    "self.network.to(cfg['inference_device']);self.network.eval();self.network.requires_grad_(False)")
runtime=change(runtime,"self.completion=arm in ['GUIDED_NOISE_COMPLETION','GUIDED_COMPLETION_REAL_CONTROL']",
    "assert arm in cfg['endpoints'];self.completion=True")
(S/'flow_runtime.py').write_text(runtime,'utf8')

numerical=(OLD/'check_models.py').read_text('utf8')
numerical=change(numerical,"R/'config.json'","R/'config_post.json'")
numerical=change(numerical,'net.eval();net.requires_grad_(False)',
    "net.to(C['inference_device']);net.eval();net.requires_grad_(False)")
numerical=change(numerical,"completion=arm==C['arms'][1]","completion=True")
numerical=change(numerical,"models={}",
    "historical=json.loads((R/'HISTORICAL_TRAIN_SOLVER_CHECK.json').read_text());assert historical['both_pass']\nmodels={}")
numerical=change(numerical,"'both_pass':all(r['pass']for r in models.values())",
    "'both_pass':historical['both_pass'] and all(r['pass']for r in models.values()),'historical_control_pass_reused':historical['both_pass'],'new_models':1,'device':C['inference_device']")
(S/'numerical_check.py').write_text(numerical,'utf8')

gen=(OLD/'generate_all.py').read_text('utf8')
gen=change(gen,"ROOT/'config_cpu.json'","ROOT/'config_post.json'")
gen=change(gen,"['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']","['JOINT_PULSE_REAL_CONTROL']")
gen=change(gen,'[8000,2048,1664,256]','[4000,1024,832,128]')
gen=change(gen,"progress('complete11968')","progress('complete5984')")
gen=change(gen,'from model import adc','from model import adc\nfrom canonical import canonical')
gen=change(gen,"before=ev.features(real['reference_V']);after=ev.features(cpost)",
    "before=ev.features(canonical(real['reference_V']));after=ev.features(canonical(cpost))")
(S/'generate_all.py').write_text(gen,'utf8')

replay=(OLD/'replay_all.py').read_text('utf8')
replay=change(replay,"ROOT/'config_cpu.json'","ROOT/'config_post.json'")
replay=change(replay,"'device':'cpu'","'device':CFG['inference_device']")
replay=change(replay,'assert COUNT==8000','assert COUNT==4000')
(S/'replay_all.py').write_text(replay,'utf8')

assessment=(S/'assessment.py').read_text('utf8')
assessment=change(assessment,'from common_math import distances','from common_math import distances\nfrom canonical import canonical')
assessment=change(assessment,'def data(v,ids,groups,typ,ref,qc=None):',
    'def data(v,ids,groups,typ,ref,qc=None,adc_domain=True):')
assessment=change(assessment,"d=ev.features(v);d['ids']", "d=ev.features(canonical(v) if adc_domain else v);d['ids']")
(S/'assessment.py').write_text(assessment,'utf8')
diag=(S/'flow_diagnostics.py').read_text('utf8')
diag=change(diag,'from model import adc','from model import adc\nfrom canonical import canonical')
diag=change(diag,'ev.features(post)','ev.features(canonical(post))')
diag=change(diag,'ev.features(reference)','ev.features(canonical(reference))')
(S/'flow_diagnostics.py').write_text(diag,'utf8')

for path in S.glob('*.py'):ast.parse(path.read_text('utf8'),filename=str(path))
(S/'POSTPROCESS_PREPARATION.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),
    'owner':'/root','training_config_and_training_code_untouched':True,'config':'config_post.json',
    'new_generation_rows':5984,'independent_replay_rows':4000,'new_HGB_fits':10,'new_density_slots':16,
    'raw_SOURCE_input_features_preserved':True,'common_ADC_codes_applied_to_all_ADC_domains':True,
    'numerical_reference_only_steps':256,'all_asts_parsed':True,
    'no_new_scientific_computation_in_preparation':True},indent=2)+'\n','utf8')
print(json.dumps({'prepared':str(S),'new_generation_rows':5984,'new_replay_rows':4000,'AST':'pass'}))
