from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R33_RESIDUAL_OT_FLOW_001'
src=(OLD/'evaluate_endpoints_attempt002.py').read_text('utf8')
def change(old,new):
    global src
    assert old in src,old[:120]
    src=src.replace(old,new)
change("ARMS=CFG['endpoints'];CONTROLS=['BASE']","ARMS=CFG['endpoints'];CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM']")
change("results={'BASE':old['C2ST']['results']['CONTROL_FINAL']};private={}","results={n:old['C2ST']['results'][n] for n in CONTROLS};private={}")
change("[('OT_minus_INDEPENDENT','OT_CFM','INDEPENDENT_CFM'),('OT_minus_BASE','OT_CFM','BASE'),('INDEPENDENT_minus_BASE','INDEPENDENT_CFM','BASE')]",
    "[('ORDERED_minus_BASE','PSD_ORDERED_CFM','BASE'),('ORDERED_minus_INDEPENDENT','PSD_ORDERED_CFM','INDEPENDENT_CFM')]")
change("else load(OUT/(reference+'__'+view+'_PREDICTIONS_PRIVATE.npz'))",
    "else load(pathlib.Path(CFG['R33_root'])/'results'/(reference+'__'+view+'_PREDICTIONS_PRIVATE.npz'))")
change("tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN']};tables['BASE']=old['energy_conditional_width']['tables']['CONTROL_FINAL'];grid=",
    "tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','SOURCE_CLEAN',*CONTROLS]};grid=")
change("[('all_final_endpoints',[*CONTROLS,*ARMS]),('OT_vs_independent',ARMS)]",
    "[('all_controls_and_new',[*CONTROLS,*ARMS]),('ORDERED_vs_BASE',['BASE',*ARMS]),('ORDERED_vs_INDEPENDENT',['INDEPENDENT_CFM',*ARMS])]")
change("old=json.loads((pathlib.Path(CFG['R32_root'])/'metrics.json').read_text('utf8'))",
    "old=json.loads((pathlib.Path(CFG['R33_root'])/'metrics.json').read_text('utf8'))")
change("domains['REAL']['valid']&=real['real_QC_valid'];rr=",
    "for control in ['INDEPENDENT_CFM','OT_CFM']:\n        saved=load(pathlib.Path(CFG['R33_root'])/control/'generated.npz')\n        assert np.array_equal(saved['source_ids'],ids)\n        domains[control]=aa.data(saved['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)\n    domains['REAL']['valid']&=real['real_QC_valid'];rr=")
change("'two_current_server_flow_fits_as_registered':True","'one_new_server_flow_fit_as_registered':True,'original_R33_controls_reused':True")
change("COUNTER['HGB_attempts']<=20 and COUNTER['density_EM_attempts']<=16 and COUNTER['main_generation_rows']<=4000 and COUNTER['repeat_noise_rows']<=2048 and COUNTER['source_probe_rows']<=1664",
    "COUNTER['HGB_attempts']<=10 and COUNTER['density_EM_attempts']<=8 and COUNTER['density_description_slots']<=8 and COUNTER['main_generation_rows']<=2000 and COUNTER['repeat_noise_rows']<=1024 and COUNTER['source_probe_rows']<=832")
change("'solver_check_rows':64","'solver_check_rows':32")
change("log('R33 evaluation complete ","log('R35 evaluation complete ")
change("'generation':generation,'solver_check':solver,","'generation':generation,'solver_check':solver,'boundary_differences':boundary,")
change("metric={'status':'complete'","boundary=fd.boundary_metrics(domains,OUT);write('BOUNDARY_METRICS.json',boundary)\n    metric={'status':'complete'")
ast.parse(src);(ROOT/'evaluate_endpoints.py').write_text(src,'utf8')
src=(OLD/'replay_endpoints.py').read_text('utf8');ast.parse(src);(ROOT/'replay_endpoints.py').write_text(src,'utf8')
# Retain the exact preADC endpoint for main and both auxiliary classes.
src=(OLD/'flow_runtime.py').read_text('utf8')
src=src.replace("'flow_increment_V':pre-base,", "'flow_increment_V':pre-base,'flow_increment_f64_V':pre.astype('f8')-base.astype('f8'),\n                   'base_sum_rounding_f64_V':base.astype('f8')-signal.astype('f8')-electronic.astype('f8'),")
ast.parse(src);(ROOT/'flow_runtime.py').write_text(src,'utf8')
src=(OLD/'flow_diagnostics.py').read_text('utf8')
src=src.replace('style=style,waveforms_V=post,','style=style,pre_ADC_V=pre,waveforms_V=post,')
src=src.replace('source_parent_index=parent,waveforms_V=post,','source_parent_index=parent,pre_ADC_V=pre,waveforms_V=post,')
src=src.replace("pieces=np.stack([signal,electronic,detail['flow_increment_V']],0).reshape(3,128,8,480)",
    "pieces=np.stack([signal.astype('f8'),electronic.astype('f8'),detail['base_sum_rounding_f64_V'],detail['flow_increment_f64_V']],0).reshape(4,128,8,480)")
src=src.replace("['base_signal','base_electronic','flow_increment']", "['base_signal','base_electronic','base_sum_rounding','flow_increment_f64']")
src=src.replace("'three_additive_algebraic_components_covariance_V2'", "'four_additive_algebraic_components_covariance_V2'")
src=src.replace("'cross_terms_retained':True,'ADC':ad,", "'cross_terms_retained':True,'component_order':['base_signal','base_electronic','base_sum_rounding','flow_increment_f64'],'rounding_not_physical_noise':True,'ADC':ad,")
boundary=(ROOT.parent/'R34_PROTECTED_FLOW_CORRECTION_001/protected_diagnostics.py').read_text('utf8')
boundary=boundary[boundary.index('def boundary_metrics('):]
# That file ends with this pure saved-waveform description; no P/recovery code copied.
ast.parse(boundary)
src+='\nfrom scipy.stats import wasserstein_distance\n'+boundary
ast.parse(src);(ROOT/'flow_diagnostics.py').write_text(src,'utf8')
sources={}
for p in sorted(ROOT.glob('*.py')):
    content=p.read_text('utf8');ast.parse(content)
    sources[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'EVALUATION_SOURCE_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'sources':sources,'main_and_aux_exact_preADC_retained':True,'new_projection_code_imported':False,
    'original_evaluation_definition_unchanged':True,'science_here':0},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_AST_passed':True,'sources':len(sources),
    'new_HGB_cap':10,'new_density_slots':8,'science_here':0}),flush=True)
