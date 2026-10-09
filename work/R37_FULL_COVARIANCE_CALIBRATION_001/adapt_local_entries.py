from pathlib import Path
import ast,json,datetime,hashlib,os
R=Path(__file__).resolve().parent;old=R.parent/'R36_CONDITIONAL_DIFFUSION_EDIT_001'
s=(old/'evaluate_saved_repair.py').read_text('utf8')
s=s.replace("SOURCE_ARM='CONDITIONAL_DIFFUSION_EDIT'","SOURCE_ARM='FULL_COVARIANCE_CALIBRATION'")
s=s.replace("CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM']","CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT']")
s=s.replace("density_EM_attempts']+attempt<=16 and COUNTER['density_description_slots']<16","density_EM_attempts']+attempt<=8 and COUNTER['density_description_slots']<8")
s=s.replace("DIFFUSION_vs_BASE","CALIBRATION_vs_BASE").replace("DIFFUSION_vs_ORDERED","CALIBRATION_vs_ORDERED")
s=s.replace("old=json.loads((pathlib.Path(CFG['R35_root'])/'metrics.json')","old=json.loads((pathlib.Path(CFG['R36_root'])/'metrics.json')")
s=s.replace("('PSD_ORDERED_CFM',pathlib.Path(CFG['R35_root'])/'PSD_ORDERED_CFM/generated.npz'),(SOURCE_ARM", "('PSD_ORDERED_CFM',pathlib.Path(CFG['R35_root'])/'PSD_ORDERED_CFM/generated.npz'),\n        ('CONDITIONAL_DIFFUSION_EDIT',pathlib.Path(CFG['R36_root'])/'CONDITIONAL_DIFFUSION_EDIT/generated.npz'),(SOURCE_ARM")
start=s.index("    a=load(ROOT/REAL_ARM/'generated.npz'");end=s.index("    generation=json.loads",start)
s=s[:start]+s[end:]
s=s.replace("json.loads((ROOT/SOURCE_ARM/'TRAINING_RECEIPT.json')", "json.loads((ROOT/SOURCE_ARM/'STATISTICS_FIT_RECEIPT.json')")
s=s.replace("COUNTER['HGB_attempts']==20 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16", "COUNTER['HGB_attempts']==10 and COUNTER['density_description_slots']==8 and COUNTER['density_EM_attempts']<=8")
s=s.replace("'REAL_control_seen_TRAIN_reference_not_unseen_acquisition':True", "'new_REAL_guide_control':False,'RAW_RR_same_acquisition_reference_reused':True")
s=s.replace("log('R36 saved evaluation complete ", "log('R37 saved evaluation complete ")
s=s.replace("    metrics={'status':'complete'", "    from covariance_description import describe_covariance\n    cv=describe_covariance(domains,['BASE',SOURCE_ARM],OUT)\n    write('DEV_COVARIANCE_DESCRIPTION.json',cv)\n    metrics={'covariance_description':cv,'status':'complete'")
ast.parse(s);(R/'evaluate_saved.py').write_text(s,'utf8')
s=(old/'replay_all.py').read_text('utf8')
s=s.replace("actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');seeds=load(ROOT/'ACTUAL_DIFFUSION_SEED_MAPS_PRIVATE.npz',keys=['SOURCE_MAIN','REAL_GUIDE'])", "actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz')")
s=s.replace("name='CONDITIONAL_DIFFUSION_EDIT';pipeline=Pipeline(CFG,ROOT,name);pipeline.set_seeds(seeds['SOURCE_MAIN'],'SOURCE_MAIN')", "name='FULL_COVARIANCE_CALIBRATION';pipeline=Pipeline(CFG,ROOT,name)")
start=s.index('    del pipeline,saved,pre,post');end=s.index('    receipt=',start);s=s[:start]+s[end:]
s=s.replace('complete_source_to_BASE_to_finite_noise_to_50_step_DDPM_to_ADC_replayed', 'complete_source_to_BASE_to_TRAIN_calibration_to_ADC_replayed')
s=s.replace("'REAL_control_guide_and_outside_support_original_ADC_replayed':True,", "'original_source_bin_outside_support_BASE_fallback_replayed':True,")
s=s.replace('assert COUNT==4000','assert COUNT==2000');ast.parse(s);(R/'replay_all.py').write_text(s,'utf8')
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in R.glob('*.py')}
(R/'LOCAL_PREGEN_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entries':files,'ast_pass':all(ast.parse(p.read_text('utf8'))is not None for p in R.glob('*.py'))},indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'source_only':True,'AST':True,'entries_created':['evaluate_saved.py','replay_all.py']}))
