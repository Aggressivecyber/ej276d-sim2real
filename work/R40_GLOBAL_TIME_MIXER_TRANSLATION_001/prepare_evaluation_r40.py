from pathlib import Path
from datetime import datetime,timezone
import json,ast,hashlib,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R39_FULL_JOINT_KERNEL_TRANSLATION_001'
A='GLOBAL_TIME_MIXER_FINAL';R='GLOBAL_TIME_MIXER_REAL_CONTROL'
s=(OLD/'generate_all.py').read_text('utf8').replace('JOINT_KERNEL_FINAL',A).replace('JOINT_KERNEL_REAL_CONTROL',R)
s=s.replace("for name in ['BASE_REAL_CONTROL','"+R+"']:","for name in ['"+R+"']:")
s=s.replace("COUNTER['main_generation_rows']==6000","COUNTER['main_generation_rows']==4000")
needle="    write('REAL_CONTROL_INPUT_BINDING.json'"
start=s.index(needle)
extra='''    # Final-module-zero is descriptive, not another fitted/evaluated candidate.
    pipeline.base.disable_time_mixer=True
    zp,zs,zn=infer(pipeline,source[:128],actual['z'][:128],actual['white'][:128],actual['style'][:128],CFG['generation_chunk'])
    pipeline.base.disable_time_mixer=False
    diffs={'preADC':zp.astype('f8')-pre[:128].astype('f8'),'signal':zs.astype('f8')-signal[:128].astype('f8'),'electronic':zn.astype('f8')-electronic[:128].astype('f8')}
    np.savez_compressed(OUT/'FINAL_MODULE_ZERO_DESCRIPTION_PRIVATE.npz',parent_index=np.arange(128),source_ids=ids[:128],zero_module_preADC_V=zp,zero_module_signal_V=zs,zero_module_electronic_V=zn,full_module_preADC_V=pre[:128],full_module_signal_V=signal[:128],full_module_electronic_V=electronic[:128])
    summary={'rows':128,'same_exact_first128_source_noise':True,'new_fitted_candidate':False,'fit_attempts':0,'not_no_module_training_counterfactual':True,'components':{k:{'full480_RMS_difference_V':ev.summary(np.sqrt(np.mean(v*v,axis=1))),'nonidentical_fraction':float(np.mean(np.any(v!=0,axis=1)))}for k,v in diffs.items()}}
    write('FINAL_MODULE_ZERO_DESCRIPTION.json',summary);COUNTER['module_zero_description_rows']=128
    old=json.loads((pathlib.Path(CFG['R39_root'])/'GENERATION_RECEIPT.json').read_text('utf8'))
    generations['BASE_REAL_CONTROL']={**old['domains']['BASE_REAL_CONTROL'],'reused_from':CFG['old_BASE_REAL_generated'],'new_inference_rows':0,'new_fits':0,'old_independent_replay_from':str(pathlib.Path(CFG['R39_root'])/'REPLAY_RECEIPT.json')}
'''
s=s[:start]+extra+s[start:]
(ROOT/'generate_all.py').write_text(s,'utf8')
s=(OLD/'evaluate_saved.py').read_text('utf8').replace('JOINT_KERNEL_FINAL',A).replace('JOINT_KERNEL_REAL_CONTROL',R)
s=s.replace("NAMES=[*CONTROLS,*ARMS]","NAMES=[*CONTROLS,'BASE_REAL_CONTROL',*ARMS]")
s=s.replace("result['RAW_REAL_REAL']=rrrows;private={};paired={}","result['RAW_REAL_REAL']=rrrows;result['BASE_REAL_CONTROL']=old['C2ST']['results']['BASE_REAL_CONTROL'];private={};paired={'BASE_REAL_CONTROL':old['C2ST']['paired']['BASE_REAL_CONTROL']}")
s=s.replace("pb=load(OUT/('BASE_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz')","pb=load(pathlib.Path(CFG['R39_root'])/'results'/('BASE_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz')")
s=s.replace("['REAL','SOURCE_CLEAN',*CONTROLS]","['REAL','SOURCE_CLEAN','BASE_REAL_CONTROL',*CONTROLS]")
s=s.replace("<=24 and COUNTER['density_description_slots']<24","<=16 and COUNTER['density_description_slots']<16")
s=s.replace("CFG['R38_root'])/'metrics.json'","CFG['R39_root'])/'metrics.json'")
s=s.replace("a=load(ROOT/name/'generated.npz',keys=['waveforms_V','reference_ids','reference_groups'])","path=pathlib.Path(CFG['old_BASE_REAL_generated'])if name=='BASE_REAL_CONTROL'else ROOT/name/'generated.npz'\n        a=load(path,keys=['waveforms_V','reference_ids','reference_groups'])")
s=s.replace("COUNTER['HGB_attempts']==30 and COUNTER['density_description_slots']==24 and COUNTER['density_EM_attempts']<=24","COUNTER['HGB_attempts']==20 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16")
s=s.replace("'covariance_description':cv,'status'","'final_module_zero_description':json.loads((ROOT/'FINAL_MODULE_ZERO_DESCRIPTION.json').read_text('utf8')),'covariance_description':cv,'status'")
s=s.replace('R39 saved evaluation complete','R40 saved evaluation complete')
(ROOT/'evaluate_saved.py').write_text(s,'utf8')
s=(OLD/'replay_all.py').read_text('utf8').replace('JOINT_KERNEL_FINAL',A).replace('JOINT_KERNEL_REAL_CONTROL',R)
s=s.replace("['"+A+"','BASE_REAL_CONTROL','"+R+"']","['"+A+"','"+R+"']")
s=s.replace('COUNT==6000','COUNT==4000')
(ROOT/'replay_all.py').write_text(s,'utf8')
bindings={}
for name in ['generate_all.py','evaluate_saved.py','replay_all.py','flow_runtime.py']:
    p=ROOT/name;b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'sources':bindings,'science_generation_started':False,'new_main_rows':4000,'old_BASE_REAL_regeneration_or_refits':0,'HGB':20,'density_slots':16,'module_zero_description':128}
(ROOT/'EVALUATION_SOURCE_FREEZE.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
