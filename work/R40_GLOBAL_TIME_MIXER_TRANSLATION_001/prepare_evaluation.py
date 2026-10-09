"""Historical R39 helper intentionally disabled; use prepare_evaluation_r40.py."""
raise RuntimeError('R40 evaluation was frozen by prepare_evaluation_r40.py: two new endpoints, 20 HGB, 16 slots, BASE_REAL reuse. This legacy R39 helper is not executable.')
from pathlib import Path
from datetime import datetime,timezone
import os,json,ast,hashlib
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R38_SLICED_MONOTONE_TRANSPORT_001'
s=(OLD/'evaluate_saved.py').read_text('utf8').replace('SLICED_MONOTONE_TRANSPORT','JOINT_KERNEL_FINAL').replace("REAL_ARM='REAL_GUIDE_EDIT'","REAL_ARMS=['BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']")
s=s.replace("'FULL_COVARIANCE_CALIBRATION'];NAMES","'FULL_COVARIANCE_CALIBRATION','SLICED_MONOTONE_TRANSPORT'];NAMES")
s=s.replace('name==REAL_ARM','name in REAL_ARMS')
s=s.replace("CFG['R37_root'])/'metrics.json'","CFG['R38_root'])/'metrics.json'")
s=s.replace("(SOURCE_ARM,ROOT/SOURCE_ARM/'generated.npz')","('SLICED_MONOTONE_TRANSPORT',Path_placeholder/'SLICED_MONOTONE_TRANSPORT/generated.npz'),(SOURCE_ARM,ROOT/SOURCE_ARM/'generated.npz')")
s=s.replace('Path_placeholder',"pathlib.Path(CFG['R38_root'])")
needle="    generation=json.loads((ROOT/'GENERATION_RECEIPT.json').read_text('utf8'))"
extra="""    for name in REAL_ARMS:
        a=load(ROOT/name/'generated.npz',keys=['waveforms_V','reference_ids','reference_groups'])
        assert np.array_equal(a['reference_ids'],real['reference_ids'])and np.array_equal(a['reference_groups'],real['reference_groups'])
        domains[name]=aa.data(a['waveforms_V'],real['reference_ids'],real['reference_groups'],'TRAIN_REFERENCE',ref)
"""
assert s.count(needle)==1;s=s.replace(needle,extra+needle)
s=s.replace("<=8 and COUNTER['density_description_slots']<8","<=24 and COUNTER['density_description_slots']<24")
s=s.replace("['STATISTICS_FIT_RECEIPT.json']","['TRAINING_RECEIPT.json']").replace("/'STATISTICS_FIT_RECEIPT.json'","/'TRAINING_RECEIPT.json'")
s=s.replace("COUNTER['HGB_attempts']==10 and COUNTER['density_description_slots']==8 and COUNTER['density_EM_attempts']<=8","COUNTER['HGB_attempts']==30 and COUNTER['density_description_slots']==24 and COUNTER['density_EM_attempts']<=24")
s=s.replace("'new_REAL_guide_control':False","'two_same_treated_REAL_controls':True")
s=s.replace("'SLICED_vs_BASE'","'JOINT_vs_BASE'").replace("'SLICED_vs_ORDERED'","'JOINT_vs_ORDERED'").replace('R38 saved evaluation','R39 saved evaluation')
needle="    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private"
extra="""    same_treated={}
    for v in views:
        pn=load(OUT/('JOINT_KERNEL_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster'])
        pb=load(OUT/('BASE_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster'])
        same_treated[v]=aa.paired_bootstrap(pn,pb,CFG);same_treated[v]['reference']='BASE_REAL_CONTROL'
    write('SAME_TREATED_REAL_CONTROL_PAIRED.json',same_treated)
"""
assert s.count(needle)==1;s=s.replace(needle,extra+needle)
s=s.replace("return{'results':result,'paired':paired,","return{'results':result,'paired':paired,'same_treated_REAL_control_paired':same_treated,")
(ROOT/'evaluate_saved.py').write_text(s,'utf8')
for p in [ROOT/'evaluate_saved.py',ROOT/'generate_all.py',ROOT/'flow_runtime.py']:ast.parse(p.read_text('utf8'))
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in [ROOT/'evaluate_saved.py',ROOT/'generate_all.py',ROOT/'flow_runtime.py']},'numerical_generation_evaluation_started':False}
(ROOT/'EVALUATION_SOURCE_FREEZE.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps(r),flush=True)
