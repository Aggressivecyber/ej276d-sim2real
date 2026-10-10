"""Bounded engineering adaptation of the already used fixed evaluators to two arms."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent;OLD=R.parent/'R45_EXTENDED_BUDGET_NEURAL_FLOW_001'
s=(OLD/'evaluate_saved.py').read_bytes().decode().replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
while '\n\n\n' in s:s=s.replace('\n\n\n','\n\n')
s=s.replace("ARMS=CFG['endpoints'];SOURCE_ARM='EXTENDED_FLOW_FINAL';REAL_ARMS=['BASE_REAL_CONTROL','EXTENDED_FLOW_REAL_CONTROL']",
 "ARMS=CFG['endpoints'];SOURCE_ARMS=CFG['arms'];REAL_ARMS=['BASE_REAL_CONTROL','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']")
start=s.index('    same_treated={}\n');end=s.index("    np.savez_compressed(OUT/'C2ST_PRIVATE.npz'",start)
s=s[:start]+'''    same_treated={}
    for name in REAL_ARMS[1:]:
        same_treated[name]={}
        for v in views:
            pn=load(OUT/(name+'__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster'])
            pb=load(pathlib.Path(CFG['R39_root'])/'results'/('BASE_REAL_CONTROL__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster'])
            same_treated[name][v]=aa.paired_bootstrap(pn,pb,CFG);same_treated[name][v]['reference']='BASE_REAL_CONTROL'
    write('SAME_TREATED_REAL_CONTROL_PAIRED.json',same_treated)

'''+s[end:]
s=s.replace("COUNTER['density_EM_attempts']+attempt<=16 and COUNTER['density_description_slots']<16",
 "COUNTER['density_EM_attempts']+attempt<=32 and COUNTER['density_description_slots']<32")
s=s.replace("for label,names in [('JOINT_vs_BASE',['BASE',SOURCE_ARM]),('JOINT_vs_ORDERED',['PSD_ORDERED_CFM',SOURCE_ARM]),('all_new_and_BASE',['BASE',*ARMS])]:",
 "for label,names in [(arm+'_vs_BASE',['BASE',arm])for arm in SOURCE_ARMS]+[('two_SOURCE_arms_and_BASE',['BASE',*SOURCE_ARMS]),('all_new_and_BASE',['BASE',*ARMS])]:")
s=s.replace(",('SLICED_MONOTONE_TRANSPORT',pathlib.Path(CFG['R38_root'])/'SLICED_MONOTONE_TRANSPORT/generated.npz'),(SOURCE_ARM,ROOT/SOURCE_ARM/'generated.npz')]:",
 ",('SLICED_MONOTONE_TRANSPORT',pathlib.Path(CFG['R38_root'])/'SLICED_MONOTONE_TRANSPORT/generated.npz')]+[(n,ROOT/n/'generated.npz')for n in SOURCE_ARMS]:")
s=s.replace('[*CONTROLS,SOURCE_ARM]', '[*CONTROLS,*SOURCE_ARMS]')
s=s.replace("['REAL','SOURCE_CLEAN','SOURCE_INPUT',*CONTROLS,SOURCE_ARM]", "['REAL','SOURCE_CLEAN','SOURCE_INPUT',*CONTROLS,*SOURCE_ARMS]")
start=s.index("    gp=load(OUT/'GUIDE_PERTURBATION_PRIVATE.npz'");end=s.index('    from covariance_description import describe_covariance',start)
s=s[:start]+"    gps=json.loads((ROOT/'GUIDE_PERTURBATION_COMPLETE_SUMMARY.json').read_text('utf8'))\n\n"+s[end:]
s=s.replace("describe_covariance(domains,['BASE',SOURCE_ARM],OUT)", "describe_covariance(domains,['BASE',*SOURCE_ARMS],OUT)")
s=s.replace("'training':json.loads((ROOT/SOURCE_ARM/'TRAINING_RECEIPT.json').read_text('utf8'))",
 "'training':{'new_updates':0,'optimizer_loaded':False,'frozen_historical_training':CFG['historical_training'],'LONG_only_primary':True,'selection':'fixed two weights; no DEV checkpoint selection'}")
s=s.replace("metrics['short_budget_R43_comparison']=budget_compare(metrics,CFG,ROOT,SOURCE_ARM)",
 "metrics['long_short_common64_comparison']=budget_compare(metrics,CFG,ROOT)")
s=s.replace("assert COUNTER['HGB_attempts']==20 and COUNTER['density_description_slots']==16 and COUNTER['density_EM_attempts']<=16",
 "assert COUNTER['HGB_attempts']==40 and COUNTER['density_description_slots']==32 and COUNTER['density_EM_attempts']<=32")
s=s.replace("log('R45 saved evaluation complete '", "log('R46 saved evaluation complete '")
s=s.replace("    old=json.loads((pathlib.Path(CFG['R39_root'])/'metrics.json').read_text('utf8'));", "    assert json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']\n    old=json.loads((pathlib.Path(CFG['R39_root'])/'metrics.json').read_text('utf8'));")
assert 'SOURCE_ARM' not in s.replace('SOURCE_ARMS','')
assert 'EXTENDED_FLOW' not in s
ast.parse(s)
(R/'evaluate_saved.py').write_bytes(s.encode())
p=R/'budget_comparison.py';q=p.read_text('utf8').replace("['BA_difference_CI95']", "['BA_delta_95CI']");p.write_bytes(q.encode())
print('R46 two SOURCE/two REAL evaluator adapted; fixed40HGB/32slots, no new science executed')
