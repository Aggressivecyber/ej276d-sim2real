from pathlib import Path
import ast,json,os,datetime
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R28_FRESH_PROJECTION_TRANSLATION_001'
def rename_endpoints(s):
    return s.replace('FRESH_SELECTED','ENERGY_SELECTED').replace('FRESH_FINAL','ENERGY_FINAL').replace('HOMO_SELECTED','FRESH_SELECTED').replace('HOMO_FINAL','FRESH_FINAL')
t=rename_endpoints((OLD/'evaluate_endpoints.py').read_text('utf8'))
t=t.replace("ROOT/'FRESH_FINE", "ROOT/'SLICED_ENERGY")
t=t.replace("CFG['R26_root']", "CFG['R28_root']")
t=t.replace('paired_vs_corresponding_R26','paired_vs_corresponding_R28').replace('paired_vs_R26','paired_vs_R28')
t=t.replace("    oldpure=json.loads((pathlib.Path(CFG['R27_root'])/'C_CLASSIFIER_METRICS.json').read_text('utf8'))['results']\n",'')
oldblock="""            if view in CFG['C2ST_views']:
                cp={k:oldpred[control+'__'+view+'__'+k] for k in ['probability','label','cluster']}
                control_result=old['C2ST']['results'][control][view]
            else:
                cp=load(pathlib.Path(CFG['R27_root'])/'results'/(control+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
                control_result=oldpure[control][view]
"""
assert oldblock in t
t=t.replace(oldblock,"""            cp={k:oldpred[control+'__'+view+'__'+k] for k in ['probability','label','cluster']}
            control_result=old['C2ST']['results'][control][view]
""")
t=t.replace("old['C2ST']['RR_definition']", "old['C2ST']['REAL_REAL_reference']")
t=t.replace("online={'FRESH_FINE':common.online_auc('FRESH_FINE'),'HOMOGENEOUS_REORDER':old['online_D_validation_AUC']['HOMOGENEOUS_REORDER']}", "online={'SLICED_ENERGY':common.online_auc('SLICED_ENERGY'),'FRESH_FINE':old['online_D_validation_AUC']['FRESH_FINE']}")
t=t.replace("'training':{'FRESH_FINE':training}", "'training':{'SLICED_ENERGY':training}")
t=t.replace("old['training']['HOMOGENEOUS_REORDER']", "old['training']['FRESH_FINE']")
t=t.replace("'sole_training_intervention':'fresh independent fine projection directions once per G update'", "'sole_training_intervention':'three-diagonal-excluded signed energy U instead of quantile-L1, same actual fresh directions and all other losses'")
t=t.replace('R28 evaluation complete','R29 evaluation complete')
assert "ROOT/'FRESH_FINE" not in t and 'oldpure' not in t
(ROOT/'evaluate_endpoints.py').write_text(t,'utf8');ast.parse(t)
r=rename_endpoints((OLD/'render_figures.py').read_text('utf8'))
r=r.replace("cfg['R26_root']", "cfg['R28_root']").replace('paired_vs_corresponding_R26','paired_vs_corresponding_R28')
r=r.replace("m['training']['FRESH_FINE']", "m['training']['SLICED_ENERGY']")
r=r.replace("n.startswith('FRESH')", "n.startswith('ENERGY')")
r=r.replace("[('fixed',m['control_training_reused'],'#377eb8'),('fresh',m['training']['SLICED_ENERGY'],'#d95f02')]", "[('fresh W1',m['control_training_reused'],'#377eb8'),('energy U',m['training']['SLICED_ENERGY'],'#d95f02')]")
r=r.replace("'Fixed selected2750'", "'Fresh selected'+str(m['control_training_reused']['selected_step'])").replace("'Fixed final2950'", "'Fresh final'+str(m['control_training_reused']['updates'])")
r=r.replace("'Fresh selected2250'", "'Energy selected'+str(m['training']['SLICED_ENERGY']['selected_step'])").replace("'Fresh final2950'", "'Energy final'+str(m['training']['SLICED_ENERGY']['updates'])")
r=r.replace('Fixed\\nSELECTED','Fresh W1\\nSELECTED').replace('Fixed\\nFINAL','Fresh W1\\nFINAL').replace('Fresh\\nSELECTED','Energy U\\nSELECTED').replace('Fresh\\nFINAL','Energy U\\nFINAL')
r=r.replace("[v.replace('full_historical_shape','shape493').replace('_','\\n') for v in views]", "['shape493','full\\nvoltage','prompt\\nlocal','postpeak\\nlocal','tail\\nlocal','baseline\\nlocal','Q / PSD\\n2D','pure u480','seen\\n480 points','complement\\n480 points']")
(ROOT/'render_figures.py').write_text(r,'utf8');ast.parse(r)
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluation_sources_prepared':True,'old_R28_control_recomputed':False,'fits_generation_started':False}),flush=True)
