from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent
R27=ROOT.parent/'R27_FIXED_PROJECTION_COVERAGE_DIAGNOSTIC_001'
src=(R27/'numeric_worker.py').read_text('utf8');tree=ast.parse(src)
parts=[]
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name in ['means_coverage','fine_distances']:
        parts.append(ast.get_source_segment(src,n))
t='\n\n'.join(parts)
start=t.index("    states={};save={}")
stop=t.index("    comparisons=[*NAMES,'TRAIN_REFERENCE'];",start)
t=t[:start]+"    actual=load(pathlib.Path(CFG['R27_root'])/'results/NEW_PROJECTIONS_PRIVATE.npz')\n    for seed in [20261009271,20261009272,20261009273,20261009274]:\n        label='NEW_'+str(seed);draws[label]={name:actual[label+'__'+name] for name in GROUPS}\n"+t[stop:]
t=t.replace("CFG['quantile_levels']", "{'first':.005,'last':.995,'count':128,'method':'linear'}")
(ROOT/'temporal_common.py').write_text(t+'\n','utf8');ast.parse(t)
# Preserve the numerical classifier, adding a persistence callback before bootstrap.
p=ROOT/'diagnostic_core.py';t=p.read_text('utf8')
assert "'heldout_CI':bootstrap(yt,p,labels,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])" in t
t=t.replace("'heldout_CI':bootstrap(yt,p,labels,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])", "'heldout_CI':None")
t=t.replace('    return result,private\n', "    if globals().get('PERSIST_HGB_CALLBACK') is not None:PERSIST_HGB_CALLBACK(result,private)\n    result['heldout_CI']=bootstrap(yt,p,labels,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])\n    result['fit_and_bootstrap_s']=time.monotonic()-clock\n    return result,private\n")
assert "PERSIST_HGB_CALLBACK" in t
p.write_text(t,'utf8');ast.parse(t)
print('Existing temporal A/B definitions and HGB numerical algorithm reused; save each fit before bootstrap.')
