"""Static support assembly. Never decode experimental arrays."""
from pathlib import Path
import ast,json,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
apply(P/'support_runtime')
source=(R/'work/R71_REAL_CHARGE_FULL_TRAIN_001/postcheck.py').read_text('utf8');tree=ast.parse(source)
needed={'write','canonical','psd','bins','rms','acf','boundary','summary','compare','migrations'}
lines=['"""Preserved descriptive TRAIN metrics; no model inference at import."""','import json','import numpy as np',
       'from scipy.stats import wasserstein_distance,ks_2samp','from numpy_reference import q','']
lines.extend(ast.get_source_segment(source,node)for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in needed)
(S/'train_metrics.py').write_text('\n\n'.join(lines)+'\n','utf8')
old=(R/'reviews/R71_REAL_CHARGE_FULL_TRAIN_001/collect.py').read_text('utf8')
(P/'collect.py').write_text(old.replace('R71_TERMINAL_OUTPUT','R72_TERMINAL_OUTPUT'),'utf8')
for f in [*S.glob('*.py'),P/'control.py',P/'collect.py']:
    compile(f.read_text('utf8'),str(f),'exec')
print(json.dumps({'static_compile_pass':True,'science_arrays_decoded':False}))
