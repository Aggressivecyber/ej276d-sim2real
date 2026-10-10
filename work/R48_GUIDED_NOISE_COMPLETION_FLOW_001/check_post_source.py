from pathlib import Path
from datetime import datetime,timezone
import ast,json,os,hashlib
R=Path(__file__).resolve().parent;C=json.loads((R/'config_cpu.json').read_text('utf8'))
names=['flow_runtime.py','flow_diagnostics.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_figures.py','render_train.py']
rows=[]
for name in names:
    d=(R/name).read_bytes();tree=ast.parse(d.decode('utf8'));used=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript)and isinstance(node.value,ast.Name)and node.value.id in ['CFG','C','cfg']and isinstance(node.slice,ast.Constant)and isinstance(node.slice.value,str):
            used.append(node.slice.value)
    missing=sorted(set(used)-set(C));assert not missing,(name,missing)
    rows.append({'name':name,'size':len(d),'sha256':hashlib.sha256(d).hexdigest(),'AST_pass':True,'literal_configuration_keys_present':True})
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'rows':rows,'source_only_no_scientific_import':True,
     'scientific_array_read':0,'HGB_EM_neural_forward':0,'frozen_eval_input_root':C['frozen_eval_input_root'],
     'old_REF_paths_not_copied':{k:C[k]for k in ['R7_root','R26_root','R27_root','R32_root','R33_root','R39_root']}}
(R/'POST_ENTRY_AST.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps(rec),flush=True)
