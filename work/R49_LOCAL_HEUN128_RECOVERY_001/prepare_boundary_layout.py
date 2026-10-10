from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os
root=Path(__file__).resolve().parent
history=root/'display_initial';history.mkdir(exist_ok=True)
assert not(root/'render_boundary_layout.py.used').exists()
for source,target in [('results/figures/boundary_differences.png','boundary_differences.png'),('render_figures.py','render_figures.boundary_initial.py'),('supervise_local.py','supervise_local.boundary_initial.py')]:
    p=history/target
    assert not p.exists(),p
    p.write_bytes((root/source).read_bytes())
src=(root/'supervise_local.py').read_text('utf8')
old="if entry not in CFG['allowed_scientific_entries']:raise RuntimeError('unregistered science entry')"
assert src.count(old)==1
src=src.replace(old,"if entry!='render_boundary_layout.py':raise RuntimeError('unregistered display-only entry')")
old="minimum=expected[entry]+400*1024**2"
assert src.count(old)==1
src=src.replace(old,"expected['render_boundary_layout.py']=500*1024**2\n"+old)
ast.parse(src);(root/'supervise_boundary_layout.py').write_bytes(src.encode('utf8'))
ast.parse((root/'render_boundary_layout.py').read_text('utf8'))
receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'reason':'two-row legend overlapped boundary title','scope':'only display layout; nine stored boundary values and limits unchanged','scientific_jobs_repeated':0,'new_HGB':0,'new_EM':0,'new_generation':0,'previous_PNG':'display_initial/boundary_differences.png','previous_render_source':'display_initial/render_figures.boundary_initial.py','original_supervisor':'display_initial/supervise_local.boundary_initial.py','new_entry':'render_boundary_layout.py','supervisor':'supervise_boundary_layout.py','config_original_unchanged':True,'actual_PID_guard_reused':True,'files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['display_initial/boundary_differences.png','display_initial/render_figures.boundary_initial.py','display_initial/supervise_local.boundary_initial.py','render_boundary_layout.py','supervise_boundary_layout.py']}}
(root/'BOUNDARY_LAYOUT_CORRECTION_001.json').write_bytes((json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode('utf8'))
print(json.dumps(receipt),flush=True)
