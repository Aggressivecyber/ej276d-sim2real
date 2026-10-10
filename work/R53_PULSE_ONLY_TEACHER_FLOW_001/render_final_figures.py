from pathlib import Path
from datetime import datetime,timezone
import json,os,time,shutil
R=Path(__file__).resolve().parent;start=time.monotonic()
previous=json.loads((R/'render_figures_only.py.NATIVE_RECEIPT.json').read_text())
assert previous['actual_worker_native_exit']==1 and 'TypeError' in previous['worker_self_receipt']['error']
archive=R/'display_originals';archive.mkdir(exist_ok=True)
for name in ['training_and_checks.png','canonical_five_view_response.png']:
    p=R/'figures'/name;dest=archive/(p.stem+'.before_fixed_grid.png');assert not dest.exists();shutil.copy2(p,dest)
C=json.loads((R/'config_post.json').read_text());M=json.loads((R/'metrics.json').read_text())
gate=json.loads((R/'GATE_METRICS.json').read_text());acceptance=json.loads((R/'ACCEPTANCE_A_C.json').read_text())
import render_R53_figures
render_R53_figures.run(R,C,M,gate,acceptance)
rec={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'elapsed_s':time.monotonic()-start,
     'correction':'all display histograms use saved canonical features on the original fixed PSD grid; historical REAL grid was null',
     'new_fits':0,'new_neural_inference':0,'gate_and_metric_files_rewritten':False,
     'both_original_native_failures_preserved':True,'original_partial_images_preserved':True,'status':'complete'}
(R/'PLOT_RECOVERY_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps(rec),flush=True)
