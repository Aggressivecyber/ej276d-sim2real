from pathlib import Path
from datetime import datetime,timezone
import json,os,time
R=Path(__file__).resolve().parent;start=time.monotonic()
original=json.loads((R/'summarize_and_plot.py.NATIVE_RECEIPT.json').read_text())
assert original['actual_worker_native_exit']==1 and original['worker_self_receipt']['error']=="KeyError('histogram_counts')"
archive=R/'display_originals';archive.mkdir(exist_ok=True)
import shutil
for p in (R/'figures').glob('*.png'):
    if p.name in ['training_and_checks.png','canonical_five_view_response.png']:
        dest=archive/(p.stem+'.before_histogram_fix.png');assert not dest.exists();shutil.copy2(p,dest)
C=json.loads((R/'config_post.json').read_text());M=json.loads((R/'metrics.json').read_text())
gate=json.loads((R/'GATE_METRICS.json').read_text());acceptance=json.loads((R/'ACCEPTANCE_A_C.json').read_text())
import render_R53_figures
render_R53_figures.run(R,C,M,gate,acceptance)
rec={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'elapsed_s':time.monotonic()-start,
     'error_corrected':'historical density tables need display histograms from already saved canonical features',
     'new_fits':0,'new_neural_inference':0,'gate_and_metric_files_rewritten':False,
     'original_native_failure_preserved':True,'original_partial_images_preserved':True,'status':'complete'}
(R/'PLOT_RECOVERY_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8')
print(json.dumps(rec),flush=True)
