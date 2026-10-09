from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent
src=(ROOT/'summarize_and_plot.py').read_text('utf8')
training=src[src.index('fig,axs=plt.subplots(2,2,figsize=(13,7))'):src.index("cov=load(ROOT/'results/DEV_COVARIANCE_DESCRIPTION_PRIVATE.npz'")]
training=training.replace("    x=np.arange(1,33)","    x=np.arange(1,33);cc=plt.get_cmap('tab10')(k)")
training=training.replace("lw=.8", "lw=.8,color=cc").replace("lw=.9", "lw=.9,color=cc")
guide=src[src.index("gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz')"):src.index('files={p.name:')]
guide=guide.replace('Had fraction','Fast/slow ')
head='''from __future__ import annotations
import pathlib,json,time,datetime,hashlib,os,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';START=time.monotonic()
A='SLICED_MONOTONE_TRANSPORT';colors={A:'#0867ad'}
m=json.loads((ROOT/'metrics.json').read_text('utf8'));stats=m['training'];tt=np.arange(480)-120
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
prior=OUT/'layout_initial';prior.mkdir(exist_ok=True)
for n in ['training_sliced_maps.png','guide_and_random_decomposition.png']:
    assert not(prior/n).exists();shutil.copyfile(OUT/n,prior/n)
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
'''
tail='''files={n:{'bytes':(OUT/n).stat().st_size,'sha256':hashlib.sha256((OUT/n).read_bytes()).hexdigest()}for n in ['training_sliced_maps.png','guide_and_random_decomposition.png']}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'reason':'same bin color for dotted/solid direction traces; source probe names explicitly Fast/slow, not particle fraction','original_figures_preserved':'results/figures/layout_initial','new_fit_generation':0,'saved_data_only':True}
(ROOT/'DISPLAY_REPAIR_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\\n','utf8');print(json.dumps(receipt),flush=True)
'''
out=head+training+guide+tail
ast.parse(out)
(ROOT/'repair_display.py').write_text(out,'utf8')
print('saved display-only repair entry')
