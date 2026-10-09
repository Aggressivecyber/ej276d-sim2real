"""Redraw saved aggregate C2ST points only; preserves original figure/source."""
from pathlib import Path
import os,time,json,datetime
ROOT=Path(__file__).resolve().parent;start=time.monotonic()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=json.loads((ROOT/'metrics.json').read_text());cfg=json.loads((ROOT/'config.json').read_text());cs=m['C2ST'];views=cfg['C2ST_views'];colors={'GAUSSIAN_BASE':'#6c4c9b','RESIDUAL_FLOW':'#009e73','REAL_REAL':'#777777'}
fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
labels=['full shape','full voltage','prompt local','postpeak local','tail local','baseline local'];plt.rcParams.update({'font.size':10})
for j,(name,res) in enumerate(cs['results'].items()):
    y=np.array([res[v]['balanced_accuracy'] for v in views]);ci=np.array([res[v]['heldout_CI']['balanced_accuracy_95CI'] for v in views])
    axes[0].errorbar(np.arange(6)+.07*(j-1),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=name,color=colors[name])
pair=cs['paired_contrasts']['RESIDUAL_FLOW_minus_GAUSSIAN_BASE'];y=np.array([pair[v]['BA_delta_first_minus_second'] for v in views]);ci=np.array([pair[v]['BA_delta_95CI'] for v in views])
axes[1].errorbar(np.arange(6),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label='RESIDUAL_FLOW minus GAUSSIAN_BASE',color=colors['RESIDUAL_FLOW'])
for ax in axes:ax.set_xticks(np.arange(6),labels,rotation=25,ha='right');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(.4,1.04);axes[0].axhline(.5,color='gray',ls='--');axes[0].set_ylabel('Heldout balanced accuracy');axes[1].axhline(0,color='gray',ls='--');axes[1].set_ylabel('Paired flow minus Gaussian BA; 500-group 95% CI')
fig.suptitle('Same fixed-position common QC; REAL-REAL is a sampling reference')
target=ROOT/'results/figures/c2st_paired_corrected.png';fig.savefig(target,dpi=140,bbox_inches='tight');plt.close(fig)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'saved_aggregate_plot_label_fixed','source':'metrics.json only','output':str(target.relative_to(ROOT)),'original_figure':'c2st_paired.png retains inherited wrong contrast legend','new_fit':0,'new_generation':0,'scientific_metric_recomputation':0,'local_graphics_counted_in_numeric_budget':True}
(ROOT/'PLOT_LABEL_FIX_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
