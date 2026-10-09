import os,json,time
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
ROOT=Path(__file__).resolve().parent;START=time.monotonic();OUT=ROOT/'figures'
M=json.loads((ROOT/'METRICS_COMPACT.json').read_text('utf8'));C=M['classification'];N=['FRESH_FINAL','ENERGY_FINAL','TRAIN_REFERENCE'];V=['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'];VL=['Raw','Global center','Q-bin center','Q-bin center + scale'];COL=['#0072B2','#D55E00','#009E73']
original=OUT/'paired_prompt_exclusion.png';old=OUT/'paired_prompt_exclusion.layout_initial.png'
if old.exists():raise RuntimeError('layout repair already used')
original.rename(old)
fig,axes=plt.subplots(1,4,figsize=(16,4.3),sharey=True,layout='constrained')
for j,v in enumerate(V):
    ax=axes[j]
    for i,n in enumerate(N):
        r=C['comparisons'][n][v]['paired430_minus480'];d=r['BA_delta_first_minus_second'];ci=r['BA_delta_95CI']
        ax.errorbar(d,i,xerr=[[max(0,d-ci[0])],[max(0,ci[1]-d)]],fmt='o',color=COL[i],capsize=3)
    ax.axvline(0,color='.5',ls=':');ax.set_title(VL[j]);ax.set_xlabel('430 − saved480 BA, paired95% CI');ax.grid(axis='x',alpha=.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0].set_yticks(range(3),['Fresh final','Energy final','REAL–REAL']);axes[0].invert_yaxis()
fig.suptitle('Pairing uses identical original holdout events and artificial groups',fontsize=14)
fig.savefig(original,dpi=180,bbox_inches='tight');plt.close(fig)
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'reason':'Reduce crowded x-axis ticks only','original_preserved':str(old),'final_figure':str(original),'new_metrics_or_fits':False}
(ROOT/'LAYOUT_FIX_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps(r),flush=True)
