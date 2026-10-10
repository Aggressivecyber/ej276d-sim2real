from pathlib import Path
from datetime import datetime, timezone
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
cfg=json.loads((root/'config.json').read_text('utf8'))
m=json.loads((root/'metrics.json').read_text('utf8'))
s,l=cfg['arms'];bd=m['boundary_differences']['results'];keys=list(bd[s])
assert len(keys)==9
colors={'BASE':'#999999',s:'#2474ad',l:'#d45d1c'}
labels={'BASE':'Frozen BASE',s:'FIXED noise',l:'NOISE completion'}
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'legend.fontsize':8,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(12,5.2))
for j,n in enumerate(['BASE',s,l]):
    ax.plot(np.arange(9)+(j-1)*.12,[bd[n][k]['W1_V']*1000 for k in keys],'o-',color=colors[n],label=labels[n])
limits=[max(bd['BASE'][k]['W1_V'],bd['INDEPENDENT_CFM'][k]['W1_V'])+1/16383 for k in keys]
ax.plot(range(9),np.asarray(limits)*1000,'k--',label='Frozen BASE/Independent max +1 LSB')
ax.set_xticks(range(9),[k.replace('first_','D1 ').replace('second_center_','D2 ')for k in keys],rotation=25,ha='right')
ax.set_ylabel('Native-voltage boundary W1 (mV)')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=2)
fig.suptitle('Exactly 9 fixed boundary locations; no smoothing or post-projection',y=.82,fontsize=10)
fig.subplots_adjust(top=.73,bottom=.25)
fig.savefig(root/'results/figures/boundary_differences.png',bbox_inches='tight')
plt.close(fig)
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'only_boundary_layout_redrawn':True,'new_HGB':0,'new_EM':0,'new_generation':0,'scientific_values_reused_unchanged':True}),flush=True)
