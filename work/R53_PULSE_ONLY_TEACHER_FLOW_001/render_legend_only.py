from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, time
R=Path(__file__).resolve().parent;start=time.monotonic()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
name='PSD_all_eight_bins.png';original=R/'figures'/name
dest=R/'display_originals/PSD_all_eight_bins.before_legend_layout.png'
assert not dest.exists();shutil.copy2(original,dest)
M=json.loads((R/'metrics.json').read_text());arm='JOINT_PULSE_RESPONSE'
with np.load(R/'results/FEATURES_PRIVATE.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
plt.rcParams.update({'font.size':10,'figure.dpi':145})
fig,axes=plt.subplots(2,4,figsize=(13.4,6.5),sharex=True)
tables=M['energy_conditional_width']['tables'];names={'REAL':'REAL','BASE':'BASE',arm:'R53 source'}
colors={'REAL':'#111111','BASE':'#888888',arm:'#136bc2'}
for i,a in enumerate(axes.flat):
 failures=[]
 for n in ['REAL','BASE',arm]:
  row=tables[n]['slices'][i];edges=np.linspace(-.1,.6,129)
  q=f[n+'__Qtotal_mVns'];qe=np.asarray(M['energy_conditional_width']['edges_logQ'])
  z=np.log(np.maximum(q,1e-30));bins=np.searchsorted(qe,z,side='right')-1;bins[z==qe[-1]]=7
  hist=np.histogram(f[n+'__PSD'][f[n+'__valid']&(bins==i)],bins=edges)[0].astype(float)
  a.stairs(hist/max(hist.sum(),1)/np.diff(edges),edges,label=names[n],color=colors[n],lw=1.2)
  if not row['valid']:failures.append(names[n])
 a.set_title('Q'+str(i+1)+(' [required]' if i in range(1,7) else ' [edge]'),fontsize=10)
 a.set(xlim=(-.1,.6),xlabel='Signed PSD',ylabel='Density')
 a.text(.02,.96,'Unqualified: '+(', '.join(failures) if failures else 'none'),transform=a.transAxes,va='top',fontsize=7.7,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.963),ncol=3,fontsize=9)
fig.suptitle('All fixed Q intervals retained; failed density fits remain visible',y=.998)
fig.tight_layout(rect=(0,0,1,.91));fig.savefig(original,bbox_inches='tight');plt.close(fig)
rec={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'elapsed_s':time.monotonic()-start,'figure':name,'change':'move shared legend outside panels so Q1 failure annotation remains visible','new_fits':0,'new_inference':0,'scientific_metrics_changed':False,'original_retained':str(dest.relative_to(R))}
(R/'LEGEND_LAYOUT_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec))
