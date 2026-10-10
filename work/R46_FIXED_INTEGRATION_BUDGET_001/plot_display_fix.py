"""Redraw one saved-data figure with the fixed-support rule in its title."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, shutil, hashlib, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
start=time.monotonic()
figpath=R/'results/figures/long_short_budget_comparison.png'
old=R/'display_initial/long_short_budget_comparison.png'
old.parent.mkdir(exist_ok=True)
assert not old.exists()
shutil.copyfile(figpath,old)
m=json.loads((R/'METRICS_COMPACT.json').read_text('utf8'))['long_short_common64_comparison']
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for i,v in enumerate(['full_historical_shape','normalized_u480']):
    row=m['paired_BA_long_minus_short'][v]
    d=row['BA_delta_first_minus_second'];lo,hi=row['BA_delta_95CI']
    axs[0].errorbar(i,d,yerr=[[d-lo],[hi-d]],fmt='o',color='#d65f23',capsize=5)
axs[0].axhline(0,color='black',ls='--')
axs[0].set_xticks([0,1],['shape493','pure u480'])
axs[0].set(ylabel='LONG minus SHORT BA',title='Paired500 group bootstrap; fitted models fixed')
rows=m['eight_prespecified_distance_differences']
values=[row['relative_change_long_minus_short_over_short']for row in rows.values()]
axs[1].barh(range(8),[100*x if x is not None else 0 for x in values],color=['#d65f23'if x is not None else'#cccccc'for x in values])
axs[1].axvline(10,color='red',ls='--',lw=1);axs[1].axvline(0,color='black',lw=.7)
axs[1].set_yticks(range(8),['PSD W1','RMS W1','ACF1 W1','Q-cond PSD W1','Low center MAE','Low width MAE','High center MAE','High width MAE'])
for i,x in enumerate(values):
    if x is None:axs[1].text(0,i,' n/a',va='center')
axs[1].set(xlabel='Relative distance change (%)',title='Fixed required bins; n/a retained')
fig.tight_layout();fig.savefig(figpath,bbox_inches='tight');plt.close(fig)
record={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,
        'change':'title only: same qualified bins -> fixed required bins; n/a retained',
        'figure':str(figpath),'original_figure':str(old),
        'original_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
        'final_sha256':hashlib.sha256(figpath.read_bytes()).hexdigest(),
        'new_fits_generations_statistics':0,'saved_metrics_only':True}
(R/'DISPLAY_CORRECTION_RECEIPT.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record),flush=True)
