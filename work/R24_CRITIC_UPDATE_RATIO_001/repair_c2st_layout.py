from pathlib import Path
import os,json,time,datetime,shutil
START=time.monotonic();ROOT=Path(__file__).resolve().parent
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
M=json.loads((ROOT/'metrics.json').read_text('utf8'));views=list(M['C2ST']['results']['D5']);xx=np.arange(len(views));COL={'R21_GLOBAL_STYLE':'#9467bd','STANDARDIZED':'#e66101','D5':'#1f77b4'}
folder=ROOT/'results/figures';shutil.copy2(folder/'c2st.png',folder/'c2st_layout_initial.png')
fig,axs=plt.subplots(1,2,figsize=(14,5.3),constrained_layout=True)
for off,n in [(-.24,'R21_GLOBAL_STYLE'),(0,'STANDARDIZED'),(.24,'D5')]:
    r=M['C2ST']['reference_results_reused'][n]if n=='R21_GLOBAL_STYLE'else M['C2ST']['results'][n]
    axs[0].bar(xx+off,[r[v]['balanced_accuracy']for v in views],width=.24,color=COL[n],label='D1 (R23 frozen)'if n=='STANDARDIZED'else n)
axs[0].axhline(.5,c='gray',ls='--');axs[0].set_ylim(.45,1.02);axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=8,frameon=False);axs[0].set_ylabel('independent HGB holdout BA')
ci=M['C2ST']['D5_minus_STANDARDIZED_paired_CI'];d=np.array([ci[v]['BA_delta_first_minus_second']for v in views]);lo=np.array([ci[v]['BA_delta_95CI'][0]for v in views]);hi=np.array([ci[v]['BA_delta_95CI'][1]for v in views]);axs[1].errorbar(xx,d,yerr=np.maximum(0,np.vstack([d-lo,hi-d])),fmt='o',c=COL['D5']);axs[1].axhline(0,c='gray',ls='--');axs[1].set_ylabel('D5 minus frozen D1 BA\n500 paired group bootstrap, fixed classifiers');axs[1].set_title('Selected step0 outputs identical',fontsize=10)
labels=[v.replace('full_historical_shape','full shape').replace('_local','').replace('full_voltage','full V').replace('Q_PSD_2D','Q/PSD 2D')for v in views]
for ax in axs:ax.set_xticks(xx,labels,rotation=40,ha='right')
fig.savefig(folder/'c2st.png',dpi=150,bbox_inches='tight');plt.close(fig)
rec={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'layout_only':True,'metric_fit_or_generation':0,'old_image_retained':'results/figures/c2st_layout_initial.png','change':'legend moved outside bars; labels shortened; no values changed'}
(ROOT/'PLOT_LAYOUT_REPAIR_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec),flush=True)
