from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,time,json,datetime,resource
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotting import COLORS
names=['REAL','R3B','R10_FIXED','RAW_PCA_GMM','FACTOR_PCA_GMM'];counts={}
with np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False) as z:
    allq=np.concatenate([z[n+'__Qtotal_mVns'][z[n+'__valid']] for n in names]);qlim=(allq.min()*.92,allq.max()*1.08)
    fig,axes=plt.subplots(1,5,figsize=(21,5),sharex=True,sharey=True,layout='constrained')
    for name,ax in zip(names,axes):
        use=z[name+'__valid'];q=z[name+'__Qtotal_mVns'][use];p=z[name+'__PSD'][use];outside=(p<0)|(p>.6)|~np.isfinite(p)
        counts[name]={'QC_n':len(p),'PSD_below0':int((p<0).sum()),'PSD_above0_6':int((p>.6).sum()),'nonfinite_PSD':int((~np.isfinite(p)).sum()),'outside_total':int(outside.sum())}
        ax.scatter(q,p,s=4,alpha=.3,color=COLORS[name]);ax.set_xscale('log');ax.set_xlim(qlim);ax.set_ylim(0,.6);ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.set_xlabel('Observed signed charge [mV ns]');ax.set_ylabel('Signed PSD');ax.set_title(name+'\nQC '+str(len(p))+'; outsidePSD='+str(outside.sum()))
    fig.suptitle('Shared main-body view PSD[0,0.6]; full-range original retained, no metric recalculation');fig.savefig(ROOT/'results/figures/charge_psd_main_body.png',dpi=145,bbox_inches='tight');plt.close(fig)
result={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'source':'saved FEATURES_PRIVATE.npz','shared_PSD_range':[0,.6],'shared_charge_range_mVns':qlim,'domain_counts':counts,'metric_recomputation':0,'model_fits':0,'generation':0,'full_range_figure_retained':True,'threads':4}
(ROOT/'LAYOUT_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
