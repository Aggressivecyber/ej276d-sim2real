import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullFormatter,FuncFormatter
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
names=['REAL','R3B','FULL_WAVE_STOCHASTIC','PULSE_STOCHASTIC_OFF','FIXED_SOURCE_STOCHASTIC'];data={}
with np.load(ROOT/'results/FEATURES_PRIVATE.npz',allow_pickle=False) as z:
    for n in names:
        u=z[n+'__valid'];data[n]=(z[n+'__Qtotal_mVns'][u],z[n+'__PSD'][u])
lo=min(v[0].min() for v in data.values());hi=max(v[0].max() for v in data.values());yl=min(-.12,min(v[1].min() for v in data.values())-.02);yh=max(.62,max(v[1].max() for v in data.values())+.02)
limits=[float(lo/1.08),float(hi*1.08)];fig,ax=plt.subplots(1,5,figsize=(19,4.2),sharex=True,sharey=True,constrained_layout=True)
for a,n in zip(ax,names):
    q,psd=data[n];a.hexbin(q,psd,xscale='log',gridsize=36,mincnt=1,cmap='viridis',extent=(np.log10(limits[0]),np.log10(limits[1]),yl,yh));a.set(xlim=limits,ylim=(yl,yh),xlabel='Qtotal [mV ns]',title=n.replace('_','\n')+'\nQC='+str(len(q)))
    a.xaxis.set_major_locator(FixedLocator([500,1000,3000,10000]));a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(x):,}'));a.xaxis.set_minor_formatter(NullFormatter());a.tick_params(axis='x',labelsize=8)
ax[0].set_ylabel('signed PSD');fig.suptitle('Same logQ and PSD axes for complete QC-valid populations; fixed-source rows are random model draws')
fig.savefig(ROOT/'results/figures/charge_psd_uniform.png',dpi=160,bbox_inches='tight');plt.close(fig)
r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'source':'saved results/FEATURES_PRIVATE.npz only','output':'results/figures/charge_psd_uniform.png','common_xlim_mV_ns':limits,'common_ylim_PSD':[yl,yh],'counts':{n:len(v[0]) for n,v in data.items()},'metric_recompute':0,'fits':0,'model_inference':0,'EVAL_read':0,'purpose':'shared-axis and tick layout repair only'}
(ROOT/'LAYOUT_REPAIR_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
